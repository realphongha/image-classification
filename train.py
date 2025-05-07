import os
import argparse
import logging
import yaml
import json
import random

import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, \
    f1_score, confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, default_collate
import torchvision.transforms.v2 as T

from model.model import ClassificationModel
from dataset.dataset import Dataset
from optimizer import get_optimizer
from scheduler import get_scheduler
from losses import get_loss
from utils.env import init_cuda_cudnn
from utils.log import setup_logger
from utils.weights import load_checkpoint, save_checkpoint


def draw_confusion_matrix(output_path, conf_matrix):
    fig = plt.figure()
    df_cm = pd.DataFrame(
        conf_matrix, range(conf_matrix.shape[0]), range(conf_matrix.shape[0]))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='.3f')
    fig.savefig(os.path.join(output_path, 'confusion_matrix.png'),
                bbox_inches='tight')
    plt.close("all")


def train(model, criterion, optimizer, lr_scheduler, train_loader, device):
    logging.info("Training...")
    model.train()
    losses = []

    for data, label in tqdm(train_loader):
        data = data.float().to(device)
        if len(label.shape) == 1:
            label = label.long().to(device)
        else:
            assert len(label.shape) == 2
            label = label.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    mean_loss = np.mean(losses)
    logging.info(f"Loss: {mean_loss}")
    lr_scheduler.step()

    return mean_loss


def evaluate(model, criterion, val_loader, device):
    logging.info("Evaluating...")
    model.eval()
    losses = []
    pred = []
    gt = []

    for data, label in tqdm(val_loader):
        data = data.float().to(device)
        label = label.long().to(device)
        output = model(data)
        loss = criterion(output, label)
        output_label = torch.max(output, 1).indices.cpu().detach().numpy()
        gt_label = label.cpu().detach().numpy()
        losses.append(loss.item())
        for i in range(output_label.shape[0]):
            pred.append(round(output_label[i]))
            gt.append(round(gt_label[i]))

    mean_loss = np.mean(losses)
    logging.info(f"Loss: {mean_loss}")
    acc = accuracy_score(gt, pred)
    logging.info(f"Accuracy: {acc}")
    f1 = f1_score(gt, pred, average="macro")
    logging.info(f"Macro avg F1 score: {f1}")
    clf_report = classification_report(gt, pred)
    logging.info(f"Classification report:\n {clf_report}")
    conf_matrix = confusion_matrix(gt, pred, normalize="true")

    return f1, acc, clf_report, mean_loss, conf_matrix


def main(cfg, opt):
    # configs
    exp_name = cfg["data"]["name"] + "_" + os.path.split(opt.exp_dir)[-1]
    wandb.init(
        project="image-classification",
        name=exp_name,
        config=cfg
    )
    init_cuda_cudnn(cfg)
    device = cfg["device"]
    os.makedirs(opt.exp_dir, exist_ok=True)
    setup_logger(os.path.join(opt.exp_dir, "train.log"))
    logging.info("Configs:")
    logging.info(cfg)

    # init model
    model = ClassificationModel(cfg, training=True)
    model.to(device)

    # datasets
    train_ds = Dataset(cfg["data"]["train_path"], True, cfg)
    val_ds = Dataset(cfg["data"]["val_path"], False, cfg)

    cutmix = T.CutMix(num_classes=len(cfg["data"]["cls"]))
    mixup = T.MixUp(num_classes=len(cfg["data"]["cls"]))
    cutmix_or_mixup = T.RandomChoice([cutmix, mixup])

    assert cfg["data"]["augmentation"]["cutmix_mixup"]["prob"] <= 0.0 or \
           cfg["train"]["loss"]["name"] in ("ce", )
    def collate_fn(batch):
        if random.random() > cfg["data"]["augmentation"]["cutmix_mixup"]["prob"]:
            return cutmix_or_mixup(*default_collate(batch))
        return default_collate(batch)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=cfg["workers"],
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["test"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["workers"])

    # optimizer
    optimizer = get_optimizer(model, cfg["train"]["optimizer"])

    # loss
    criterion = get_loss(cfg["train"]["loss"], device, train_ds.cls_count)

    metric = cfg["test"]["metric"]
    assert metric in ("accuracy", "f1")

    begin_epoch = 0
    last_epoch = -1
    best_acc = -1
    best_clf_report = None
    best_conf_matrix = None
    train_loss = []
    val_loss = []

    # load checkpoint
    if os.path.isdir(opt.exp_dir) and os.path.isfile(os.path.join(opt.exp_dir, "configs.txt")):
        ckpt_file = os.path.join(opt.exp_dir, "last.pth")
        assert os.path.isfile(ckpt_file), "Exp dir exists but no checkpoint found!"
        output_path = os.path.split(ckpt_file)[0]
        logging.info(f"Loading checkpoint from {ckpt_file}...")
        checkpoint = torch.load(ckpt_file, map_location=device)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        best_acc, best_clf_report, best_conf_matrix, train_loss, val_loss = best_perf
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(f"Loaded checkpoint '{ckpt_file}' (epoch {checkpoint['epoch']})")
    else:
        os.makedirs(opt.exp_dir, exist_ok=True)
        with open(os.path.join(opt.exp_dir, "configs.txt"), "w") as output_file:
            json.dump(cfg, output_file, indent=4)

    epochs = cfg["train"]["epochs"]
    # scheduler
    lr_scheduler = get_scheduler(cfg["train"]["lr_scheduler"], optimizer,
                                 last_epoch, epochs)

    pretrained_path = opt.weights
    if not pretrained_path:
        pretrained_path = cfg["train"].get("pretrained", None)
    if pretrained_path:
        load_checkpoint(model, pretrained_path, strict=False)

    # freeze parts
    wait_for_unfreeze = False
    if cfg["train"]["freeze"]:
        assert isinstance(cfg["train"]["freeze"], list)
        model.freeze(cfg["train"]["freeze"])
        wait_for_unfreeze = cfg["train"]["unfreeze_after"]

    # compiles model
    compile = cfg["train"].get("compile", None)
    if compile and not wait_for_unfreeze:
        logging.info(f"Compiling model in {compile} mode...")
        model = torch.compile(model, mode=compile)

    # train loop
    for epoch in range(begin_epoch, epochs):
        logging.info(f"EPOCH {epoch}:")

        if wait_for_unfreeze and epoch >= wait_for_unfreeze:
            model.free(cfg["train"]["freeze"])
            wait_for_unfreeze = False
            logging.info(f"Compiling model in {compile} mode...")
            model = torch.compile(model, mode=compile)

        # trains
        loss = train(model, criterion, optimizer, lr_scheduler,
                     train_loader, device)
        train_loss.append(loss)

        # evaluates
        f1, acc, clf_report, loss, conf_matrix = evaluate(
            model, criterion, val_loader, device)
        if metric == "accuracy":
            m = acc
        elif metric == "f1":
            m = f1
        val_loss.append(loss)
        best_model = False
        if m > best_acc:
            best_acc = m
            best_clf_report = clf_report
            best_conf_matrix = conf_matrix
            best_model = True

        # wandb logging
        wandb.log({
            "train_loss": train_loss[-1],
            "val_loss": val_loss[-1],
            "accuracy": acc,
            "f1": f1,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

        # saves checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'perf': (best_acc, best_clf_report, best_conf_matrix, train_loss, val_loss),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, best_model, opt.exp_dir, cfg["train"]["save_all_epochs"])
        draw_confusion_matrix(opt.exp_dir, best_conf_matrix)

    logging.info("Done training!")
    logging.info("Best %s: %.4f" % (metric, best_acc))
    logging.info(f"Classification report:\n {best_clf_report}")


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        type=str,
                        default="configs\cifar100_mobilenetv3\cifar100.yaml",
                        help='path to config file')
    parser.add_argument('--exp-dir',
                        type=str,
                        required=True,
                        help='path to experiment directory')
    parser.add_argument('--weights',
                        type=str,
                        default="",
                        help='path to pretrained weights')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = handle_args()
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            quit()

    main(cfg, opt)
