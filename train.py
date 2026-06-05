import os
import argparse
import logging
import yaml
import json
import random

import torch
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, default_collate
import torchvision.transforms.v2 as T
import torchmetrics

from model.model import ClassificationModel
from dataset.dataset import Dataset
from optimizer import get_optimizer
from scheduler import get_scheduler
from losses import get_loss
from utils.env import init_cuda_cudnn, seed_everything
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


def train(model, ema_model, criterion, optimizer, lr_scheduler, scaler, train_loader, device, num_classes):
    logging.info("Training...")
    model.train()
    losses = []
    use_amp = scaler is not None

    for data, label in tqdm(train_loader):
        data = data.to(device)
        if len(label.shape) == 1:
            label = label.long().to(device)
        else:
            assert len(label.shape) == 2
            label = label.float().to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            output = model(data)
            loss = criterion(output, label)

        losses.append(loss.item())
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        ema_model.update_parameters(model)

    mean_loss = np.mean(losses)
    logging.info(f"Loss: {mean_loss}")
    lr_scheduler.step()

    return mean_loss


def evaluate(model, criterion, val_loader, device, num_classes):
    logging.info("Evaluating...")
    model.eval()
    losses = []

    acc_metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes).to(device)
    f1_metric = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes).to(device)

    with torch.no_grad():
        for data, label in tqdm(val_loader):
            data = data.to(device)
            label = label.long().to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(loss.item())
            preds = torch.argmax(output, dim=1)
            acc_metric.update(preds, label)
            f1_metric.update(preds, label)

    mean_loss = np.mean(losses)
    acc = acc_metric.compute().item()
    f1 = f1_metric.compute().item()
    logging.info(f"Loss: {mean_loss}")
    logging.info(f"Accuracy: {acc}")
    logging.info(f"Macro avg F1 score: {f1}")

    conf_matrix = None

    return f1, acc, mean_loss, conf_matrix


def main(cfg, opt):
    seed = cfg.get("seed", 42)
    seed_everything(seed)

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

    num_classes = len(cfg["data"]["cls"])

    model = ClassificationModel(cfg, training=True)
    model.to(device)

    train_ds = Dataset(cfg["data"]["train_path"], True, cfg)
    val_ds = Dataset(cfg["data"]["val_path"], False, cfg)

    cutmix = T.CutMix(num_classes=num_classes)
    mixup = T.MixUp(num_classes=num_classes)
    cutmix_or_mixup = T.RandomChoice([cutmix, mixup])

    assert cfg["data"]["augmentation"]["cutmix_mixup"]["prob"] <= 0.0 or \
           cfg["train"]["loss"]["name"] in ("ce", "poly")
    def collate_fn(batch):
        if random.random() < cfg["data"]["augmentation"]["cutmix_mixup"]["prob"]:
            return cutmix_or_mixup(*default_collate(batch))
        return default_collate(batch)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=cfg["workers"],
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["test"]["batch_size"],
                            shuffle=False,
                            num_workers=cfg["workers"])

    optimizer = get_optimizer(model, cfg["train"]["optimizer"])

    criterion = get_loss(cfg["train"]["loss"], device, train_ds.cls_count)

    metric = cfg["test"]["metric"]
    assert metric in ("accuracy", "f1")

    begin_epoch = 0
    last_epoch = -1
    best_acc = -1
    train_loss = []
    val_loss = []
    saved_scheduler_state = None

    scaler = torch.amp.GradScaler(enabled=cfg["train"].get("amp", False))

    if os.path.isdir(opt.exp_dir) and os.path.isfile(os.path.join(opt.exp_dir, "configs.txt")):
        ckpt_file = os.path.join(opt.exp_dir, "last.pth")
        assert os.path.isfile(ckpt_file), "Exp dir exists but no checkpoint found!"
        logging.info(f"Loading checkpoint from {ckpt_file}...")
        checkpoint = torch.load(ckpt_file, map_location=device)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        best_acc, train_loss, val_loss = best_perf
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            saved_scheduler_state = checkpoint['lr_scheduler']
        if 'scaler' in checkpoint and cfg["train"].get("amp", False):
            scaler.load_state_dict(checkpoint['scaler'])
        logging.info(f"Loaded checkpoint '{ckpt_file}' (epoch {checkpoint['epoch']})")
    else:
        os.makedirs(opt.exp_dir, exist_ok=True)
        with open(os.path.join(opt.exp_dir, "configs.txt"), "w") as output_file:
            json.dump(cfg, output_file, indent=4)

    epochs = cfg["train"]["epochs"]
    lr_scheduler = get_scheduler(cfg["train"]["lr_scheduler"], optimizer,
                                 last_epoch, epochs)

    if saved_scheduler_state is not None:
        lr_scheduler.load_state_dict(saved_scheduler_state)

    pretrained_path = opt.weights
    if not pretrained_path:
        pretrained_path = cfg["train"].get("pretrained", None)
    if pretrained_path:
        load_checkpoint(model, pretrained_path, strict=False)

    wait_for_unfreeze = False
    if cfg["train"]["freeze"]:
        assert isinstance(cfg["train"]["freeze"], list)
        model.freeze(cfg["train"]["freeze"])
        wait_for_unfreeze = cfg["train"]["unfreeze_after"]

    compile_mode = cfg["train"].get("compile", None)
    if compile_mode and not wait_for_unfreeze:
        logging.info(f"Compiling model in {compile_mode} mode...")
        model = torch.compile(model, mode=compile_mode)

    ema_decay_rate = cfg["train"]["ema"]["decay_rate"]
    ema_model = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay_rate),
        device=device,
        use_buffers=True,
    )
    if os.path.isfile(os.path.join(opt.exp_dir, "ema.pth")):
        logging.info("Loading EMA model from %s..." % os.path.join(opt.exp_dir, "ema.pth"))
        ema_model.load_state_dict(torch.load(os.path.join(opt.exp_dir, "ema.pth")))

    for epoch in range(begin_epoch, epochs):
        logging.info(f"EPOCH {epoch}:")

        if wait_for_unfreeze and epoch >= wait_for_unfreeze:
            model.free(cfg["train"]["freeze"])
            wait_for_unfreeze = False
            scaler = torch.amp.GradScaler(enabled=cfg["train"].get("amp", False))
            if compile_mode:
                logging.info(f"Compiling model in {compile_mode} mode...")
                model = torch.compile(model, mode=compile_mode)

        loss = train(model, ema_model, criterion, optimizer, lr_scheduler,
                     scaler, train_loader, device, num_classes)
        train_loss.append(loss)

        logging.info(f"Evaluating the current model...")
        f1, acc, loss, _ = evaluate(model, criterion, val_loader, device, num_classes)
        if metric == "accuracy":
            m = acc
        elif metric == "f1":
            m = f1
        val_loss.append(loss)
        best_model = False
        if m > best_acc:
            best_acc = m
            best_model = True

        ema_model.train()
        torch.optim.swa_utils.update_bn(train_loader, ema_model, device=device)
        logging.info(f"Evaluating the EMA model...")
        ema_f1, ema_acc, ema_loss, _ = evaluate(
            ema_model, criterion, val_loader, device, num_classes)
        if metric == "accuracy" and ema_acc > best_acc:
            best_acc = ema_acc
        elif metric == "f1" and ema_f1 > best_acc:
            best_acc = ema_f1

        wandb.log({
            "train_loss": train_loss[-1],
            "val_loss": val_loss[-1],
            "accuracy": acc,
            "f1": f1,
            "ema_accuracy": ema_acc,
            "ema_f1": ema_f1,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

        save_checkpoint({
            'epoch': epoch + 1,
            'perf': (best_acc, train_loss, val_loss),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
        }, best_model, opt.exp_dir, cfg["train"]["save_all_epochs"])
        torch.save(ema_model.state_dict(), os.path.join(opt.exp_dir, "ema.pth"))

    logging.info("Done training!")
    logging.info("Best %s: %.4f" % (metric, best_acc))


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        type=str,
                        default="configs/cifar100/cifar100_mobilenetv3_small.yaml",
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
