from train import *


def main(cfg, opt):
    init_cuda_cudnn(cfg)
    device = cfg["device"]

    os.makedirs(opt.exp_dir, exist_ok=True)
    setup_logger(os.path.join(opt.exp_dir, "test.log"))
    logging.info("Configs:")
    logging.info(cfg)

    model = ClassificationModel(cfg, training=False)
    model.to(device)

    test_ds = Dataset(cfg["data"]["test_path"], False, cfg)
    test_loader = DataLoader(test_ds, batch_size=cfg["test"]["batch_size"],
                             shuffle=False, num_workers=cfg["workers"])

    criterion = get_loss(cfg["train"]["loss"], device, test_ds.cls_count)

    metric = cfg["test"]["metric"]
    assert metric in ("accuracy", "f1")

    pretrained_path = opt.weights
    assert pretrained_path
    load_checkpoint(model, pretrained_path, strict=True)

    # evaluates
    f1, acc, clf_report, loss, conf_matrix = evaluate(
        model, criterion, test_loader, device)
    draw_confusion_matrix(opt.exp_dir, conf_matrix)

    logging.info("Done evaluating!")


if __name__ == "__main__":
    opt = handle_args()
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            quit()
    main(cfg, opt)

