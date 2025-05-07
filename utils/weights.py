import logging
import torch
import collections
import os


def load_checkpoint(model, checkpoint, strict=False):
    logging.info(f"Loading weights from {checkpoint}...")
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint, weights_only=False, map_location=device)
    checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    source_state_ = checkpoint
    source_state = {}

    target_state = model.state_dict()
    new_target_state = collections.OrderedDict()

    for k in source_state_:
        if k.startswith("_orig_mod."):
            new_k = k[10:]
            source_state[new_k] = source_state_[k]
        elif k.startswith('module.'):
            new_k = k[7:]
            source_state[new_k] = source_state_[k]
        else:
            source_state[k] = source_state_[k]

    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            logging.warning(f'Not found pre-trained parameters for {target_key}')

    model.load_state_dict(new_target_state, strict=strict)

    return model


def save_checkpoint(states, is_best, output_dir, save_all_epoches):
    logging.info("Saving checkpoint to %s..." % os.path.join(output_dir, "last.pth"))
    torch.save(states, os.path.join(output_dir, "last.pth"))
    if save_all_epoches:
        torch.save(states, os.path.join(output_dir, "epoch_%i.pth" % states['epoch']))
    if is_best:
        logging.info("Saving best checkpoint to %s..." % os.path.join(output_dir, "best.pth"))
        torch.save(states, os.path.join(output_dir, "best.pth"))

def simplify_checkpoint(path, out_path):
    checkpoint = torch.load(path, weights_only=False)
    state_dict = checkpoint["state_dict"]
    torch.save(state_dict, out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-o", "--out-path", type=str, required=True)
    args = parser.parse_args()

    simplify_checkpoint(args.path, args.out_path)

