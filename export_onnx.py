import argparse
import yaml

import torch
import numpy as np
import onnx
import onnxsim

from model.model import ClassificationModel
from utils.weights import load_checkpoint


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main(opt, cfg):
    if not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(opt.device)
    model = ClassificationModel(cfg, training=False)
    model.to(device)
    load_checkpoint(model, opt.weights, strict=True)
    if opt.remove_fc:
        model.remove_fc()
    model.eval()

    if opt.dynamic and opt.batch == 1:
        opt.batch = 5
    dummy_input = torch.zeros(opt.batch, 3,
                              cfg["data"]["input_h"],
                              cfg["data"]["input_w"]).to(device)
    # Exports
    torch.onnx.export(
        model, dummy_input, opt.output,
        verbose=False, opset_version=opt.opset,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        } if opt.dynamic else None
    )

    print(f"Exported to {opt.output}!")

    # Checks
    print("Testing onnx model...")

    model_onnx = onnx.load(opt.output)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # print(onnx.helper.printable_graph(model_onnx.graph))

    # Simplifies
    if opt.simplify:
        model_onnx, check = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, opt.output)

    # Checks even more
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(opt.output)

    # computes ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compares ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(model(dummy_input)), ort_outs[0], rtol=1e-03, atol=1e-03)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True,
                        help='path to model weights')
    parser.add_argument('--cfg', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--output', type=str, required=True,
                        help='output file path to export')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true',
                        help='dynamic axes')
    parser.add_argument('--remove-fc', action='store_true',
                        help='remove fully connected layers')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX: opset version')
    parser.add_argument('--simplify', action='store_true', help='simplify model')
    opt = parser.parse_args()

    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()
    main(opt, cfg)

