import time
import shutil
import os
import yaml
import argparse
from abc import ABCMeta, abstractmethod

import numpy as np
import cv2

from utils.perf_monitor import PerfMonitorMixin


class ClassifierAbs(PerfMonitorMixin, metaclass=ABCMeta):
    def __init__(self, model_path, input_shape, device):
        self.model_path = model_path
        self.input_shape = tuple(input_shape)
        self.w, self.h = self.input_shape
        self.device = device
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def letterbox(self, im, fill=(127, 127, 127), auto=False, stride=32):
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = (self.h, self.w)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=fill)  # add border
        return im

    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.letterbox(img)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1]).astype(np.float32)
        return img[None]

    def _postprocess(self, output):
        cls = np.argmax(output)
        cls_prob = ClassifierAbs.softmax(output)
        return cls, cls_prob

    @abstractmethod
    def infer(self, img):
        pass


class ClassifierTorch(ClassifierAbs):
    def __init__(self, model_path, input_shape, device, cfg):
        super().__init__(model_path, input_shape, device)
        import torch
        from utils.weights import load_checkpoint
        self.torch = torch
        from model.model import ClassificationModel
        self.device = device
        self.model = ClassificationModel(cfg, training=False)
        self.model.to(device)
        load_checkpoint(self.model, model_path, strict=True)
        self.model.eval()

    def infer(self, img):
        np_input = self._preprocess(img)
        inp = self.torch.Tensor(np_input).float().to(self.device)
        begin = time.time()
        output = self.model(inp)[0]
        self.update_perf("infer", time.time()-begin)
        np_output = output.cpu().detach().numpy() \
            if output.requires_grad else output.cpu().numpy()
        cls, cls_prob = self._postprocess(np_output)
        return cls, cls_prob

    def infer_batch(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = self._preprocess(imgs[i])
        inp = np.concatenate(imgs, axis=0)
        inp = self.torch.Tensor(inp).float().to(self.device)
        begin = time.time()
        outputs = self.model(inp)
        self.update_perf("infer_batch", time.time()-begin)
        clss, cls_probs = list(), list()
        for output in outputs:
            np_output = output.cpu().detach().numpy() if output.requires_grad else output.cpu().numpy()
            cls, cls_prob = self._postprocess(np_output)
            clss.append(cls)
            cls_probs.append(cls_prob)
        return clss, cls_probs


class ClassifierOnnx(ClassifierAbs):
    def __init__(self, model_path, input_shape, device):
        super().__init__(model_path, input_shape, device)
        import onnxruntime
        print("Start infering using device: %s" % device)
        if device == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise NotImplementedError(f"Device {device} is not implemented!")
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name

    def infer(self, img):
        inp = self._preprocess(img)
        begin = time.time()
        output = self.ort_session.run(None, {self.input_name: inp})[0][0]
        self.update_perf("infer", time.time()-begin)
        cls, cls_prob = self._postprocess(output)
        return cls, cls_prob

    def infer_batch(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = self._preprocess(imgs[i])
        inp = np.concatenate(imgs, axis=0)
        begin = time.time()
        np_outputs = self.ort_session.run(None, {self.input_name: inp})[0]
        self.update_perf("infer_batch", time.time()-begin)
        clss, cls_probs = [], []
        for np_output in np_outputs:
            cls, cls_prob = self._postprocess(np_output)
            clss.append(cls)
            cls_probs.append(cls_prob)
        return clss, cls_probs


def main(opt):
    cfg = None
    if opt.cfg:
        with open(opt.cfg, "r") as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                quit()
    if opt.engine == "torch":
        assert cfg
        from utils.env import init_cuda_cudnn
        init_cuda_cudnn(cfg)

        engine = ClassifierTorch(
            opt.model, (cfg["data"]["input_w"], cfg["data"]["input_h"]),
            opt.device, cfg)
    elif opt.engine == "onnx":
        engine = ClassifierOnnx(opt.model, opt.input_shape, opt.device)
    else:
        raise NotImplementedError("Engine %s is not supported!" % opt.engine)
    print("Image:", opt.img_path)
    img = cv2.imread(opt.img_path)
    cls, cls_prob = engine.infer(img)

    classes = None
    if opt.cls:
        classes = opt.cls
    elif cfg:
        classes = cfg["data"]["cls"]
    print("Result:")
    cls_name = classes[cls] if classes else str(cls)
    print("Classes probability:", cls_prob)
    print("Class: %i (%s), score: %.4f" % (cls, cls_name, cls_prob[cls]))
    PerfMonitorMixin.get_all_perfs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-path',
                        type=str,
                        required=True,
                        help='path to image')
    parser.add_argument('--engine',
                        type=str,
                        default='onnx',
                        help='engine type (onnx, torch)')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='path to model weights')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=(224, 224),
                        help='input shape for classification model')
    parser.add_argument('--cls',
                        type=str,
                        nargs="+",
                        help='class names for classification')
    parser.add_argument('--cfg',
                        type=str,
                        required=True,
                        help='path to config file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='device to run infer on')

    opt = parser.parse_args()
    main(opt)
