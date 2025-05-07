# image-classification
Training, testing, inference and export scripts for Image classification with Pytorch.

# Environment and dependencies
- Python 3.10
- `pip install -r requirements.txt`

# Usage
## Prepare a dataset for training/testing
In this format:
```
dataset_name
├── train
│   ├── class1
│   ├── class2
│   └── ...
├── val
│   ├── class1
│   ├── class2
│   └── ...
```
`train` and `val` do not need to be in the same directory. 
You can specify their paths in the config file.
Multiple paths are supported for `train` and `val`.

## Training a Image classification model
See `train.sh`

## Testing a trained model
See `test.sh`

## Exporting a trained model to ONNX
See `export_onnx.py`

## Inference Pytorch or ONNX model
See `infer.py`

# Currently supported:
## Backbones:
- ResNet
- ViT
- MobileNetV3
- ShuffleNetV2
- EfficientNetV2

## Necks:
- GlobalAveragePool

## Heads:
- Linear
- StackedLinear

## Losses:
- CrossEntropy (with weighted)
- FocalLoss

## Optimizers:
- Adam
- AdamW
- SGD

## Schedulers:
(with warmup/freeze supports)
- CosineAnnealingLR
- MultiStepLR
- Cosine

## Sample configs for datasets:
- CIFAR100
- ImageNette
