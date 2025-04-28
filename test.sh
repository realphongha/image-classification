datetime=$(date +"%Y-%m-%d_%H-%M-%S")
weights_path=runs/2025-05-04_17-22-50_train/best.pth
exp_dir=./runs/${datetime}_test
mkdir -p ${exp_dir}

CUDA_VISIBLE_DEVICES=0 python test.py \
    --cfg configs/cifar100/cifar100_mobilenetv3_small.yaml \
    --weights ${weights_path} \
    --exp-dir ${exp_dir}

