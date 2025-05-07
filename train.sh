datetime=$(date +"%Y-%m-%d_%H-%M-%S")
exp_dir=./runs/${datetime}_train
mkdir -p ${exp_dir}

python train.py \
    --cfg ./configs/cifar100/cifar100_mobilenetv3_small.yaml \
    --exp-dir ${exp_dir}

