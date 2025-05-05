datetime=$(date +"%Y-%m-%d_%H-%M-%S")
exp_dir=./runs/${datetime}_train
mkdir -p ${exp_dir}

CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg ./configs/imagenette/imagenette_mobilenetv3_small.yaml \
    --exp-dir ${exp_dir}

