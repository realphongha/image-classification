datetime=$(date +"%Y-%m-%d_%H-%M-%S")
weights_path=./runs/2026-06-04_18-05-03_train/best.pth
exp_dir=./runs/${datetime}_test
mkdir -p ${exp_dir}

CUDA_VISIBLE_DEVICES=1 python test.py \
    --cfg ./configs/_personview/satudora10k_convnextv2_tiny.yaml \
    --weights ${weights_path} \
    --exp-dir ${exp_dir}

