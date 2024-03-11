#!/bin/bash

# Define lambda values
lmdas=(0.1 0.3 0.5 0.7 0.9)

# Define GPU cores
gpus=(0 1 2 3 4 5 6 7)

# fix mix_ratio to 0.25

# Function to run training for ResNet
run_resnet() {
    local dataset=$1
    local model_name=$2
    local gpu_index=0

    for lmda in "${lmdas[@]}"; do
        CUDA_VISIBLE_DEVICES=${gpus[gpu_index]} python resnet_train_standard_ae_diff_dataset.py --k 1 --mix_ratio 0.25 --lmda $lmda \
        --output_model_path "param_study_standard_ae_${model_name}_from_tinyimagenet_to_${dataset}_mix${lmda/./}.pt" \
        --dataset $dataset --ft_epoch 100 --pt_epoch 100 --enable_wandb --info "diff dataset" ${model_name:+--model_name $model_name} &

        gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
    done
    wait # Wait for all processes to finish
}

# Function to run training for ViT
run_vit() {
    local dataset=$1
    local script_name=$2
    local gpu_index=0

    for lmda in "${lmdas[@]}"; do
        CUDA_VISIBLE_DEVICES=${gpus[gpu_index]} python $script_name --k 1 --mix_ratio 0.25 --lmda $lmda \
        --output_model_path "param_study_${script_name%_*}_${lmda/./}_from_tinyimagenet_to_${dataset}_mix${lmda/./}.pt" \
        --dataset $dataset --ft_epoch 100 --pt_epoch 100 --enable_wandb --info "diff dataset" &

        gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
    done
    wait # Wait for all processes to finish
}

# Run ResNet models
run_resnet cifar10 resnet18
run_resnet cifar10 resnet26
run_resnet cifar10 resnet34
run_resnet cifar100 resnet18
run_resnet cifar100 resnet26
run_resnet cifar100 resnet34

# Run ViT models
run_vit cifar10 vit_train_standard_ae_diff_dataset.py
run_vit cifar10 vit_train_mae_diff_dataset.py
run_vit cifar100 vit_train_standard_ae_diff_dataset.py
run_vit cifar100 vit_train_mae_diff_dataset.py

wait