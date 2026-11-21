#!/usr/bin/env bash
set -eu

usage() {
    cat <<'EOF'
Usage: bash shell/sweep.sh STUDY MODEL DATASET

Studies:  ratio, weight, scratch, transfer, transfer-weight, linear-probe,
          cross-ratio, cross-lambda, cross-ablation
Models:   resnet18, resnet26, resnet34, vit-ae, vit-mae
Datasets: cifar10, cifar100, svhn, tinyimagenet

Cross-dataset studies train from Tiny ImageNet to CIFAR-10 or CIFAR-100.
Transfer and linear-probe studies start from an ImageNet-pretrained ResNet.
Runs execute sequentially and write checkpoints under ./checkpoints/.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ $# != 3 ]]; then
    usage >&2
    exit 2
fi
study=$1
model=$2
dataset=$3

case "$dataset" in
    cifar10|cifar100|svhn|tinyimagenet) ;;
    *) usage >&2; exit 2 ;;
esac

case "$study" in
    cross-ratio|cross-lambda|cross-ablation)
        if [[ $dataset != cifar10 && $dataset != cifar100 ]]; then
            echo "Cross-dataset studies require a CIFAR target." >&2
            exit 2
        fi
        suffix=_diff_dataset
        ;;
    transfer|transfer-weight|linear-probe)
        if [[ $model != resnet* ]]; then
            echo "Choose a ResNet for ImageNet transfer." >&2
            exit 2
        fi
        suffix= ;;
    ratio|weight|scratch) suffix= ;;
    *) usage >&2; exit 2 ;;
esac

case "$model" in
    resnet18|resnet26|resnet34) entry="resnet_train_standard_ae${suffix}.py"; backbone=$model ;;
    vit-ae) entry="vit_train_standard_ae${suffix}.py"; backbone=vit-t ;;
    vit-mae)
        if [[ -n $suffix ]]; then entry=vit_train_mae_diff_dataset.py; else entry=mix_train.py; fi
        backbone=vit-t ;;
    *) usage >&2; exit 2 ;;
esac

cv_dir=$(cd "$(dirname "$0")/../cv" && pwd)
mkdir -p checkpoints
run() {
    local checkpoint="checkpoints/$study-$model-$dataset-$1.pt"
    shift
    python "$cv_dir/$entry" --model_name "$backbone" --dataset "$dataset" \
        --output_model_path "$checkpoint" "$@"
}

case "$study" in
    ratio|cross-ratio)
        for ratio in 0 0.25 0.5 0.75 1; do
            run "ratio$ratio" --pt_epoch 100 --ft_epoch 100 --mix_ratio "$ratio"
        done
        ;;
    weight)
        for alpha in 0.01 0.1 0.5 0.9 0.99; do
            run "alpha$alpha" --pt_epoch 100 --ft_epoch 100 --mix_ratio 0.75 --mix_loss_alpha "$alpha"
        done
        ;;
    scratch)
        run classifier --pt_epoch 0 --ft_epoch 200 --mix_ratio 0 ;;
    transfer)
        for ratio in 0 0.25 0.5 0.75 1; do
            run "ratio$ratio" --from_pretrain --pt_epoch 0 --ft_epoch 100 --mix_ratio "$ratio"
        done
        ;;
    transfer-weight)
        for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
            run "alpha$alpha" --from_pretrain --pt_epoch 0 --ft_epoch 100 --mix_ratio 0.25 --mix_loss_alpha "$alpha"
        done
        ;;
    linear-probe)
        run classifier --from_pretrain --linear_probe --pt_epoch 0 --ft_epoch 100 --mix_ratio 0 ;;
    cross-lambda)
        for lambda in 0.1 0.3 0.5 0.7 0.9; do
            run "lambda$lambda" --pt_epoch 100 --ft_epoch 100 --mix_ratio 0.25 --lmda "$lambda"
        done
        ;;
    cross-ablation)
        run blend --pt_epoch 100 --ft_epoch 100 --mix_ratio 0.25 --lmda 0.5
        run noise --pt_epoch 100 --ft_epoch 100 --mix_ratio 0.25 --lmda 0.5 --data_mix_noise
        run separate --pt_epoch 100 --ft_epoch 100 --mix_ratio 0.25 --data_no_mix ;;
esac
