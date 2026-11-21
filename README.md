# MixTraining: A Better Trade-Off Between Compute and Performance

**MixTraining** overlaps part of self-supervised pretraining with supervised fine-tuning. For image classification, a mixed epoch sends an image through the encoder once and uses its features for both reconstruction and classification. For language tasks, an auxiliary language-modeling loss accompanies the task loss during the first part of fine-tuning. [Paper](https://openreview.net/forum?id=NVpS2g9KRo)

## What it does

- **Controls how much training overlaps.** Given two vision epoch budgets, the trainer merges `floor(mix_ratio × min(pt_epoch, ft_epoch))` pairs of epochs when training from scratch. With budgets of 2 and 2, a ratio of 0 runs four sequential epochs. A ratio of 0.5 runs one reconstruction epoch, one mixed epoch and one classification epoch. [Vision trainer](cv/mix_train.py)
- **Shares image features across objectives.** The mixed loss is `alpha × reconstruction_loss + (1 − alpha) × classification_loss`. ResNets reconstruct pixels from their feature maps. ViT can reconstruct from all patch features or from a masked subset. In cross-dataset runs, each target image is blended with a sampled source image and keeps its target label. [ResNet](cv/model_resnet.py) · [ViT](cv/model_vit.py)
- **Adds language modeling to GLUE fine-tuning.** BERT uses a masked copy of each input for its token loss and the original text for the GLUE task. GPT-2 uses the same hidden states for next-token prediction and the task head. After `mix_end_epoch`, training uses the task loss alone. [Language trainer](nlp/mix_trainer.py)

## Layout

| Path | Contents |
|---|---|
| [`cv/mix_train.py`](cv/mix_train.py) and [`cv/vision_data.py`](cv/vision_data.py) | Vision schedules, optimization, checkpoints and dataset transforms |
| [`cv/mae_pretrain.py`](cv/mae_pretrain.py) and [`cv/train_classifier.py`](cv/train_classifier.py) | Two-stage CIFAR-10 baseline with patch masking before the encoder |
| [`shell/sweep.sh`](shell/sweep.sh) | Vision sweeps for epoch overlap, loss weighting, transfer, linear probing and cross-dataset training |
| [`nlp/run_glue.py`](nlp/run_glue.py) and [`nlp/run_glue.sh`](nlp/run_glue.sh) | BERT and GPT-2 fine-tuning, evaluation and prediction |
| [`tests/`](tests) | Offline checks for gradients, scheduling, checkpoint selection and language-model losses |

## Reproduce

Run the commands from the repository root. The implementation was checked with Python 3.10 and PyTorch 2.7.0. Install the requirements for the experiment you want to run. For NVIDIA GPUs, choose the PyTorch 2.7.0 and torchvision 0.22.0 wheels matching your CUDA setup first. [PyTorch install commands](https://pytorch.org/get-started/previous-versions/#v270)

**Run the vision checks on CPU.** These use small local tensors and temporary checkpoints.

```bash
python -m pip install -r cv/requirements.txt
python -m unittest discover -s tests -p 'test_vision*.py'
python -m unittest discover -s cv -p 'test_*.py'
```

**Train a ViT on CIFAR-10.** This runs one epoch per phase. Compare against sequential training by setting `--mix_ratio 0` and choosing a second output path.

```bash
python cv/mix_train.py --model_name vit-t --dataset cifar10 \
  --pt_epoch 2 --ft_epoch 2 --mix_ratio 0.5 --mix_loss_alpha 0.5 \
  --load_batch_size 32 --output_model_path checkpoints/vit-cifar10.pt
```

The trainer saves the highest-accuracy checkpoint among epochs that trained the classifier. A reconstruction-only run saves the final weights. CIFAR and SVHN use their official test splits for checkpoint selection. Reported accuracy for these runs is measured on the split used to choose the checkpoint. Tiny ImageNet and ImageNet use their validation splits. For ImageNet, supply `--data_root` with `train/<class>/` and `val/<class>/` directories.

**Compare vision settings.** A ratio sweep runs five settings sequentially and gives each checkpoint a separate name. Use `--help` to list the available models, datasets and studies. The cross-dataset presets use Tiny ImageNet as the source and CIFAR as the target. Optional vision tools use `wandb` for run logging, `thop` for MAC counts, and `matplotlib` and `scikit-learn` for [t-SNE plots](cv/explain_tsne_vit.py).

```bash
bash shell/sweep.sh --help
bash shell/sweep.sh ratio resnet18 cifar10
```

**Fine-tune BERT on SST-2.** The command downloads the public model and GLUE data, runs one mixed epoch followed by two task-only epochs, and writes metrics and checkpoints to `results/bert-sst2/`. Use `gpt2` for next-token prediction, or `--mix_end_epoch 0` for task-only fine-tuning.

```bash
python -m pip install -r nlp/requirements.txt
python -m unittest discover -s tests -p 'test_nlp.py'
python nlp/run_glue.py --model_name_or_path bert-base-cased --task_name sst2 \
  --do_train --do_eval --num_train_epochs 3 --mix_end_epoch 1 \
  --per_device_train_batch_size 16 --learning_rate 2e-5 \
  --report_to none --output_dir results/bert-sst2
```

## Reference

Zexin Li, Jiancheng Zhang, Yufei Li, Yinglun Zhu, and Cong Liu. “MixTraining: A Better Trade-Off Between Compute and Performance.” *Transactions on Machine Learning Research*, 2026. [OpenReview](https://openreview.net/forum?id=NVpS2g9KRo) · [arXiv](https://arxiv.org/abs/2502.19513)

```bibtex
@article{li2026mixtraining,
  title   = {{MixTraining}: A Better Trade-Off Between Compute and Performance},
  author  = {Li, Zexin and Zhang, Jiancheng and Li, Yufei and Zhu, Yinglun and Liu, Cong},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=NVpS2g9KRo}
}
```
