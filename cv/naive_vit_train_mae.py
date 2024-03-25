import math
import os

import numpy as np
import tinyimagenet
import torch
import torchvision
import wandb
from model_vit import (  # Assuming that MAE_Decoder is also defined in model_vit.py
    MAE_Mix,
    load_from_config_yaml,
)
from naive_mix_trainer import NaiveMixTrainer
from randomaug import RandAugment
from thop import profile
from torchvision import transforms
from transformers import HfArgumentParser
from utils import setup_seed  # Assuming this is a function to set the random seed


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Convert to PIL image and ensure RGB format
        # image = Image.fromarray(self.hf_dataset[idx]['image']).convert('RGB')

        # assume image is alerady in JPEG format
        image = self.hf_dataset[idx]["image"].convert("RGB")
        image = self.transform(image)

        # Convert image to tensor
        image = torch.tensor(np.array(image))

        label = self.hf_dataset[idx]["label"]
        return image, label


torch.autograd.set_detect_anomaly(True)

from dataclasses import dataclass
from typing import cast


@dataclass
class MixTrainingArguments:
    seed: int = 42
    batch_size: int = 128
    max_device_batch_size: int = 256
    base_learning_rate: float = 1e-3
    weight_decay: float = 0.05
    total_epoch: int = 100
    warmup_epoch: int = 5
    pretrained_model_path: str = None
    output_model_path: str = "test.pt"
    model_config: str = os.path.join(
        os.path.dirname(__file__), "config/cifar10_mix_vit.yml"
    )
    enable_wandb: bool = True
    pt_dataset: str = "tinyimagenet"
    ft_dataset: str = "cifar10"
    device: str = "cuda"
    mask_ratio: float = 0.75
    mode: str = "mix"
    debug: bool = False
    epoch_mix_ratio: float = 0
    loss_mix_alpha: float = 0.5
    k: int = 1
    load_batch_size: int = 256
    profile: bool = False
    ft_epoch: int = 100
    pt_epoch: int = 100
    ft_warmup_epoch: int = 5
    pt_warmup_epoch: int = 20
    from_pretrain: bool = False
    linear_probe: bool = False
    model_name: str = "vit-t"
    dataset_mix_lambda: float = 0.5
    info: str = "test"


def prepare_cifar10():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_validate = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        "data", train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        "data", train=False, download=True, transform=transform_validate
    )
    return train_dataset, val_dataset


def prepare_cifar100():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_validate = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(
        "data", train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR100(
        "data", train=False, download=True, transform=transform_validate
    )
    return train_dataset, val_dataset


def prepare_svhn():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)
            ),
        ]
    )
    transform_validate = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)
            ),
        ]
    )
    train_dataset = torchvision.datasets.SVHN(
        "data", split="train", download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.SVHN(
        "data", split="test", download=True, transform=transform_validate
    )
    return train_dataset, val_dataset


def prepare_tinyimagenet():
    transform_train = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            # Randaug as in paper https://openreview.net/forum?id=KXfjZPL5pqr
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=64,
                scale=(0.6, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            RandAugment(n=2, m=9),
            # call AugMix with small magnitude
            # transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    transform_validate = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = tinyimagenet.TinyImageNet(
        "data", split="train", transform=transform_train
    )
    val_dataset = tinyimagenet.TinyImageNet(
        "data", split="val", transform=transform_validate
    )
    return train_dataset, val_dataset


def prepare_imagenet():
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # Randaug as in paper https://openreview.net/forum?id=KXfjZPL5pqr
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.6, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            RandAugment(n=2, m=9),
            # call AugMix with small magnitude
            # transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # ImageNet stats
        ]
    )
    transform_validate = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # ImageNet stats
        ]
    )
    train_dataset = HFDataset(train_dataset, transform_train)
    val_dataset = HFDataset(val_dataset, transform_validate)
    return train_dataset, val_dataset


def prepare_dataset(dataset: str):
    if dataset == "cifar10":
        return prepare_cifar10()
    elif dataset == "cifar100":
        return prepare_cifar100()
    elif dataset == "svhn":
        return prepare_svhn()
    elif dataset == "tinyimagenet":
        return prepare_tinyimagenet()
    elif dataset == "imagenet":
        return prepare_imagenet()
    else:
        raise f"Unknown dataset: {dataset}"


def main():
    parser = HfArgumentParser(MixTrainingArguments)
    args = cast(MixTrainingArguments, parser.parse_args())
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_dataset, val_dataset = prepare_dataset(args)
    pt_training_dataset, _ = prepare_dataset(args.pt_dataset)
    ft_training_dataset, val_dataset = prepare_dataset(args.ft_dataset)

    pt_training_dataloader = torch.utils.data.DataLoader(
        pt_training_dataset, args.load_batch_size, shuffle=True, num_workers=32
    )
    ft_training_dataloader = torch.utils.data.DataLoader(
        ft_training_dataset, args.load_batch_size, shuffle=True, num_workers=32
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, args.load_batch_size, shuffle=False, num_workers=32
    )

    model = load_from_config_yaml(args.model_config)

    if args.profile:
        input = torch.randn(1, 3, 224, 224).to(device)
        macs, params = profile(model, inputs=(input,))
        macs_readable = f"{macs / 1e9:.2f} G"
        params_readable = f"{params / 1e6:.2f} M"

        print(f"MACs: {macs_readable}, Parameters: {params_readable}")
        exit(0)

    mae_parameter_groups = list(model.encoder.parameters()) + list(
        model.decoder.parameters()
    )
    classifier_parameter_groups = list(model.encoder.parameters()) + list(
        model.classifier.parameters()
    )

    if args.ft_dataset == "tinyimagenet":
        # Pre-Training Optimizer Configuration
        pt_optimizer = torch.optim.AdamW(
            mae_parameter_groups,
            lr=5e-4 * args.load_batch_size / 128,  # base learning rate for pre-training
            betas=(0.9, 0.95),  # beta values for pre-training
            weight_decay=0.15,
        )  # weight decay for pre-training
        # Fine-Tuning Optimizer Configuration
        ft_optimizer = torch.optim.AdamW(
            classifier_parameter_groups,
            lr=1e-3 * args.load_batch_size / 128,
            # base learning rate for fine-tuning
            betas=(0.9, 0.999),  # beta values for fine-tuning
            weight_decay=0.05,
        )  # weight decay for fine-tuning
    else:
        pt_optimizer = torch.optim.AdamW(
            mae_parameter_groups,
            lr=1.5e-4 * args.load_batch_size / 256 * args.k,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        ft_optimizer = torch.optim.AdamW(
            classifier_parameter_groups,
            lr=1e-3 * args.load_batch_size / 256 * args.k,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    if args.pt_epoch == 0:
        # enforce no pretraining
        pt_lr_fn = lambda _: 1
    else:
        pt_lr_fn = lambda epoch: min(
            (epoch + 1) / (args.pt_warmup_epoch + 1e-8),
            0.5 * (math.cos(epoch / args.pt_epoch * math.pi) + 1),
        )

    lr_func_mix = lambda epoch: min(
        (epoch + 1) / (args.ft_warmup_epoch + 1e-8),
        0.5 * (math.cos(epoch / args.ft_epoch * math.pi) + 1),
    )

    pt_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        pt_optimizer, lr_lambda=pt_lr_fn, verbose=True
    )
    ft_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        ft_optimizer, lr_lambda=lr_func_mix, verbose=True
    )

    if args.enable_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="mixtrain",
            name="[{}, warmup, t-{} standard AE {}][{}mix, alpha {}]-{}".format(
                args.info,
                args.from_pretrain,
                args.model_name,
                args.epoch_mix_ratio,
                args.loss_mix_alpha,
                args.pt_dataset,
            ),
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.base_learning_rate,
                "architecture": "ViT-T",
                "dataset": args.ft_dataset,
                "epochs": args.total_epoch,
            },
        )

    def pt_forwardfn(model: MAE_Mix, images, labels):
        features = model.unmasked_generate(images)
        features_truncated = features[:-1, :, :]  # [seq_len, batch_size, embed_dim]
        masked_features, forward_indexes, backward_indexes = (
            model.apply_mask_to_features(features_truncated)
        )
        reconstructed_img, mask = model.generate(masked_features, backward_indexes)
        return features, reconstructed_img, mask

    def ft_forwardfn(model: MAE_Mix, inputs, labels, pt_outputs):
        features, reconstructed_img, mask = pt_outputs
        logits = model.predict_from_features(features)
        return logits

    ft_lossfn = torch.nn.CrossEntropyLoss()
    ft_acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    naive_mix_trainer = NaiveMixTrainer(
        model=model,
        eval_dataset=val_dataloader,
        compute_metrics=ft_acc_fn,
        epoch_mix_ratio=args.epoch_mix_ratio,
        data_mix_lambda=args.dataset_mix_lambda,
        loss_mix_alpha=args.loss_mix_alpha,
        ft_epochs=args.ft_epoch,
        pt_epochs=args.pt_epoch,
        from_pretrained=args.from_pretrain,
        device=device,
        pt_forwardfn=pt_forwardfn,
        pt_lossfn=ft_lossfn,
        pt_optimizer=pt_optimizer,
        pt_scheduler=pt_lr_scheduler,
        pt_dataloader=pt_training_dataloader,
        ft_forwardfn=ft_forwardfn,
        ft_lossfn=ft_lossfn,
        ft_optimizer=ft_optimizer,
        ft_scheduler=ft_lr_scheduler,
        ft_dataloader=ft_training_dataloader,
        same_dataset=(args.ft_dataset == args.pt_dataset),
    )

    naive_mix_trainer.train()


if __name__ == "__main__":
    main()
