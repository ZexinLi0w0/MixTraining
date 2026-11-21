"""Training utilities for the two-stage CIFAR-10 ViT baseline."""

import argparse
import math
from pathlib import Path

import torch
from torchvision import datasets, transforms
from tqdm import tqdm


def training_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_device_batch_size', type=int)
    parser.add_argument('--base_learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int)
    parser.add_argument('--warmup_epoch', type=int)
    parser.add_argument('--dataset', choices=['cifar10'], default='cifar10')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


def make_loader(args, train, image_size=32):
    microbatch = min(args.batch_size, args.max_device_batch_size)
    if microbatch <= 0 or args.batch_size % microbatch:
        raise ValueError('Choose a positive microbatch size that divides the effective batch size.')
    preprocessing = [transforms.Resize((image_size, image_size))] if image_size != 32 else []
    transform = transforms.Compose(preprocessing + [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    data = datasets.CIFAR10(args.data_root, train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(data, microbatch, shuffle=train, num_workers=args.num_workers)


def optimizer_and_scheduler(model, args, betas):
    if args.total_epoch < 1 or args.warmup_epoch < 0:
        raise ValueError('total_epoch must be positive and warmup_epoch must be nonnegative.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256,
                                  betas=betas, weight_decay=args.weight_decay)

    def lr_multiplier(epoch):
        cosine = (1 + math.cos(math.pi * epoch / args.total_epoch)) / 2
        return min((epoch + 1) / args.warmup_epoch, cosine) if args.warmup_epoch else cosine

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def run_epoch(model, loader, device, objective, optimizer=None, steps_per_update=1):
    """An objective returns a mean loss and a count of correct predictions."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen, window = 0.0, 0, 0, 0
    if training:
        optimizer.zero_grad()
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(tqdm(loader), start=1):
            loss, hits = objective(model, images.to(device), labels.to(device))
            count = len(images)
            total_loss += loss.item() * count
            correct += int(hits)
            seen += count
            if training:
                (loss * count).backward()
                window += count
                if step % steps_per_update == 0 or step == len(loader):
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.div_(window)
                    optimizer.step()
                    optimizer.zero_grad()
                    window = 0
    return total_loss / seen, correct / seen


def save_checkpoint(path, model_name, model_kwargs, model):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_name': model_name, 'model_kwargs': model_kwargs, 'state_dict': model.state_dict()}, path)
