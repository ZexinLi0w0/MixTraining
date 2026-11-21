"""MixTraining for vision: reconstruction, shared-feature mixed epochs, and classification."""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from model_resnet import MixModel, SingletonBatchNorm2d
from model_vit import MAE_Encoder, MAE_Mix, MAE_ViT, ViT_Classifier
from utils import setup_seed
from vision_data import IMAGE_SIZE, NUM_CLASSES, make_dataset

MODELS = ("vit-t", "resnet18", "resnet26", "resnet34")


def epoch_schedule(pt_epochs, ft_epochs, mix_ratio, from_pretrain=False):
    available = ft_epochs if from_pretrain else min(pt_epochs, ft_epochs)
    mixed = math.floor(available * mix_ratio)
    pretrain = 0 if from_pretrain else pt_epochs - mixed
    return ["ssl"] * pretrain + ["mix"] * mixed + ["sl"] * (ft_epochs - mixed)


def build_model(name, kwargs):
    if name == "vit-t":
        return MAE_Mix(**kwargs)
    if name == "vit-t-mae-pretrain":
        return MAE_ViT(**kwargs)
    if name == "vit-t-classifier":
        kwargs = kwargs.copy()
        return ViT_Classifier(MAE_Encoder(**kwargs.pop("encoder_kwargs", {})), **kwargs)
    if name in MODELS:
        return MixModel(**kwargs)
    raise ValueError(f"Unknown model {name!r}")


def load_checkpoint(path, device="cpu", *, trusted_legacy=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=not trusted_legacy)
    if trusted_legacy and isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(device)
    model = build_model(checkpoint["model_name"], checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def get_batches(phase, source, target, args):
    """Yield reconstruction inputs, classification inputs, and target labels."""
    if phase != "mix" or source is target:
        loader = source if phase == "ssl" else target
        for images, labels in loader:
            images = images.to(args.device)
            yield images, images, labels.to(args.device)
        return
    # Draw one independent source image for every target, including its final partial batch.
    sampler = RandomSampler(source.dataset, replacement=True, num_samples=len(target.dataset))
    paired_source = DataLoader(source.dataset, sampler=sampler, batch_size=target.batch_size,
                               num_workers=source.num_workers)
    for (src, _), (dst, labels) in zip(paired_source, target):
        src, dst, labels = src.to(args.device), dst.to(args.device), labels.to(args.device)
        if args.data_mix_noise:
            src = torch.randn_like(src)
        if args.data_no_mix:
            yield src, dst, labels
        else:
            images = (1 - args.lmda) * src + args.lmda * dst
            yield images, images, labels


def train_epoch(model, source, target, optimizers, phase, args):
    model.train()
    if args.linear_probe:
        model.backbone.eval()  # Keep frozen batch-normalization statistics.
    pt_optimizer, ft_optimizer = optimizers
    alpha = {"ssl": 1.0, "mix": args.mix_loss_alpha, "sl": 0.0}[phase]
    ssl_sum = sl_sum = correct = count = 0
    batches = get_batches(phase, source, target, args)
    for ssl_images, sl_images, labels in tqdm(batches, desc=phase):
        model.zero_grad(set_to_none=True)
        features = None
        n = len(labels)
        loss = 0
        if alpha > 0:
            features = model.get_features(ssl_images)
            reconstruction, mask = model.reconstruct_features(features)
            if mask is None:
                ssl_loss = F.mse_loss(reconstruction, ssl_images)
            else:
                ssl_loss = ((reconstruction - ssl_images).square() * mask).sum() / mask.sum()
            ssl_sum += ssl_loss.item() * n
            loss = alpha * ssl_loss
        if alpha < 1:
            if features is None or sl_images is not ssl_images:
                features = model.get_features(sl_images)
            logits = model.predict_from_features(features)
            sl_loss = F.cross_entropy(logits, labels)
            sl_sum += sl_loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()
            loss = loss + (1 - alpha) * sl_loss
        loss.backward()
        if alpha < 1:
            ft_optimizer.step()
            # Both optimizers own the backbone. Clear its gradient before
            # stepping the reconstruction decoder so each parameter steps once.
            ft_optimizer.zero_grad(set_to_none=True)
        if alpha > 0:
            pt_optimizer.step()
        count += n
    metrics = {}
    if alpha > 0:
        metrics["ssl_loss"] = ssl_sum / count
    if alpha < 1:
        metrics.update(sl_loss=sl_sum / count, train_acc=correct / count)
    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = correct = count = 0
    for images, labels in loader:
        labels = labels.to(device)
        logits = model.predict(images.to(device))
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return {"val_loss": total_loss / count, "val_acc": correct / count}


def make_optimizers(model, args):
    if args.model_name == "vit-t":
        backbone, head = model.encoder, model.classifier
    else:
        backbone, head = model.backbone, model.head
    pt_params = [*backbone.parameters(), *model.decoder.parameters()]
    ft_params = [*backbone.parameters(), *head.parameters()]
    if args.model_name != "vit-t":
        if args.from_pretrain:
            return tuple(torch.optim.AdamW(p, lr=1e-5, weight_decay=5e-4) for p in (pt_params, ft_params))
        return tuple(torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=5e-4) for p in (pt_params, ft_params))

    batch_scale = args.load_batch_size / 256
    if args.dataset == "tinyimagenet":
        return (torch.optim.AdamW(pt_params, lr=1e-3 * batch_scale, betas=(0.9, 0.95), weight_decay=0.15),
                torch.optim.AdamW(ft_params, lr=2e-3 * batch_scale, betas=(0.9, 0.999), weight_decay=0.05))
    return (torch.optim.AdamW(pt_params, lr=1.5e-4 * batch_scale * args.k, betas=(0.9, 0.95), weight_decay=args.weight_decay),
            torch.optim.AdamW(ft_params, lr=args.base_learning_rate * batch_scale * args.k, betas=(0.9, 0.95), weight_decay=args.weight_decay))


def lr_schedule(epoch, warmup, total):
    return min((epoch + 1) / max(1, warmup), 0.5 * (1 + math.cos(epoch * math.pi / max(total, 1))))


def parse_args(default_model, default_source, default_mask):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", choices=MODELS, default=default_model)
    p.add_argument("--dataset", choices=IMAGE_SIZE, default="cifar10", help="supervised target dataset")
    p.add_argument("--source_dataset", choices=IMAGE_SIZE, default=default_source, help="defaults to the target")
    p.add_argument("--data_root", default="data")
    p.add_argument("--pt_epoch", type=int, default=100)
    p.add_argument("--ft_epoch", type=int, default=100)
    p.add_argument("--mix_ratio", type=float, default=0.0, help="fraction of overlapping SSL/SL epochs")
    p.add_argument("--mix_loss_alpha", type=float, default=0.5, help="SSL weight in the mixed objective")
    p.add_argument("--lmda", type=float, default=0.5, help="target image weight in cross-dataset batches")
    p.add_argument("--mask_ratio", type=float, default=default_mask, help="fraction of ViT patch features masked; zero selects AE")
    p.add_argument("--data_no_mix", action="store_true", help="use separate source and target forward passes")
    p.add_argument("--data_mix_noise", action="store_true", help="replace source images by Gaussian noise during mixing")
    p.add_argument("--load_batch_size", "--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weight_decay", type=float, default=0.05, help="ViT weight decay on CIFAR and SVHN")
    p.add_argument("--base_learning_rate", type=float, default=1e-3, help="ViT supervised base rate on CIFAR and SVHN")
    p.add_argument("--k", type=int, default=1, help="ViT learning-rate multiplier on CIFAR and SVHN")
    p.add_argument("--pt_warmup_epoch", type=int, default=20)
    p.add_argument("--ft_warmup_epoch", type=int, default=5)
    p.add_argument("--from_pretrain", action="store_true", help="transfer from pretrained ResNet weights or a supplied checkpoint")
    p.add_argument("--linear_probe", action="store_true", help="train the classification head of a pretrained ResNet")
    p.add_argument("--pretrained_model_path", help="initialize from a MixTraining checkpoint of the same architecture")
    p.add_argument("--trusted_legacy_checkpoint", action="store_true", help="allow loading a trusted full-model pickle")
    p.add_argument("--output_model_path", default="test.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable_wandb", action="store_true")
    p.add_argument("--info", default="", help="optional W&B run name")
    p.add_argument("--profile", action="store_true", help="print inference MACs and parameters without loading data")
    p.add_argument("--debug", action="store_true", help="enable autograd anomaly detection")
    # Historical launchers accepted these fields, but the phase budgets
    # and loader batch size always determined the effective values.
    for name, typ in (("total_epoch", int), ("warmup_epoch", int), ("max_device_batch_size", int), ("mode", str)):
        p.add_argument("--" + name, type=typ, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not all(0 <= x <= 1 for x in (args.mix_ratio, args.mix_loss_alpha, args.lmda, args.mask_ratio)):
        p.error("mix_ratio, mix_loss_alpha, lmda and mask_ratio must be in [0, 1]")
    if min(args.ft_epoch, args.pt_epoch, args.num_workers, args.ft_warmup_epoch, args.pt_warmup_epoch) < 0:
        p.error("epoch and worker counts must be nonnegative")
    if args.load_batch_size < 1 or args.k < 1:
        p.error("batch size and k must be positive")
    if args.from_pretrain and args.model_name == "vit-t" and not args.pretrained_model_path:
        p.error("ViT transfer requires --pretrained_model_path")
    if args.linear_probe and (args.model_name == "vit-t" or not args.from_pretrain or args.mix_ratio):
        p.error("linear probing requires a pretrained ResNet with --mix_ratio 0")
    args.source_dataset = args.source_dataset or args.dataset
    if (args.data_mix_noise or args.data_no_mix) and args.source_dataset == args.dataset:
        p.error("data_mix_noise and data_no_mix require a distinct source dataset")
    if not epoch_schedule(args.pt_epoch, args.ft_epoch, args.mix_ratio, args.from_pretrain) and not args.profile:
        p.error("the schedule must include at least one epoch")
    return args


def main(default_model="vit-t", default_source=None, default_mask=0.75):
    args = parse_args(default_model, default_source, default_mask)
    setup_seed(args.seed)
    torch.autograd.set_detect_anomaly(args.debug)
    image_size = max(IMAGE_SIZE[args.source_dataset], IMAGE_SIZE[args.dataset])
    if args.from_pretrain and args.model_name != "vit-t":
        image_size = 224
    if args.model_name == "vit-t":
        kwargs = dict(image_size=image_size, patch_size={32: 2, 64: 4, 224: 16}[image_size],
                      emb_dim=192, decoder_layer=2, num_classes=NUM_CLASSES[args.dataset], mask_ratio=args.mask_ratio)
    else:
        # timm ResNet has stride 32; the decoder upsamples each feature cell by 32.
        kwargs = dict(backbone_name=args.model_name, num_classes=NUM_CLASSES[args.dataset], pretrained=False)
    if args.from_pretrain and args.model_name != "vit-t" and not args.pretrained_model_path:
        model = MixModel(**{**kwargs, "pretrained": True})
    else:
        model = build_model(args.model_name, kwargs)
    if args.pretrained_model_path:
        loaded = load_checkpoint(args.pretrained_model_path, trusted_legacy=args.trusted_legacy_checkpoint)
        model.load_state_dict(loaded.state_dict())
    model.to(args.device)
    if args.linear_probe:
        model.freeze_backbone_and_decoder()
    if args.profile:
        from thop import profile
        from thop.vision.basic_hooks import count_normalization
        sample = torch.randn(1, 3, image_size, image_size, device=args.device)
        modes = ("predict", "reconstruct") if args.model_name == "vit-t" else ("predict", "generate")
        for mode in modes:
            macs, params = profile(model.eval(), inputs=(sample, mode), verbose=False,
                                   custom_ops={SingletonBatchNorm2d: count_normalization})
            print(f"{mode}: {macs / 1e9:.3f} G MACs, {params / 1e6:.3f} M parameters")
        return

    def loader(name, train):
        data = make_dataset(name, args.data_root, image_size, train, transfer=args.from_pretrain)
        return DataLoader(data, batch_size=args.load_batch_size, shuffle=train, num_workers=args.num_workers)

    phases = epoch_schedule(args.pt_epoch, args.ft_epoch, args.mix_ratio, args.from_pretrain)
    mix_epochs = phases.count("mix")
    ssl_updates = phases.count("ssl") + (mix_epochs if args.mix_loss_alpha > 0 else 0)
    sl_updates = phases.count("sl") + (mix_epochs if args.mix_loss_alpha < 1 else 0)
    needs_target = args.source_dataset == args.dataset or mix_epochs or sl_updates
    target = loader(args.dataset, True) if needs_target else None
    validation = loader(args.dataset, False) if sl_updates else None
    source = target
    if args.source_dataset != args.dataset and ("ssl" in phases or "mix" in phases):
        source = loader(args.source_dataset, True)
    optimizers = make_optimizers(model, args)
    schedulers = (
        torch.optim.lr_scheduler.LambdaLR(optimizers[0], lambda e: lr_schedule(e, args.pt_warmup_epoch, ssl_updates)),
        torch.optim.lr_scheduler.LambdaLR(optimizers[1], lambda e: lr_schedule(e, args.ft_warmup_epoch, sl_updates)),
    )
    if args.enable_wandb:
        import wandb
        wandb.init(project="MixTraining", name=args.info or None, config=vars(args))

    print(f"Schedule: {phases.count('ssl')} SSL, {phases.count('mix')} mixed, {phases.count('sl')} supervised epochs")
    best_acc = None
    output = Path(args.output_model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch, phase in enumerate(phases, start=1):
        start = time.monotonic()
        metrics = train_epoch(model, source, target, optimizers, phase, args)
        metrics["training_seconds"] = time.monotonic() - start
        if "ssl_loss" in metrics:
            schedulers[0].step()
        save = not sl_updates and epoch == len(phases)
        if "sl_loss" in metrics:
            schedulers[1].step()
            metrics.update(evaluate(model, validation, args.device))
            if best_acc is None or metrics["val_acc"] > best_acc:
                best_acc = metrics["val_acc"]
                save = True
        print(f"Epoch {epoch}/{len(phases)} ({phase}) {metrics}")
        if args.enable_wandb:
            wandb.log(metrics, step=epoch)
        if save:
            torch.save({"model_name": args.model_name, "model_kwargs": kwargs, "state_dict": model.state_dict(),
                        "epoch": epoch, "val_acc": best_acc, "dataset": args.dataset}, output)
    if args.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
