"""Dataset transforms and splits shared by the vision experiments."""

from pathlib import Path

from torchvision import datasets, transforms

from randomaug import RandAugment

IMAGE_SIZE = {"cifar10": 32, "cifar100": 32, "svhn": 32, "tinyimagenet": 64, "imagenet": 224}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "svhn": 10, "tinyimagenet": 200, "imagenet": 1000}
_CIFAR = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
NORMALIZATION = {
    "cifar10": _CIFAR,
    "cifar100": _CIFAR,
    "svhn": ((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
    "tinyimagenet": _IMAGENET,
    "imagenet": _IMAGENET,
}


def make_dataset(name, root, image_size, train, transfer=False):
    ops = []
    if train and name in ("tinyimagenet", "imagenet"):
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1), ratio=(0.9, 1.1),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            RandAugment(n=2, m=9),
        ])
    else:
        if train:
            if not transfer:
                ops.append(transforms.RandomCrop(32, padding=4))
            ops.append(transforms.RandomHorizontalFlip())
        if image_size != IMAGE_SIZE[name] or name == "imagenet":
            ops.append(transforms.Resize((image_size, image_size),
                                         interpolation=transforms.InterpolationMode.BICUBIC))
    ops.extend([transforms.ToTensor(), transforms.Normalize(*NORMALIZATION[name])])
    transform = transforms.Compose(ops)
    if name in ("cifar10", "cifar100"):
        cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        return cls(root, train=train, download=True, transform=transform)
    if name == "svhn":
        return datasets.SVHN(root, split="train" if train else "test", download=True, transform=transform)
    if name == "tinyimagenet":
        import tinyimagenet
        return tinyimagenet.TinyImageNet(str(root), split="train" if train else "val", transform=transform,
                                         target_transform=int)
    return datasets.ImageFolder(Path(root) / ("train" if train else "val"), transform=transform)
