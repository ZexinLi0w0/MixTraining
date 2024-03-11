import random
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.initial_lr = optimizer.defaults['lr']
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warm-up phase
            lr = ((self.initial_lr - self.min_lr) / self.warmup_epochs) * self.last_epoch + self.min_lr
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_term = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_term
        print("current lr: {}".format(lr))
        return [lr for _ in self.optimizer.param_groups]


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
        image = self.hf_dataset[idx]['image'].convert('RGB')
        image = self.transform(image)

        # Convert image to tensor
        image = torch.tensor(np.array(image))

        label = self.hf_dataset[idx]['label']
        return image, label