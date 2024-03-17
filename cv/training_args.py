from dataclasses import dataclass

# parser.add_argument('--seed', type=int, default=42)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--max_device_batch_size', type=int, default=256)
# parser.add_argument('--base_learning_rate', type=float, default=1e-3)
# parser.add_argument('--weight_decay', type=float, default=0.05)
# parser.add_argument('--total_epoch', type=int, default=100)
# parser.add_argument('--warmup_epoch', type=int, default=5)
# parser.add_argument('--pretrained_model_path', type=str, default=None)
# parser.add_argument('--output_model_path', type=str, default='test.pt')
# parser.add_argument('--enable_wandb', action='store_true')
# parser.add_argument('--dataset', type=str, default='cifar10')
# parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--mask_ratio', type=float, default=0.75)
# parser.add_argument('--mode', type=str, default='mix')
# parser.add_argument('--debug', action='store_true')
# parser.add_argument('--mix_ratio', type=float, default=0.0)
# parser.add_argument('--mix_loss_alpha', type=float, default=0.5)
# parser.add_argument('--k', type=int, default=1)
# parser.add_argument('--load_batch_size', type=int, default=256)
# parser.add_argument('--profile', action='store_true')
# parser.add_argument('--ft_epoch', type=int, default=100)
# parser.add_argument('--pt_epoch', type=int, default=100)
# parser.add_argument('--ft_warmup_epoch', type=int, default=5)
# parser.add_argument('--pt_warmup_epoch', type=int, default=20)
# parser.add_argument('--from_pretrain', action='store_true')
# parser.add_argument('--linear_probe', action='store_true')
# parser.add_argument('--info', type=str, default='')
# parser.add_argument('--model_name', type=str, default='vit-t', choices=['vit-t'])


@dataclass(frozen=True)
class MixTrainingArgs:
    """
    Training arguments for MixTrainer.

    Attributes:
    """

    seed: int = 42
    batch_size: int = 128
    max_device_batch_size: int = 256
    base_learning_rate: float = 1e-3
    weight_decay: float = 0.05
    total_epoch: int = 100
    pt_epoch: int = 100
    ft_epoch: int = 100
    warmup_epoch: int = 5
    pt_warmup_epoch: int = 20
    ft_warmup_epoch: int = 5
