"""Pretrain a ViT-T masked autoencoder on CIFAR-10."""

import time

import torch
from torchvision.utils import save_image

from cifar_training import make_loader, optimizer_and_scheduler, run_epoch, save_checkpoint, training_parser
from model_vit import MAE_ViT
from utils import setup_seed


def reconstruction_loss(model, images, labels):
    prediction, mask = model(images)
    return ((prediction - images).square() * mask).sum() / mask.sum(), 0


@torch.no_grad()
def save_reconstructions(model, data, device, path):
    model.eval()
    images = torch.stack([data[i][0] for i in range(min(16, len(data)))]).to(device)
    prediction, mask = model(images)
    visible = images * (1 - mask)
    panels = torch.stack([visible, visible + prediction * mask, images], dim=1).flatten(0, 1)
    save_image((panels + 1) / 2, path, nrow=6)


def main():
    parser = training_parser(__doc__)
    parser.set_defaults(batch_size=4096, max_device_batch_size=512, base_learning_rate=1.5e-4,
                        total_epoch=2000, warmup_epoch=200)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--model_path', default='vit-t-mae.pt')
    parser.add_argument('--save_images', action='store_true', help='Save masked, reconstructed and original validation images each epoch.')
    args = parser.parse_args()
    if not 0 < args.mask_ratio <= 1:
        parser.error('--mask_ratio must be greater than 0 and at most 1')
    setup_seed(args.seed)
    train_loader = make_loader(args, train=True)
    if args.save_images:
        val_data = make_loader(args, train=False).dataset

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(args.device)
    optimizer, scheduler = optimizer_and_scheduler(model, args, betas=(0.9, 0.95))
    if args.enable_wandb:
        import wandb
        wandb.init(project='MixTraining', name='pretrain-vit-cifar10', config=vars(args))

    training_time = 0.0
    for epoch in range(args.total_epoch):
        start = time.monotonic()
        loss, _ = run_epoch(model, train_loader, args.device, reconstruction_loss, optimizer,
                            steps_per_update=args.batch_size // train_loader.batch_size)
        training_time += time.monotonic() - start
        scheduler.step()
        print(f'Epoch {epoch + 1}: reconstruction loss {loss:.6f}')
        if args.save_images:
            save_reconstructions(model, val_data, args.device, f'mae-pretrain-cifar10-{epoch}.png')
        save_checkpoint(args.model_path, 'vit-t-mae-pretrain', {'mask_ratio': args.mask_ratio}, model)
        if args.enable_wandb:
            wandb.log({'unsupervised_loss': loss, 'accumulated_times': training_time})

    if args.enable_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
