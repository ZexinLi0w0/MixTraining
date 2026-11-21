"""Fine-tune a pretrained ViT encoder or train a CIFAR-10 classifier from scratch."""

import inspect
import math
import time

from torch.nn.functional import cross_entropy

from cifar_training import make_loader, optimizer_and_scheduler, run_epoch, save_checkpoint, training_parser
from model_vit import MAE_Encoder, ViT_Classifier
from utils import setup_seed


def classification_loss(model, images, labels):
    logits = model(images)
    return cross_entropy(logits, labels), (logits.argmax(dim=1) == labels).sum()


def main():
    parser = training_parser(__doc__)
    parser.set_defaults(batch_size=128, max_device_batch_size=256, base_learning_rate=1e-3,
                        total_epoch=100, warmup_epoch=5)
    parser.add_argument('--pretrained_model_path', help='Path to a ViT autoencoder checkpoint.')
    parser.add_argument('--trusted_legacy_checkpoint', action='store_true', help='Allow loading a trusted pickled model.')
    parser.add_argument('--output_model_path', default='vit-t-classifier-from_scratch.pt')
    args = parser.parse_args()
    setup_seed(args.seed)
    classifier_kwargs = {'num_classes': 10}
    image_size = 32
    if args.pretrained_model_path:
        from mix_train import load_checkpoint
        source = load_checkpoint(args.pretrained_model_path, device='cpu', trusted_legacy=args.trusted_legacy_checkpoint)
        if not isinstance(getattr(source, 'encoder', None), MAE_Encoder):
            parser.error('--pretrained_model_path must contain a ViT autoencoder')
        encoder = source.encoder
        patch_size = encoder.patchify.kernel_size[0]
        image_size = patch_size * math.isqrt(len(encoder.pos_embedding))
        architecture = {
            'image_size': image_size, 'patch_size': patch_size, 'emb_dim': encoder.pos_embedding.shape[-1],
            'num_layer': len(encoder.transformer),
            'num_head': encoder.transformer[0].attn.num_heads if len(encoder.transformer) else 1,
        }
        defaults = inspect.signature(MAE_Encoder).parameters
        if any(value != defaults[key].default for key, value in architecture.items()):
            classifier_kwargs['encoder_kwargs'] = architecture
        del source
    else:
        encoder = MAE_Encoder()

    train_loader = make_loader(args, train=True, image_size=image_size)
    val_loader = make_loader(args, train=False, image_size=image_size)
    model = ViT_Classifier(encoder, num_classes=10).to(args.device)
    optimizer, scheduler = optimizer_and_scheduler(model, args, betas=(0.9, 0.999))
    if args.enable_wandb:
        import wandb
        wandb.init(project='MixTraining', name='finetune-vit-cifar10', config=vars(args))

    best_val_acc = -1.0
    training_time = 0.0
    for epoch in range(args.total_epoch):
        start = time.monotonic()
        train_loss, train_acc = run_epoch(model, train_loader, args.device, classification_loss, optimizer,
                                          steps_per_update=args.batch_size // train_loader.batch_size)
        training_time += time.monotonic() - start
        scheduler.step()
        val_loss, val_acc = run_epoch(model, val_loader, args.device, classification_loss)
        print(f'Epoch {epoch + 1}: train loss {train_loss:.6f}, accuracy {train_acc:.4f}, '
              f'validation loss {val_loss:.6f}, accuracy {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.output_model_path, 'vit-t-classifier', classifier_kwargs, model)
        if args.enable_wandb:
            wandb.log({'supervised_loss': train_loss, 'acc': train_acc, 'val_loss': val_loss,
                       'val_acc': val_acc, 'accumulated_times': training_time})

    if args.enable_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
