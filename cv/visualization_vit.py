"""Export reconstructions of the first validation batch from a mixed ViT."""

import argparse
from math import isqrt
from pathlib import Path

import torch
from torchvision.utils import save_image

from mix_train import load_checkpoint
from model_vit import MAE_Mix
from utils import setup_seed
from vision_data import IMAGE_SIZE, NORMALIZATION, make_dataset


def validation_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pretrained_model_path', type=Path,
                        help='Mixed ViT checkpoint. Omit to inspect a randomly initialized ViT.')
    parser.add_argument('--trusted_legacy_checkpoint', action='store_true')
    parser.add_argument('--dataset', choices=tuple(IMAGE_SIZE), default='cifar10')
    parser.add_argument('--data_root', type=Path, default=Path('data'))
    parser.add_argument('--load_batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--from_pretrain', action='store_true',
                        help='Use 224-pixel inputs for the randomly initialized comparison.')
    return parser


def validation_inputs(args):
    setup_seed(args.seed)
    if args.pretrained_model_path:
        model = load_checkpoint(args.pretrained_model_path, args.device,
                                trusted_legacy=args.trusted_legacy_checkpoint)
        if not isinstance(model, MAE_Mix):
            raise ValueError('This visualization requires a mixed ViT checkpoint.')
        image_size = isqrt(model.encoder.pos_embedding.shape[0]) * model.encoder.patchify.stride[0]
    else:
        image_size = 224 if args.from_pretrain else IMAGE_SIZE[args.dataset]
        patch_size = {32: 2, 64: 4, 224: 16}[image_size]
        model = MAE_Mix(image_size=image_size, patch_size=patch_size, decoder_layer=2).to(args.device)
    dataset = make_dataset(args.dataset, args.data_root, image_size, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.load_batch_size)
    return model.eval(), loader


def main():
    parser = validation_parser(__doc__)
    parser.add_argument('--output_dir', type=Path, default=Path('vit_mae_mix_images'))
    args = parser.parse_args()
    model, loader = validation_inputs(args)
    images, _ = next(iter(loader))
    with torch.inference_mode():
        features = model.unmasked_generate(images.to(args.device))
        reconstructions, _ = model.reconstruct_features(features)
    mean, std = NORMALIZATION[args.dataset]
    reconstructions = reconstructions.cpu()
    mean = reconstructions.new_tensor(mean).view(-1, 1, 1)
    std = reconstructions.new_tensor(std).view(-1, 1, 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(reconstructions):
        save_image(image * std + mean, args.output_dir / f'{i}_generated.jpg')
    print(f'Saved {len(reconstructions)} reconstructions to {args.output_dir}')


if __name__ == '__main__':
    main()
