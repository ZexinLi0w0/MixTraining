"""Plot a t-SNE projection of the validation images' ViT class-token embeddings."""

from pathlib import Path

import numpy as np
import torch

from visualization_vit import validation_inputs, validation_parser


def main():
    parser = validation_parser(__doc__)
    parser.add_argument('--output', type=Path, default=Path('tsne.png'))
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model, loader = validation_inputs(args)
    embeddings, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            embeddings.append(model.unmasked_generate(images.to(args.device))[0].cpu())
            labels.append(targets)
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()
    if len(labels) < 2:
        parser.error('t-SNE requires at least two validation images.')
    points = TSNE(n_components=2, random_state=args.seed,
                  perplexity=min(30, len(labels) - 1)).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label in np.unique(labels):
        xy = points[labels == label]
        ax.scatter(xy[:, 0], xy[:, 1], label=str(label), s=8)
    if len(np.unique(labels)) <= 20:
        ax.legend()
    ax.set(title=f'{args.dataset} validation embeddings', xlabel='t-SNE 1', ylabel='t-SNE 2')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f'Saved {len(labels)} embeddings to {args.output}')


if __name__ == '__main__':
    main()
