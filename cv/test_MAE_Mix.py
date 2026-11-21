"""Exercise both heads of a small ViT on CPU without downloading data."""

import unittest

import torch
from model_vit import MAE_Mix


class ViTMixTest(unittest.TestCase):
    def test_reconstruction_and_classification_share_patch_features(self):
        images = torch.randn(2, 3, 8, 8)
        for ratio in (0.0, 0.75):
            with self.subTest(mask_ratio=ratio):
                model = MAE_Mix(image_size=8, patch_size=2, emb_dim=12,
                                encoder_layer=1, encoder_head=3, decoder_layer=1,
                                decoder_head=3, mask_ratio=ratio, num_classes=3)
                features = model.get_features(images)
                logits = model.predict_from_features(features)
                reconstruction, mask = model.reconstruct_features(features)
                self.assertEqual(logits.shape, (2, 3))
                self.assertEqual(reconstruction.shape, images.shape)
                self.assertEqual(mask.mean().item(), ratio or 1.0)
                loss = ((reconstruction - images).square() * mask).sum() / mask.sum()
                (loss + logits.square().mean()).backward()
                for layer in (model.encoder.patchify, model.decoder.head, model.classifier):
                    self.assertGreater(layer.weight.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
