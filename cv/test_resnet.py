"""Check the shared ResNet encoder and its two heads on CPU."""

import unittest

import torch
from model_resnet import MixModel


class ResNetMixTest(unittest.TestCase):
    def test_shared_features_match_individual_heads(self):
        model = MixModel('resnet18', num_classes=3, pretrained=False, resolution=32).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            features = model(images, mode='get_feature')
            logits = model(features, mode='predict_from_features')
            prediction = model(features, mode='generate_from_features')
            torch.testing.assert_close(logits, model(images, mode='predict'))
            torch.testing.assert_close(prediction, model(images, mode='generate'))
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(prediction.shape, images.shape)


if __name__ == '__main__':
    unittest.main()
