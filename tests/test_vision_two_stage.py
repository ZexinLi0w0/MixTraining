"""CIFAR-10 baseline checks using synthetic data."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FakeData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cv'))
import mae_pretrain
import train_classifier
from cifar_training import run_epoch, save_checkpoint
from mix_train import load_checkpoint
from model_vit import MAE_Mix


class TwoStageTest(unittest.TestCase):
    def test_short_accumulation_window_matches_logical_batches(self):
        data = TensorDataset(torch.arange(27, dtype=torch.float32).view(9, 3) / 10, torch.arange(9) % 2)
        baseline = torch.nn.Linear(3, 2)
        accumulated = copy.deepcopy(baseline)
        for model, batch, steps in [(baseline, 4, 1), (accumulated, 2, 2)]:
            run_epoch(model, DataLoader(data, batch), 'cpu', train_classifier.classification_loss,
                      torch.optim.SGD(model.parameters(), lr=0.2), steps)
        torch.testing.assert_close(accumulated.weight, baseline.weight)
        torch.testing.assert_close(accumulated.bias, baseline.bias)
        loss, acc = run_epoch(accumulated, DataLoader(data, 4), 'cpu', train_classifier.classification_loss)
        expected, hits = train_classifier.classification_loss(accumulated, *data.tensors)
        self.assertAlmostEqual(loss, expected.item(), places=6)
        self.assertAlmostEqual(acc, hits.item() / len(data))

    def test_two_stage_checkpoints_and_reconstruction_export(self):
        def synthetic_cifar(root, train, download, transform):
            return FakeData(size=3, image_size=(3, 32, 32), num_classes=10, transform=transform)

        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, patch('cifar_training.datasets.CIFAR10', synthetic_cifar):
            mae = Path(tmp, 'mae.pt')
            classifier = Path(tmp, 'classifier.pt')
            common = ['test', '--device', 'cpu', '--total_epoch', '1', '--batch_size', '2',
                      '--max_device_batch_size', '1', '--num_workers', '0']
            with patch.object(sys, 'argv', common + ['--model_path', str(mae), '--mask_ratio', '0.7']):
                mae_pretrain.main()
            restored = load_checkpoint(mae)
            self.assertEqual(restored.encoder.mask_ratio, 0.7)
            data = TensorDataset(torch.rand(2, 3, 32, 32), torch.zeros(2))
            preview = Path(tmp, 'reconstruction.png')
            mae_pretrain.save_reconstructions(restored, data, 'cpu', preview)
            self.assertTrue(preview.is_file())

            for pretrained in (mae, Path(tmp, 'mixed.pt')):
                if pretrained != mae:
                    kwargs = dict(image_size=16, patch_size=4, emb_dim=12, encoder_layer=1,
                                  decoder_layer=1, encoder_head=2, decoder_head=2)
                    save_checkpoint(pretrained, 'vit-t', kwargs, MAE_Mix(**kwargs))
                args = ['--pretrained_model_path', str(pretrained), '--output_model_path', str(classifier)]
                with patch.object(sys, 'argv', common + args):
                    train_classifier.main()
                saved_kwargs = torch.load(classifier, weights_only=True)['model_kwargs']
                self.assertEqual('encoder_kwargs' in saved_kwargs, pretrained != mae)
                restored_classifier = load_checkpoint(classifier)
                size = 32 if pretrained == mae else 16
                self.assertEqual(restored_classifier(torch.rand(1, 3, size, size)).shape, (1, 10))
