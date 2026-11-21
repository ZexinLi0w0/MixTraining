"""Offline regression checks for mixed epochs and shared-backbone updates."""

import copy
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cv"))
from mix_train import epoch_schedule, evaluate, get_batches, main, train_epoch
from model_resnet import MixModel
from model_vit import MAE_Mix


class VisionTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.args = SimpleNamespace(device="cpu", linear_probe=False, mix_loss_alpha=0.7,
                                    data_mix_noise=False, data_no_mix=False, lmda=0.25)

    def test_epoch_budgets(self):
        self.assertEqual(epoch_schedule(3, 4, 0.5), ["ssl"] * 2 + ["mix"] + ["sl"] * 3)
        self.assertEqual(epoch_schedule(0, 100, .25, from_pretrain=True), ["mix"] * 25 + ["sl"] * 75)
        self.assertEqual(epoch_schedule(4, 2, 0), ["ssl"] * 4 + ["sl"] * 2)

    def test_cross_dataset_covers_all_targets_and_keeps_labels(self):
        source = DataLoader(TensorDataset(torch.ones(1, 1), torch.tensor([99])), batch_size=2)
        target = DataLoader(TensorDataset(torch.zeros(3, 1), torch.tensor([2, 1, 0])), batch_size=2)
        batches = list(get_batches("mix", source, target, self.args))
        self.assertEqual(torch.cat([b[2] for b in batches]).tolist(), [2, 1, 0])
        for a, b, _ in batches:
            self.assertIs(a, b)
            torch.testing.assert_close(a, torch.full_like(a, .75))
        self.args.data_no_mix = True
        for a, b, _ in get_batches("mix", source, target, self.args):
            self.assertTrue(a.eq(1).all() and b.eq(0).all())

    def test_shared_step_matches_joint_sgd(self):
        # Compare every parameter against a single optimizer applied to Eq. 3.
        model = MAE_Mix(image_size=8, patch_size=2, emb_dim=12, num_classes=3,
                        encoder_layer=1, decoder_layer=1, mask_ratio=0)
        reference = copy.deepcopy(model)
        images = torch.randn(2, 3, 8, 8)
        labels = torch.tensor([0, 2])
        data = DataLoader(TensorDataset(images, labels), batch_size=2)
        optimizers = (torch.optim.SGD([*model.encoder.parameters(), *model.decoder.parameters()], lr=.01),
                      torch.optim.SGD([*model.encoder.parameters(), *model.classifier.parameters()], lr=.01))
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        features = reference.get_features(images)
        image, _ = reference.reconstruct_features(features)
        logits = reference.predict_from_features(features)
        loss = .7 * F.mse_loss(image, images) + .3 * F.cross_entropy(logits, labels)
        loss.backward()
        reference_optimizer.step()
        with patch.object(model, "get_features", wraps=model.get_features) as enc:
            metrics = train_epoch(model, data, data, optimizers, "mix", self.args)
        self.assertEqual(enc.call_count, 1)
        self.assertGreater(metrics["ssl_loss"], 0)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_resnet_singleton_batch_keeps_learning(self):
        model = MixModel("resnet18", num_classes=2, pretrained=False)
        optimizers = (torch.optim.SGD(model.decoder.parameters(), lr=.01),
                      torch.optim.SGD([*model.backbone.parameters(), *model.head.parameters()], lr=.01))
        data = TensorDataset(torch.randn(5, 3, 32, 32), torch.tensor([0, 1, 1, 0, 1]))
        batches = DataLoader(data, batch_sampler=[[0, 1], [2], [3, 4]])
        last_bn = [m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)][-1]
        counts, head_weights = [], []

        def record_step(*_):
            counts.append(last_bn.num_batches_tracked.item())
            head_weights.append(model.head[-1].weight.detach().clone())

        with optimizers[1].register_step_post_hook(record_step):
            train_epoch(model, batches, batches, optimizers, "sl", self.args)
        self.assertEqual(counts, [1, 1, 2])
        self.assertFalse(torch.equal(head_weights[0], head_weights[1]))
        model.freeze_backbone_and_decoder()
        self.args.linear_probe = True
        train_epoch(model, batches, batches, optimizers, "sl", self.args)
        self.assertEqual(last_bn.num_batches_tracked.item(), 2)

    def test_checkpoint_selection(self):
        images = torch.rand(2, 3, 32, 32)
        data = TensorDataset(images, torch.tensor([0, 1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            for ft_epochs, expected_epoch in ((2, 3), (0, 2)):
                argv = ["train", "--pt_epoch", "2", "--ft_epoch", str(ft_epochs), "--load_batch_size", "2",
                        "--num_workers", "0", "--device", "cpu", "--output_model_path", str(path)]
                # Tied zero accuracy must still store the first trained classifier.
                with patch.object(sys, "argv", argv), patch("mix_train.make_dataset", return_value=data), \
                        patch("mix_train.evaluate", return_value={"val_acc": 0., "val_loss": 1.}) as evaluate_mock:
                    main()
                checkpoint = torch.load(path, weights_only=True)
                self.assertEqual(evaluate_mock.call_count, ft_epochs)
                self.assertEqual(checkpoint["epoch"], expected_epoch)
                self.assertEqual(checkpoint["val_acc"], 0. if ft_epochs else None)

    def test_sample_weighted_validation(self):
        model = MAE_Mix(image_size=8, patch_size=2, emb_dim=12, num_classes=2,
                        encoder_layer=1, decoder_layer=1)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([1., 0.]))
        labels = torch.tensor([0, 0, 1])
        data = DataLoader(TensorDataset(torch.rand(3, 3, 8, 8), labels), batch_size=2)
        metrics = evaluate(model, data, "cpu")
        self.assertAlmostEqual(metrics["val_acc"], 2 / 3)
        self.assertAlmostEqual(metrics["val_loss"], F.cross_entropy(torch.tensor([[1., 0.]] * 3), labels).item())


if __name__ == "__main__":
    unittest.main()
