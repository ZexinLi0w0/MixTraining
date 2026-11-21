"""Run with python -m unittest discover -s tests -p test_nlp.py."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import BertConfig, BertTokenizerFast, GPT2Config, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nlp"))
from data_collator import mask_tokens
from mix_trainer import MixTrainer
from modeling_bert import BertForSequenceClassification
from modeling_gpt2 import GPT2ForSequenceClassification


class MixLossTest(unittest.TestCase):
    def test_bert_clean_inputs_and_epoch_cutoff(self):
        with tempfile.TemporaryDirectory() as temp:
            vocab = Path(temp, "vocab.txt")
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nred\nblue\n")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            model = BertForSequenceClassification(BertConfig(
                vocab_size=len(tokenizer), hidden_size=8, intermediate_size=12,
                num_attention_heads=2, num_hidden_layers=1, hidden_dropout_prob=0,
                attention_probs_dropout_prob=0, classifier_dropout=0,
            ))
            batch = tokenizer(["red blue", "red"], padding=True, return_tensors="pt")
            batch["labels"] = torch.tensor([0, 1])
            original = batch["input_ids"].clone()
            _, targets = mask_tokens(batch["input_ids"], tokenizer, 1)
            self.assertEqual(targets.ne(-100).sum().item(), 3)
            torch.testing.assert_close(batch["input_ids"], original)
            trainer = MixTrainer(
                model=model, args=TrainingArguments(temp, use_cpu=True, report_to=[]),
                processing_class=tokenizer, mix_end_epoch=1, mlm_probability=1,
            )
            task_loss = model(**batch).loss
            model.train()
            mixed = trainer.compute_loss(model, batch)
            self.assertGreater(mixed.item(), task_loss.item())
            mixed.backward()
            self.assertGreater(model.mlm_head.predictions.bias.grad.abs().sum().item(), 0)
            torch.testing.assert_close(batch["input_ids"], original)
            model.eval()
            torch.testing.assert_close(trainer.compute_loss(model, batch), task_loss)
            model.train()
            trainer.mlm_probability = 0
            torch.testing.assert_close(trainer.compute_loss(model, batch), task_loss)
            trainer.mlm_probability = 1
            trainer.state.epoch = 1
            torch.testing.assert_close(trainer.compute_loss(model, batch), task_loss)

    def test_gpt2_padding_and_single_token(self):
        model = GPT2ForSequenceClassification(GPT2Config(
            vocab_size=8, n_embd=8, n_layer=1, n_head=2, pad_token_id=0,
        )).eval()
        labels = torch.tensor([1])

        def forward(ids, mask, mixed, second_view=False):
            ids, mask = torch.tensor([ids]), torch.tensor([mask])
            return model(
                input_ids=ids, attention_mask=mask, labels=labels, pretrain_mlm=mixed,
                mlm_input_ids=ids.clone() if second_view else None,
                mlm_attention_mask=mask if second_view else None, mlm_gts=ids.masked_fill(mask == 0, -100),
            )

        clean = forward([3, 4], [1, 1], False)
        with patch.object(model.transformer, "forward", wraps=model.transformer.forward) as backbone:
            mixed = forward([3, 4], [1, 1], True)
            self.assertEqual(backbone.call_count, 1)
            backbone.reset_mock()
            separate = forward([3, 4], [1, 1], True, second_view=True)
            self.assertEqual(backbone.call_count, 2)
        torch.testing.assert_close(mixed.loss, separate.loss)
        parameters = (model.lm_head.weight, model.transformer.wte.weight)
        torch.testing.assert_close(
            torch.autograd.grad(mixed.loss, parameters), torch.autograd.grad(separate.loss, parameters)
        )
        padded = forward([3, 4, 0, 0], [1, 1, 0, 0], True)
        torch.testing.assert_close(padded.logits, clean.logits)
        torch.testing.assert_close(padded.loss, mixed.loss)
        self.assertGreater(mixed.loss.item(), clean.loss.item())
        torch.testing.assert_close(forward([3], [1], True).loss, forward([3], [1], False).loss)


if __name__ == "__main__":
    unittest.main()
