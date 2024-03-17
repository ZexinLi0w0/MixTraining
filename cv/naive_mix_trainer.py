import logging
from sched import scheduler
from typing import Any, Callable, Dict, Union

import torch
from mix_trainer import MixTrainer
from torch import nn


class NaiveMixTrainer(MixTrainer):
    def __init__(
        self,
        mix_ratio: float = 0,
        # k_mix: int = 1,
        ft_epochs: int = 100,
        pt_epochs: int = 100,
        from_pretrained: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mix_ratio < 0 or mix_ratio > 1:
            raise ValueError("mix_ratio should be in the range [0, 1]")

        # self.k_mix = k_mix

        if from_pretrained:
            mixable_epoch = ft_epochs * mix_ratio
            self.pt_epochs = 0
            self.mix_epochs = round(mixable_epoch)
            self.ft_epochs = round(ft_epochs - mixable_epoch)
            self.total_epochs = self.mix_epochs + self.ft_epochs
        else:
            mixable_epoch = min(pt_epochs, ft_epochs)
            self.pt_epochs = round(pt_epochs - mixable_epoch * mix_ratio)
            self.mix_epochs = round(mixable_epoch * mix_ratio)
            self.ft_epochs = round(ft_epochs - mixable_epoch * mix_ratio)

        assert set(self.optimizers.keys()) == set(
            "pretrain", "finetune"
        ), "optimizers should be a dict with keys 'pretrain', 'finetune'"
        assert set(self.schedulers.keys()) == set(
            "pretrain", "finetune"
        ), "schedulers should be a dict with keys 'pretrain', 'finetune'"
        assert set(self.forward_fns.keys()) == set(
            "pretrain", "finetune"
        ), "forward_fns should be a dict with keys 'pretrain', 'finetune'"
        assert set(self.loss_fns.keys()) == set(
            "pretrain", "finetune"
        ), "loss_fns should be a dict with keys 'pretrain', 'finetune'"

        logging.info(
            f"Pretrain: {self.pt_epochs}, Mixtrain: {self.mix_epochs}, Finetune: {self.ft_epochs}"
        )

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    def pretrain_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> torch.Tensor:
        scheduler = self.schedulers["pretrain"]
        optimizer = self.optimizers["pretrain"]
        forward_fn = self.forward_fns["pretrain"]
        loss_fn = self.loss_fns["pretrain"]
        outputs = forward_fn(model, inputs, mode="pretrain")
        loss: torch.Tensor = loss_fn(inputs, **outputs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    def mixtrain_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> torch.Tensor:
        for mode in ["pretrain", "finetune"]:
            scheduler = self.schedulers[mode]
            optimizer = self.optimizers[mode]
            forward_fn = self.forward_fns[mode]
            loss_fn = self.loss_fns[mode]
            outputs = forward_fn(model, inputs, mode=mode)
            loss: torch.Tensor = loss_fn(inputs, **outputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    def finetune_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        forward_fn: Callable,
        loss_fn: Callable,
    ) -> torch.Tensor:
        scheduler = self.schedulers["finetune"]
        optimizer = self.optimizers["finetune"]
        outputs = forward_fn(model, inputs, mode="finetune")
        loss: torch.Tensor = loss_fn(inputs, **outputs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    def _make_training_steps(self):
        return [
            [self.pretrain_step] * self.pt_epochs,
            [self.mixtrain_step] * self.ft_epochs,
            [self.finetune_step] * self.ft_epochs,
        ]
