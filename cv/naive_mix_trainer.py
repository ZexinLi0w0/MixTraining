import logging
from ftplib import FTP_PORT
from sched import scheduler
from typing import Any, Callable, Dict, Union

import torch
from mix_trainer import MixTrainer
from torch import nn
from tqdm import tqdm


class NaiveMixTrainer(MixTrainer):
    def __init__(
        self,
        model,
        eval_dataset,
        compute_metrics: Callable,
        epoch_mix_ratio: float = 0,
        loss_mix_alpha: float = 0.5,
        ft_epochs: int = 100,
        pt_epochs: int = 100,
        from_pretrained: bool = False,
        dataset_mix_ratio: float = 0.5,
        dataset_mix_noise: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        logging.info("Initializing NaiveMixTrainer...")
        self.model = model
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.epoch_mix_ratio = epoch_mix_ratio
        self.loss_mix_alpha = loss_mix_alpha
        self.dataset_mix_ratio = dataset_mix_ratio
        self.dataset_mix_noise = dataset_mix_noise

        if from_pretrained:
            mixable_epoch = ft_epochs * epoch_mix_ratio
            self.pt_epochs = 0
            self.mix_epochs = round(mixable_epoch)
            self.ft_epochs = round(ft_epochs - mixable_epoch)
            self.total_epochs = self.mix_epochs + self.ft_epochs
        else:
            mixable_epoch = min(pt_epochs, ft_epochs)
            self.pt_epochs = round(pt_epochs - mixable_epoch * epoch_mix_ratio)
            self.mix_epochs = round(mixable_epoch * epoch_mix_ratio)
            self.ft_epochs = round(ft_epochs - mixable_epoch * epoch_mix_ratio)

            logging.info(
                f"Epochs:\nPretrain: {self.pt_epochs}, Mixtrain: {self.mix_epochs}, Finetune: {self.ft_epochs}"
            )

        self.training_epochs = self._make_training_epochs()

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    def pretrain_epoch(
        self,
    ):
        model = self.model
        scheduler = self.pt_scheduler
        optimizer = self.pt_optimizer
        for inputs in tqdm(self.pt_dataloader):
            model.train()
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = self.pt_lossfn(outputs, inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()

    def mixtrain_epoch(
        self,
    ):
        model = self.model
        pt_scheduler = self.pt_scheduler
        pt_optimizer = self.pt_optimizer
        pt_scheduler = self.pt_scheduler
        ft_scheduler = self.ft_scheduler
        ft_optimizer = self.ft_optimizer
        ft_scheduler = self.ft_scheduler
        for pt_input, ft_input in tqdm(zip(self.pt_dataloader, self.ft_dataloader)):
            model.train()
            pt_optimizer.zero_grad()
            ft_optimizer.zero_grad()
            pt_outputs = model(**pt_input)
            ft_outputs = model(**ft_input)
            pt_loss = self.pt_lossfn(pt_outputs, pt_input)
            ft_loss = self.ft_lossfn(ft_outputs, ft_input)
            if self.dataset_mix_ratio == 0:
                pt_loss.backward()
                ft_loss.backward()
            else:
                loss = (
                    self.loss_mix_alpha * pt_loss + (1 - self.loss_mix_alpha) * ft_loss
                )
            pt_loss.backward()
            ft_loss.backward()
            pt_optimizer.step()
            ft_optimizer.step()
            pt_scheduler.step()
            ft_scheduler.step()

    def finetune_epoch(
        self,
    ):
        model = self.model
        scheduler = self.ft_scheduler
        optimizer = self.ft_optimizer
        for inputs in tqdm(self.ft_dataloader):
            model.train()
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = self.pt_lossfn(outputs, inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()

    def _make_training_epochs(self):
        pass
