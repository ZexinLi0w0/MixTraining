import logging
from typing import Callable, List, Tuple

import torch
from abstract_model import AbstractModel
from mix_trainer import MixTrainer
from tqdm import tqdm


class NaiveMixTrainer(MixTrainer):
    def __init__(
        self,
        eval_dataset,
        compute_metrics: Callable,
        epoch_mix_ratio: float = 0,
        data_mix_lambda: float = 0,
        loss_mix_alpha: float = 0.5,
        ft_epochs: int = 100,
        pt_epochs: int = 100,
        from_pretrained: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        logging.info("Initializing NaiveMixTrainer...")
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.data_mix_ratio = data_mix_lambda
        self.epoch_mix_ratio = epoch_mix_ratio
        self.loss_mix_alpha = loss_mix_alpha

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

    def pretrain_epoch(
        self,
    ):
        self.model.train()
        for inputs, labels in tqdm(self.pt_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.pt_forwardfn(self.model, inputs, labels)
            loss: torch.Tensor = self.pt_lossfn(outputs, inputs, labels)
            loss.backward()
            self.pt_optimizer.step()
            self.pt_optimizer.zero_grad()

    def mixtrain_diff_dataset_epoch(
        self,
    ):
        self.model.train()
        for (pt_inputs, pt_labels), (ft_inputs, ft_labels) in tqdm(
            zip(self.pt_dataloader, self.ft_dataloader)
        ):
            pt_inputs, pt_labels = pt_inputs.to(self.device), pt_labels.to(self.device)
            ft_inputs, ft_labels = ft_inputs.to(self.device), ft_labels.to(self.device)
            if len(pt_inputs) != len(ft_inputs):
                break
            inputs = (
                1 - self.data_mix_ratio
            ) * pt_inputs + self.data_mix_ratio * ft_inputs
            labels = ft_labels
            pt_outputs = self.pt_forwardfn(self.model, inputs, labels)
            pt_loss = self.pt_lossfn(pt_outputs, inputs, labels)
            pt_loss.backward()
            ft_outputs = self.ft_forwardfn(self.model, inputs, labels, pt_outputs)
            ft_loss = self.ft_lossfn(ft_outputs, inputs, labels)
            loss: torch.Tensor = (
                1 - self.loss_mix_alpha
            ) * pt_loss + self.loss_mix_alpha * ft_loss
            loss.backward()
            self.pt_optimizer.step()
            self.pt_optimizer.zero_grad()
            self.ft_optimizer.step()
            self.ft_optimizer.zero_grad()
        for _ in self.pt_dataloader:
            pass
        for _ in self.ft_dataloader:
            pass
        self.pt_scheduler.step()
        self.ft_scheduler.step()

    def mixtrain_same_dataset_epoch(
        self,
    ):
        self.model.train()
        for inputs, labels in tqdm(iter(self.pt_dataloader)):
            pt_outputs: torch.Tensor = self.pt_forwardfn(self.model, inputs, labels)
            pt_loss: torch.Tensor = self.pt_lossfn(pt_outputs, labels)
            ft_outputs: torch.Tensor = self.ft_forwardfn(
                self.model, inputs, labels, pt_outputs
            )
            ft_loss: torch.Tensor = self.ft_lossfn(ft_outputs, labels)
            loss: torch.Tensor = (
                1 - self.loss_mix_alpha
            ) * pt_loss + self.loss_mix_alpha * ft_loss
            loss.backward()
            self.pt_optimizer.step()
            self.pt_optimizer.zero_grad()
            self.ft_optimizer.step()
            self.ft_optimizer.zero_grad()
        self.pt_scheduler.step()
        self.ft_scheduler.step()

    def finetune_epoch(
        self,
    ):
        self.model.train()
        for inputs, labels in tqdm(iter(self.ft_dataloader)):
            outputs = self.ft_forwardfn(self.model, inputs, labels, None)
            loss: torch.Tensor = self.pt_lossfn(outputs, inputs)
            loss.backward()
            self.ft_optimizer.step()
            self.ft_optimizer.zero_grad()
        self.ft_scheduler.step()

    def _make_training_epochs(self) -> List[Callable[[], None]]:
        if self.same_dataset:
            mix_training_epoch = self.mixtrain_same_dataset_epoch
        else:
            mix_training_epoch = self.mixtrain_diff_dataset_epoch
        return (
            [self.pretrain_epoch] * self.pt_epochs
            + [mix_training_epoch] * self.mix_epochs
            + [self.finetune_epoch] * self.ft_epochs
        )
