import logging
import time
from typing import Callable, List

import torch
from mix_trainer import MixTrainer
from tqdm import tqdm


class NaiveMixTrainer(MixTrainer):
    def __init__(
        self,
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
        training_begin_time = time.time()
        self.model.train()
        losses = []
        for inputs, labels in tqdm(iter(self.pt_dataloader)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.pt_forwardfn(self.model, inputs, labels)
            loss: torch.Tensor = self.pt_lossfn(outputs, inputs, labels)
            losses.append(loss.item())
            loss.backward()
            self.pt_optimizer.step()
            self.pt_optimizer.zero_grad()
        self.pt_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        training_time = time.time() - training_begin_time
        log: dict = self.val_fn(self.model, self.val_dataloader)
        log.update(
            {
                "pt_loss": avg_loss,
                "acc": 0.0,
                "ft_loss": 0.0,
                "training_time": training_time,
            }
        )
        return log

    def mixtrain_diff_dataset_epoch(
        self,
    ):
        train_begin_time = time.time()
        self.model.train()
        pt_losses = []
        ft_losses = []
        accs = []
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
            pt_losses.append(pt_loss.item())
            pt_loss.backward()
            ft_outputs = self.ft_forwardfn(self.model, inputs, labels, pt_outputs)
            ft_loss = self.ft_lossfn(ft_outputs, inputs, labels)
            ft_losses.append(ft_loss.item())
            loss: torch.Tensor = (
                1 - self.loss_mix_alpha
            ) * pt_loss + self.loss_mix_alpha * ft_loss
            loss.backward()
            self.pt_optimizer.step()
            self.pt_optimizer.zero_grad()
            self.ft_optimizer.step()
            self.ft_optimizer.zero_grad()
            accs.append(self.compute_metrics(ft_outputs, inputs, labels).item())
        for _ in self.pt_dataloader:
            pass
        for _ in self.ft_dataloader:
            pass
        self.pt_scheduler.step()
        self.ft_scheduler.step()
        training_time = time.time() - train_begin_time

        avg_acc = sum(accs) / len(accs)
        avg_pt_loss = sum(pt_losses) / len(pt_losses)
        avg_ft_loss = sum(ft_losses) / len(ft_losses)

        log: dict = self.val_fn(self.model, self.val_dataloader)
        log.update(
            {
                "pt_loss": avg_pt_loss,
                "ft_loss": avg_ft_loss,
                "training_acc": avg_acc,
                "training_time": training_time,
            }
        )
        return log

    def mixtrain_same_dataset_epoch(
        self,
    ):
        training_begin_time = time.time()
        pt_losses = []
        ft_losses = []
        acces = []
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
            pt_losses.append(pt_loss.item())
            ft_losses.append(ft_loss.item())
            acces.append(self.compute_metrics(ft_outputs, inputs, labels).item())
        self.pt_scheduler.step()
        self.ft_scheduler.step()
        training_time = time.time() - training_begin_time
        log: dict = self.val_fn(self.model, self.val_dataloader)
        avg_acc = sum(acces) / len(acces)
        avg_pt_loss = sum(pt_losses) / len(pt_losses)
        avg_ft_loss = sum(ft_losses) / len(ft_losses)
        log.update(
            {
                "pt_loss": avg_pt_loss,
                "ft_loss": avg_ft_loss,
                "training_acc": avg_acc,
                "training_time": training_time,
            }
        )
        return log

    def finetune_epoch(
        self,
    ):
        training_begin_time = time.time()
        losses = []
        acces = []
        self.model.train()
        for inputs, labels in tqdm(iter(self.ft_dataloader)):
            outputs = self.ft_forwardfn(self.model, inputs, labels, None)
            loss: torch.Tensor = self.pt_lossfn(outputs, inputs)
            loss.backward()
            self.ft_optimizer.step()
            self.ft_optimizer.zero_grad()
            losses.append(loss.item())
            acces.append(self.compute_metrics(outputs, inputs, labels).item())
        self.ft_scheduler.step()
        training_time = time.time() - training_begin_time
        log: dict = self.val_fn(self.model, self.val_dataloader)
        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(acces) / len(acces)
        log.update(
            {
                "pt_loss": 0.0,
                "ft_loss": avg_loss,
                "training_acc": avg_acc,
                "training_time": training_time,
            }
        )
        return log

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
