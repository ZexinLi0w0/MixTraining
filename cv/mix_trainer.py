import abc
import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn


class MixTrainer(abc.ABC):
    """
    TODO
    """

    def __init__(
        self,
        model,
        device,
        pt_forwardfn: Callable[
            [nn.Module, torch.Tensor, torch.Tensor],
            Union[torch.Tensor, Tuple[torch.Tensor]],
        ],
        pt_lossfn: Callable,
        pt_optimizer,
        pt_scheduler,
        pt_dataloader,
        ft_forwardfn: Callable[
            [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor],
            Union[torch.Tensor, Tuple[torch.Tensor]],
        ],
        ft_lossfn: Callable,
        ft_optimizer,
        ft_scheduler,
        val_dataloader,
        val_fn: Callable,
        ft_dataloader=None,
        same_dataset: bool = False,
    ):
        self.model = model
        self.device = device
        self.pt_forwardfn = pt_forwardfn
        self.pt_lossfn = pt_lossfn
        self.pt_optimizer = pt_optimizer
        self.pt_scheduler = pt_scheduler
        self.pt_dataloader = pt_dataloader
        self.ft_forwardfn = ft_forwardfn
        self.ft_lossfn = ft_lossfn
        self.ft_optimizer = ft_optimizer
        self.ft_scheduler = ft_scheduler
        self.ft_dataloader = ft_dataloader
        self.same_dataset = same_dataset
        self.val_dataloader = val_dataloader
        self.val_fn = val_fn
        if same_dataset:
            # TODO: check if pt_dataloader and ft_dataloader are the same, if not warning
            logging.info(
                "As same_dataset is True, pt_dataloader will be used for ft_dataloader"
            )
            self.ft_dataloader = pt_dataloader

    @abc.abstractmethod
    def _make_training_epochs(
        self,
    ) -> List[Callable]:
        """
        make_mode defines how to partition the training process into pre-training, mix-training and fine-tuning
        to implement this method, you need to gene

        Returns:
            a 2-d list of functions that defines the training process
            each list of functions defines the training process for one epoch
            one function defines the training process for one step
        """
        raise NotImplementedError

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        """
        Entry point for training.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
        """
        # TODO: resume from checkpoint
        for i, epoch in enumerate(self.training_epochs):
            logging.info(f"Epoch {i}")
            epoch()
