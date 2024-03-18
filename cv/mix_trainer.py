import abc
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from tqdm import tqdm


class MixTrainer(abc.ABC):
    """
    TODO
    """

    def __init__(
        self,
        pt_lossfn: Callable,
        pt_optimizer,
        pt_scheduler,
        pt_dataloader,
        ft_lossfn: Callable,
        ft_optimizer,
        ft_scheduler,
        ft_dataloader,
    ):
        self.pt_lossfn = pt_lossfn
        self.pt_optimizer = pt_optimizer
        self.pt_scheduler = pt_scheduler
        self.pt_dataloader = pt_dataloader
        self.ft_lossfn = ft_lossfn
        self.ft_optimizer = ft_optimizer
        self.ft_scheduler = ft_scheduler
        self.ft_dataloader = ft_dataloader

    @abc.abstractmethod
    def _make_training_epochs(
        self,
    ) -> list[Callable[["MixTrainer"], None]]:
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
