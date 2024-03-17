import abc
import dataclasses
import json
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn


class MixTrainer(abc.ABC):
    """
    Args:
        model:
            The model to train
        args:
            Training arguments
        data_collator:
        pre_train_dataset:
        eval_dataset:
        model_init:
        compute_metrics:
        callbacks:
        optimizers:
    """

    def __init__(
        self,
        model,
        pt_dataset,
        ft_dataset,
        eval_dataset,
        compute_metrics: Callable,
        optimizers,
        schedulers,
    ):
        self.model = model
        self.pt_dataset = pt_dataset
        self.ft_dataset = ft_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.training_steps = self._make_training_steps()

    @abc.abstractmethod
    def _make_training_steps(
        self,
    ) -> list[
        list[
            Callable[
                ["MixTrainer", nn.Module, Dict[str, Union[torch.Tensor, Any]]],
                torch.Tensor,
            ]
        ]
    ]:
        """
        make_mode defines how to partition the training process into pre-training, mix-training and fine-tuning
        to implement this method, you need to gene

        Returns:
            a 2-d list of functions that defines the training process
            each list of functions defines the training process for one epoch
            one function defines the training process for one step
        """
        raise NotImplementedError

    def inner_training_loop(self):
        for epoch, epoch_steps in enumerate(self.training_steps):
            self.state.epoch = epoch
            for step in epoch_steps:
                step(self)

    def training_step(self, step: int):
        """
        Train one epoch

        Args:
            mode: a function that defines the training process for one epoch
        """
        self.training_steps[step]()

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        """
        Entry point for training.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
        """
        # TODO: training_steps based training
        pass
