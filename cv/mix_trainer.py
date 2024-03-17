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
        model,
        pt_dataloader,
        ft_dataloader,
        eval_dataset,
        compute_metrics: Callable,
        optimizers: Dict[str, Any],
        schedulers: Dict[str, Any],
        forward_fns: Dict[str, Callable],
        loss_fns: Dict[str, Callable],
    ):
        self.model = model
        self.pt_dataloader = pt_dataloader
        self.ft_dataloader = ft_dataloader
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.forward_fns = forward_fns
        self.loss_fns = loss_fns
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
        for epoch, epoch_steps in enumerate(self.training_steps):
            logging.info(f"Epoch {epoch}")
            for step in tqdm(epoch_steps):
                step(self)
