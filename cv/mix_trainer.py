import abc
from typing import Callable

from training_args import MixTrainingArgs


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
        args: MixTrainingArgs,
        pre_train_dataset,
        eval_dataset,
        model_init: Callable,
        compute_metrics: Callable,
        callbacks,
        optimizers,
    ):
        self.model = model
        self.args = args
        self.pre_train_dataset = pre_train_dataset
        self.eval_dataset = eval_dataset
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.optimizers = optimizers

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
