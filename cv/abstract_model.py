# abstract_model.py

import abc

import torch.nn as nn


class AbstractModel(nn.Module, abc.ABC):
    def __init__(self):
        super(AbstractModel, self).__init__()

    @abc.abstractmethod
    def forward(self, x, mode="train"):
        """
        Forward pass of the model.

        Args:
        x: input tensor
        mode: used for efficient training implementation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_weights(self):
        """
        Method for weight initialization.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load_from_config_yaml(self, config_yaml):
        """
        Method to load model from config yaml.
        """
        raise NotImplementedError
