# abstract_model.py

import torch.nn as nn
import abc

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