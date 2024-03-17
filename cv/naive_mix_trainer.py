from typing import Any, Callable

from mix_trainer import MixTrainer


class NaiveMixTrainer(MixTrainer):
    def __init__(
        self,
        model,
        args,
        pre_train_dataset,
        eval_dataset,
        model_init: Callable[..., Any],
        compute_metrics: Callable[..., Any],
        callbacks,
        optimizers,
    ):
        super().__init__(
            model,
            args,
            pre_train_dataset,
            eval_dataset,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
        )

    def train(self):
        pass
