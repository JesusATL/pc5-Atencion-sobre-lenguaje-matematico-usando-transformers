"""A wrapper class for optimizer """
import numpy as np
from torch import optim


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer: optim.Optimizer, d_model: int, n_warmup_steps: int) -> None:
        self._optimizer = optimizer
        self.n_warmup_steps: int = n_warmup_steps
        self.n_current_steps: int = 0
        self.init_lr: float = 1 / np.sqrt(d_model)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def _get_lr_scale(self) -> float:
        return np.min(
            [
                self.n_current_steps**(-0.5),
                self.n_warmup_steps**(-1.5) * self.n_current_steps
            ]

        )

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
