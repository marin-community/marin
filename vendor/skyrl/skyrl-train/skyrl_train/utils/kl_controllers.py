"""KL coefficient controllers.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def get_kl_controller(algorithm_cfg: DictConfig):
    if algorithm_cfg.kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=algorithm_cfg.kl_loss_coef)
    elif algorithm_cfg.kl_ctrl.type == "adaptive":
        if algorithm_cfg.kl_ctrl.horizon <= 0:
            raise ValueError(f"horizon must be larger than 0. Got {algorithm_cfg.kl_ctrl.horizon}")
        return AdaptiveKLController(
            init_kl_coef=algorithm_cfg.kl_loss_coef,
            target=algorithm_cfg.kl_ctrl.kl_target,
            horizon=algorithm_cfg.kl_ctrl.horizon,
        )
    else:
        raise ValueError(f"Invalid KL controller type: {algorithm_cfg.kl_ctrl.type}")
