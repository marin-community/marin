# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
from typing import Generic, TypeVar

import draccus
import equinox as eqx
import jax.numpy as jnp
import optax


S = TypeVar("S")
M = TypeVar("M")


class ModelAveraging(eqx.Module, Generic[M]):
    """
    This is the interface for model averaging algorithms. Model averaging algorithms are used to average
    the parameters of a model over multiple training steps. This is useful for improving generalization
    """

    @abc.abstractmethod
    def update(self: S, model: M, step: int) -> S:
        pass

    @property
    @abc.abstractmethod
    def model_params(self) -> M:
        pass


class EmaModelAveraging(ModelAveraging[M]):
    """
    Exponential moving average model averaging
    """

    model: M
    beta: float = eqx.field(static=True)

    def update(self: S, new_model: M, step: int) -> S:
        del step
        # 1 - beta because increment_update expects the weight of the new model
        return dataclasses.replace(self, model=optax.incremental_update(new_model, self.model, 1 - self.beta))  # type: ignore

    @property
    def model_params(self) -> M:
        return self.model


class EmaDecaySqrtModelAveraging(ModelAveraging[M]):
    """Hybrid EMA followed by :math:`1 - \sqrt{x}` decay.

    This implementation keeps a running total of the weight mass so the
    average can be queried at any step. After ``decay_steps`` updates the
    raw weight becomes zero and the average stops changing.
    """

    model: M
    switch_step: int = eqx.field(static=True)
    decay_steps: int = eqx.field(static=True)
    beta: float = eqx.field(static=True, default=0.999)
    tot_weight: float = 0.0

    def _raw_weight(self, step: int) -> float:
        if step < self.switch_step:
            return 1.0 - self.beta
        t = step - self.switch_step
        frac = jnp.clip(t / self.decay_steps, 0.0, 1.0)
        return float(1.0 - jnp.sqrt(frac))

    def update(self, new_model: M, step: int) -> "EmaDecaySqrtModelAveraging[M]":
        w = self._raw_weight(step)
        new_tot_w = self.tot_weight + w
        alpha = 0.0 if new_tot_w == 0.0 else w / new_tot_w
        updated = optax.incremental_update(new_model, self.model, alpha)
        return dataclasses.replace(self, model=updated, tot_weight=new_tot_w)  # type: ignore[arg-type]

    @property
    def model_params(self) -> M:
        return self.model


class ModelAveragingConfig(abc.ABC, draccus.ChoiceRegistry, Generic[M]):
    @abc.abstractmethod
    def create(self, model: M) -> ModelAveraging[M]:
        pass


@ModelAveragingConfig.register_subclass("ema")
@dataclasses.dataclass(frozen=True)
class EmaModelAveragingConfig(ModelAveragingConfig[M]):
    beta: float = 0.999

    def create(self, model: M) -> EmaModelAveraging[M]:
        return EmaModelAveraging(model=model, beta=self.beta)


@ModelAveragingConfig.register_subclass("ema_decay_sqrt")
@dataclasses.dataclass(frozen=True)
class EmaDecaySqrtConfig(ModelAveragingConfig[M]):
    """EMA followed by :math:`1 - \sqrt{x}` decay."""

    beta: float = 0.999
    switch_step: int = 100_000
    decay_steps: int = 100_000

    def create(self, model: M) -> EmaDecaySqrtModelAveraging[M]:
        return EmaDecaySqrtModelAveraging(
            model=model,
            beta=self.beta,
            switch_step=self.switch_step,
            decay_steps=self.decay_steps,
        )
