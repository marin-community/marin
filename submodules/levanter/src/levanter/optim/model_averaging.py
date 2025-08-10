import abc
import dataclasses
from typing import Generic, TypeVar

import draccus
import equinox as eqx
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
