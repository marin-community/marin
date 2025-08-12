import typing

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import haliax as hax


Arrayish: typing.TypeAlias = hax.NamedArray | np.ndarray | jnp.ndarray


class RunningMean(eqx.Module):
    mean: Arrayish
    total: Arrayish

    @staticmethod
    def zeros_like(x: Arrayish) -> "RunningMean":
        return RunningMean(x * 0.0, x * 0.0)

    def add(self, x: Arrayish, total: Arrayish) -> "RunningMean":
        delta = x - self.mean
        # careful: total and self.total can be 0
        new_total = self.total + total
        ratio = hax.where(new_total, total / new_total, 0.0)
        new_mean = self.mean + delta * ratio
        new_total = self.total + total
        return RunningMean(new_mean, new_total)

    def __add__(self, other: "RunningMean"):
        return self.add(other.mean, other.total)

    def __str__(self):
        return f"RunningMean(mean={self.mean}, total={self.total})"
