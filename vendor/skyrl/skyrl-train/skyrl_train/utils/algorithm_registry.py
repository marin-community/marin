"""Registries for policy losses and advantage estimators.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from functools import wraps
from dataclasses import dataclass
from typing import Callable, Union, cast

import ray
from loguru import logger

from marinskyrl.runtime_options import AdvantageEstimator, PolicyLossType
from skyrl_train.config.objective_spec import BUILTIN_LOSS_SPECS, LossSpec
from skyrl_train.utils.function_registry import BaseFunctionRegistry


@dataclass(frozen=True)
class ExactPhysicalGroup:
    """Require the estimator's complete physical rollout group."""


@dataclass(frozen=True)
class MinimumBaselineEligibleGroup:
    """Allow a ragged baseline cohort down to a user-configured floor."""


@dataclass(frozen=True)
class NoGroupAdvantage:
    """Declare that the estimator does not compute group-relative advantages."""


GroupAdvantageContract = ExactPhysicalGroup | MinimumBaselineEligibleGroup | NoGroupAdvantage


class AdvantageEstimatorRegistry(BaseFunctionRegistry):
    """
    Registry for advantage estimator functions.

    This registry allows users to register custom advantage estimators without modifying
    the skyrl_train package. Custom estimators can be registered by calling
    AdvantageEstimatorRegistry.register() directly or by using the @register_advantage_estimator
    decorator.

    See examples/algorithms/custom_advantage_estimator for a simple example of how to
    register and use custom advantage estimators.
    """

    _actor_name = "advantage_estimator_registry"
    _function_type = "advantage estimator"
    # Driver-only metadata: validate_cfg materializes it into primitive config before Ray dispatch.
    _group_contracts: dict[str, GroupAdvantageContract] = {}

    @classmethod
    def register(cls, name: str, func: Callable, *, group_contract: GroupAdvantageContract | None = None):
        if group_contract is None:
            raise ValueError(f"advantage estimator '{name}' must declare a group_contract")
        super().register(name, func)
        cls._group_contracts[name] = group_contract

    @classmethod
    def group_contract(cls, name: str) -> GroupAdvantageContract:
        try:
            return cls._group_contracts[name]
        except KeyError as error:
            raise ValueError(f"advantage estimator '{name}' has no local group contract") from error

    @classmethod
    def unregister(cls, name: str):
        super().unregister(name)
        cls._group_contracts.pop(name, None)


@dataclass(frozen=True)
class PolicyLossRegistration:
    """A policy loss callable and its declaration, serialized together across Ray processes."""

    function: Callable
    spec: LossSpec

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class PolicyLossRegistry(BaseFunctionRegistry):
    """
    Registry for policy loss functions.

    This registry allows users to register custom policy loss functions without modifying
    the skyrl_train package. Custom functions can be registered by calling
    PolicyLossRegistry.register() directly or by using the @register_policy_loss
    decorator.

    See examples/algorithms/custom_policy_loss for a simple example of how to
    register and use custom policy loss functions.
    """

    _actor_name = "policy_loss_registry"
    _function_type = "policy loss"

    @classmethod
    def register(cls, name: str, func: Callable, *, spec: LossSpec | None = None):
        spec = BUILTIN_LOSS_SPECS.get(name, spec)
        if spec is None:
            raise ValueError(f"custom policy loss {name!r} requires a LossSpec")
        super().register(name, PolicyLossRegistration(func, spec))

    @classmethod
    def spec(cls, name: str) -> LossSpec:
        if name in BUILTIN_LOSS_SPECS:
            return BUILTIN_LOSS_SPECS[name]
        return cast(PolicyLossRegistration, cls.get(name)).spec


def register_advantage_estimator(name: Union[str, AdvantageEstimator], *, group_contract: GroupAdvantageContract):
    """Decorator to register an advantage estimator function."""
    registry_name = name.value if isinstance(name, AdvantageEstimator) else name

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        AdvantageEstimatorRegistry.register(registry_name, wrapper, group_contract=group_contract)
        return wrapper

    return decorator


def register_policy_loss(name: Union[str, PolicyLossType], *, spec: LossSpec | None = None):
    """Decorator to register a policy loss function."""
    registry_name = name.value if isinstance(name, PolicyLossType) else name

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        PolicyLossRegistry.register(registry_name, wrapper, spec=spec)
        return wrapper

    return decorator


def sync_registries():
    """Sync the registries with the ray actor once ray is initialized"""
    if not ray.is_initialized():
        raise ValueError("Ray is not initialized, cannot sync registries")
    PolicyLossRegistry.sync_with_actor()
    AdvantageEstimatorRegistry.sync_with_actor()
    logger.info("Synced registries to ray actor")
