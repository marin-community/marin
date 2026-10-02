# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage complete HF weight publications before replacing a serving model."""

import math
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax.state_dict import flatten_modules_for_export, from_torch_compatible_state_dict, to_state_dict

from levanter.models.lm_model import LmHeadModel
from levanter.trainer import TrainerConfig


@dataclass(frozen=True)
class WeightTransferConfig:
    """Opt-in, single-device Torch broadcast receiver configuration."""

    backend: Literal["gloo", "nccl"]
    max_staging_bytes: int
    timeout: float = 120.0


class WeightReceiver(Protocol):
    def receive(self, dtype: str, shape: tuple[int, ...]) -> jax.Array: ...


class WeightPublicationContext(Protocol):
    model: LmHeadModel
    model_version: int
    admission_lock: threading.Lock

    def reset_prefix_cache(self) -> None: ...

    def reload(self, source: Callable[[LmHeadModel], LmHeadModel], *, expected_version: int) -> int: ...


@dataclass(frozen=True)
class WeightPublication:
    publication_id: str
    model_version: int


@dataclass
class _StagedWeights:
    publication: WeightPublication
    model: LmHeadModel
    specs: dict[str, jax.ShapeDtypeStruct]
    tensors: dict[str, jax.Array]


_EXPERT_PARAMETER = re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def _weight_shapes(model: LmHeadModel) -> dict[str, jax.ShapeDtypeStruct]:
    return eqx.filter_eval_shape(lambda value: to_state_dict(flatten_modules_for_export(value)), model)


def _canonical_weight(name: str, specs: dict[str, jax.ShapeDtypeStruct]) -> tuple[str, int | None]:
    if name in specs:
        return name, None
    match = _EXPERT_PARAMETER.fullmatch(name)
    if match is not None:
        bank = f"{match[1]}.{match[3]}.weight"
        expert = int(match[2])
        if bank in specs and 0 <= expert < specs[bank].shape[0]:
            return bank, expert
    raise ValueError(f"Unknown weight: {name}")


class WeightReloadSession:
    """Accept one complete publication; failures leave the serving model unchanged."""

    def __init__(self, context: WeightPublicationContext, trainer: TrainerConfig, config: WeightTransferConfig):
        if jax.device_count() != 1 or jax.process_count() != 1:
            raise ValueError("Remote weight transfer currently requires one serving device and one process")
        self.context = context
        self.trainer = trainer
        self.config = config
        self.lock = threading.Lock()
        self.staged: _StagedWeights | None = None

    def begin(self) -> WeightPublication:
        """Start a publication, invalidating any unfinished previous transfer."""
        with self.lock:
            with self.context.admission_lock:
                model, version = self.context.model, self.context.model_version
            with self.trainer.use_device_mesh(), hax.axis_mapping(self.trainer.compute_axis_mapping):
                specs = _weight_shapes(model)
            required_bytes = sum(math.prod(spec.shape) * spec.dtype.itemsize for spec in specs.values())
            if required_bytes > self.config.max_staging_bytes:
                raise ValueError(
                    f"Publication requires {required_bytes} staging bytes, budget is {self.config.max_staging_bytes}"
                )
            publication = WeightPublication(uuid.uuid4().hex, version)
            self.staged = _StagedWeights(publication, model, specs, {})
            return publication

    def _current(self, publication: WeightPublication) -> _StagedWeights:
        if self.staged is None or self.staged.publication != publication:
            raise ValueError("Weight publication is no longer active")
        return self.staged

    def receive(
        self, publication: WeightPublication, name: str, dtype: str, shape: tuple[int, ...], receiver: WeightReceiver
    ) -> None:
        """Receive and stage one tensor after validating its wire metadata."""
        with self.lock:
            staged = self._current(publication)
            try:
                canonical, expert = _canonical_weight(name, staged.specs)
                spec = staged.specs[canonical]
                expected_shape = spec.shape if expert is None else spec.shape[1:]
                if shape != expected_shape or jnp.dtype(dtype.removeprefix("torch.")) != spec.dtype:
                    raise ValueError(f"Weight {name} does not match the serving shape and dtype")
                if name in staged.tensors or canonical in staged.tensors:
                    raise ValueError(f"Weight {name} was already received")
                if expert is None and any(
                    _canonical_weight(key, staged.specs)[0] == canonical for key in staged.tensors
                ):
                    raise ValueError(f"Weight {name} mixes bank and individual expert tensors")
                tensor = receiver.receive(dtype, shape)
                if tensor.shape != expected_shape or tensor.dtype != spec.dtype:
                    raise ValueError(f"Received weight {name} does not match its metadata")
                staged.tensors[name] = tensor
            except Exception:
                self.staged = None
                raise

    def finish(self, publication: WeightPublication) -> int:
        """Install a complete candidate at the serving pause barrier."""
        with self.lock:
            staged = self._current(publication)
            self.staged = None
        with self.trainer.use_device_mesh(), hax.axis_mapping(self.trainer.compute_axis_mapping):
            weights = dict(staged.tensors)
            for name, spec in staged.specs.items():
                if name in weights:
                    continue
                parts = [
                    (expert, tensor)
                    for key, tensor in staged.tensors.items()
                    for canonical, expert in [_canonical_weight(key, staged.specs)]
                    if canonical == name and expert is not None
                ]
                if (
                    not parts
                    or len(parts) != spec.shape[0]
                    or [expert for expert, _ in sorted(parts)] != list(range(spec.shape[0]))
                ):
                    raise ValueError(f"Publication is incomplete: missing weight {name}")
                weights[name] = jnp.stack([tensor for _, tensor in sorted(parts)])
            candidate = from_torch_compatible_state_dict(staged.model, weights)
            candidate = jax.tree.map(
                lambda old, new: jax.device_put(new, old.sharding) if eqx.is_array(old) else new,
                staged.model,
                candidate,
            )
            return self.context.reload(lambda _: candidate, expected_version=publication.model_version)

    def reset_transport(self, reset_receiver: Callable[[], None]) -> None:
        """Discard staged weights and reset their transport atomically."""
        with self.lock:
            self.staged = None
            reset_receiver()
