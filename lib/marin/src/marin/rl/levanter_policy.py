# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stateful Levanter policy operations for the MarinSkyRL compatibility shim."""

import asyncio
import io
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import requests
from haliax._src.state_dict import flatten_modules_for_export, to_state_dict
from haliax.jax_utils import is_jax_array_like
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmHeadModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route


@dataclass(frozen=True)
class PolicyBatch:
    """The small subset of a SkyRL training batch consumed by Levanter."""

    sequences: np.ndarray
    action_count: int
    attention_mask: np.ndarray
    old_action_log_probs: np.ndarray | None = None
    advantages: np.ndarray | None = None
    loss_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.sequences.ndim != 2:
            raise ValueError(f"sequences must have shape [batch, sequence], got {self.sequences.shape}")
        if self.attention_mask.shape != self.sequences.shape:
            raise ValueError(f"attention_mask must have shape {self.sequences.shape}, got {self.attention_mask.shape}")
        if not 0 < self.action_count < self.sequences.shape[1]:
            raise ValueError(f"action_count must be between one and sequence_length - 1, got {self.action_count}")
        expected = (self.sequences.shape[0], self.action_count)
        for name in ("old_action_log_probs", "advantages", "loss_mask"):
            value = getattr(self, name)
            if value is not None and value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


@dataclass(frozen=True)
class PolicyForwardOutput:
    action_log_probs: np.ndarray


@dataclass(frozen=True)
class PolicyTrainOutput:
    action_log_probs: np.ndarray
    loss: float
    step: int


class WeightPublisher(Protocol):
    """Publishes HF-named JAX weight arrays to a rollout backend."""

    def publish(self, weights: Mapping[str, jax.Array], *, step: int) -> None: ...


def encode_policy_batch(batch: PolicyBatch) -> bytes:
    """Encode a batch as an NPZ payload for the intentionally thin HTTP shim."""
    buffer = io.BytesIO()
    arrays: dict[str, np.ndarray] = {
        "sequences": batch.sequences,
        "action_count": np.asarray(batch.action_count, dtype=np.int32),
        "attention_mask": batch.attention_mask,
    }
    for name in ("old_action_log_probs", "advantages", "loss_mask"):
        value = getattr(batch, name)
        if value is not None:
            arrays[name] = value
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def decode_policy_batch(payload: bytes) -> PolicyBatch:
    """Decode a payload produced by :func:`encode_policy_batch`."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return PolicyBatch(
            sequences=arrays["sequences"],
            action_count=int(arrays["action_count"]),
            attention_mask=arrays["attention_mask"],
            old_action_log_probs=arrays["old_action_log_probs"] if "old_action_log_probs" in arrays else None,
            advantages=arrays["advantages"] if "advantages" in arrays else None,
            loss_mask=arrays["loss_mask"] if "loss_mask" in arrays else None,
        )


def encode_array(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def decode_array(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def encode_train_output(output: PolicyTrainOutput) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        action_log_probs=output.action_log_probs,
        loss=np.asarray(output.loss, dtype=np.float32),
        step=np.asarray(output.step, dtype=np.int64),
    )
    return buffer.getvalue()


def decode_train_output(payload: bytes) -> PolicyTrainOutput:
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return PolicyTrainOutput(
            action_log_probs=arrays["action_log_probs"],
            loss=float(arrays["loss"]),
            step=int(arrays["step"]),
        )


def hf_named_jax_weights(model: LmHeadModel) -> dict[str, jax.Array]:
    """Return the model's HF-compatible state dictionary without copying to the host."""
    arrays_only = eqx.filter(model, is_jax_array_like)
    flattened = flatten_modules_for_export(arrays_only)
    return {name: value for name, value in to_state_dict(flattened).items() if isinstance(value, jax.Array)}


def _action_log_probs(
    model: LmHeadModel,
    sequences: jax.Array,
    attention_mask: jax.Array,
    action_count: int,
) -> jax.Array:
    batch_axis = hax.Axis("batch", sequences.shape[0])
    position_axis = model.Pos.resize(sequences.shape[1])
    tokens = hax.named(sequences, (batch_axis, position_axis))
    segment_ids = hax.named(jnp.where(attention_mask, 0, -1), (batch_axis, position_axis))
    logits = model(tokens, AttentionMask.causal().with_segment_ids(segment_ids)).array
    token_log_probs = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    targets = sequences[:, 1:]
    selected = jnp.take_along_axis(token_log_probs, targets[..., None], axis=-1)[..., 0]
    return selected[:, -action_count:]


class LevanterPolicy:
    """A deliberately small stateful policy implementation for SkyRL calls."""

    def __init__(
        self,
        model: LmHeadModel,
        *,
        learning_rate: float,
        clip_epsilon: float = 0.2,
        weight_publisher: WeightPublisher | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        self.clip_epsilon = clip_epsilon
        self.weight_publisher = weight_publisher
        self.step = 0

    def forward(self, batch: PolicyBatch) -> PolicyForwardOutput:
        action_log_probs = _action_log_probs(
            self.model,
            jnp.asarray(batch.sequences, dtype=jnp.int32),
            jnp.asarray(batch.attention_mask, dtype=jnp.bool_),
            batch.action_count,
        )
        return PolicyForwardOutput(np.asarray(action_log_probs, dtype=np.float32))

    def ppo_train(self, batch: PolicyBatch) -> PolicyTrainOutput:
        if batch.old_action_log_probs is None or batch.advantages is None:
            raise ValueError("ppo_train requires old_action_log_probs and advantages")
        loss_mask = np.ones_like(batch.advantages) if batch.loss_mask is None else batch.loss_mask
        sequences = jnp.asarray(batch.sequences, dtype=jnp.int32)
        attention_mask = jnp.asarray(batch.attention_mask, dtype=jnp.bool_)
        old_action_log_probs = jnp.asarray(batch.old_action_log_probs)
        advantages = jnp.asarray(batch.advantages)
        mask = jnp.asarray(loss_mask)

        def loss_fn(model: LmHeadModel) -> tuple[jax.Array, jax.Array]:
            action_log_probs = _action_log_probs(model, sequences, attention_mask, batch.action_count)
            ratio = jnp.exp(action_log_probs - old_action_log_probs)
            unclipped = ratio * advantages
            clipped = jnp.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            denominator = jnp.maximum(mask.sum(), 1.0)
            loss = -(jnp.minimum(unclipped, clipped) * mask).sum() / denominator
            return loss, action_log_probs

        (loss, action_log_probs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(self.model)
        updates, self.opt_state = self.optimizer.update(
            grads,
            self.opt_state,
            eqx.filter(self.model, eqx.is_inexact_array),
        )
        self.model = eqx.apply_updates(self.model, updates)
        self.step += 1
        return PolicyTrainOutput(np.asarray(action_log_probs, dtype=np.float32), float(loss), self.step)

    def broadcast_weights(self) -> int:
        if self.weight_publisher is None:
            raise RuntimeError("No weight publisher is configured")
        self.weight_publisher.publish(hf_named_jax_weights(self.model), step=self.step)
        return self.step


class TorchDistributedWeightPublisher:
    """Broadcast Levanter weights through an existing torch NCCL process group."""

    def __init__(
        self,
        process_group,
        announce: Callable[[str, str, tuple[int, ...]], None],
        complete: Callable[[], None],
    ) -> None:
        self.process_group = process_group
        self.announce = announce
        self.complete = complete

    def publish(self, weights: Mapping[str, jax.Array], *, step: int) -> None:
        import torch  # noqa: PLC0415 - optional GPU bridge

        del step
        for name, array in weights.items():
            tensor = torch.utils.dlpack.from_dlpack(array)
            self.announce(name, str(tensor.dtype).removeprefix("torch."), tuple(tensor.shape))
            torch.distributed.broadcast(tensor, 0, group=self.process_group)
            self.complete()


def build_levanter_policy_app(
    policy: LevanterPolicy,
    configure_weight_sync: Callable[[dict], WeightPublisher] | None = None,
) -> Starlette:
    """Expose policy operations as a serialized, single-flight HTTP API."""
    operation_lock = asyncio.Lock()

    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok", "step": policy.step})

    async def forward(request: Request) -> Response:
        batch = decode_policy_batch(await request.body())
        async with operation_lock:
            output = await asyncio.to_thread(policy.forward, batch)
        return Response(encode_array(output.action_log_probs), media_type="application/x-npy")

    async def ppo_train(request: Request) -> Response:
        batch = decode_policy_batch(await request.body())
        async with operation_lock:
            output = await asyncio.to_thread(policy.ppo_train, batch)
        return Response(encode_train_output(output), media_type="application/x-npz")

    async def broadcast_weights(_request: Request) -> Response:
        async with operation_lock:
            step = await asyncio.to_thread(policy.broadcast_weights)
        return JSONResponse({"step": step})

    async def configure(request: Request) -> Response:
        if configure_weight_sync is None:
            return JSONResponse({"error": "weight sync configuration is disabled"}, status_code=404)
        payload = await request.json()
        async with operation_lock:
            policy.weight_publisher = await asyncio.to_thread(configure_weight_sync, payload)
        return JSONResponse({"status": "ready"})

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/forward", forward, methods=["POST"]),
            Route("/ppo_train", ppo_train, methods=["POST"]),
            Route("/broadcast_weights", broadcast_weights, methods=["POST"]),
            Route("/configure_weight_sync", configure, methods=["POST"]),
        ]
    )


@dataclass(frozen=True)
class LevanterPolicyClient:
    """Synchronous client used behind SkyRL's Ray compatibility actor."""

    base_url: str
    timeout_seconds: float = 1800.0

    def forward(self, batch: PolicyBatch) -> PolicyForwardOutput:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/forward",
            data=encode_policy_batch(batch),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return PolicyForwardOutput(decode_array(response.content))

    def ppo_train(self, batch: PolicyBatch) -> PolicyTrainOutput:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/ppo_train",
            data=encode_policy_batch(batch),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return decode_train_output(response.content)

    def broadcast_weights(self) -> int:
        response = requests.post(
            f"{self.base_url.rstrip('/')}/broadcast_weights",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return int(response.json()["step"])


@dataclass(frozen=True)
class LevanterPolicyGroupClient:
    """Fan out SkyRL operations to every process in a Levanter gang."""

    clients: tuple[LevanterPolicyClient, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("A Levanter policy group needs at least one client")

    def _batch_shards(self, batch: PolicyBatch) -> tuple[PolicyBatch, ...]:
        if batch.sequences.shape[0] % len(self.clients) != 0:
            raise ValueError(
                f"batch size {batch.sequences.shape[0]} must divide evenly across {len(self.clients)} policy processes"
            )
        indices = np.array_split(np.arange(batch.sequences.shape[0]), len(self.clients))
        return tuple(
            PolicyBatch(
                sequences=batch.sequences[index],
                action_count=batch.action_count,
                attention_mask=batch.attention_mask[index],
                old_action_log_probs=(
                    batch.old_action_log_probs[index] if batch.old_action_log_probs is not None else None
                ),
                advantages=batch.advantages[index] if batch.advantages is not None else None,
                loss_mask=batch.loss_mask[index] if batch.loss_mask is not None else None,
            )
            for index in indices
        )

    def forward(self, batch: PolicyBatch) -> PolicyForwardOutput:
        shards = self._batch_shards(batch)
        with ThreadPoolExecutor(max_workers=len(self.clients), thread_name_prefix="levanter-policy") as executor:
            outputs = tuple(executor.map(lambda pair: pair[0].forward(pair[1]), zip(self.clients, shards, strict=True)))
        return PolicyForwardOutput(np.concatenate([output.action_log_probs for output in outputs]))

    def ppo_train(self, batch: PolicyBatch) -> PolicyTrainOutput:
        shards = self._batch_shards(batch)
        with ThreadPoolExecutor(max_workers=len(self.clients), thread_name_prefix="levanter-policy") as executor:
            outputs = tuple(
                executor.map(lambda pair: pair[0].ppo_train(pair[1]), zip(self.clients, shards, strict=True))
            )
        steps = {output.step for output in outputs}
        if len(steps) != 1:
            raise RuntimeError(f"Levanter policy processes returned different optimizer steps: {sorted(steps)}")
        return PolicyTrainOutput(
            action_log_probs=np.concatenate([output.action_log_probs for output in outputs]),
            loss=float(np.mean([output.loss for output in outputs])),
            step=steps.pop(),
        )

    def broadcast_weights(self) -> int:
        with ThreadPoolExecutor(max_workers=len(self.clients), thread_name_prefix="levanter-policy") as executor:
            steps = set(executor.map(lambda client: client.broadcast_weights(), self.clients))
        if len(steps) != 1:
            raise RuntimeError(f"Levanter policy processes published different optimizer steps: {sorted(steps)}")
        return steps.pop()
