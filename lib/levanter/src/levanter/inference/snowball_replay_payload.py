# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Portable, token-bound evidence from Snowball production engine decoding."""

import dataclasses
import hashlib
import json
import struct
from dataclasses import dataclass

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import load, save

from levanter.grug.grug_moe import PreparedMoERouteCapture
from levanter.inference.jit_scheduler import _DecodeOutputs
from levanter.kernels.pallas.autotune_utils import named_sharding_of
from levanter.models.lm_model import LmExample
from levanter.models.snowball import (
    SnowballConfig,
    SnowballDecodeRouteCaptureBatch,
    SnowballLMHeadModel,
    SnowballRouteCaptureBatch,
)

_MAGIC = b"SNOWBALL-ENGINE-REPLAY\n"
_SCHEMA_VERSION = 1
_MASK = "contiguous-causal-v1"


def model_binding(model: SnowballLMHeadModel) -> str:
    """Bind configuration and vocabulary; the caller's policy ID binds weights."""
    return json.dumps(
        {"config": draccus.encode(model.config), "vocab_size": model.Vocab.size},
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class SnowballReplayLayer:
    selected_experts: np.ndarray
    token_valid: np.ndarray
    qb_alpha: np.ndarray


@dataclass(frozen=True)
class SnowballReplaySampling:
    """Effective engine sampling settings, including the explicit request key."""

    temperature: float
    top_p: float
    max_num_tokens: int
    stop_tokens: tuple[tuple[int, ...], ...]
    key: tuple[int, int]


@dataclass(frozen=True)
class SnowballReplayPayload:
    """Host evidence for one request/choice, including the real terminal input.

    Rows have absolute positions 0..len(tokens)-1. Row p predicts token p+1;
    only rows prompt_length-1..len(tokens)-2 produced generated targets.
    Sampled logprobs follow the engine's temperature/top-p transformed logits
    (temperature zero uses unscaled logits with greedy token selection).
    """

    request_id: int
    choice: int
    tokens: np.ndarray
    positions: np.ndarray
    prompt_length: int
    layers: tuple[SnowballReplayLayer, ...]
    sampled_logprobs: np.ndarray
    sampling: SnowballReplaySampling
    finish_reason: str
    policy_id: str
    model_binding: str
    binding: str

    def _metadata(self) -> dict:
        return {
            "version": _SCHEMA_VERSION,
            "mask": _MASK,
            "route_source": "engine.decode_with_route_captures",
            "logprob_semantics": "engine-sampling-logits-v1",
            "request_id": self.request_id,
            "choice": self.choice,
            "prompt_length": self.prompt_length,
            "sampling": dataclasses.asdict(self.sampling),
            "finish_reason": self.finish_reason,
            "policy_id": self.policy_id,
            "model_binding": self.model_binding,
            "num_layers": len(self.layers),
        }

    def _arrays(self) -> dict[str, np.ndarray]:
        arrays = {"tokens": self.tokens, "positions": self.positions, "sampled_logprobs": self.sampled_logprobs}
        for index, layer in enumerate(self.layers):
            for name in ("selected_experts", "token_valid", "qb_alpha"):
                arrays[f"layer.{index}.{name}"] = getattr(layer, name)
        return arrays

    def _digest(self) -> str:
        digest = hashlib.sha256(json.dumps(self._metadata(), sort_keys=True, separators=(",", ":")).encode())
        for name, array in sorted(self._arrays().items()):
            digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
        return digest.hexdigest()

    def validate(self) -> None:
        """Reject missing rows, incompatible schemas and altered evidence."""
        self._validate_arrays()
        if self.binding != self._digest():
            raise ValueError("replay evidence binding differs: tokens, row order, prompt, policy or routes altered")

    def _validate_arrays(self) -> None:
        n = len(self.tokens)
        if self.tokens.dtype != np.int32 or self.tokens.shape != (n,) or np.any(self.tokens < 0):
            raise ValueError("replay tokens must be a flat int32 vocabulary sequence")
        if self.positions.dtype != np.int32 or not np.array_equal(self.positions, np.arange(n)):
            raise ValueError("replay rows must cover every contiguous input, including the terminal token")
        if not 0 < self.prompt_length < n or self.choice != 0 or not self.policy_id:
            raise ValueError("replay requires a nonempty prompt, continuation and immutable policy identity")
        if self.finish_reason not in ("STOP", "LENGTH"):
            raise ValueError("replay supports completed STOP/LENGTH requests")
        if self.sampled_logprobs.dtype != np.float32 or self.sampled_logprobs.shape != (n - self.prompt_length,):
            raise ValueError("sampled logprobs must cover every generated target")
        if not np.all(np.isfinite(self.sampled_logprobs)) or not self.layers:
            raise ValueError("replay requires finite sampled logprobs and per-layer evidence")
        for layer in self.layers:
            if layer.selected_experts.dtype != np.int32 or layer.selected_experts.ndim != 2:
                raise ValueError("expert IDs must be int32 rows by top-k")
            if layer.selected_experts.shape[0] != n or np.any(layer.selected_experts < 0):
                raise ValueError("expert rows must cover every input")
            if layer.token_valid.dtype != np.bool_ or layer.token_valid.shape != (n,) or not layer.token_valid.all():
                raise ValueError("replay requires fully valid per-request routes")
            if layer.qb_alpha.dtype != np.float32 or layer.qb_alpha.shape != (n, 1):
                raise ValueError("qb_alpha must have shape (input rows, 1) and dtype float32")
            if not np.all(np.isfinite(layer.qb_alpha)):
                raise ValueError("qb_alpha must be finite")

    @property
    def raw_policy_logprobs(self) -> np.ndarray:
        """Sampled probabilities eligible for raw-policy comparison at T=1, p=1."""
        self.validate()
        if self.sampling.temperature != 1.0 or self.sampling.top_p != 1.0:
            raise ValueError("raw-policy comparison requires temperature=1 and top_p=1")
        return self.sampled_logprobs

    def validate_model(self, model: SnowballLMHeadModel, *, policy_id: str) -> None:
        """Match architecture and policy while allowing backends to select their own kernels."""
        self.validate()
        if policy_id != self.policy_id:
            raise ValueError("replay immutable policy identity differs")
        state = json.loads(self.model_binding)
        serving_config = draccus.decode(SnowballConfig, state["config"])
        serving_architecture = serving_config.to_hf_config(state["vocab_size"]).to_dict()
        learner_architecture = model.config.to_hf_config(model.Vocab.size).to_dict()
        if serving_architecture != learner_architecture:
            raise ValueError("replay model architecture differs")
        if len(self.tokens) > model.config.max_seq_len or np.any(self.tokens >= model.Vocab.size):
            raise ValueError("replay tokens exceed the bound model limits")
        if len(self.layers) != model.config.num_layers or any(
            layer.selected_experts.shape[1] != model.config.num_experts_per_token
            or np.any(layer.selected_experts >= model.config.num_experts)
            for layer in self.layers
        ):
            raise ValueError("replay expert layout differs from the bound model")

    def training_inputs(
        self, model: SnowballLMHeadModel, *, policy_id: str
    ) -> tuple[LmExample, SnowballRouteCaptureBatch]:
        """Reconstruct causal learner inputs using only transported serving routes."""
        self.validate_model(model, policy_id=policy_id)
        tokens = hax.named(jnp.asarray(self.tokens), "position")
        example = LmExample.from_prompt_and_completion(tokens.axes[0], tokens, self.prompt_length)
        routes = tuple(
            PreparedMoERouteCapture(
                jnp.asarray(layer.selected_experts), jnp.asarray(layer.token_valid), jnp.asarray(layer.qb_alpha)
            )
            for layer in self.layers
        )
        return example, SnowballRouteCaptureBatch(routes, jnp.asarray(self.tokens[None, :]), None)

    def to_bytes(self) -> bytes:
        """Serialize versioned JSON metadata and safetensors arrays; never pickle."""
        self.validate()
        metadata = json.dumps({**self._metadata(), "binding": self.binding}, sort_keys=True).encode()
        return _MAGIC + struct.pack(">I", len(metadata)) + metadata + save(self._arrays())

    @classmethod
    def from_bytes(cls, data: bytes) -> "SnowballReplayPayload":
        """Reload a payload and verify its full token/route/metadata binding."""
        prefix = len(_MAGIC)
        if not data.startswith(_MAGIC) or len(data) < prefix + 4:
            raise ValueError("incompatible or truncated Snowball replay payload")
        length = struct.unpack(">I", data[prefix : prefix + 4])[0]
        start = prefix + 4
        if length > len(data) - start:
            raise ValueError("truncated Snowball replay metadata")
        try:
            metadata = json.loads(data[start : start + length])
            if type(metadata["version"]) is not int or metadata.pop("version") != _SCHEMA_VERSION:
                raise ValueError("incompatible Snowball replay schema version")
            if (
                metadata.pop("mask") != _MASK
                or metadata.pop("route_source") != "engine.decode_with_route_captures"
                or metadata.pop("logprob_semantics") != "engine-sampling-logits-v1"
            ):
                raise ValueError("incompatible Snowball replay causal or sampling contract")
            arrays = load(data[start + length :])
            layers = tuple(
                SnowballReplayLayer(
                    *(arrays.pop(f"layer.{index}.{name}") for name in ("selected_experts", "token_valid", "qb_alpha"))
                )
                for index in range(metadata.pop("num_layers"))
            )
            sampling = metadata.pop("sampling")
            sampling["stop_tokens"] = tuple(tuple(row) for row in sampling["stop_tokens"])
            sampling["key"] = tuple(sampling["key"])
            payload = cls(layers=layers, sampling=SnowballReplaySampling(**sampling), **arrays, **metadata)
        except (KeyError, TypeError, ValueError, SafetensorError) as exc:
            raise ValueError(f"invalid Snowball replay payload: {exc}") from exc
        payload.validate()
        return payload


class _SnowballRouteBuffer(eqx.Module):
    """One bounded engine chunk. No KV pages or vocabulary logits are retained."""

    token_ids: jax.Array
    position_ids: jax.Array
    slot_ids: jax.Array
    token_valid: jax.Array
    selected_experts: jax.Array
    qb_alpha: jax.Array
    size: jax.Array

    @staticmethod
    def init(model: SnowballLMHeadModel, capacity: int) -> "_SnowballRouteBuffer":
        cfg = model.config
        expert_dtype = jnp.uint8 if cfg.num_experts <= 256 else jnp.int32
        return _SnowballRouteBuffer(
            jnp.full((capacity,), -1, jnp.int32),
            jnp.full((capacity,), -1, jnp.int32),
            jnp.full((capacity,), -1, jnp.int32),
            jnp.zeros((capacity,), jnp.bool_),
            jnp.zeros((capacity, cfg.num_layers, cfg.num_experts_per_token), expert_dtype),
            jnp.zeros((capacity, cfg.num_layers, 1), jnp.float32),
            jnp.asarray(0, jnp.int32),
        )

    def append(self, capture: SnowballDecodeRouteCaptureBatch, slots: hax.NamedArray) -> "_SnowballRouteBuffer":
        def append(array, values):
            sharding = named_sharding_of(array)
            if sharding is not None:
                values = jax.sharding.reshard(values, sharding)
            return jax.lax.dynamic_update_slice_in_dim(array, values, self.size, axis=0)

        return _SnowballRouteBuffer(
            append(self.token_ids, capture.token_ids),
            append(self.position_ids, capture.position_ids),
            append(self.slot_ids, slots.array),
            append(self.token_valid, capture.token_valid),
            append(
                self.selected_experts,
                jnp.stack(
                    [route.selected_experts.astype(self.selected_experts.dtype) for route in capture.route_captures],
                    axis=1,
                ),
            ),
            append(self.qb_alpha, jnp.stack([route.qb_alpha for route in capture.route_captures], axis=1)),
            self.size + capture.token_ids.shape[0],
        )


class _SnowballDecodeOutputs(eqx.Module):
    outputs: _DecodeOutputs
    routes: _SnowballRouteBuffer


@dataclass(frozen=True)
class _RequestRows:
    tokens: np.ndarray
    positions: np.ndarray
    selected_experts: np.ndarray
    qb_alpha: np.ndarray

    def prefix(self, length: int) -> "_RequestRows":
        return _RequestRows(
            self.tokens[:length], self.positions[:length], self.selected_experts[:length], self.qb_alpha[:length]
        )


class _SnowballReplayCollector:
    """Resolve serving slot ownership at extraction, before slots are reused."""

    def __init__(self):
        self.rows: dict[tuple[int, int], list[_RequestRows]] = {}
        self.retain: set[int] = set()

    def append(self, buffer: _SnowballRouteBuffer, local_map: dict[int, tuple[int, int]]) -> None:
        n = int(buffer.size)
        valid = buffer.token_valid[:n]
        valid_rows = np.flatnonzero(valid)
        slots = buffer.slot_ids[valid_rows]
        if not valid_rows.size:
            return
        order = np.argsort(slots, kind="stable")
        rows = valid_rows[order]
        ordered_slots = slots[order]
        boundaries = np.r_[0, np.flatnonzero(ordered_slots[1:] != ordered_slots[:-1]) + 1, len(rows)]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            identity = local_map.get(int(ordered_slots[start]))
            if identity is None:
                raise ValueError("serving capture contains an unmapped request slot")
            if identity[0] not in self.retain:
                continue
            indices = rows[start:stop]
            self.rows.setdefault(identity, []).append(
                _RequestRows(
                    buffer.token_ids[indices],
                    buffer.position_ids[indices],
                    buffer.selected_experts[indices],
                    buffer.qb_alpha[indices],
                )
            )

    def prefix(self, request_id: int, length: int) -> tuple[_RequestRows, ...]:
        chunks = self.rows[(request_id, 0)]
        retained = []
        covered = 0
        for chunk in chunks:
            count = min(len(chunk.tokens), length - covered)
            if count:
                retained.append(chunk if len(chunk.tokens) == count else chunk.prefix(count))
                covered += count
            if covered == length:
                return tuple(retained)
        raise ValueError("Retained KV prefix has no corresponding routes")

    def reuse_prefix(self, request_id: int, rows: tuple[_RequestRows, ...], length: int) -> None:
        self.rows[(request_id, 0)] = list(rows)
        self.rows[(request_id, 0)] = list(self.prefix(request_id, length))

    def payload(
        self,
        *,
        request_id: int,
        tokens: list[int],
        prompt_length: int,
        sampling: SnowballReplaySampling,
        logprobs: list[float],
        finish_reason: str,
        policy_id: str,
        model: SnowballLMHeadModel,
    ) -> SnowballReplayPayload:
        chunks = self.rows.pop((request_id, 0))
        positions = np.concatenate([chunk.positions for chunk in chunks])
        order = np.argsort(positions) if np.any(positions[1:] < positions[:-1]) else slice(None)
        captured_tokens = np.concatenate([chunk.tokens for chunk in chunks])[order]
        if not np.array_equal(captured_tokens, tokens):
            raise ValueError("serving capture token binding differs from generated request")
        selected = np.concatenate([chunk.selected_experts for chunk in chunks])[order]
        qb_alpha = np.concatenate([chunk.qb_alpha for chunk in chunks])[order]
        valid = np.ones(len(captured_tokens), np.bool_)
        layers = tuple(
            SnowballReplayLayer(
                np.ascontiguousarray(selected[:, layer], dtype=np.int32),
                valid,
                np.ascontiguousarray(qb_alpha[:, layer]),
            )
            for layer in range(model.config.num_layers)
        )
        payload = SnowballReplayPayload(
            request_id=request_id,
            choice=0,
            tokens=captured_tokens,
            positions=positions[order],
            prompt_length=prompt_length,
            layers=layers,
            sampled_logprobs=np.asarray(logprobs, np.float32),
            sampling=sampling,
            finish_reason=finish_reason,
            policy_id=policy_id,
            model_binding=model_binding(model),
            binding="",
        )
        payload._validate_arrays()
        return dataclasses.replace(payload, binding=payload._digest())
