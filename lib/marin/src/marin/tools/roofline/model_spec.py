# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model presets and derived Grug/MoE roofline dimensions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class MeshSpec:
    replica_dcn: int
    data: int
    expert: int
    model: int

    @property
    def devices(self) -> int:
        return self.replica_dcn * self.data * self.expert * self.model

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeshSpec:
        return cls(
            replica_dcn=int(payload["replica_dcn"]),
            data=int(payload["data"]),
            expert=int(payload["expert"]),
            model=int(payload["model"]),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "replica_dcn": self.replica_dcn,
            "data": self.data,
            "expert": self.expert,
            "model": self.model,
        }


@dataclass(frozen=True)
class ModelSpec:
    model_preset: str
    sequence_length: int
    sliding_window: int
    long_attention_every: int | None
    num_layers: int
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    top_k: int
    num_heads: int
    head_dim: int
    vocab_size: int
    global_batch_sequences: int
    global_batch_tokens: int
    remat: str
    attention_impl: str
    cross_entropy_impl: str
    moe_impl: str
    optimizer: str
    muon_ns_iters: int
    mesh: MeshSpec

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelSpec:
        return cls(
            model_preset=str(payload["model_preset"]),
            sequence_length=int(payload["sequence_length"]),
            sliding_window=int(payload["sliding_window"]),
            long_attention_every=_optional_positive_int(payload["long_attention_every"]),
            num_layers=int(payload["num_layers"]),
            hidden_dim=int(payload["hidden_dim"]),
            intermediate_dim=int(payload["intermediate_dim"]),
            num_experts=int(payload["num_experts"]),
            top_k=int(payload["top_k"]),
            num_heads=int(payload["num_heads"]),
            head_dim=int(payload["head_dim"]),
            vocab_size=int(payload["vocab_size"]),
            global_batch_sequences=int(payload["global_batch_sequences"]),
            global_batch_tokens=int(payload["global_batch_tokens"]),
            remat=str(payload["remat"]),
            attention_impl=str(payload["attention_impl"]),
            cross_entropy_impl=str(payload["cross_entropy_impl"]),
            moe_impl=str(payload["moe_impl"]),
            optimizer=str(payload["optimizer"]),
            muon_ns_iters=int(payload["muon_ns_iters"]),
            mesh=MeshSpec.from_dict(payload["mesh"]),
        )

    def with_overrides(self, overrides: dict[str, Any]) -> ModelSpec:
        if not overrides:
            return self
        payload = self.to_dict()
        payload.update(overrides)
        if "mesh" not in overrides:
            payload["mesh"] = self.mesh.to_dict()
        return replace(ModelSpec.from_dict(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_preset": self.model_preset,
            "sequence_length": self.sequence_length,
            "sliding_window": self.sliding_window,
            "long_attention_every": self.long_attention_every,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "global_batch_sequences": self.global_batch_sequences,
            "global_batch_tokens": self.global_batch_tokens,
            "remat": self.remat,
            "attention_impl": self.attention_impl,
            "cross_entropy_impl": self.cross_entropy_impl,
            "moe_impl": self.moe_impl,
            "optimizer": self.optimizer,
            "muon_ns_iters": self.muon_ns_iters,
            "mesh": self.mesh.to_dict(),
            "derived": {
                "mesh_devices": self.mesh.devices,
                "tokens_per_sequence": self.sequence_length,
                "long_attention_layers": _long_attention_layers(self.num_layers, self.long_attention_every),
                "expert_weight_shapes": {
                    "w_gate_up": [self.num_experts, self.hidden_dim, 2 * self.intermediate_dim],
                    "w_down": [self.num_experts, self.intermediate_dim, self.hidden_dim],
                },
            },
        }


def grug_moe_d2560_may_spec() -> ModelSpec:
    return ModelSpec(
        model_preset="grug_moe_d2560_may",
        sequence_length=4096,
        sliding_window=2048,
        long_attention_every=4,
        num_layers=26,
        hidden_dim=2560,
        intermediate_dim=1280,
        num_experts=256,
        top_k=4,
        num_heads=20,
        head_dim=128,
        vocab_size=128_256,
        global_batch_sequences=16,
        global_batch_tokens=16 * 4096,
        remat="recompute_all",
        attention_impl="gpu_fa4_cute",
        cross_entropy_impl="pallas",
        moe_impl="gmm",
        optimizer="muonh",
        muon_ns_iters=3,
        mesh=MeshSpec(replica_dcn=2, data=1, expert=8, model=1),
    )


def model_preset(name: str) -> ModelSpec:
    if name == "grug_moe_d2560_may":
        return grug_moe_d2560_may_spec()
    raise ValueError(f"Unknown model preset '{name}'. Available presets: grug_moe_d2560_may")


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _long_attention_layers(num_layers: int, long_attention_every: int | None) -> int:
    if long_attention_every is None:
        return 0
    return num_layers // long_attention_every
