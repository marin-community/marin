"""Megatron Grug log scaling against the committed JAX Hero oracle.

The oracle is a tiny Hero-shaped model whose floor is 16 keys, so most positions of its two documents
are scaled. Megatron runs in bf16, which its packed and context-parallel attention kernels require, so
the gate is set by the noise in the same run: queries that see at most 16 keys have a factor of 1, and
Megatron's error there is the bf16 noise floor, capped at NOISE_FLOOR_CAP. Past the floor, the error against the right oracle must
stay within NOISE_RATIO of that floor, and the wrong oracle must be WRONG_ORACLE_RATIO further away.
Megatron must pass unpacked, packed, and packed under CP2, with the log scaling fields set and null. A
checkpoint whose power is 1 instead of the oracle's 2 must fail the same gate.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import TOKENIZER
from tests.gpu.test_grug_megatron import (
    _config,
    _init_policy,
    _megatron_response_logprobs,
)

ORACLE_DIR = Path(__file__).parents[1] / "fixtures" / "grug_log_scaling_oracle"
LOG_SCALING_FIELDS = ("log_scaling_n_floor", "log_scaling_alpha", "log_scaling_power")
LEFT_PADDING = (3, 0)
PAD_TOKEN_ID = 0
# Queries before this prediction index see at most the floor's 16 keys, so their factor is exactly 1.
FLOOR_PREDICTIONS = 16
NOISE_RATIO = 1.5
# An error at every position also raises the noise floor, so the floor itself is capped. On H100 the
# measured floor was 0.0096 nats in every geometry.
NOISE_FLOOR_CAP = 0.03
WRONG_ORACLE_RATIO = 2.0


def _write_checkpoint(path: Path, config_overrides: dict) -> None:
    path.mkdir()
    shutil.copy(ORACLE_DIR / "model.safetensors", path / "model.safetensors")
    config = json.loads((ORACLE_DIR / "config.json").read_text())
    config.update(config_overrides)
    (path / "config.json").write_text(json.dumps(config))
    AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(path)


def _documents_batch(documents: list[np.ndarray]) -> TrainingInputBatch:
    """One document per row, with left and right padding; every next-token prediction is a response."""
    width = max(pad + len(document) for pad, document in zip(LEFT_PADDING, documents, strict=True)) + 2
    sequences = torch.full((len(documents), width), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(documents), width), dtype=torch.long)
    for row, (pad, document) in enumerate(zip(LEFT_PADDING, documents, strict=True)):
        sequences[row, pad : pad + len(document)] = torch.from_numpy(document.astype(np.int64))
        attention_mask[row, pad : pad + len(document)] = 1
    response_length = width - 1
    response_mask = attention_mask[:, -response_length:]
    zeros = torch.zeros(len(documents), response_length, dtype=torch.float32)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "values": zeros.clone(),
            "returns": zeros.clone(),
            "advantages": zeros.clone(),
            "loss_mask": response_mask.clone(),
            "response_mask": response_mask.clone(),
        }
    )
    batch.metadata = {"response_length": response_length, "global_step": 0}
    return batch


def _document_logprobs(output: torch.Tensor, documents: list[np.ndarray]) -> list[np.ndarray]:
    # Output column t holds log p(token[t + 1] | tokens[: t + 1]).
    return [
        output[row, pad : pad + len(document) - 1].numpy()
        for row, (pad, document) in enumerate(zip(LEFT_PADDING, documents, strict=True))
    ]


@dataclass(frozen=True)
class Gate:
    noise_floor: float
    past_floor_error: float
    wrong_oracle_error: float

    @property
    def passed(self) -> bool:
        return (
            self.noise_floor <= NOISE_FLOOR_CAP
            and self.past_floor_error <= NOISE_RATIO * self.noise_floor
            and self.wrong_oracle_error >= WRONG_ORACLE_RATIO * self.past_floor_error
        )


def _gate(actual: list[np.ndarray], expected: list[np.ndarray], wrong: list[np.ndarray]) -> Gate:
    def errors(oracle, below_floor):
        part = slice(None, FLOOR_PREDICTIONS) if below_floor else slice(FLOOR_PREDICTIONS, None)
        return float(np.concatenate([np.abs(a - o)[part] for a, o in zip(actual, oracle, strict=True)]).mean())

    gate = Gate(errors(expected, True), errors(expected, False), errors(wrong, False))
    print(
        f"log scaling gate: noise floor {gate.noise_floor:.4f}, past-floor error {gate.past_floor_error:.4f}, "
        f"wrong oracle {gate.wrong_oracle_error:.4f}, passed {gate.passed}"
    )
    return gate


def _megatron_logprobs(tmp_path: Path, config_overrides: dict, packing: bool, context_parallel_size: int, documents):
    model_path = tmp_path / "model"
    _write_checkpoint(model_path, config_overrides)
    cfg = _config(str(model_path), world_size=context_parallel_size, pp=1, ep=1)
    cfg.trainer.flash_attn = True
    cfg.trainer.use_sample_packing = packing
    cfg.trainer.policy.megatron_config.context_parallel_size = context_parallel_size
    cfg.trainer.train_batch_size = len(documents)
    cfg.trainer.policy_mini_batch_size = len(documents)
    cfg.trainer.micro_forward_batch_size_per_gpu = len(documents)
    cfg.trainer.micro_train_batch_size_per_gpu = len(documents)
    batch = _documents_batch(documents)
    initialize_ray(cfg)
    try:
        policy = _init_policy(cfg, context_parallel_size)
        return _document_logprobs(_megatron_response_logprobs(policy, batch), documents)
    finally:
        ray.shutdown()


def _oracle() -> dict[str, list[np.ndarray]]:
    with np.load(ORACLE_DIR / "observations.npz") as archive:
        return {name: [archive[f"{name}_0"], archive[f"{name}_1"]] for name in ("tokens", "logprobs_on", "logprobs_off")}


@pytest.mark.parametrize("log_scaling", [True, False], ids=["log_scaling", "null_fields"])
@pytest.mark.parametrize(
    ("packing", "context_parallel_size"),
    [(False, 1), (True, 1), (True, 2)],
    ids=["unpacked", "packed_cp1", "packed_cp2"],
)
def test_grug_megatron_log_scaling_matches_jax(tmp_path, log_scaling, packing, context_parallel_size):
    require_hoppers(context_parallel_size)
    oracle = _oracle()
    overrides = {} if log_scaling else dict.fromkeys(LOG_SCALING_FIELDS)
    expected, wrong = (
        (oracle["logprobs_on"], oracle["logprobs_off"]) if log_scaling else (oracle["logprobs_off"], oracle["logprobs_on"])
    )

    actual = _megatron_logprobs(tmp_path, overrides, packing, context_parallel_size, oracle["tokens"])

    assert _gate(actual, expected, wrong).passed


def test_gate_rejects_the_wrong_log_scaling_power(tmp_path):
    require_hoppers(1)
    oracle = _oracle()

    actual = _megatron_logprobs(tmp_path, {"log_scaling_power": 1.0}, False, 1, oracle["tokens"])

    assert not _gate(actual, oracle["logprobs_on"], oracle["logprobs_off"]).passed
