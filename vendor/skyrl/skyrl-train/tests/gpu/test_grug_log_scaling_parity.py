"""Megatron Grug log scaling against the committed JAX Hero oracle.

The oracle is a tiny Hero-shaped model whose floor is 16 keys, so most positions of its two documents
are scaled, and whose log-probabilities move by about 0.6 nats when log scaling is turned off. Megatron
must match the scaled oracle unpacked, packed, and packed under CP2, where each document's positions
come from the zigzag chunks. With the config fields null, it must match the unscaled oracle instead.
"""

from __future__ import annotations

import json
import shutil
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
    LOGPROB_MAX_ABS_TOLERANCE,
    LOGPROB_MEAN_ABS_TOLERANCE,
    _config,
    _init_policy,
    _megatron_response_logprobs,
)

ORACLE_DIR = Path(__file__).parents[1] / "fixtures" / "grug_log_scaling_oracle"
LOG_SCALING_FIELDS = ("log_scaling_n_floor", "log_scaling_alpha", "log_scaling_power")
LEFT_PADDING = (3, 0)
PAD_TOKEN_ID = 0
# The wrong oracle must sit this many times further from Megatron than the right one.
NEGATIVE_CONTROL_RATIO = 5.0


def _write_checkpoint(path: Path, *, log_scaling: bool) -> None:
    path.mkdir()
    shutil.copy(ORACLE_DIR / "model.safetensors", path / "model.safetensors")
    config = json.loads((ORACLE_DIR / "config.json").read_text())
    if not log_scaling:
        config.update(dict.fromkeys(LOG_SCALING_FIELDS))
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


def _mean_and_max_error(actual: list[np.ndarray], expected: list[np.ndarray]) -> tuple[float, float]:
    errors = np.concatenate([np.abs(a - e) for a, e in zip(actual, expected, strict=True)])
    return float(errors.mean()), float(errors.max())


@pytest.mark.parametrize("log_scaling", [True, False], ids=["log_scaling", "null_fields"])
@pytest.mark.parametrize(
    ("packing", "context_parallel_size"),
    [(False, 1), (True, 1), (True, 2)],
    ids=["unpacked", "packed_cp1", "packed_cp2"],
)
def test_grug_megatron_log_scaling_matches_jax(tmp_path, log_scaling, packing, context_parallel_size):
    require_hoppers(context_parallel_size)
    with np.load(ORACLE_DIR / "observations.npz") as archive:
        oracle = {name: archive[name] for name in archive.files}
    documents = [oracle["tokens_0"], oracle["tokens_1"]]
    scaled = [oracle["logprobs_on_0"], oracle["logprobs_on_1"]]
    unscaled = [oracle["logprobs_off_0"], oracle["logprobs_off_1"]]
    expected, other = (scaled, unscaled) if log_scaling else (unscaled, scaled)

    model_path = tmp_path / "model"
    _write_checkpoint(model_path, log_scaling=log_scaling)
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
        actual = _document_logprobs(_megatron_response_logprobs(policy, batch), documents)
    finally:
        ray.shutdown()

    mean_error, max_error = _mean_and_max_error(actual, expected)
    control_error, _ = _mean_and_max_error(actual, other)
    print(f"log scaling parity: mean {mean_error:.4f}, max {max_error:.4f}; wrong oracle mean {control_error:.4f}")
    assert mean_error < LOGPROB_MEAN_ABS_TOLERANCE
    assert max_error < LOGPROB_MAX_ABS_TOLERANCE
    assert control_error > NEGATIVE_CONTROL_RATIO * mean_error
