# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Llama policy learning with save/resume in separate processes."""

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jmp
import numpy as np
import optax
from iris.jax.init import initialize_jax

from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.grpo import grpo_advantages, grpo_objective_weights
from levanter.layers.attention import AttentionBackend
from levanter.models.llama import LlamaConfig
from levanter.policy import ClippedPolicyConfig
from levanter.policy_learner import SpmdPolicyLearner
from levanter.policy_model import PolicyBatch, policy_model_loss
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig


_PROJECTION_BLOCK_SIZE = 8


@dataclass(frozen=True)
class SmokeConfig:
    batch_size: int
    per_device_parallelism: int
    updates: int
    compute_dtype: str
    hidden_dim: int = 8
    intermediate_dim: int = 16
    num_layers: int = 1
    num_heads: int = 2
    vocab_size: int = 16
    sequence_length: int = 5
    response_length: int = 2
    group_size: int = 4
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    kl_loss_coef: float = 0.01
    eps_clip: float = 0.2


def _model(config: SmokeConfig, *, seed: int):
    return LlamaConfig(
        max_seq_len=config.sequence_length,
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_kv_heads=config.num_heads,
        gradient_checkpointing=False,
        attn_backend=AttentionBackend.VANILLA,
    ).build(hax.Axis("vocab", config.vocab_size), key=jax.random.PRNGKey(seed))


def _batch(config: SmokeConfig) -> PolicyBatch:
    Batch = hax.Axis("batch", config.batch_size)
    Position = hax.Axis("position", config.sequence_length)
    Response = hax.Axis("response", config.response_length)
    groups = config.batch_size // config.group_size
    rows = np.arange(config.batch_size)
    lengths = np.resize([0, 1, 2, 2], config.batch_size)
    attention = (np.arange(Position.size)[None, :] < (Position.size - Response.size + lengths)[:, None]).astype(
        np.int32
    )
    tokens = (rows[:, None] * 3 + np.arange(Position.size)[None, :]) % config.vocab_size
    tokens = np.where(attention, tokens, 0).astype(np.int32)
    response_mask = hax.named(attention[:, -Response.size :].astype(np.float32), (Batch, Response))
    token_rewards = np.zeros((Batch.size, Response.size), dtype=np.float32)
    token_rewards[:, -1] = np.asarray([0.0, 1.0, 3.0, -2.0])[rows // groups]
    advantages = grpo_advantages(
        hax.named(token_rewards, (Batch, Response)),
        response_mask,
        hax.named((rows % groups).astype(np.int32), Batch),
        Batch=Batch,
        Position=Response,
        num_groups=groups,
        normalize_by_std=True,
    )
    policy_weights, kl_weights = grpo_objective_weights(response_mask, Batch=Batch, Position=Response)
    return PolicyBatch(
        tokens=hax.named(tokens, (Batch, Position)),
        attention_mask=hax.named(attention, (Batch, Position)),
        position_ids=hax.named(np.maximum(attention.cumsum(-1) - 1, 0), (Batch, Position)),
        response_mask=response_mask,
        advantages=advantages,
        policy_weights=policy_weights,
        kl_weights=kl_weights,
        old_logprobs=hax.full((Batch, Response), -9.0),
        reference_logprobs=None,
    )


def _arrays(tree):
    return [np.asarray(value) for value in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def _fingerprint(tree) -> str:
    digest = hashlib.sha256()
    for value in _arrays(tree):
        digest.update(json.dumps({"shape": value.shape, "dtype": str(value.dtype)}).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _compare(actual, expected) -> float:
    largest = 0.0
    for value, reference in zip(_arrays(actual), _arrays(expected), strict=True):
        if np.issubdtype(reference.dtype, np.integer) or np.issubdtype(reference.dtype, np.bool_):
            np.testing.assert_array_equal(value, reference)
        else:
            np.testing.assert_allclose(value, reference, atol=1e-6, rtol=1e-5)
        largest = max(largest, float(np.max(np.abs(value.astype(np.float64) - reference.astype(np.float64)))))
    return largest


def _metadata(path: Path, config: SmokeConfig) -> dict:
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["smoke_config"] != dataclasses.asdict(config):
        raise ValueError("Checkpoint policy learner smoke configuration does not match")
    return metadata


def run(config: SmokeConfig, *, phase: str, output_dir: Path, require_accelerator: bool) -> dict:
    """Run a save or resume phase on one process with all test devices visible."""
    if jax.process_count() != 1:
        raise ValueError("Policy learner smoke requires one JAX process with all test devices visible")
    if require_accelerator and jax.default_backend() == "cpu":
        raise RuntimeError("Policy learner smoke requires accelerator devices")
    if config.updates < 2 or config.batch_size % config.group_size:
        raise ValueError("Smoke requires at least two updates and complete reward groups")
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "resume" / "step-1"
    expected_path = output_dir / "expected" / f"step-{config.updates}"
    trainer_config = TrainerConfig(
        id="policy-learner-smoke",
        train_batch_size=config.batch_size,
        per_device_parallelism=config.per_device_parallelism,
        num_train_steps=config.updates,
        tracker=NoopConfig(),
        require_accelerator=require_accelerator,
        mp=jmp.get_policy(f"params=float32,compute={config.compute_dtype},output=float32"),
        log_dir=str(output_dir / "logs"),
        log_jaxprs=False,
        log_xla_hlo=False,
        load_checkpoint=False,
    )
    microbatch_size = trainer_config.microbatch_size
    if microbatch_size is None or config.batch_size % microbatch_size:
        raise ValueError("Smoke batch must divide evenly into execution microbatches")
    loss = partial(
        policy_model_loss,
        config=ClippedPolicyConfig(config.eps_clip, config.eps_clip, config.kl_loss_coef),
        accumulation_steps=config.batch_size // microbatch_size,
        block_size=_PROJECTION_BLOCK_SIZE,
    )
    optimizer = optax.adamw(
        config.learning_rate,
        b1=config.adam_beta1,
        b2=config.adam_beta2,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    with Trainer(trainer_config, optimizer, loss, add_default_hooks=False) as trainer:
        state = trainer.initial_state(
            jax.random.PRNGKey(17 if phase == "save" else 99),
            model_init=partial(_model, config, seed=7 if phase == "save" else 97),
        )
        backend = SpmdPolicyLearner(trainer, block_size=_PROJECTION_BLOCK_SIZE)
        batch = _batch(config)
        if phase == "save":
            initial_fingerprint = _fingerprint(state.model)
            batch = backend.score(state, batch)
            # The initial policy is also the fixed reference for this synthetic run.
            batch = dataclasses.replace(batch, reference_logprobs=batch.old_logprobs)
        else:
            batch = dataclasses.replace(batch, reference_logprobs=hax.zeros_like(batch.old_logprobs))
            metadata = _metadata(resume_path, config)
            restored = load_checkpoint(
                {"state": state, "batch": batch},
                resume_path,
                axis_mapping=trainer.parameter_axis_mapping,
                mesh=trainer.device_mesh,
            )
            assert _fingerprint(restored) == metadata["payload_sha256"], "Restored checkpoint tensor bytes differ"
            state, batch = restored["state"], restored["batch"]
            assert int(state.step) == metadata["step"] == 1

        old_scores = _fingerprint(batch.old_logprobs)
        losses = []
        while int(state.step) < config.updates:
            result = backend.update(state, batch)
            state = result.state
            assert np.isfinite(result.loss), "Policy learner loss is nonfinite"
            assert _fingerprint(batch.old_logprobs) == old_scores, "Old-policy snapshot changed during updates"
            losses.append({"step": int(state.step), "loss": float(result.loss)})
            if phase == "save" and int(state.step) in (1, config.updates):
                payload = {"state": state.saveable_state, "batch": batch}
                save_checkpoint(
                    payload,
                    step=int(state.step),
                    checkpoint_path=resume_path if int(state.step) == 1 else expected_path,
                    metadata={
                        "smoke_config": dataclasses.asdict(config),
                        "payload_sha256": _fingerprint(payload),
                        "loss": float(result.loss),
                    },
                )

        deviations = None
        if phase == "save":
            assert _fingerprint(state.model) != initial_fingerprint, "Policy learner did not update model parameters"
        else:
            metadata = _metadata(expected_path, config)
            expected = load_checkpoint(
                {"state": state, "batch": batch},
                expected_path,
                axis_mapping=trainer.parameter_axis_mapping,
                mesh=trainer.device_mesh,
            )
            assert int(expected["state"].step) == config.updates
            deviations = {
                name: _compare(getattr(state, name), getattr(expected["state"], name))
                for name in ("model", "opt_state", "training_key", "step", "model_averaging")
            }
            _compare(batch, expected["batch"])
            np.testing.assert_allclose(result.loss, metadata["loss"], atol=1e-6, rtol=1e-5)

        receipt = {
            "phase": phase,
            "backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "process_count": jax.process_count(),
            "device_kinds": sorted({device.device_kind for device in jax.devices()}),
            "config": dataclasses.asdict(config),
            "policy_reduction": "token_mean",
            "kl_reduction": "nonempty_sequence_mean",
            "step": int(state.step),
            "checkpoint": str(expected_path if phase == "save" else resume_path),
            "old_scores_sha256": old_scores,
            "losses": losses,
            "max_abs_deviation": deviations,
        }
    (output_dir / f"{phase}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("save", "resume"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-device-parallelism", type=int, default=1)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--require-accelerator", action="store_true")
    args = parser.parse_args()
    initialize_jax()
    config = SmokeConfig(
        batch_size=4 * jax.device_count(),
        per_device_parallelism=args.per_device_parallelism,
        updates=args.updates,
        compute_dtype=args.compute_dtype,
    )
    run(config, phase=args.phase, output_dir=args.output_dir, require_accelerator=args.require_accelerator)


if __name__ == "__main__":
    main()
