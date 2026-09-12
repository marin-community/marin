# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check captured Torch GRPO values and logprob gradients against JAX."""

import argparse
import json
from dataclasses import asdict, dataclass

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grpo import GrpoConfig, KlGradient, grpo_advantages, grpo_loss, grpo_objective_weights
from marin.rl.grpo_artifact import read_golden_rollout
from rigging.filesystem.storage_path import StoragePath

POLICY_DIAGNOSTICS = (
    "ppo_clip_ratio",
    "ppo_clip_ratio_low",
    "ppo_clip_ratio_high",
    "ppo_clip_pressure_low",
    "ppo_clip_pressure_high",
    "ppo_ratio_exact_unit_fraction",
)
MARINSKYRL_ORACLE_COMMIT = "8e33e01707b7225ecde1d6b8ad172a3dd4dc8661"


@dataclass(frozen=True)
class ReplayComparison:
    max_absolute_error: float
    mean_absolute_error: float


@dataclass(frozen=True)
class GrpoReplayResult:
    trajectories: int
    response_tokens: int
    objective_partitions: int
    loss: float
    advantages: ReplayComparison
    loss_error: ReplayComparison
    logprob_gradients: ReplayComparison
    diagnostic_errors: dict[str, ReplayComparison]


def _compare(actual, expected, *, atol: float, rtol: float, name: str, scale=None) -> ReplayComparison:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    checked_actual = actual if scale is None else actual / scale
    checked_expected = expected if scale is None else expected / scale
    np.testing.assert_allclose(checked_actual, checked_expected, atol=atol, rtol=rtol, err_msg=name, equal_nan=False)
    error = np.abs(actual - expected)
    return ReplayComparison(float(error.max()), float(error.mean()))


def replay_golden_rollout(uri: str, *, atol: float, rtol: float) -> GrpoReplayResult:
    """Verify a persisted oracle batch, raising on unsupported recipes or mismatch.

    This checks the loss boundary only. It does not load a language model or
    establish full-model optimizer-update parity.
    """
    batch, manifest = read_golden_rollout(uri)
    provenance = manifest["provenance"]
    if provenance["oracle_base_commit"] != MARINSKYRL_ORACLE_COMMIT or provenance["kl_gradient"] != "detached":
        raise ValueError("Replay requires the pinned MarinSkyRL oracle with detached KL")
    algorithm = manifest["config"]["trainer"]["algorithm"]
    required = {
        "advantage_estimator": "grpo",
        "policy_loss_type": "regular",
        "loss_reduction": "token_mean",
        "advantage_batch_normalize": False,
        "use_kl_in_reward": False,
        "use_entropy_loss": False,
        "use_tis": False,
        "think_token_weight": 1.0,
    }
    for key, expected in required.items():
        if algorithm[key] != expected:
            raise ValueError(f"GRPO replay requires {key}={expected!r}, got {algorithm[key]!r}")
    if not np.array_equal(batch.loss_mask, batch.response_mask):
        raise ValueError("This oracle recipe requires loss_mask to equal response_mask")
    if batch.logprob_gradients is None:
        raise ValueError("Capture must include independent Torch logprob gradients")
    use_kl = algorithm["use_kl_loss"]
    if use_kl and (algorithm["kl_estimator_type"] != "k3" or batch.reference_logprobs is None):
        raise ValueError("KL replay requires k3 and captured reference logprobs")
    config = GrpoConfig(
        eps_clip_low=algorithm["eps_clip_low"],
        eps_clip_high=algorithm["eps_clip_high"],
        kl_loss_coef=algorithm["kl_loss_coef"] if use_kl else 0.0,
        kl_gradient=KlGradient(manifest["provenance"]["kl_gradient"]),
    )
    Batch, Position = (hax.Axis("batch", batch.old_logprobs.shape[0]), hax.Axis("position", batch.old_logprobs.shape[1]))
    _, group_ids = np.unique(batch.group_ids, return_inverse=True)
    _, partition_ids = np.unique(batch.objective_partition_ids, return_inverse=True)
    num_groups = int(group_ids.max()) + 1
    num_partitions = int(partition_ids.max()) + 1
    mask = hax.named(jnp.asarray(batch.response_mask), (Batch, Position))
    advantages = grpo_advantages(
        hax.named(jnp.asarray(batch.rewards), (Batch, Position)),
        mask,
        hax.named(jnp.asarray(group_ids), Batch),
        Batch=Batch,
        Position=Position,
        num_groups=num_groups,
        normalize_by_std=algorithm["grpo_norm_by_std"],
    )
    policy_weights, kl_weights = grpo_objective_weights(
        mask,
        hax.named(jnp.asarray(partition_ids), Batch),
        Batch=Batch,
        Position=Position,
        num_partitions=num_partitions,
    )
    old = hax.named(jnp.asarray(batch.old_logprobs), (Batch, Position))
    reference = hax.named(
        jnp.asarray(batch.reference_logprobs) if use_kl else jnp.zeros_like(old.array), (Batch, Position)
    )

    def loss(logprobs):
        return grpo_loss(
            hax.named(logprobs, (Batch, Position)),
            old,
            reference,
            advantages,
            policy_weights,
            kl_weights,
            config=config,
            accumulation_steps=1,
        )

    current = batch.old_logprobs if batch.current_logprobs is None else batch.current_logprobs
    (value, metrics), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(jnp.asarray(current))
    oracle_metrics = manifest["oracle"]["metrics"]
    if set(oracle_metrics) != set(POLICY_DIAGNOSTICS):
        raise ValueError("Capture must contain exactly the six regular-PPO clipping diagnostics")
    diagnostic_errors = {
        name: _compare(metrics[name].value(), oracle_metrics[name], atol=atol, rtol=rtol, name=name)
        for name in POLICY_DIAGNOSTICS
    }
    masked = batch.response_mask == 0
    np.testing.assert_array_equal(np.asarray(gradient)[masked], 0, err_msg="JAX padding gradient")
    np.testing.assert_array_equal(batch.logprob_gradients[masked], 0, err_msg="Torch padding gradient")
    return GrpoReplayResult(
        trajectories=Batch.size,
        response_tokens=int(batch.response_mask.sum()),
        objective_partitions=num_partitions,
        loss=float(value),
        diagnostic_errors=diagnostic_errors,
        advantages=_compare(advantages.array, batch.advantages, atol=atol, rtol=rtol, name="advantages"),
        loss_error=_compare(value, manifest["oracle"]["loss"], atol=atol, rtol=rtol, name="loss"),
        logprob_gradients=_compare(
            gradient,
            batch.logprob_gradients,
            atol=atol,
            rtol=rtol,
            name="logprob gradients",
            scale=np.where(np.asarray(policy_weights.array) > 0, np.asarray(policy_weights.array), 1.0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="Local or regional object-store capture URI")
    parser.add_argument("--output-uri", help="Write the comparison JSON to this local or regional URI")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    result = asdict(replay_golden_rollout(args.capture, atol=args.atol, rtol=args.rtol))
    if args.output_uri:
        with StoragePath(args.output_uri).open("wt") as output:
            json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
