# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import numpy as np

from marin.rl.objectives.recipes import make_rloo_objective
from marin.rl.objectives.runtime import ObjectiveRuntimeConfig, build_objective_runtime
from marin.rl.objectives.signals import compute_rloo_advantages_from_rewards
from marin.rl.rl_losses import rloo_loss_with_importance_sampling
from marin.rl.scoring import ScoreBundle, ScoreRequirements
from marin.rl.train_batch import create_sequence_batch_from_rollouts, create_training_batch_from_rollouts
from marin.rl.types import Rollout, RolloutMetadata, RolloutWithAdvantage


def create_test_rollout(
    *,
    group_id: str,
    trace_id: str,
    unique_id: int,
    episode_reward: float,
    is_truncated: bool = False,
) -> Rollout:
    prompt_tokens = np.full(3, unique_id, dtype=np.int32)
    response_tokens = np.array([1000 + unique_id, 1001 + unique_id], dtype=np.int32)
    response_logprobs = np.array([-0.7, -0.4], dtype=np.float32)
    token_rewards = np.array([0.1, 0.2], dtype=np.float32)

    return Rollout(
        env_name="test_env",
        env_example_id=f"example-{unique_id}",
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        token_rewards=token_rewards,
        episode_reward=episode_reward,
        temperature=1.0,
        top_k=16,
        is_truncated=is_truncated,
        metadata=RolloutMetadata(
            worker_id="worker-1",
            timestamp=111.0 + unique_id,
            weight_step=9,
            run_id="run-1",
            lesson_id="lesson-a",
            group_id=group_id,
            trace_id=trace_id,
            task_name="math",
            task_version="v1",
        ),
    )


@dataclass(frozen=True)
class StaticScoreSource:
    score_requirements: ScoreRequirements
    student_logprobs: jax.Array
    behavior_logprobs: jax.Array | None = None
    reference_logprobs: jax.Array | None = None
    backend_name: str = "static"

    def requirements(self) -> ScoreRequirements:
        return self.score_requirements

    def score(self, batch, info, roles, *, key):
        del batch, info, roles, key
        return ScoreBundle(
            student_logprobs=self.student_logprobs,
            behavior_logprobs=self.behavior_logprobs,
            reference_logprobs=self.reference_logprobs,
            student_pass_count=1 if self.student_logprobs is not None else 0,
            reference_pass_count=1 if self.reference_logprobs is not None else 0,
        )


def _metric_values(metrics: dict[str, object], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in keys}


def test_rloo_signal_builder_computes_groupwise_advantages_and_group_sizes():
    rollouts = [
        create_test_rollout(group_id="group-a", trace_id="trace-a", unique_id=1, episode_reward=0.0),
        create_test_rollout(group_id="group-a", trace_id="trace-a", unique_id=2, episode_reward=1.0),
        create_test_rollout(group_id="group-b", trace_id="trace-b", unique_id=3, episode_reward=0.5),
    ]
    sequence_batch, batch_info = create_sequence_batch_from_rollouts(rollouts, max_tokens=12, pad_token_id=0)

    runtime = build_objective_runtime(
        ObjectiveRuntimeConfig(
            objective=make_rloo_objective(kl_coef=0.0),
        )
    )
    prepared_batch = runtime.prepare_batch(sequence_batch, batch_info)

    np.testing.assert_allclose(prepared_batch.signals.sequence_advantages.array[:2], np.array([-1.0, 1.0]))
    np.testing.assert_allclose(prepared_batch.signals.sequence_advantages.array[2:], np.array([0.0]))
    np.testing.assert_array_equal(prepared_batch.signals.group_size.array, np.array([2, 2, 1], dtype=np.int32))
    np.testing.assert_allclose(
        prepared_batch.signals.token_weights.array,
        prepared_batch.sequence_batch.response_mask.array * prepared_batch.signals.sequence_advantages.array[:, None],
    )


def test_build_objective_runtime_requires_reference_scores_for_kl_recipe():
    objective = make_rloo_objective(kl_coef=0.2)
    sequence_runtime_config = ObjectiveRuntimeConfig(objective=objective)
    bad_score_source = StaticScoreSource(
        score_requirements=ScoreRequirements(student_logprobs=True, behavior_logprobs=True),
        student_logprobs=jax.numpy.zeros((1, 1), dtype=jax.numpy.float32),
        behavior_logprobs=jax.numpy.zeros((1, 1), dtype=jax.numpy.float32),
    )

    try:
        build_objective_runtime(sequence_runtime_config, score_source=bad_score_source)
    except ValueError as exc:
        assert "reference_logprobs" in str(exc)
    else:
        raise AssertionError("expected build_objective_runtime to reject missing reference scores")


def test_rloo_objective_runtime_matches_current_rloo_loss_path():
    rollouts = [
        create_test_rollout(group_id="group-a", trace_id="trace-a", unique_id=1, episode_reward=0.0),
        create_test_rollout(group_id="group-a", trace_id="trace-a", unique_id=2, episode_reward=1.0),
    ]
    sequence_batch, batch_info = create_sequence_batch_from_rollouts(rollouts, max_tokens=12, pad_token_id=0)
    advantages = compute_rloo_advantages_from_rewards(np.array([rollout.episode_reward for rollout in rollouts]))
    training_batch = create_training_batch_from_rollouts(
        [
            RolloutWithAdvantage(rollout=rollout, advantage=float(advantage))
            for rollout, advantage in zip(rollouts, advantages, strict=True)
        ],
        max_tokens=12,
        pad_token_id=0,
    )

    behavior_logprobs = sequence_batch.behavior_logprobs.array
    response_mask = sequence_batch.response_mask.array
    student_logprobs = behavior_logprobs + 0.1 * response_mask
    reference_logprobs = behavior_logprobs - 0.05 * response_mask
    score_source = StaticScoreSource(
        score_requirements=ScoreRequirements(
            student_logprobs=True,
            behavior_logprobs=True,
            reference_logprobs=True,
        ),
        student_logprobs=student_logprobs,
        behavior_logprobs=behavior_logprobs,
        reference_logprobs=reference_logprobs,
    )

    runtime = build_objective_runtime(
        ObjectiveRuntimeConfig(
            objective=make_rloo_objective(kl_coef=0.2, do_trainer_inference_mismatch_importance_sampling=True)
        ),
        score_source=score_source,
    )
    prepared_batch = runtime.prepare_batch(sequence_batch, batch_info)
    new_loss_fn = runtime.create_loss_fn(reference_model=object())
    new_loss, new_metrics = new_loss_fn(object(), prepared_batch, key=None)

    old_loss, old_metrics = rloo_loss_with_importance_sampling(
        object(),
        object(),
        training_batch,
        score_source,
        key=None,
        kl_coef=0.2,
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.2,
        tis_importance_sampling_ratio_max=2.0,
        do_trainer_inference_mismatch_importance_sampling=True,
        synchronous=False,
        do_overlong_filtering=False,
    )

    np.testing.assert_allclose(new_loss, old_loss, atol=1e-6)
    metric_keys = (
        "ratio_mean",
        "clipped_ratio_mean",
        "clip_fraction",
        "reinforce_loss",
        "kl_loss",
        "kl_penalty",
        "trainer_inference_importance_sampling_ratio_mean",
        "mean_advantages",
        "current_entropy",
        "policy_entropy",
        "scoring/student_pass_count",
        "scoring/reference_pass_count",
    )
    np.testing.assert_allclose(
        list(_metric_values(new_metrics, metric_keys).values()),
        list(_metric_values(old_metrics, metric_keys).values()),
        atol=1e-6,
    )
