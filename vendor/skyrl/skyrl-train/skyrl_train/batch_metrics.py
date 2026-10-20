import re
from typing import Literal
import torch
from torch import Tensor
from omegaconf import DictConfig
from marinskyrl.runtime_options import AdvantageEstimator
from skyrl_train.config.objective_spec import off_policy_correction
from skyrl_train.objective.correction import compute_correction
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.telemetry import ConsumedWork
from skyrl_train.utils.advantage_estimators import GRPO_FLAT_REWARD_STD_TOLERANCE
from skyrl_train.utils.importance_ratio_diagnostics import LocalReduction, mismatch_ratio_metrics
from collections import defaultdict
from collections.abc import Sequence
from typing import Dict, List

import numpy as np

from skyrl_train.rollouts.buffer import RowFacts
from skyrl_train.trajectory_runners.trajectory_reward_shaping import (
    aggregate_reward_shaping_components,
    reward_shaping_metrics,
    shaping_groups,
)
from skyrl_train.trajectory_runners.types import BatchFields
from skyrl_train.trajectory_runners.rollout_metrics import rollout_length_metrics
from skyrl_train.trajectory_runners.trajectory_processing import TitoFullDeclineReason, merge_batch_failure_metrics
from skyrl_train.trajectory_runners.trajectory_retention import RETENTION_METRIC_PREFIX

from skyrl_train.metric_names import (
    IDENTITY_AWARE_REWARD_METRIC_PREFIX,
    TASK_ROLLOUT_METRIC_PREFIX,
    TIS_ALIGNED_TOKENS_METRIC,
    TIS_ALIGNMENT_ALERT_METRIC,
    TIS_EXACT_MATCH_FRACTION_METRIC,
    TIS_LCS_FALLBACK_FRACTION_METRIC,
    TIS_UNALIGNED_FRACTION_METRIC,
    TIS_ALIGNMENT_FAIL_COUNT_METRIC,
    TIS_LCS_FALLBACK_MESSAGES_METRIC,
    TIS_LCS_FALLBACK_ALERT_METRIC,
    TIS_TITO_FULL_ATTEMPTS_METRIC,
    TIS_TITO_FULL_SUCCESS_FRACTION_METRIC,
    TIS_TITO_FULL_DECLINE_COUNT_METRIC,
    TIS_TITO_FULL_DECLINE_METRIC_PREFIX,
)


def rollout_metrics(facts: Sequence[RowFacts], *, tis_lcs_alert_threshold: float) -> dict[str, float]:
    """Aggregate admitted rollout counters and row values without reading payloads."""
    token_rewards = any(group.fields.token_rewards for group in facts)
    lengths = [int(length) for group in facts for length in group.response_len]
    successes = [
        passed if has_verdict or not (token_rewards and scalar is not None and length == 0) else False
        for group in facts
        for passed, has_verdict, scalar, length in zip(
            group.rollout_passes, group.verifier_present, group.scalar_values, group.response_len, strict=True
        )
    ]
    has_environment = any(group.env_metrics is not None for group in facts)
    env_metrics = (
        [value for group in facts for value in (group.env_metrics or ({},) * len(group.response_len))]
        if has_environment
        else None
    )
    env_classes = (
        [value for group in facts for value in (group.env_classes or ("",) * len(group.response_len))]
        if has_environment
        else None
    )
    rollout_metrics = rollout_length_metrics(lengths, successes, env_metrics, env_classes)
    rollout_metrics.update(
        rollout_counter_metrics(
            [group.group_metrics for group in facts], tis_lcs_alert_threshold=tis_lcs_alert_threshold
        )
    )
    rollout_metrics.update(_facts_shaping_metrics(facts))
    return rollout_metrics


def rollout_counter_metrics(
    group_metrics: Sequence[dict[str, float]], *, tis_lcs_alert_threshold: float
) -> dict[str, float]:
    """Aggregate token-weighted alignment and per-group failure and retention counters."""
    rollout_metrics = {}
    # TIS alignment metrics use token-weighted fractions across batches.
    total_aligned = 0.0
    sum_exact = sum_lcs = sum_unaligned = 0.0
    sum_fail = sum_lcs_msgs = 0.0
    saw_tis = False
    for output in group_metrics:
        rm = output
        n = rm.get(TIS_ALIGNED_TOKENS_METRIC)
        if n is None:
            continue
        saw_tis = True
        total_aligned += n
        sum_exact += rm.get(TIS_EXACT_MATCH_FRACTION_METRIC, 0.0) * n
        sum_lcs += rm.get(TIS_LCS_FALLBACK_FRACTION_METRIC, 0.0) * n
        sum_unaligned += rm.get(TIS_UNALIGNED_FRACTION_METRIC, 0.0) * n
        sum_fail += rm.get(TIS_ALIGNMENT_FAIL_COUNT_METRIC, 0.0)
        sum_lcs_msgs += rm.get(TIS_LCS_FALLBACK_MESSAGES_METRIC, 0.0)
    if saw_tis:
        denom = max(total_aligned, 1.0)
        rollout_metrics[TIS_ALIGNED_TOKENS_METRIC] = total_aligned
        rollout_metrics[TIS_EXACT_MATCH_FRACTION_METRIC] = sum_exact / denom
        rollout_metrics[TIS_LCS_FALLBACK_FRACTION_METRIC] = sum_lcs / denom
        rollout_metrics[TIS_UNALIGNED_FRACTION_METRIC] = sum_unaligned / denom
        rollout_metrics[TIS_ALIGNMENT_FAIL_COUNT_METRIC] = sum_fail
        rollout_metrics[TIS_LCS_FALLBACK_MESSAGES_METRIC] = sum_lcs_msgs
        lcs_alert = 1.0 if (sum_lcs / denom) > tis_lcs_alert_threshold else 0.0
        rollout_metrics[TIS_LCS_FALLBACK_ALERT_METRIC] = lcs_alert
        total_tito_attempts = sum((output).get(TIS_TITO_FULL_ATTEMPTS_METRIC, 0.0) for output in group_metrics)
        total_tito_successes = sum(
            (output).get(TIS_TITO_FULL_SUCCESS_FRACTION_METRIC, 0.0) * (output).get(TIS_TITO_FULL_ATTEMPTS_METRIC, 0.0)
            for output in group_metrics
        )
        rollout_metrics[TIS_TITO_FULL_ATTEMPTS_METRIC] = total_tito_attempts
        rollout_metrics[TIS_TITO_FULL_SUCCESS_FRACTION_METRIC] = (
            total_tito_successes / total_tito_attempts if total_tito_attempts else 0.0
        )
        total_tito_declines = sum((output).get(TIS_TITO_FULL_DECLINE_COUNT_METRIC, 0.0) for output in group_metrics)
        rollout_metrics[TIS_TITO_FULL_DECLINE_COUNT_METRIC] = total_tito_declines
        rollout_metrics[TIS_ALIGNMENT_ALERT_METRIC] = (
            1.0 if sum_unaligned > 0 or lcs_alert or total_tito_declines > 0 else 0.0
        )
        for reason in TitoFullDeclineReason:
            name = f"{TIS_TITO_FULL_DECLINE_METRIC_PREFIX}{reason.value}"
            rollout_metrics[name] = sum((output).get(name, 0.0) for output in group_metrics)

    rollout_metrics.update(merge_batch_failure_metrics(group_metrics))

    # Retention counts per group, and this rebuilds rollout_metrics from responses and rewards, so
    # the per-group counters have to be carried across or the archives are written unobserved.
    for output in group_metrics:
        for name, value in (output).items():
            if name.startswith(
                (RETENTION_METRIC_PREFIX, IDENTITY_AWARE_REWARD_METRIC_PREFIX, TASK_ROLLOUT_METRIC_PREFIX)
            ):
                rollout_metrics[name] = rollout_metrics.get(name, 0.0) + value

    return rollout_metrics


def _facts_shaping_metrics(facts: Sequence[RowFacts]) -> dict[str, float]:
    fields = BatchFields.from_groups([group.fields for group in facts])
    if "reward_shaping_components" not in fields.first_group_list_fields or any(group.loop_credit for group in facts):
        return {}
    components = [values for group in facts for values in group.shaping_components]
    totals = [
        0.0 if fields.token_rewards and scalar is not None and length == 0 else total
        for group in facts
        for scalar, length, total in zip(
            group.scalar_values, group.response_len, group.rewards.optimization, strict=True
        )
    ]
    outcomes = [value for group in facts for value in group.rewards.outcome]
    lengths = [value for group in facts for value in group.shaping_response_tokens]
    stops = (
        [value for group in facts for value in group.stop_reasons]
        if "stop_reasons" in fields.first_group_list_fields
        else [None] * len(lengths)
    )
    last = (
        [value for group in facts for value in group.is_last_step]
        if "is_last_step" in fields.first_group_list_fields
        else None
    )
    incidence = [value for group in facts for value in group.loop_incidence]
    groups = shaping_groups(last, len(lengths))
    return reward_shaping_metrics(
        trajectory_components=[aggregate_reward_shaping_components(components, rows) for rows in groups],
        shaped_totals=[totals[rows[-1]] for rows in groups],
        trajectory_outcomes=[outcomes[rows[-1]] for rows in groups],
        trajectory_lengths=[sum(lengths[row] for row in rows) for rows in groups],
        trajectory_stops=[stops[rows[-1]] for rows in groups],
        loop_incidence=[any(incidence[row] for row in rows) for rows in groups],
        trajectory_loop_advantages=[0] * len(groups),
        trajectory_loop_token_counts=[0] * len(groups),
        charged_loop_advantages=[],
    )


MAX_DOMAIN_REWARD_METRICS = 32


def domain_metric_source_key(source: str | None) -> str:
    """Encode one source as a distinct, tracker-safe metric path segment.

    Lowercase ASCII names stay readable. ``_missing`` denotes absent metadata;
    other names use fixed-width UTF-8 byte escapes under ``_source_``.
    """
    if source is None:
        return "_missing"
    if re.fullmatch(r"[a-z_][a-z0-9_]*", source) and source != "_missing" and not source.startswith("_source_"):
        return source
    encoded = "".join(
        chr(byte) if byte in b"abcdefghijklmnopqrstuvwxyz0123456789" else f"_{byte:02x}"
        for byte in source.encode("utf-8")
    )
    return f"_source_{encoded}"


def domain_reward_metrics(data_sources: List[str | None], rewards: List[float]) -> Dict[str, float]:
    """Return bounded per-source means with distinct, stable metric names."""
    if len(data_sources) != len(rewards):
        raise ValueError(
            f"Expected one data source per reward, got {len(data_sources)} sources and {len(rewards)} rewards"
        )

    rewards_by_source: Dict[str, List[float]] = defaultdict(list)
    for source, reward in zip(data_sources, rewards, strict=True):
        rewards_by_source[domain_metric_source_key(source)].append(reward)

    sources = sorted(rewards_by_source)
    metrics = {
        f"reward/domain/{source}/avg_raw_reward": float(np.mean(rewards_by_source[source]))
        for source in sources[:MAX_DOMAIN_REWARD_METRICS]
    }
    if len(sources) > MAX_DOMAIN_REWARD_METRICS:
        overflow = [reward for source in sources[MAX_DOMAIN_REWARD_METRICS:] for reward in rewards_by_source[source]]
        metrics["reward/domain_overflow/avg_raw_reward"] = float(np.mean(overflow))
    return metrics


def reward_metrics(facts: Sequence[RowFacts], uids: Sequence[str], *, n_samples_per_prompt: int) -> dict[str, float]:
    """Report graded reward, pass, verifier, and domain values from row facts."""
    token_rewards = any(group.fields.token_rewards for group in facts)
    has_unshaped = any("unshaped_rewards" in group.fields.present for group in facts)
    raw_totals, sign_totals, passes, graded = [], [], [], []
    scores, agents, sources = [], [], []
    for group in facts:
        for row, length in enumerate(group.response_len):
            empty_promoted = token_rewards and group.scalar_values[row] is not None and length == 0
            raw_totals.append(0 if empty_promoted else group.metric_totals[row])
            sign_totals.append(0.0 if empty_promoted else group.reward_sign_totals[row])
            passed = group.passes[row]
            if empty_promoted and not has_unshaped and group.verifier_task_pass[row] is None:
                passed = False
            passes.append(passed)
            graded.append(group.graded[row])
            scores.append(group.verifier_scores[row] if group.verifier_scores is not None else None)
            agents.append(group.verifier_agents[row] if group.verifier_agents is not None else None)
            sources.append(group.data_sources[row] if group.data_sources is not None else None)
    indices = [index for index, included in enumerate(graded) if included]
    if not indices:
        return {}
    grouped_passes, grouped_rewards, by_agent = defaultdict(list), defaultdict(list), defaultdict(list)
    for index in indices:
        grouped_passes[uids[index]].append(passes[index])
        grouped_rewards[uids[index]].append(sign_totals[index])
        if scores[index] is not None and isinstance(agents[index], str):
            by_agent[domain_metric_source_key(agents[index])].append(scores[index])
    metrics = {
        f"reward/avg_pass_at_{n_samples_per_prompt}": sum(any(values) for values in grouped_passes.values())
        / len(grouped_passes),
        "reward/avg_raw_reward": float(np.mean([raw_totals[index] for index in indices])),
        "reward/informative_group_fraction": sum(max(values) > min(values) for values in grouped_rewards.values())
        / len(grouped_rewards),
    }
    if any(group.verifier_scores is not None for group in facts):
        included_scores = [scores[index] for index in indices if scores[index] is not None]
        metrics["reward/verifier_score_coverage"] = len(included_scores) / len(indices)
        if included_scores:
            metrics["reward/avg_verifier_score"] = float(np.mean(included_scores))
        metrics.update(
            {
                f"reward/agent/{agent}/avg_verifier_score": float(np.mean(by_agent[agent]))
                for agent in sorted(by_agent)[:MAX_DOMAIN_REWARD_METRICS]
            }
        )
    if any(group.data_sources is not None for group in facts):
        metrics.update(
            domain_reward_metrics(
                [sources[index] for index in indices], [float(raw_totals[index]) for index in indices]
            )
        )
    return metrics


def zero_std_group_fraction(uids: Sequence[str], rewards: Tensor) -> float:
    """Count complete groups whose sample reward deviation is below the GRPO threshold."""
    groups: dict[str, list[Tensor]] = {}
    for uid, reward in zip(uids, rewards, strict=True):
        groups.setdefault(uid, []).append(reward)
    if not groups:
        return 0.0
    flat = sum(
        len(group) > 1 and torch.std(torch.stack(group)).item() <= GRPO_FLAT_REWARD_STD_TOLERANCE
        for group in groups.values()
    )
    return flat / len(groups)


def install_correction(
    batch: TrainingInputBatch, algorithm: DictConfig, *, reduction: LocalReduction
) -> dict[str, float]:
    """Install configured policy correction weights and return their diagnostics."""
    correction = off_policy_correction(algorithm)
    if not correction.rules:
        return {}
    rollout_logprobs = batch.get("rollout_logprobs")
    if rollout_logprobs is None:
        raise ValueError("off_policy_correction requires rollout_logprobs")
    result = compute_correction(
        batch["action_log_probs"], rollout_logprobs, batch["loss_mask"], correction, reduction=reduction
    )
    batch["correction_weights"] = result.weights
    return result.metrics


def training_metrics(
    batch: TrainingInputBatch,
    *,
    algorithm: DictConfig,
    reduction: LocalReduction,
    phase: Literal["forward", "advantages"],
    step_wise: bool,
    training_metrics_enabled: bool = False,
    rollout_logprobs_enabled: bool = False,
    group_scores: tuple[Sequence[str], Tensor] | None = None,
) -> dict[str, float]:
    """Run shared training diagnostic producers at their forward or advantage boundary."""
    metrics = {}
    if phase == "forward":
        staleness = batch.pop("rollout_staleness")
        if batch.get("rollout_logprobs") is None:
            return metrics
        if training_metrics_enabled:
            metrics.update(
                mismatch_ratio_metrics(
                    batch["action_log_probs"],
                    batch["rollout_logprobs"],
                    batch["loss_mask"],
                    staleness,
                    eps_clip_low=algorithm.eps_clip_low,
                    eps_clip_high=algorithm.eps_clip_high,
                    reduction=reduction,
                )
            )
        if rollout_logprobs_enabled:
            valid = batch["loss_mask"] > 0
            differences = (batch["rollout_logprobs"][valid] - batch["action_log_probs"][valid]).exp().abs()
            metrics.update(
                {
                    "policy/rollout_train_prob_diff_mean": reduction.mean(differences),
                    "policy/rollout_train_prob_diff_std": reduction.std(differences),
                }
            )
        return metrics
    rows = batch.batch_size - batch.metadata.get("pad_size", 0)
    uids, rewards = (batch.metadata["uids"][:rows], batch["rewards"][:rows]) if group_scores is None else group_scores
    if algorithm.advantage_estimator == AdvantageEstimator.GRPO and not step_wise:
        metrics["reward/zero_std_group_fraction"] = zero_std_group_fraction(uids, rewards.sum(-1))
    rows = batch.batch_size - batch.metadata.get("pad_size", 0)
    rewards = batch["rewards"].sum(-1)[:rows]
    if step_wise:
        rewards = rewards[batch["is_last_step"][:rows]]
    advantages = torch.masked_select(batch["advantages"][:rows], batch["response_mask"][:rows].bool())
    average_reward = reduction.mean(rewards)
    average_advantage = reduction.mean(advantages)
    average_absolute = reduction.mean(advantages.abs())
    batch.metadata.setdefault("metrics", {}).update(
        avg_final_rewards=average_reward,
        avg_response_length=batch.metadata["avg_response_length"],
        avg_advantages=average_advantage,
        avg_advantages_abs=average_absolute,
    )
    metrics.update(
        {
            "loss/avg_final_rewards": average_reward,
            "loss/avg_raw_advantages": average_advantage,
            "loss/avg_raw_advantages_abs": average_absolute,
        }
    )
    return metrics


def token_staleness_metrics(
    batch: TrainingInputBatch, facts: Sequence[RowFacts], *, reduction: LocalReduction
) -> dict[str, float]:
    """Report loss-bearing token staleness and compact policy-stamp coverage."""
    trained = batch["loss_mask"] > 0
    count = reduction.sum(trained).item()
    if not count:
        return {}
    charged = batch["rollout_staleness"].masked_fill(~trained, 0)
    spanning = [value for group in facts for value in group.spanning_policy_steps]
    return {
        "async/staleness_tokens_mean": reduction.sum(charged).item() / count,
        "async/staleness_tokens_max": reduction.combine(charged.max(), "max").item(),
        "async/staleness_tokens_ratio": reduction.sum(charged != 0).item() / count,
        "async/staleness_tokens_stamped": sum(sum(group.stamped_loss_tokens) for group in facts) / count,
        "async/staleness_tokens_spanning": torch.tensor(spanning).float().mean().item() if spanning else 0.0,
    }


def consumed_work(training_input: TrainingInputBatch) -> ConsumedWork:
    """Count the rows and tokens an optimizer step consumed, excluding data-parallel padding."""
    real_rows = training_input.batch_size - training_input.metadata.get("pad_size", 0)
    return ConsumedWork(
        sequences=real_rows,
        response_tokens=int(training_input["response_mask"][:real_rows].sum().item()),
        loss_tokens=int(training_input["loss_mask"][:real_rows].sum().item()),
    )


def consumed_stop_metrics(stop_reasons: Sequence[str | None] | None, sequence_count: int) -> dict[str, float]:
    """Count length stops on admitted sequences; the fraction needs every stop reason."""
    reasons = [None] * sequence_count if stop_reasons is None else stop_reasons
    known = sum(reason is not None and reason != "" for reason in reasons)
    length_stops = sum(reason == "length" for reason in reasons)
    metrics = {
        "sequences": float(sequence_count),
        "length_stop_count": float(length_stops),
        "known_stop_count": float(known),
        "unknown_stop_count": float(sequence_count - known),
    }
    if sequence_count:
        metrics["stop_reason_coverage"] = known / sequence_count
        if known == sequence_count:
            metrics["length_stop_fraction"] = length_stops / sequence_count
    return {f"consumed/{name}": value for name, value in metrics.items()}
