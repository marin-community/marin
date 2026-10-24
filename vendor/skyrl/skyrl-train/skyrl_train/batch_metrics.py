from omegaconf import DictConfig
from skyrl_train.config.objective_spec import off_policy_correction
from skyrl_train.objective.correction import compute_correction
from skyrl_train.utils.importance_ratio_diagnostics import LocalReduction
from skyrl_train.training_batch import TrainingInputBatch
from collections.abc import Sequence
from collections import defaultdict
from typing import Dict, List
import re
import numpy as np

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
