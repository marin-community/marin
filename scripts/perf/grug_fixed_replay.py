# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light helpers for the fixed Grug replay benchmark."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_REPRESENTATIVE_POSITIONS = ("first", "middle", "last")
_COORDINATE_FIELDS = (
    "microbatch",
    "representative_position",
    "worker_sample_index",
    "action_index",
    "model_token_index",
)


def build_loss_weight(loss_mask: np.ndarray, sequence_length: int) -> np.ndarray:
    """Align SkyRL action-token weights with native next-token loss positions."""

    if loss_mask.ndim != 2:
        raise ValueError(f"loss_mask must be rank 2, got {loss_mask.shape}")
    action_length = int(loss_mask.shape[1])
    start = sequence_length - action_length - 1
    end = sequence_length - 1
    if start < 0:
        raise ValueError(f"action length {action_length} does not fit sequence length {sequence_length}")
    result = np.zeros((loss_mask.shape[0], sequence_length), dtype=np.float32)
    result[:, start:end] = loss_mask.astype(np.float32, copy=False)
    return result


def representative_action_coordinates(
    *, sequence_length: int, action_length: int, microbatch_count: int
) -> list[dict[str, int | str]]:
    """Name the same three action samples per microbatch as MarinSkyRL."""

    if sequence_length <= 1 or action_length <= 0 or action_length >= sequence_length:
        raise ValueError(f"invalid sequence/action lengths: sequence={sequence_length}, actions={action_length}")
    if microbatch_count <= 0:
        raise ValueError(f"microbatch_count must be positive, got {microbatch_count}")

    action_indices = (0, action_length // 2, action_length - 1)
    model_start = sequence_length - action_length - 1
    return [
        {
            "microbatch": microbatch,
            "representative_position": position,
            "worker_sample_index": microbatch * len(action_indices) + ordinal,
            "action_index": action_index,
            "model_token_index": model_start + action_index,
        }
        for microbatch in range(microbatch_count)
        for ordinal, (position, action_index) in enumerate(zip(_REPRESENTATIVE_POSITIONS, action_indices, strict=True))
    ]


def _sample_map(workers: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], float]:
    samples: dict[tuple[Any, ...], float] = {}
    for worker in workers:
        rank = int(worker["rank"])
        values = worker["representative_action_log_probs"]
        coordinates = worker["representative_action_log_prob_coordinates"]
        if len(values) != len(coordinates):
            raise ValueError(f"rank {rank} has {len(values)} values but {len(coordinates)} coordinates")
        for value, coordinate in zip(values, coordinates, strict=True):
            if set(coordinate) != set(_COORDINATE_FIELDS):
                raise ValueError(f"rank {rank} has unexpected coordinate fields: {sorted(coordinate)}")
            key = (rank, *(coordinate[field] for field in _COORDINATE_FIELDS))
            if key in samples:
                raise ValueError(f"duplicate sampled-action coordinate: {key}")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"non-finite sampled action log probability at {key}: {value}")
            samples[key] = value
    return samples


def _distance_summary(
    oracle: np.ndarray,
    reference: np.ndarray,
    keys: Sequence[tuple[Any, ...]],
) -> dict[str, Any]:
    signed = reference - oracle
    absolute = np.abs(signed)
    maximum_index = int(np.argmax(absolute))
    maximum_key = keys[maximum_index]
    coordinate = {"rank": int(maximum_key[0])}
    coordinate.update(dict(zip(_COORDINATE_FIELDS, maximum_key[1:], strict=True)))
    return {
        "checked": int(absolute.size),
        "mean_signed_difference": float(np.mean(signed)),
        "mean_abs_difference": float(np.mean(absolute)),
        "median_abs_difference": float(np.median(absolute)),
        "p95_abs_difference": float(np.percentile(absolute, 95)),
        "max_abs_difference": float(absolute[maximum_index]),
        "root_mean_squared_difference": float(np.sqrt(np.mean(np.square(signed)))),
        "max_abs_difference_sample": {
            "coordinate": coordinate,
            "levanter": float(oracle[maximum_index]),
            "reference": float(reference[maximum_index]),
            "signed_difference": float(signed[maximum_index]),
        },
    }


def compare_sampled_action_log_probs(
    levanter_workers: Sequence[Mapping[str, Any]],
    reference_arms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare frozen MSRL arms with Levanter at exact shared coordinates.

    This deliberately reports descriptive distances rather than declaring a
    cross-framework semantic tolerance.
    """

    if len(reference_arms) != 2:
        raise ValueError(f"expected exactly two reference arms, got {sorted(reference_arms)}")
    oracle_map = _sample_map(levanter_workers)
    if not oracle_map:
        raise ValueError("no Levanter sampled action log probabilities")
    keys = sorted(oracle_map)
    oracle = np.asarray([oracle_map[key] for key in keys], dtype=np.float64)

    reference_values: dict[str, np.ndarray] = {}
    distances = {}
    for arm_name, workers in reference_arms.items():
        arm_map = _sample_map(workers)
        missing = [key for key in keys if key not in arm_map]
        if missing:
            raise ValueError(f"reference arm {arm_name!r} is missing {len(missing)} Levanter coordinates")
        values = np.asarray([arm_map[key] for key in keys], dtype=np.float64)
        reference_values[arm_name] = values
        distances[arm_name] = _distance_summary(oracle, values, keys)

    left_name, right_name = reference_arms
    left = reference_values[left_name]
    right = reference_values[right_name]
    changed = left != right
    left_error = np.abs(left - oracle)
    right_error = np.abs(right - oracle)
    left_closer = changed & (left_error < right_error)
    right_closer = changed & (right_error < left_error)
    ties = changed & (left_error == right_error)
    changed_count = int(np.count_nonzero(changed))
    pair = {
        "left_arm": left_name,
        "right_arm": right_name,
        "checked": int(oracle.size),
        "changed": changed_count,
        "unchanged": int(oracle.size - changed_count),
        "left_closer_on_changed": int(np.count_nonzero(left_closer)),
        "right_closer_on_changed": int(np.count_nonzero(right_closer)),
        "ties_on_changed": int(np.count_nonzero(ties)),
        "mean_abs_error_delta_right_minus_left": float(np.mean(right_error - left_error)),
        "sum_squared_error_left": float(np.sum(np.square(left - oracle))),
        "sum_squared_error_right": float(np.sum(np.square(right - oracle))),
    }
    pair["sum_squared_error_right_over_left"] = (
        pair["sum_squared_error_right"] / pair["sum_squared_error_left"] if pair["sum_squared_error_left"] != 0 else None
    )
    pair["right_closer_fraction_of_changed"] = pair["right_closer_on_changed"] / changed_count if changed_count else None
    return {
        "coordinate_contract": list(_COORDINATE_FIELDS),
        "distances_from_levanter": distances,
        "paired_preference": pair,
        "interpretation": (
            "Descriptive only: the step-630 export path is provenance-backed but exact tensor identity is not "
            "independently certified, and shared framework differences can dominate the small within-MSRL arm "
            "delta. No cross-framework pass threshold is applied."
        ),
    }


def repacked_operational_micro_loss(
    cross_entropy_sum,
    router_aux_loss,
    *,
    global_loss_tokens: int,
    microbatch_count: int,
):
    """Scale one repacked microbatch like one logical production batch.

    Token losses are additive, so each microbatch contributes its CE sum
    divided by the logical batch's total loss-token count. Router loss is
    already a mean statistic, so the logical update uses its arithmetic mean
    across the repacked microbatches.
    """

    if global_loss_tokens <= 0:
        raise ValueError("global_loss_tokens must be positive")
    if microbatch_count <= 0:
        raise ValueError("microbatch_count must be positive")
    return cross_entropy_sum / global_loss_tokens + router_aux_loss / microbatch_count
