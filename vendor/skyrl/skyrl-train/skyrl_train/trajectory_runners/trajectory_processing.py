from typing import List, Tuple, Union, Optional, Dict, Any, Sequence
from collections import defaultdict
from math import fsum
from enum import StrEnum
import numpy as np
from skyrl_train.group_admission import group_is_fully_excluded_from_training
from skyrl_train.task_inputs import TaskInput, PreferenceInput
from skyrl_train.trajectory_runners.types import (
    BatchFields,
    TrajectoryBatch,
    TrajectoryRequestBatch,
    TrajectoryID,
    BatchMetadata,
    TrainingPhase,
    TaskRequestBatch,
    PreferenceRequestBatch,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import (
    NormalizedReward,
    refresh_trajectory_reward_shaping_metrics,
)
from skyrl_train.metric_names import ROLLOUT_FAILURE_FRACTION_METRIC
from skyrl_train.inference_engines.base import UNSTAMPED_POLICY_STEP
from loguru import logger
from skyrl_train.trajectory_runners.rollout_metrics import rollout_length_metrics
from skyrl_gym.verification import VerificationResult, VerificationStatus, normalized_verifier_score


BATCH_ERROR_METRIC_PREFIX = "generate/errors/"
_NUM_TRIALS_METRIC = "generate/num_trials"
_NUM_FAILED_INSTANCES_METRIC = "generate/num_failed_instances"
_NUM_FAILED_TRAJECTORIES_METRIC = "generate/num_failed_trajectories"
_NUM_MASKED_TRAJECTORIES_METRIC = "generate/num_masked_trajectories"


class TitoFullDeclineReason(StrEnum):
    """Reason exact full-token trajectory assembly could not be proven safe."""

    MISSING_STREAMS = "missing_streams"
    EMPTY_STREAMS = "empty_streams"
    TURN_COUNT_MISMATCH = "turn_count_mismatch"
    ASSISTANT_MESSAGE_COUNT_MISMATCH = "assistant_message_count_mismatch"
    MALFORMED_TURN_STREAM = "malformed_turn_stream"
    PREFIX_MISMATCH = "prefix_mismatch"
    INITIAL_PROMPT_TOO_SHORT = "initial_prompt_too_short"
    GENERATION_PROMPT_MISMATCH = "generation_prompt_mismatch"
    COMPLETION_REGION_MISMATCH = "completion_region_mismatch"


def get_metrics_from_trajectory_batch(trajectory_batch: TrajectoryBatch, uids: List[str]) -> Tuple[float, float]:
    """
    Get the mean optimization reward and `pass_at_n` from a trajectory batch.

    The `n` in `pass_at_n` is the number of trajectories we generate for each example. It is
    calculated as `len(trajectory_batch["rewards"]) / len(uids)`, where `len(uids)` is the number of
    unique examples.

    Rewards can be either per-trajectory or per-token. The returned mean describes
    the optimization reward. ``pass_at_n`` uses ``unshaped_rewards`` when supplied,
    so optimization-specific shaping cannot change the task-success metric.
    Explicit verifier pass verdicts take precedence over positive partial rewards.
    """
    rewards: Union[List[float], List[List[float]]] = trajectory_batch["rewards"]
    if not len(rewards):
        raise ValueError(f"`rewards` must be a non-empty list, got {rewards}")

    trajectory_passes = get_trajectory_passes(trajectory_batch)

    if isinstance(rewards[0], list):
        # Token-level rewards: rewards is List[List[float]]
        # For each trajectory, sum token rewards before computing the batch mean.
        mean_reward = fsum(fsum(trajectory_rewards) for trajectory_rewards in rewards) / len(rewards)
    else:
        # Evaluation batches arrive in completion order, including after checkpoint restore.
        mean_reward = fsum(rewards) / len(rewards)

    uid_to_trajectory_passes = defaultdict(list)
    for uid, passed in zip(uids, trajectory_passes, strict=True):
        uid_to_trajectory_passes[uid].append(passed)

    pass_at_n = sum(any(passes) for passes in uid_to_trajectory_passes.values()) / len(uid_to_trajectory_passes)

    return mean_reward, pass_at_n


def get_outcome_rewards(trajectory_batch: TrajectoryBatch) -> List[float]:
    """Return the unshaped task outcome associated with each trajectory."""
    rewards = trajectory_batch["rewards"]
    unshaped_rewards = trajectory_batch.get("unshaped_rewards")
    if unshaped_rewards is not None:
        if len(unshaped_rewards) != len(rewards):
            raise ValueError(
                "`unshaped_rewards` must have one entry per trajectory: "
                f"got {len(unshaped_rewards)} unshaped rewards and {len(rewards)} optimization rewards"
            )
        return [float(reward) for reward in unshaped_rewards]
    return [NormalizedReward.from_output(reward).outcome for reward in rewards]


def normalized_verifier_scores(trajectory_batch: TrajectoryBatch) -> List[float | None] | None:
    """Return bounded task scores, or None if the batch has no verdict channel.

    Entries are None whenever no verifier verdict exists, including verifier
    errors. Verified failures retain their actual zero scores. This keeps
    GenRM's 1–5 ratings comparable with 0–1 verifiers in cross-task averages,
    while leaving the optimization rewards and raw verifier scores intact.
    """
    results = trajectory_batch.get("verification_results")
    if results is None:
        return None
    if len(results) != len(trajectory_batch["rewards"]):
        raise ValueError("verification_results must have one entry per reward")
    scores: List[float | None] = []
    for result in results:
        if result is None or result.status is not VerificationStatus.VERIFIED:
            scores.append(None)
        else:
            scores.append(normalized_verifier_score(result))
    return scores


def verifier_score_summary(scores: List[float | None]) -> tuple[float, float | None]:
    """Return included-row coverage and mean bounded score."""
    included = [score for score in scores if score is not None]
    coverage = len(included) / len(scores) if scores else 0.0
    return coverage, float(np.mean(included)) if included else None


def graded_row_indices(trajectory_batch: TrajectoryBatch) -> List[int]:
    """Return rows whose environment ran grading; skipped rows carry no reward to report."""
    results = trajectory_batch.get("verification_results")
    if results is None:
        return list(range(len(trajectory_batch["rewards"])))
    return [
        index
        for index, result in enumerate(results)
        if result is None or result.status is not VerificationStatus.SKIPPED
    ]


def get_trajectory_passes(trajectory_batch: TrajectoryBatch) -> List[bool]:
    """Return task success, honoring explicit verifier verdicts when available."""
    outcomes = get_outcome_rewards(trajectory_batch)
    results = trajectory_batch.get("verification_results")
    if results is None:
        return [outcome > 0.0 for outcome in outcomes]
    passes = []
    for outcome, result in zip(outcomes, results, strict=True):
        if result is None:
            passes.append(outcome > 0.0)
        elif result.status is not VerificationStatus.VERIFIED:
            passes.append(False)
        else:
            passes.append(result.passed if result.passed is not None else outcome > 0.0)
    return passes


def _rollout_logprob_presence(trajectory_batches: List[TrajectoryBatch], *, required: bool) -> List[bool]:
    """Validate missing logprobs and return their per-group presence mask."""
    presence = [output.get("rollout_logprobs") is not None for output in trajectory_batches]
    if not required:
        return presence
    for output, has_logprobs in zip(trajectory_batches, presence, strict=True):
        if not has_logprobs and not group_is_fully_excluded_from_training(output):
            raise ValueError("rollout_logprobs are required for every generated group")
    return presence


def scalar_reward_token_credit(reward: float, response_ids: Sequence[int]) -> List[float]:
    """Convert a response-level reward to token-level credit on the last response token.

    A zero-token response has nowhere to place the reward, so it stays empty.
    """
    token_rewards = [0.0] * len(response_ids)
    if token_rewards:
        token_rewards[-1] = float(reward)
    return token_rewards


def token_reward_rows(rewards, response_ids) -> List[List[float]]:
    """Retain token rewards or credit each scalar on its final response token."""
    if rewards and isinstance(rewards[0], list):
        return rewards
    return [scalar_reward_token_credit(reward, response) for reward, response in zip(rewards, response_ids)]


def _concatenate_rewards(
    trajectory_batches: Sequence[TrajectoryBatch], *, token_rewards: bool
) -> Union[List[float], List[List[float]]]:
    """Concatenate rewards while preserving token-level credit from any child batch."""
    if not token_rewards:
        return [float(reward) for output in trajectory_batches for reward in output["rewards"]]

    rewards: List[List[float]] = []
    for output in trajectory_batches:
        for reward, response_ids in zip(output["rewards"], output["response_ids"], strict=True):
            if isinstance(reward, list):
                rewards.append(reward)
                continue
            rewards.append(scalar_reward_token_credit(reward, response_ids))
    return rewards


def _reward_sign_successes(rewards: Sequence[float | List[float]]) -> List[bool]:
    return [float(np.sum(reward)) > 0.0 for reward in rewards]


def _concatenate_environment_metrics(
    result: TrajectoryBatch, batches: Sequence[TrajectoryBatch], *, fields: BatchFields
) -> None:
    if "env_metrics" in fields.present:
        for batch in batches:
            if ("env_metrics" in batch) != ("env_classes" in batch):
                raise ValueError("environment metrics and classes must be carried together")
        result["env_metrics"] = [
            metrics for batch in batches for metrics in batch.get("env_metrics", [{} for _ in batch["response_ids"]])
        ]
        result["env_classes"] = [
            env_class for batch in batches for env_class in batch.get("env_classes", [""] * len(batch["response_ids"]))
        ]


def normalize_trajectory_batches(
    trajectory_batches: Sequence[TrajectoryBatch], *, fields: BatchFields
) -> TrajectoryBatch:
    """Concatenate rows using whole-batch field presence and route geometry."""
    assert trajectory_batches
    any_has_logprobs = "rollout_logprobs" in fields.present

    # Handle mixed rollout_logprobs: if some batches have logprobs and others don't,
    # fill in placeholder [0.0] values for the batches that don't have them.
    # This can happen when all trials in a batch fail (returns None) while other batches succeed.
    rollout_logprobs_concat = None
    if any_has_logprobs:
        rollout_logprobs_concat = []
        for output in trajectory_batches:
            if output.get("rollout_logprobs") is not None:
                rollout_logprobs_concat.extend(output["rollout_logprobs"])
            else:
                # Fill in placeholder logprobs for batches that don't have them
                # Each trajectory needs logprobs matching its response_ids length
                for response_ids in output["response_ids"]:
                    rollout_logprobs_concat.append(np.zeros(len(response_ids), dtype=np.float32))

    selected_topk_concat = None
    behavior_topk_concat = None
    if "student_topk_indices" in fields.present:
        if any(
            output.get("student_topk_indices") is None or output.get("behavior_topk_logprobs") is None
            for output in trajectory_batches
        ):
            raise ValueError("student-selected top-K evidence cannot be concatenated with missing rollout scores")
        selected_topk_concat = [row for output in trajectory_batches for row in output["student_topk_indices"]]
        behavior_topk_concat = [row for output in trajectory_batches for row in output["behavior_topk_logprobs"]]
    elif "behavior_topk_logprobs" in fields.present:
        raise ValueError("student-selected behavior scores require selected token IDs")

    data_sources_concat = None
    if "data_sources" in fields.present:
        data_sources_concat = [
            source
            for output in trajectory_batches
            for source in (output.get("data_sources") or [None] * len(output["response_ids"]))
        ]

    unshaped_rewards_concat = None
    unshaped_reward_available_concat = None
    if "unshaped_rewards" in fields.present:
        unshaped_rewards_concat = [reward for output in trajectory_batches for reward in get_outcome_rewards(output)]
        if "unshaped_reward_available" in fields.present:
            unshaped_reward_available_concat = [
                available
                for output in trajectory_batches
                for available in (
                    output.get("unshaped_reward_available")
                    or [output.get("unshaped_rewards") is not None] * len(output["response_ids"])
                )
            ]

    disposition_channels: dict[str, list[Any]] = {}
    for key in ("exception_types", "error_treatments", "server_errors"):
        if key in fields.present:
            disposition_channels[key] = [
                value
                for output in trajectory_batches
                for value in (output.get(key) or [None] * len(output["response_ids"]))
            ]

    baseline_exclusions_concat = None
    if "exclude_from_baseline" in fields.present:
        baseline_exclusions_concat = [
            excluded
            for output in trajectory_batches
            for excluded in (output.get("exclude_from_baseline") or [False] * len(output["response_ids"]))
        ]

    # Missing batches keep the geometry learned from the first captured sample.
    rollout_routed_experts_concat = None
    if "rollout_routed_experts" in fields.present:
        _concat_sentinel_row = None
        if fields.route_geometry is not None:
            layers, top_k, dtype = fields.route_geometry
            _concat_sentinel_row = np.zeros((layers, top_k), dtype=dtype)
        rollout_routed_experts_concat = []
        for output in trajectory_batches:
            if "rollout_routed_experts" in output and output.get("rollout_routed_experts") is not None:
                rollout_routed_experts_concat.extend(output["rollout_routed_experts"])
            else:
                for response_ids in output["response_ids"]:
                    rollout_routed_experts_concat.append(_re_sentinel_rows(len(response_ids), _concat_sentinel_row))

    # Loop-behavior reward shaping (Stage B / F5 + F4): mix the per-token shaping
    # channel + span tags the same way as routed_experts — sentinel-fill (zeros)
    # any batch that lacks them so the concatenated list stays 1:1 with
    # response_ids. When the channel is off NO batch carries the keys (the
    # runner omits them), so these stay None and the result is byte-identical.
    token_level_shaping_concat = None
    if "token_level_shaping" in fields.present:
        token_level_shaping_concat = []
        for output in trajectory_batches:
            if "token_level_shaping" in output and output.get("token_level_shaping") is not None:
                token_level_shaping_concat.extend(output["token_level_shaping"])
            else:
                for response_ids in output["response_ids"]:
                    token_level_shaping_concat.append([0.0] * len(response_ids))

    response_span_tags_concat = None
    if "response_span_tags" in fields.present:
        response_span_tags_concat = []
        for output in trajectory_batches:
            if "response_span_tags" in output and output.get("response_span_tags") is not None:
                response_span_tags_concat.extend(output["response_span_tags"])
            else:
                for response_ids in output["response_ids"]:
                    response_span_tags_concat.append([0] * len(response_ids))

    result: TrajectoryBatch = {
        "prompt_token_ids": sum([output["prompt_token_ids"] for output in trajectory_batches], []),
        "response_ids": sum([output["response_ids"] for output in trajectory_batches], []),
        "rewards": _concatenate_rewards(trajectory_batches, token_rewards=fields.token_rewards),
        "loss_masks": sum([output["loss_masks"] for output in trajectory_batches], []),
        "stop_reasons": (
            sum([output["stop_reasons"] for output in trajectory_batches], [])
            if "stop_reasons" in fields.first_group_list_fields
            else None
        ),
        "rollout_logprobs": rollout_logprobs_concat,
    }
    if rollout_routed_experts_concat is not None:
        result["rollout_routed_experts"] = rollout_routed_experts_concat
    if "rollout_policy_steps" in fields.present:
        result["rollout_policy_steps"] = [
            row
            for output in trajectory_batches
            for row in (
                output.get("rollout_policy_steps")
                or [np.full(len(ids), UNSTAMPED_POLICY_STEP, dtype=np.int32) for ids in output["response_ids"]]
            )
        ]
    if selected_topk_concat is not None:
        result["student_topk_indices"] = selected_topk_concat
        result["behavior_topk_logprobs"] = behavior_topk_concat
    if data_sources_concat is not None:
        result["data_sources"] = data_sources_concat
    if token_level_shaping_concat is not None:
        result["token_level_shaping"] = token_level_shaping_concat
    if response_span_tags_concat is not None:
        result["response_span_tags"] = response_span_tags_concat
    if unshaped_rewards_concat is not None:
        result["unshaped_rewards"] = unshaped_rewards_concat
    if unshaped_reward_available_concat is not None:
        result["unshaped_reward_available"] = unshaped_reward_available_concat
    for key, values in disposition_channels.items():
        result[key] = values
    if baseline_exclusions_concat is not None:
        result["exclude_from_baseline"] = baseline_exclusions_concat

    _concatenate_environment_metrics(result, trajectory_batches, fields=fields)
    for key in ("verification_results", "evidence_messages"):
        if key in fields.present:
            result[key] = [
                value
                for batch in trajectory_batches
                for value in (batch.get(key) or [None] * len(batch["response_ids"]))
            ]

    # propagate additional keys with list values as-is
    additional_keys = [key for key in fields.first_group_list_fields if key not in result]
    if len(additional_keys):
        logger.info(f"Attempting to concatenate values for additional keys {additional_keys}")
    for key in additional_keys:
        result[key] = sum([trajectory_batch[key] for trajectory_batch in trajectory_batches], [])

    return result


def concatenate_trajectory_batches(
    trajectory_batches: List[TrajectoryBatch],
    *,
    require_rollout_logprobs: bool = False,
    tis_lcs_alert_threshold: float,
) -> TrajectoryBatch:
    """Concatenate rollout rows and aggregate their metrics."""
    assert trajectory_batches
    _rollout_logprob_presence(trajectory_batches, required=require_rollout_logprobs)
    fields = BatchFields.from_groups([BatchFields.from_batch(batch) for batch in trajectory_batches])
    result = normalize_trajectory_batches(trajectory_batches, fields=fields)

    from skyrl_train.rollouts.buffer import RowFacts
    from skyrl_train.batch_metrics import rollout_metrics

    result["rollout_metrics"] = rollout_metrics(
        tuple(RowFacts.from_batch(batch) for batch in trajectory_batches),
        tis_lcs_alert_threshold=tis_lcs_alert_threshold,
    )
    refresh_trajectory_reward_shaping_metrics(result)

    num_prompts = len(result["prompt_token_ids"])
    validate_trajectory_batch(num_prompts, result)

    return result


def validate_trajectory_batch(num_prompts: int, trajectory_batch: TrajectoryBatch) -> None:
    """Validate the shape and value categories of a trajectory batch."""
    if not trajectory_batch["response_ids"]:
        raise RuntimeError("No outputs generated")

    num_responses = len(trajectory_batch["response_ids"])
    data_sources = trajectory_batch.get("data_sources")
    if data_sources is not None and len(data_sources) != num_responses:
        raise ValueError(
            f"data_sources must match response_ids: got {len(data_sources)} sources for {num_responses} rows"
        )
    num_prompt_tokens = len(trajectory_batch["prompt_token_ids"])
    assert num_prompts == num_responses, f"Mismatch between prompts ({num_prompts}) and responses ({num_responses})"
    assert num_responses == num_prompt_tokens, (
        f"Mismatch between responses ({num_responses}) and prompt_token_ids ({num_prompt_tokens})"
    )

    for key in (
        "response_ids",
        "loss_masks",
        "rewards",
        "rollout_logprobs",
        "verifier_tests",
        "env_metrics",
        "env_classes",
        "verification_results",
        "evidence_messages",
    ):
        value = trajectory_batch.get(key)
        if isinstance(value, list):
            assert len(value) == num_responses, (
                f"Trajectory batch {key} length must equal response_ids length, got {len(value)} and {num_responses}"
            )

    for index, (response_ids, loss_masks, rewards) in enumerate(
        zip(trajectory_batch["response_ids"], trajectory_batch["loss_masks"], trajectory_batch["rewards"])
    ):
        assert len(response_ids) == len(loss_masks), (
            "Response ids and loss masks must have the same length, "
            f"for sample {index} got {len(response_ids)} and {len(loss_masks)}"
        )
        if isinstance(rewards, list):
            assert len(rewards) == len(response_ids), (
                "Token rewards and response ids must have the same length, "
                f"for sample {index} got {len(rewards)} and {len(response_ids)}"
            )

        rollout_logprobs = trajectory_batch.get("rollout_logprobs")
        if rollout_logprobs:
            assert len(response_ids) == len(rollout_logprobs[index]), (
                "Response ids and rollout logprobs must have the same length, "
                f"for sample {index} got {len(response_ids)} and {len(rollout_logprobs[index])}"
            )

    if np.concatenate(trajectory_batch["loss_masks"]).sum() == 0:
        logger.warning("All outputs are loss masked, which may lead to NaN loss, please check your generation logic!!")

    rewards = trajectory_batch["rewards"]
    if isinstance(rewards[0], list):
        assert all(isinstance(reward, list) for reward in rewards), (
            "rewards must be `List[float]` or `List[List[float]]`"
        )
    else:
        assert all(not isinstance(reward, list) for reward in rewards), (
            "rewards must be `List[float]` or `List[List[float]]`"
        )


def apply_overlong_filtering(
    loss_masks: List[List[int]],
    response_ids: List[List[int]],
    eos_token_id: int,
) -> List[List[int]]:
    """
    Implements DAPO Overlong Filtering: zero-out every token's mask whenever
    the response does not end with the eos token id (i.e. truncated).

    Returns:
        - The loss masks with tokens zeroed out for truncated responses
    """
    assert len(loss_masks) == len(response_ids), "loss_masks and response_ids must have the same length"
    return [
        [0] * len(mask) if not response or response[-1] != eos_token_id else mask
        for mask, response in zip(loss_masks, response_ids)
    ]


def rollout_successes(rewards, verification_results) -> list[bool]:
    """Resolve rollout success from verifier verdicts with reward-sign fallback."""
    successes = _reward_sign_successes(rewards)
    if verification_results is not None:
        for index, result in enumerate(verification_results):
            if result is None:
                continue
            if result.status is not VerificationStatus.VERIFIED:
                successes[index] = False
            else:
                successes[index] = result.passed if result.passed is not None else float(result.score) > 0.0
    return successes


def get_rollout_metrics(
    responses: List[List[int]],
    rewards: Union[List[float], List[List[float]]],
    env_metrics: Optional[List[Dict[str, Any]]] = None,
    env_classes: Optional[List[str]] = None,
    verification_results: Optional[List[Optional[VerificationResult]]] = None,
):
    """Aggregate response lengths and verifier-aware successes by environment."""
    successes = rollout_successes(rewards, verification_results)
    return rollout_length_metrics([len(response) for response in responses], successes, env_metrics, env_classes)


_BATCH_FAILURE_COUNT_KEYS = (
    _NUM_TRIALS_METRIC,
    _NUM_FAILED_INSTANCES_METRIC,
    _NUM_FAILED_TRAJECTORIES_METRIC,
    _NUM_MASKED_TRAJECTORIES_METRIC,
)


def merge_batch_failure_metrics(metrics_by_group: Sequence[dict[str, float]]) -> dict[str, float]:
    """Sum failure counts across rollout groups and recompute their batch fraction."""
    group_metrics = [metrics for metrics in metrics_by_group if _NUM_TRIALS_METRIC in metrics]
    if not group_metrics:
        return {}
    for metrics in group_metrics:
        missing = [key for key in _BATCH_FAILURE_COUNT_KEYS if key not in metrics]
        if missing:
            raise ValueError(f"Incomplete rollout failure metrics: missing {', '.join(missing)}")

    merged = get_batch_failure_metrics(
        sum(metrics[_NUM_TRIALS_METRIC] for metrics in group_metrics),
        num_failed_trajectories=sum(metrics[_NUM_FAILED_TRAJECTORIES_METRIC] for metrics in group_metrics),
        num_failed_instances=sum(metrics[_NUM_FAILED_INSTANCES_METRIC] for metrics in group_metrics),
        num_masked_trajectories=sum(metrics[_NUM_MASKED_TRAJECTORIES_METRIC] for metrics in group_metrics),
    )
    error_keys = {key for metrics in group_metrics for key in metrics if key.startswith(BATCH_ERROR_METRIC_PREFIX)}
    merged.update({key: sum(metrics.get(key, 0) for metrics in group_metrics) for key in error_keys})
    return merged


def get_batch_failure_metrics(
    num_trials: int,
    num_failed_trajectories: int,
    num_failed_instances: int,
    num_masked_trajectories: int,
) -> Dict[str, float]:
    """Describe failed and masked trajectories within a requested rollout group.

    Args:
        num_trials: Requested trajectories and denominator of the failure fraction.
        num_failed_trajectories: Trajectories that did not complete successfully.
        num_failed_instances: Distinct instances with at least one failed trajectory.
        num_masked_trajectories: Trajectories excluded from the baseline.
    """
    return {
        _NUM_TRIALS_METRIC: num_trials,
        _NUM_FAILED_INSTANCES_METRIC: num_failed_instances,
        _NUM_FAILED_TRAJECTORIES_METRIC: num_failed_trajectories,
        _NUM_MASKED_TRAJECTORIES_METRIC: num_masked_trajectories,
        ROLLOUT_FAILURE_FRACTION_METRIC: num_failed_trajectories / num_trials if num_trials else 0.0,
    }


def prepare_trajectory_request(
    rows: List[dict],
    n_samples_per_prompt: int,
    sampling_params: Dict[str, Any],
    training_phase: TrainingPhase,
    global_step: int,
) -> Tuple[TrajectoryRequestBatch, List[str]]:
    """Return repeated typed inputs with attempt IDs and one UID per repeated input."""
    inputs = tuple(row["input"] for row in rows for _ in range(n_samples_per_prompt))
    trajectory_ids = tuple(
        TrajectoryID(instance_id=row["input"].metadata.uid, repetition_id=repetition)
        for row in rows
        for repetition in range(n_samples_per_prompt)
    )
    metadata = BatchMetadata(global_step=global_step, training_phase=training_phase)
    if all(isinstance(row, TaskInput) for row in inputs):
        request = TaskRequestBatch(
            rows=inputs,
            sampling_params=sampling_params,
            trajectory_ids=trajectory_ids,
            batch_metadata=metadata,
        )
    elif all(isinstance(row, PreferenceInput) for row in inputs):
        request = PreferenceRequestBatch(
            prompts=tuple(row.prompt for row in inputs),
            preferences=tuple(row.preference for row in inputs),
            metadata=tuple(row.metadata for row in inputs),
            sampling_params=sampling_params,
            trajectory_ids=trajectory_ids,
            batch_metadata=metadata,
        )
    else:
        raise TypeError("A request must contain TaskInput rows or static PreferenceInput rows, not a mixture")
    return request, [item.instance_id for item in trajectory_ids]


def _re_sentinel_rows(n: int, sentinel_row: np.ndarray) -> np.ndarray:
    """Return a compact sentinel block with the learned route geometry."""
    return np.zeros((n, *sentinel_row.shape), dtype=sentinel_row.dtype)
