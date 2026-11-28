"""Metric names shared by training and offline analysis."""

from rolloutengine.models import TASK_TIMING_KEYS as TASK_TIMING_KEYS

ROLLOUT_FAILURE_FRACTION_METRIC = "generate/failed_trajectory_fraction"
ENVIRONMENT_METRIC_PREFIX = "environment/"
GENERATE_OUTCOME_METRIC_PREFIX = "generate/outcome/"
EVAL_OUTCOME_METRIC_PREFIX = "eval/all/outcome/"
IDENTITY_AWARE_REWARD_METRIC_PREFIX = "generate/reward_shaping/identity_aware"
TASK_ROLLOUT_METRIC_PREFIX = "generate/task_rollout/"
SERVED_TURNS = f"{TASK_ROLLOUT_METRIC_PREFIX}served_turns"
SERVED_LENGTH_STOPS = f"{TASK_ROLLOUT_METRIC_PREFIX}served_length_stops"
UNPROJECTED_LENGTH_STOPS = f"{TASK_ROLLOUT_METRIC_PREFIX}unprojected_length_stops"
SERVED_TOKENS = f"{TASK_ROLLOUT_METRIC_PREFIX}served_response_tokens"
ABANDONED_RETRY_TOKENS = f"{TASK_ROLLOUT_METRIC_PREFIX}abandoned_retry_response_tokens"
UNOBSERVED_CALLS = f"{TASK_ROLLOUT_METRIC_PREFIX}unobserved_generation_calls"
UNOBSERVED_TOKEN_LIMIT = f"{TASK_ROLLOUT_METRIC_PREFIX}unobserved_generation_token_limit"
TIS_ALIGNED_TOKENS_METRIC = "generate/tis/aligned_tokens"
TIS_METRIC_PREFIX = TIS_ALIGNED_TOKENS_METRIC.removesuffix("aligned_tokens")
TIS_EXACT_MATCH_FRACTION_METRIC = "generate/tis/exact_match_fraction"
TIS_UNALIGNED_FRACTION_METRIC = "generate/tis/unaligned_fraction"
TIS_ALIGNMENT_FAIL_COUNT_METRIC = "generate/tis/alignment_fail_count"
TIS_ALIGNMENT_ALERT_METRIC = "generate/tis/alignment_alert"
TIS_TITO_FULL_ATTEMPTS_METRIC = "generate/tis/tito_full/attempts"
TIS_TITO_FULL_SUCCESS_FRACTION_METRIC = "generate/tis/tito_full/success_fraction"
TIS_TITO_FULL_DECLINE_COUNT_METRIC = "generate/tis/tito_full/decline_count"
TIS_TITO_FULL_DECLINE_METRIC_PREFIX = "generate/tis/tito_full/decline/"

CORRECTION_WEIGHT_MEAN_METRIC = "policy/correction/weight_mean"
CORRECTION_TRUNCATED_FRACTION_METRIC = "policy/correction/truncated_fraction"
CORRECTION_MASKED_FRACTION_METRIC = "policy/correction/masked_fraction"

PROMPTS_DROPPED_TOO_LONG_METRIC = "generate/prompts_dropped_too_long"


def curriculum_metric_prefix(source: str) -> str:
    return f"curriculum/{source}/"
