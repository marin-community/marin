from collections import defaultdict

import numpy as np
from skyrl_gym.metrics import aggregate_for_task

from skyrl_train.metric_names import ENVIRONMENT_METRIC_PREFIX


def rollout_length_metrics(response_lengths, successes, env_metrics=None, env_classes=None):
    """Aggregate response lengths, success flags, and environment observations."""
    num_tokens_arr = np.array(response_lengths)
    non_zero_rewards_arr = np.array(successes, dtype=bool)
    zero_rewards_arr = ~non_zero_rewards_arr
    # average tokens for non zero rewards
    avg_tokens_non_zero_rewards = (
        np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
    )
    # average tokens for zero rewards
    avg_tokens_zero_rewards = np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)

    rollout_metrics = {
        "generate/min_num_tokens": np.min(num_tokens_arr).item(),
        "generate/max_num_tokens": np.max(num_tokens_arr).item(),
        "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
        "generate/std_num_tokens": np.std(num_tokens_arr).item(),
        "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
        "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
    }

    if env_metrics is not None and env_classes is not None:
        env_to_metrics = defaultdict(list)
        for i, metrics in enumerate(env_metrics):
            # Skipped episodes (e.g. over-length prompts) never step the environment and
            # report an empty dict; per-environment aggregators only see stepped episodes.
            if metrics:
                env_to_metrics[env_classes[i]].append(metrics)
        for env_name, metrics in env_to_metrics.items():
            # Aggregate metrics across all trajectories for the same environment
            agg = aggregate_for_task(env_name, metrics)
            for key, value in agg.items():
                rollout_metrics[f"{ENVIRONMENT_METRIC_PREFIX}{key}"] = value

    return rollout_metrics
