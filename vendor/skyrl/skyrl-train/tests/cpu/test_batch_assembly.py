import numpy as np
import pytest
import torch
from omegaconf import open_dict

from types import SimpleNamespace
from skyrl_train.batch_assembly import assemble_slice, fields_from_facts, plan_batch
from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import resolve_group_advantage_invariant
from skyrl_train.rollouts.buffer import RolloutGroup, RowFacts
from skyrl_train.trajectory_runners.types import BatchFields
from skyrl_train.trajectory_runners.trajectory_processing import (
    normalize_trajectory_batches,
    token_reward_rows,
)
from skyrl_train.utils.advantage_estimators import (
    compute_advantages_and_returns,
    apply_loop_advantages,
    drop_advantage_inputs,
)


def assert_batch_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None
        else:
            assert actual[key] is not None
            assert actual[key].dtype == value.dtype
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            if value.is_floating_point():
                assert actual[key].numpy().tobytes() == value.numpy().tobytes(), key
    assert actual.metadata.keys() == expected.metadata.keys()
    for key, value in expected.metadata.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual.metadata[key], value)
        else:
            assert actual.metadata[key] == value
    assert actual.routed_expert_rows.response_len == expected.routed_expert_rows.response_len
    assert actual.routed_expert_rows.num_experts == expected.routed_expert_rows.num_experts
    for row, reference in zip(actual.routed_expert_rows.rows, expected.routed_expert_rows.rows, strict=True):
        assert row.dtype == reference.dtype
        np.testing.assert_array_equal(row, reference)
    torch.testing.assert_close(actual.routed_experts_tensor(), expected.routed_experts_tensor(), rtol=0, atol=0)


@pytest.mark.parametrize("dp_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("estimator", ["rloo", "rloo_n", "grpo"])
@pytest.mark.parametrize("loop_channel", [False, True])
def test_dp_slices_with_fields_and_longest_rows_on_other_ranks_match_driver(dp_size, estimator, loop_channel):
    cfg = get_default_config()
    algorithm = cfg.trainer.algorithm
    algorithm.advantage_estimator = estimator
    algorithm.group_advantage_min_size = 2 if estimator == "rloo_n" else None
    algorithm.enable_token_reward_channel = True
    invariant = resolve_group_advantage_invariant(
        advantage_estimator=estimator,
        physical_group_size=4,
        minimum_group_size=algorithm.group_advantage_min_size,
    )
    with open_dict(algorithm):
        algorithm.resolved_group_advantage = invariant.to_config()
    groups = []
    for index in range(6):
        lengths = [0 if index in (2, 3) else 1, 2, 3, 9 if index == 5 else 4]
        rewards = [-0.0, -0.0, 0.0, -0.0] if index == 1 else [9.0 if index == 3 else 0.0, 1.0, 3.0, 2.0]
        batch = {
            "prompt_token_ids": [[index + 1] * (7 if index == 5 else 2) for _ in lengths],
            "response_ids": [
                ([10 + index] + [20 + row] * (length - 1)) if length else [] for row, length in enumerate(lengths)
            ],
            "loss_masks": [[1] * length for length in lengths],
            "rewards": rewards,
            "is_last_step": [True] * 4,
            "stop_reasons": ["stop", "length", None, "stop"],
        }
        if loop_channel:
            batch["loop_advantages"] = [[0.0] * length for length in lengths]
        if index % 2 == 0:
            batch["rewards"] = [
                [0.0] * (length - 1) + [reward] if length else [] for length, reward in zip(lengths, rewards)
            ]
        if index == 0:
            batch["rewards"][1] = [1.0, 0.0]
            batch["rewards"][0] = [-0.0]
        if index >= 4:
            batch["rollout_logprobs"] = [np.full(length, -1.25, dtype=np.float32) for length in lengths]
            batch["rollout_routed_experts"] = [np.full((length, 3, 2), index, dtype=np.uint8) for length in lengths]
            batch["token_level_shaping"] = [[-0.5] * length for length in lengths]
            batch["response_span_tags"] = [[2] * length for length in lengths]
            batch["exclude_from_baseline"] = [False, False, True, False]
        groups.append(RolloutGroup(batch, f"g{index}", index % 2, {"uid": f"g{index}"}))

    batches = [group.trajectory_batch for group in groups]
    facts = [RowFacts.from_batch(batch) for batch in batches]
    uids = [group.uid for group in groups for _ in group.trajectory_batch["response_ids"]]
    layout = dict(
        batch_id=2,
        global_step=7,
        uids=uids,
        algorithm=algorithm,
        rollout_staleness=[2 - group.policy_step for group in groups for _ in group.trajectory_batch["response_ids"]],
        num_experts=128,
    )
    worker_plan = plan_batch(facts=tuple(facts), fields=fields_from_facts(facts), **layout)
    normalized = normalize_trajectory_batches(
        batches, fields=BatchFields.from_groups([BatchFields.from_batch(batch) for batch in batches])
    )
    normalized["rewards"] = token_reward_rows(normalized["rewards"], normalized["response_ids"])
    whole_plan = plan_batch(
        facts=(RowFacts.from_batch(normalized),), fields=BatchFields.from_batch(normalized), **layout
    )
    whole = assemble_slice(
        whole_plan, range(len(uids)), normalized, tokenizer=SimpleNamespace(pad_token_id=0), algorithm=algorithm
    )
    # This oracle sums the full token-width reward tensors through the public estimator.
    # It never consumes committed scalar scores or outcome_advantages.
    whole["advantages"], whole["returns"] = compute_advantages_and_returns(
        token_level_rewards=whole["rewards"],
        response_mask=whole["response_mask"],
        index=uids,
        adv_estimator=algorithm.advantage_estimator,
        config=algorithm,
        values=None,
        gamma=algorithm.gamma,
        lambd=algorithm.lambd,
        grpo_norm_by_std=algorithm.grpo_norm_by_std,
        exclude_from_baseline=whole.metadata.get("exclude_from_baseline"),
        group_advantage_invariant=invariant,
    )
    whole.metadata.update(pad_size=0, global_step=7)
    for rank, expected in enumerate(whole.chunk(whole.batch_size // dp_size)):
        rows = worker_plan.rank_rows(rank, dp_size)
        selected = []
        for group_index in worker_plan.group_indices(rank, dp_size):
            # Four rows per fixture group; DP8 intentionally cuts across groups.
            start = group_index * 4
            selected.append(
                filter_trajectory_batch(
                    batches[group_index], list(range(max(rows.start - start, 0), min(rows.stop - start, 4)))
                )
            )
        local = normalize_trajectory_batches(selected, fields=worker_plan.fields)
        local["rewards"] = token_reward_rows(local["rewards"], local["response_ids"])
        actual = assemble_slice(
            worker_plan, rows, local, tokenizer=SimpleNamespace(pad_token_id=0), algorithm=algorithm
        )
        actual.metadata.update(pad_size=0, global_step=7)
        actual["advantages"] = (
            torch.from_numpy(worker_plan.advantages[rows.start : rows.stop].copy())[:, None] * actual["response_mask"]
        )
        actual["returns"] = actual["advantages"]
        assert_batch_exact(actual, expected)
        # TensorBatch chunks share metadata; finalize an independent public projection.
        expected.metadata = dict(expected.metadata)
        assert_batch_exact(
            drop_advantage_inputs(apply_loop_advantages(actual)), drop_advantage_inputs(apply_loop_advantages(expected))
        )
