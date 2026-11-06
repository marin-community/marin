from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from skyrl_gym.verification import VerificationResult
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.inference_engines.base import UNSTAMPED_POLICY_STEP
from skyrl_train.utils.utils import validate_cfg
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from skyrl_train.batch_metrics import reward_metrics
from skyrl_train.rollouts.buffer import RowFacts

import torch
from omegaconf import OmegaConf, open_dict
from skyrl_train.batch_assembly import (
    assemble_rank_slice,
    assemble_slice,
    install_plan_advantages,
    BatchPlan,
)
from skyrl_train.config.utils import get_default_config
from skyrl_train.batch_source import require_worker_batch_rows
from skyrl_train.group_admission import resolve_group_advantage_invariant
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.trajectory_runners.trajectory_processing import (
    normalize_trajectory_batches,
    token_reward_rows,
)
from skyrl_train.trajectory_runners.types import BatchFields
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics
from skyrl_train.metric_reduction import WorldReduction
from skyrl_train.utils.advantage_estimators import (
    compute_advantages_and_returns,
)


class RecordedRows:
    def set_trajectory_sink(self, sink):
        pass


@pytest.fixture
def config():
    directory = str(Path(__file__).resolve().parents[2] / "skyrl_train/config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        return compose(
            config_name="ppo_base_config",
            overrides=[
                "+algorithm_recipe=ftpo",
                "trainer.use_sample_packing=false",
                "generator.sampling_params.logprobs=4",
                "generator.sampling_params.temperature=0",
                "trainer.logger=console",
                "trainer.training_metrics=true",
                "trainer.train_batch_size=4",
                "trainer.policy_mini_batch_size=4",
                "generator.n_samples_per_prompt=1",
                "generator.trajectory_retention.enabled=false",
            ],
        )


@pytest.mark.parametrize("lease", [None, 2])
def test_verified_closure_trainer_preserves_rollouts_and_trains_verified_repairs(config, lease):
    config.trainer.algorithm.ftpo = {
        "boundary": "verified_closure_and_continuation",
        "chosen_balance_strength": 0,
        "repair_sft_weight": 0.7,
        "auxiliary_weight": None,
    }
    validate_cfg(config)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ['assistant [{"a":1}]'],
        trainers.BpeTrainer(
            vocab_size=258, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["[PAD]", "[EOS]"]
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]")
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=tokenizer,
        train_dataset=range(4),
        inference_engine_client=None,
        trajectory_runner=RecordedRows(),
        context=None,
        callbacks=[],
    )
    trainer.global_step = 3
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=2))])
    prefix = '[{"a":1'
    prompt = tokenizer.encode("assistant " + prefix, add_special_tokens=False)
    responses = [tokenizer.encode(text, add_special_tokens=False) for text in (" ]", " ", " ]", " ]")]
    original_responses = deepcopy(responses)
    candidates = [tokenizer.encode(text, add_special_tokens=False)[0] for text in ("]", "}", "x", "\n")]
    verification = VerificationResult.verified(
        0,
        diagnostics={
            "stop_reason": "length",
            "closure": {
                "format": "json",
                "family": "json_nested",
                "control": "close_required",
                "prefix": prefix,
                "expected": '[{"a":1}]',
            },
        },
    )
    trajectory = {
        "prompt_token_ids": [prompt] * 4,
        "response_ids": responses,
        "loss_masks": [[1] * len(row) for row in responses],
        "rollout_policy_steps": [
            np.full(len(row), UNSTAMPED_POLICY_STEP if index == 1 else 1, dtype=np.int32)
            if index < 2
            else np.arange(1, len(row) + 1, dtype=np.int32)
            for index, row in enumerate(responses)
        ],
        "rewards": [[0.0] * len(row) for row in responses],
        "student_topk_indices": [np.tile(candidates, (len(row), 1)) for row in responses],
        "behavior_topk_logprobs": [
            np.tile(np.log([0.5, 0.3, 0.15, 0.05]), (len(row), 1)).astype(np.float32) for row in responses
        ],
        "verification_results": [
            verification,
            verification,
            VerificationResult.unavailable("transport failed"),
            verification,
        ],
        "is_last_step": [True] * 4,
        "exclude_from_baseline": [False, False, False, True],
    }
    uids = ["a", "b", "c", "d"]
    batch = trainer.convert_to_training_input(
        trajectory, uids, rollout_staleness=None if lease is None else [lease] * 4
    )
    assert batch.metadata["response_length"] == batch["response_mask"].shape[1] == 3
    assert trainer.all_metrics["async/staleness_tokens_mean"] == pytest.approx(0.0 if lease is None else 12 / 7)
    assert trainer.all_metrics["async/staleness_tokens_stamped"] == (0.0 if lease is None else 6 / 7)
    assert trainer.all_metrics["async/staleness_tokens_spanning"] == (0.0 if lease is None else 0.5)
    target_weights = batch["loss_mask"]
    assert target_weights.nonzero().tolist() == [[0, 1], [1, 1], [1, 2]]
    assert batch["ftpo_repair_mask"].nonzero().tolist() == [[1, 1], [1, 2]]
    assert trajectory["response_ids"] == original_responses
    repaired = batch["sequences"][1][batch["attention_mask"][1].bool()].tolist()
    assert tokenizer.decode(repaired, clean_up_tokenization_spaces=False) == 'assistant [{"a":1 }]'
    assert batch["rollout_logprobs"] is None
    assert batch["student_topk_indices"][batch["ftpo_chosen_mask"]].tolist() == [candidates[1]]
    assert trainer.all_metrics["ftpo/selected_boundaries"] == 1


@pytest.mark.parametrize(
    ("rewards", "responses", "expected", "domain_mean", "verification"),
    [
        ([1e16, 1.0, -1e16], [[1], [1], [1]], 1.0 / 3.0, 0.0, None),
        ([[1e16, 1.0, -1e16], [2e16, 1.0, -2e16]], [[1, 2, 3], [1, 2, 3]], 1.0, 1.0, None),
        (
            [[float("inf"), -float("inf")], [1.0]],
            [[1, 2], [1]],
            1.0,
            1.0,
            [VerificationResult.skipped("row did not grade"), VerificationResult.verified(1.0)],
        ),
    ],
    ids=["scalar", "token", "ungraded"],
)
def test_reward_mean_uses_compensated_totals_without_changing_domain_arithmetic(
    rewards, responses, expected, domain_mean, verification
):
    count = len(responses)
    batch = {
        "prompt_token_ids": [[0]] * count,
        "response_ids": responses,
        "rewards": rewards,
        "loss_masks": [[1] * len(row) for row in responses],
        "data_sources": ["alpha"] * count,
    }
    if verification is not None:
        batch["verification_results"] = verification
        batch["loss_masks"][0] = [0] * len(responses[0])
    facts = tuple(
        RowFacts.from_batch({key: [values[index]] for key, values in batch.items()}) for index in range(count)
    )
    metrics = reward_metrics(facts, [str(index) for index in range(count)], n_samples_per_prompt=1)
    assert metrics["reward/avg_raw_reward"] == expected
    assert metrics["reward/domain/alpha/avg_raw_reward"] == domain_mean


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


@pytest.mark.parametrize("dp_size", [1, 3, 8])
@pytest.mark.parametrize("estimator", ["rloo", "rloo_n", "grpo"])
@pytest.mark.parametrize("loop_channel", [False, True, "first", "last"])
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
        if loop_channel is True or (loop_channel == "first" and index == 0) or (loop_channel == "last" and index == 5):
            batch["loop_advantages"] = [[0.0] * length for length in lengths]
            batch["reasoning_span_violations"] = [int(length > 0) for length in lengths]
            batch["reasoning_span_violation_tokens"] = [[0] if length else [] for length in lengths]
            batch["reasoning_span_token_penalties"] = [0.25 if length else 0.0 for length in lengths]
        if index % 2 == 0:
            batch["rewards"] = [
                [0.0] * (length - 1) + [reward] if length else [] for length, reward in zip(lengths, rewards)
            ]
        if index == 0:
            batch["rewards"][1] = [0.25, 0.75]
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
        global_step=7,
        uids=uids,
        rollout_staleness=[2 - group.policy_step for group in groups for _ in group.trajectory_batch["response_ids"]],
        num_experts=128,
    )
    worker_plan = BatchPlan(
        facts=tuple(facts), fields=BatchFields.from_groups([group.fields for group in facts]), **layout
    )
    normalized = normalize_trajectory_batches(
        batches, fields=BatchFields.from_groups([BatchFields.from_batch(batch) for batch in batches])
    )
    if loop_channel:
        active = range(24) if loop_channel is True else range(4) if loop_channel == "first" else range(20, 24)
        violations = [int(row in active and row not in (8, 12)) for row in range(24)]
        assert normalized["reasoning_span_violations"] == violations
        assert normalized["reasoning_span_violation_tokens"] == [[0] if value else [] for value in violations]
        assert normalized["reasoning_span_token_penalties"] == [0.25 * value for value in violations]
        lengths = [1, 2, 3, 4] * 6
        lengths[8] = lengths[12] = 0
        lengths[-1] = 9
        assert normalized["loop_advantages"] == [[0.0] * length for length in lengths]
    normalized["rewards"] = token_reward_rows(normalized["rewards"], normalized["response_ids"])
    whole_plan = BatchPlan(
        facts=(RowFacts.from_batch(normalized),), fields=BatchFields.from_batch(normalized), **layout
    )
    whole = assemble_slice(whole_plan, range(len(uids)), normalized, pad_token_id=0, algorithm=algorithm)
    # This oracle sums the full token-width reward tensors through the public estimator.
    # It never consumes committed scalar scores or outcome_advantages.
    whole["advantages"], whole["returns"] = compute_advantages_and_returns(
        token_level_rewards=whole["rewards"],
        response_mask=whole["response_mask"],
        index=uids,
        adv_estimator=algorithm.advantage_estimator,
        config=algorithm,
        gamma=algorithm.gamma,
        grpo_norm_by_std=algorithm.grpo_norm_by_std,
        exclude_from_baseline=whole.metadata.get("exclude_from_baseline"),
        group_advantage_invariant=invariant,
    )
    whole.metadata.update(pad_size=0, global_step=7)
    for rank, expected in enumerate(whole.chunk(whole.batch_size // dp_size)):
        rows = worker_plan.rank_rows(rank, dp_size)
        indices = worker_plan.group_indices(rank, dp_size)
        actual = assemble_rank_slice(
            worker_plan, indices, [batches[index] for index in indices], rows, pad_token_id=0, algorithm=algorithm
        )
        # The collective boundary supplies scores contributed by every other DP slice.
        other_scores = whole["rewards"].sum(-1)
        other_scores[rows.start : rows.stop] = 0
        install_plan_advantages(
            actual,
            worker_plan,
            rows,
            algorithm,
            reduction=WorldReduction(lambda values, operation: values + other_scores, True),
        )
        assert_batch_exact(actual, expected)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({}, None),
        ({"batch.loop_advantages": [[0.0, 0.1]] * 4}, "nonzero loop credit"),
        ({"batch.teacher_evidence": [None] * 4}, "teacher evidence"),
        ({"batch.distillation": [None] * 4}, "teacher evidence"),
        ({"batch.student_topk_indices": [None] * 4}, "teacher evidence"),
        ({"batch.behavior_topk_logprobs": [None] * 4}, "teacher evidence"),
        ({"missing_facts": True}, "checkpoint metadata without row facts"),
        ({"rows": 3}, "rows requiring DP padding"),
        (
            {
                "trainer.policy.megatron_config.moe_router_replay": True,
                "batch.rollout_routed_experts": [np.zeros((2, 1, 1), dtype=np.uint8)] * 4,
            },
            "route dtype without the model expert count",
        ),
        ({"trainer.algorithm.use_kl_loss": True}, "DPO and reference-model objectives"),
        ({"trainer.algorithm.use_kl_in_reward": True}, "DPO and reference-model objectives"),
        ({"trainer.algorithm.policy_loss_type": "dpo"}, "DPO and reference-model objectives"),
        ({"trainer.algorithm.policy_loss_type": "ftpo"}, "DPO and reference-model objectives"),
        ({"trainer.algorithm.advantage_estimator": "reward"}, "configured loss and estimator"),
        ({"trainer.algorithm.policy_loss_type": "clip_cov"}, "configured loss and estimator"),
        ({"trainer.placement.colocate_all": True}, "colocated placement"),
        ({"trainer.algorithm.distillation": {}}, "distillation"),
        ({"trainer.trajectory_selector.type": "top_k"}, "trajectory selection"),
        ({"trainer.step_wise_training": True}, "step-wise training"),
        ({"trainer.algorithm.advantage_batch_normalize": True}, "advantage batch normalization"),
        (
            {
                "generator.trajectory_reward_shaping.enabled": True,
                "generator.trajectory_reward_shaping.loop.advantage_penalty_per_token": 0.1,
            },
            "loop advantage credit",
        ),
        (
            {
                "generator.trajectory_reward_shaping.enabled": True,
                "generator.trajectory_reward_shaping.reasoning_span.target": "marker",
                "generator.trajectory_reward_shaping.reasoning_span.penalty": 0.5,
            },
            "marker-target reasoning penalties",
        ),
        ({"trainer.policy.megatron_config.expert_tensor_parallel_size": 2, "model_experts": 4}, "ETP greater than one"),
        (
            {
                "trainer.policy.megatron_config.expert_tensor_parallel_size": None,
                "trainer.policy.megatron_config.tensor_model_parallel_size": 2,
                "model_experts": 4,
            },
            "ETP greater than one",
        ),
        (
            {
                "trainer.policy.megatron_config.expert_tensor_parallel_size": None,
                "trainer.policy.megatron_config.tensor_model_parallel_size": 2,
                "model_experts": 8,
                "model_type": "grug_moe",
            },
            "ETP greater than one",
        ),
        (
            {
                "trainer.policy.megatron_config.expert_tensor_parallel_size": None,
                "trainer.policy.megatron_config.tensor_model_parallel_size": 2,
            },
            None,
        ),
        (
            {"trainer.train_batch_size": 3, "trainer.policy_mini_batch_size": 3, "generator.n_samples_per_prompt": 3},
            "rows requiring DP padding",
        ),
        ({"trainer.dump_data_batch": True}, "dense batch dumping"),
    ],
)
def test_worker_batch_rejects_settings_and_rows_that_change_driver_training_semantics(tmp_path, overrides, message):
    cfg = get_default_config()
    cfg.trainer.batch_builder = "worker"
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.generator.n_samples_per_prompt = 2
    metadata = (
        {"model_type": overrides.get("model_type", "qwen2_moe"), "num_experts": overrides["model_experts"]}
        if "model_experts" in overrides
        else {"model_type": "qwen2"}
    )
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    cfg.trainer.policy.model.path = str(tmp_path)
    batch = {
        "prompt_token_ids": [[1]] * 4,
        "response_ids": [[2, 3]] * 4,
        "loss_masks": [[1, 1]] * 4,
        "rewards": [[0.0, 1.0]] * 4,
    }
    for key, value in overrides.items():
        if key.startswith("batch."):
            batch[key.removeprefix("batch.")] = value
        elif key not in {"rows", "missing_facts", "model_experts", "model_type"}:
            OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    if "rows" in overrides:
        batch = {key: values[: overrides["rows"]] for key, values in batch.items()}

    def check():
        validate_cfg(cfg)
        facts = [None if overrides.get("missing_facts") else RowFacts.from_batch(batch)]
        require_worker_batch_rows(cfg, facts, group_uids=("g0",), dp_size=2)

    if message is None:
        check()
    else:
        row_message = (
            "g0.*" + message
            if any(key.startswith("batch.") or key in {"rows", "missing_facts"} for key in overrides)
            else message
        )
        with pytest.raises(ValueError, match=row_message):
            check()


def test_worker_ratio_moments_match_known_finite_token_values():
    metrics = mismatch_ratio_metrics(
        torch.tensor([[-1.0, 0.0, 1.0, 2.0, float("nan")]]),
        torch.zeros(1, 5),
        torch.ones(1, 5),
        torch.zeros(1, 5),
        reduction=WorldReduction(lambda values, operation: values, True),
    )
    expected = {
        "log_ratio_mean_squared": 1.5,
        "lower_clip_pressure": 0.25,
        "upper_clip_pressure": 0.5,
        "log_ratio_abs_max": 2.0,
    }
    for scope in ("pooled", "staleness0"):
        assert {key: metrics[f"policy/mismatch/{scope}/{key}"] for key in expected} == expected
