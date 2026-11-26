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
from skyrl_train.config.driver_validation import validate_cfg, worker_batch_refusal
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from skyrl_train.batch_metrics import reward_metrics
from skyrl_train.rollouts.buffer import RowFacts

import torch
from omegaconf import OmegaConf, open_dict
from skyrl_train.batch_assembly import (
    assemble_rank_slice,
    assemble_slice,
    install_plan_advantages,
    BatchPlan,
    plan_worker_batch,
)
from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import resolve_group_advantage_invariant
from skyrl_train.rollouts.buffer import AdmittedRollout, RolloutGroup
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    normalize_trajectory_batches,
    token_reward_rows,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import SHAPING_METRIC_PREFIX, shape_trajectory_rewards
from skyrl_train.trajectory_runners.types import BatchFields
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics
from skyrl_train.utils.advantage_estimators import (
    apply_loop_advantages,
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
    assert batch["student_topk_indices"][batch["ftpo_chosen_mask"]].tolist() == [candidates[1]]
    assert trainer.all_metrics["ftpo/selected_boundaries"] == 1


@pytest.mark.parametrize("token_rewards", [False, True], ids=["scalar", "token"])
def test_reward_mean_uses_compensated_totals(token_rewards):
    values = [2.0**110, 2.0**54, 1.0, -(2.0**54), -(2.0**110)]
    response_lengths = [5, 1, 1, 1, 1] if token_rewards else [1] * 5
    batch = {
        "prompt_token_ids": [[0]] * 5,
        "response_ids": [[1] * length for length in response_lengths],
        "rewards": [values, [0.0], [0.0], [0.0], [0.0]] if token_rewards else values,
        "loss_masks": [[1] * length for length in response_lengths],
    }
    metrics = reward_metrics([RowFacts.from_batch(batch)], ["g0"] * 5, n_samples_per_prompt=1)
    assert metrics["reward/avg_raw_reward"] == 0.2


def assert_batch_exact(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None
        else:
            assert actual[key].dtype == value.dtype
            assert torch.equal(actual[key].view(torch.uint8), value.view(torch.uint8)), key
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


def test_row_facts_match_payload_shaping_with_repeated_marker_penalties(worker_config):
    config = {
        "enabled": True,
        "loop": {"advantage_penalty_per_token": 0.025},
        "reasoning_span": {"target": "marker", "penalty": 0.1},
    }
    batches = []
    for counts, tokens, responses, excluded, last in (
        ([1, 3, 6], [[0], [0, 1, 2], list(range(6))], [list(range(8))] * 3, [False] * 3, [True, False, True]),
        ([1, 0], [[], []], [[8] * 8, [9] * 8], [False, True], [True, True]),
        ([0], [[]], [list(range(8))], [False], [True]),
    ):
        batch = {
            "prompt_token_ids": [[10]] * len(counts),
            "response_ids": responses,
            "loss_masks": [[1] * 8 for _ in counts],
            "rewards": [1.0] * len(counts),
            "reasoning_span_violations": counts,
            "reasoning_span_violation_tokens": tokens,
            "exclude_from_baseline": excluded,
            "is_last_step": last,
        }
        shape_trajectory_rewards(batch, config)
        batches.append(batch)
    del batches[-1]["loop_advantages"]

    expected = concatenate_trajectory_batches(batches)["rollout_metrics"]
    groups = [
        AdmittedRollout(
            index,
            f"g{index}",
            0,
            len(batch["response_ids"]),
            sum(map(len, batch["response_ids"])),
            RowFacts.from_batch(batch),
        )
        for index, batch in enumerate(batches)
    ]
    _, actual = plan_worker_batch(
        worker_config, groups, global_step=0, dp_size=2, max_staleness_steps=0, num_experts=None
    )
    shaping = f"{SHAPING_METRIC_PREFIX}/"
    assert {key: value for key, value in actual.items() if key.startswith(shaping)} == {
        key: value for key, value in expected.items() if key.startswith(shaping)
    }
    # Five loop tokens plus six negative marker roundoff values.
    assert actual[shaping + "loop_charged_tokens_mean"] == 11 / 5


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
    marker_penalty = 0.25
    loop_credit = -0.1
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
            batch["loop_advantages"] = [
                [-marker_penalty] + [loop_credit] * (length - 1) if length else [] for length in lengths
            ]
            batch["reasoning_span_violations"] = [int(length > 0) for length in lengths]
            batch["reasoning_span_violation_tokens"] = [[0] if length else [] for length in lengths]
            batch["reasoning_span_token_penalties"] = [marker_penalty if length else 0.0 for length in lengths]
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
        assert normalized["reasoning_span_token_penalties"] == [marker_penalty * value for value in violations]
        lengths = [1, 2, 3, 4] * 6
        lengths[8] = lengths[12] = 0
        lengths[-1] = 9
        assert normalized["loop_advantages"] == [
            [-marker_penalty] + [loop_credit] * (length - 1) if violation else [0.0] * length
            for length, violation in zip(lengths, violations, strict=True)
        ]
    normalized["rewards"] = token_reward_rows(normalized["rewards"], normalized["response_ids"])
    whole_plan = BatchPlan(
        facts=(RowFacts.from_batch(normalized),), fields=BatchFields.from_batch(normalized), **layout
    )
    whole = assemble_slice(whole_plan, range(len(uids)), normalized, pad_token_id=0, algorithm=algorithm)
    # This oracle sums the full token-width reward tensors through the public estimator.
    whole["advantages"], _ = compute_advantages_and_returns(
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
    apply_loop_advantages(whole)
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
            reduction=SimpleNamespace(combine=lambda values, operation: values + other_scores),
        )
        assert_batch_exact(apply_loop_advantages(actual), expected)


@pytest.fixture
def worker_config(tmp_path):
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.train_batch_size = cfg.trainer.policy_mini_batch_size = 2
    cfg.trainer.micro_train_batch_size_per_gpu = cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.generator.n_samples_per_prompt = 2
    cfg.trainer.policy.model.path = str(tmp_path)
    return cfg


@pytest.mark.parametrize(
    ("overrides", "reason", "metadata"),
    [
        ({}, None, None),
        (
            {
                "trainer.policy.megatron_config.moe_router_replay": True,
                "generator.engine_init_kwargs.enable_return_routed_experts": True,
            },
            "router replay without the model expert count",
            {"model_type": "qwen2"},
        ),
        ({"trainer.algorithm.use_kl_loss": True}, "DPO and reference-model objectives", None),
        ({"trainer.algorithm.advantage_estimator": "reward"}, "the configured loss and estimator", None),
        ({"trainer.algorithm.policy_loss_type": "clip_cov"}, "the configured loss and estimator", None),
        ({"trainer.placement.colocate_all": True}, "colocated placement", None),
        ({"trainer.algorithm.distillation": {}}, "distillation", None),
        ({"trainer.trajectory_selector.type": "best_of_n"}, "trajectory selection", None),
        ({"trainer.step_wise_training": True}, "step-wise training", None),
        ({"trainer.algorithm.advantage_batch_normalize": True}, "advantage batch normalization", None),
        (
            {
                "generator.trajectory_reward_shaping.enabled": True,
                "generator.trajectory_reward_shaping.loop.advantage_penalty_per_token": 0.1,
            },
            None,
            None,
        ),
        (
            {
                "generator.trajectory_reward_shaping.enabled": True,
                "generator.trajectory_reward_shaping.reasoning_span.target": "marker",
                "generator.trajectory_reward_shaping.reasoning_span.penalty": 0.5,
            },
            None,
            None,
        ),
        (
            {"trainer.policy.megatron_config.expert_tensor_parallel_size": 2},
            "ETP greater than one",
            {"model_type": "qwen2_moe", "num_experts": 4},
        ),
        (
            {
                "trainer.policy.megatron_config.expert_tensor_parallel_size": None,
                "trainer.policy.megatron_config.tensor_model_parallel_size": 2,
            },
            "ETP greater than one",
            {"model_type": "qwen2_moe", "num_experts": 4},
        ),
        (
            {
                "trainer.policy.megatron_config.expert_tensor_parallel_size": None,
                "trainer.policy.megatron_config.tensor_model_parallel_size": 2,
            },
            "ETP greater than one",
            {"model_type": "grug_moe", "num_experts": 8},
        ),
        (
            {
                "trainer.policy.megatron_config.expert_tensor_parallel_size": None,
                "trainer.policy.megatron_config.tensor_model_parallel_size": 2,
            },
            None,
            {"model_type": "qwen2"},
        ),
        (
            {"trainer.train_batch_size": 3, "trainer.policy_mini_batch_size": 3, "generator.n_samples_per_prompt": 3},
            "rows requiring DP padding",
            None,
        ),
        (
            {
                "trainer.algorithm.policy_loss_type": "sft",
                "trainer.algorithm.advantage_estimator": "uniform",
                "trainer.placement.policy_num_gpus_per_node": 4,
                "trainer.train_batch_size": 6,
                "trainer.policy_mini_batch_size": 6,
                "generator.n_samples_per_prompt": 2,
            },
            None,
            None,
        ),
        (
            {
                "trainer.policy.megatron_config.moe_router_replay": True,
                "generator.engine_init_kwargs.enable_return_routed_experts": True,
            },
            None,
            {"model_type": "qwen2_moe", "num_experts": 4},
        ),
        ({"trainer.dump_data_batch": True}, "dense batch dumping", None),
    ],
)
def test_worker_batch_config_validation(worker_config, tmp_path, overrides, reason, metadata, monkeypatch):
    cfg = worker_config
    if metadata is not None:
        (tmp_path / "config.json").write_text(json.dumps(metadata))
    monkeypatch.delitem(CONFIG_MAPPING._extra_content, "grug_moe", raising=False)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    assert worker_batch_refusal(cfg) == reason
    if reason is None:
        validate_cfg(cfg)
        assert cfg.trainer.batch_builder == "worker"
    else:
        cfg.trainer.batch_builder = "worker"
        with pytest.raises(ValueError, match=reason):
            validate_cfg(cfg)


def test_auto_batch_builder_resolves_router_replay_after_planning(worker_config, tmp_path):
    cfg = worker_config
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    with open_dict(cfg.generator.engine_init_kwargs):
        cfg.generator.engine_init_kwargs.enable_return_routed_experts = True
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    validate_cfg(cfg, check_runtime_resources=False)
    assert cfg.trainer.batch_builder == "auto"
    validate_cfg(cfg)
    assert cfg.trainer.batch_builder == "driver"


def test_unreadable_policy_config_falls_back_only_for_auto(worker_config):
    cfg = worker_config
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = None
    validate_cfg(cfg)
    assert cfg.trainer.batch_builder == "driver"
    cfg.trainer.batch_builder = "worker"
    assert worker_batch_refusal(cfg, check_runtime_resources=False) is None
    with pytest.raises(ValueError, match="Unrecognized model"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("fields", "rows", "reason"),
    [
        ({"teacher_evidence": [None] * 4}, 4, "teacher evidence"),
        ({"distillation": [None] * 4}, 4, "teacher evidence"),
        ({"student_topk_indices": [None] * 4}, 4, "teacher evidence"),
        ({"behavior_topk_logprobs": [None] * 4}, 4, "teacher evidence"),
        ({}, 3, "rows requiring DP padding"),
        (
            {"rollout_routed_experts": [np.zeros((2, 1, 1), dtype=np.uint8)] * 4},
            4,
            "route dtype without the model expert count",
        ),
    ],
)
def test_worker_batch_rejects_incompatible_rows(worker_config, fields, rows, reason):
    cfg = worker_config
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    batch = {
        "prompt_token_ids": [[1]] * rows,
        "response_ids": [[2, 3]] * rows,
        "loss_masks": [[1, 1]] * rows,
        "rewards": [[0.0, 1.0]] * rows,
        **fields,
    }
    group = AdmittedRollout(0, "g0", 0, rows, sum(map(len, batch["response_ids"])), RowFacts.from_batch(batch))
    match = rf"^Policy-worker batch assembly: .*g0.*{reason}.*trainer\.batch_builder=driver"
    with pytest.raises(ValueError, match=match):
        plan_worker_batch(cfg, (group,), global_step=0, dp_size=2, max_staleness_steps=0, num_experts=None)


def test_mismatch_ratio_statistics_match_known_values_for_mixed_staleness():
    learner = torch.tensor([[-21.0, 0.0, 1.0, 8.125, float("nan")]], dtype=torch.float32)
    sampler = torch.tensor([[-1.0, 0.0, 0.0078125, 0.125, 0.0]], dtype=torch.float32)
    mask = torch.ones_like(learner)
    staleness = torch.tensor([[0, 0, 1, 2, 0]])
    metrics = mismatch_ratio_metrics(learner, sampler, mask, staleness)
    for key, value in (
        ("ess_fraction", 0.250620275328),
        ("kl_k3", 747.915720782),
        ("chi2", 2221528.63255),
        ("log_ratio_abs_p95", 18.2),
        ("log_ratio_abs_p99", 19.64),
        ("lower_clip_pressure", 0.25),
        ("upper_clip_pressure", 0.5),
        ("log_ratio_mean_squared", 116.246108949),
    ):
        assert metrics[f"policy/mismatch/pooled/{key}"] == pytest.approx(value)
    assert metrics["policy/mismatch/staleness0/log_ratio_abs_p95"] == pytest.approx(19.0)
