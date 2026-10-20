"""FTPO token selection and loss checked against the scalar definition."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from cloud.iris.launch_config import load_launch_config
from cloud.iris.tests.test_launch_config import _raw_config

from skyrl_train.batch_source import DriverBatchSource
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.training_batch import TrainingBatchIterator
from skyrl_train.config.ftpo import FTPOBoundary, FTPOConfig, ftpo_config, validate_ftpo
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.ftpo import (
    FTPOInputs,
    FTPOTargets,
    VerifiedFTPOBoundary,
    boundary_values,
    compact_boundary_logits,
    ftpo_counts,
    ftpo_loss,
    select_ftpo_candidates,
)
from skyrl_train.objective.losses import ftpo_policy_loss
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.trajectory_runners.trajectory_reward_shaping import LoopCreditConfig, tail_loop
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics
from skyrl_gym.verification import VerificationResult


@pytest.fixture
def config():
    config_dir = str(Path(__file__).resolve().parents[3] / "skyrl_train" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name="ppo_base_config",
            overrides=[
                "+algorithm_recipe=ftpo",
                "trainer.use_sample_packing=false",
                "generator.sampling_params.logprobs=4",
                "generator.sampling_params.temperature=0",
                "trainer.logger=console",
            ],
        )


def _surface(ids):
    return {0: "Wait", 1: " So", 2: " Therefore", 3: "Done", 4: "\n", 5: "WAIT"}[ids[0]]


def test_ftpo_mixed_batch_preserves_each_boundary_contract():
    surfaces = {0: "Wait", 1: "<|end_think|>", 2: "<tool_call>", 3: "{", 4: "}", 5: "Done"}
    responses = [[0] * 8, [1, 2, 2], [0, 0], [0] * 8]
    candidates = torch.tensor([0, 3, 4, 5]).expand(4, 8, 4)
    scores = torch.tensor([0.4, 0.3, 0.2, 0.1]).log().expand_as(candidates)
    chosen, weights = select_ftpo_candidates(
        responses,
        [[1] * len(response) for response in responses],
        candidates,
        scores,
        prompt_ids=[[]] * 4,
        decode=lambda tokens: "".join(surfaces[token] for token in tokens),
        loop=LoopCreditConfig(),
        config=replace(FTPOConfig(), chosen_balance_strength=0, rejected_balance_strength=0),
        seed=42,
        eligible=[True] * 4,
        verified_boundaries=[None, None, VerifiedFTPOBoundary(1, (4,)), None],
        boundary_modes=[
            FTPOBoundary.TAIL_LOOP,
            FTPOBoundary.REPEATED_TOOL_OPENER,
            FTPOBoundary.VERIFIED_CLOSURE_AND_CONTINUATION,
            FTPOBoundary.VERIFIED_CLOSURE,
        ],
    )
    assert chosen.nonzero().tolist() == [[0, 1, 1], [0, 1, 2], [0, 1, 3], [1, 2, 1], [2, 1, 2]]
    assert weights.nonzero().tolist() == [[0, 1], [1, 2], [2, 1]]
    # A closure row without proof must not silently fall back to its detectable loop.
    assert not chosen[3].any()


def test_ftpo_launch_selects_protocol_boundary_from_recipe(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["skyrl"]["config_groups"] = {"algorithm_recipe": "ftpo"}
    raw["skyrl"]["trainer"]["algorithm"]["ftpo"] = {"boundary": "repeated_tool_opener"}
    raw["skyrl"]["trainer"]["use_sample_packing"] = False
    raw["skyrl"]["generator"]["sampling_params"] = {"logprobs": 2}
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_launch_config(path)
    surfaces = {0: "<|end_think|>", 1: "<tool_call>", 2: " {"}
    candidates = torch.tensor([1, 2]).expand(1, 3, 2)
    scores = torch.tensor([0.8, 0.2]).log().expand_as(candidates)

    chosen, weights = select_ftpo_candidates(
        [[0, 1, 1]],
        [[1, 1, 1]],
        candidates,
        scores,
        prompt_ids=[[]],
        decode=lambda tokens: "".join(surfaces[token] for token in tokens),
        loop=LoopCreditConfig(),
        config=ftpo_config(config.skyrl.trainer.algorithm),
        seed=42,
        eligible=[True],
    )

    assert chosen.nonzero().tolist() == [[0, 2, 1]]
    assert weights.nonzero().tolist() == [[0, 2]]


def test_ftpo_reuses_tail_detection_but_selects_first_repeat_and_filters_alternatives():
    response = [[3, 1] + [0, 1] * 4, [0, 1] * 4 + [3], [0, 1] * 4]
    masks = [[0, 0] + [1] * 8, [1] * 9, [1] * 8]
    ids = torch.tensor([0, 5, 4, 2, 3]).expand(3, 10, 5).clone()
    scores = torch.tensor([0.5, 0.3, 0.1, 0.09, 0.01]).log().expand_as(ids).clone()
    chosen, weights = select_ftpo_candidates(
        response,
        masks,
        ids,
        scores,
        prompt_ids=[[]] * len(response),
        decode=_surface,
        loop=LoopCreditConfig(),
        config=replace(FTPOConfig(), min_p=0.2, chosen_balance_strength=0),
        seed=4,
        eligible=[True, True, False],
    )
    hit = tail_loop(response[0], masks[0], LoopCreditConfig())
    assert (hit.start, hit.period, hit.repeat_start) == (2, 2, 4)
    assert chosen.nonzero().tolist() == [[0, 4, 3]]
    assert weights.nonzero().tolist() == [[0, 4]]
    assert ids[chosen].tolist() == [2]


def test_ftpo_balances_common_tokens_deterministically_without_retokenizing():
    responses = [[0] * 6] * 6 + [[1] * 6] * 2
    ids = torch.tensor([2, 3]).expand(8, 6, 2).clone()
    ids[6:, :, 1] = 0
    scores = torch.tensor([0.7, 0.3]).log().expand_as(ids).clone()
    kwargs = dict(
        prompt_ids=[[]] * 8, decode=_surface, loop=LoopCreditConfig(), config=FTPOConfig(), seed=13, eligible=[True] * 8
    )
    chosen, weights = select_ftpo_candidates(responses, [[1] * 6] * 8, ids, scores, **kwargs)
    repeated, repeated_weights = select_ftpo_candidates(responses, [[1] * 6] * 8, ids, scores, **kwargs)
    assert torch.equal(chosen, repeated) and torch.equal(weights, repeated_weights)
    assert chosen.sum() < 16
    common = weights[:6].sum(-1)
    rare = weights[6:].sum(-1)
    assert common[common > 0].mean() < rare[rare > 0].mean()
    assert torch.all(weights[~chosen.any(-1)] == 0)


def test_ftpo_protocol_selection_excludes_thinking_observations_and_quoted_markers(config):
    surfaces = {
        0: "<tool_call>",
        1: "<|end_think|>",
        2: "<|start_think|>",
        3: " {\n",
        4: "</tool_call>",
        5: " prose",
        6: "{bad",
        7: '{"name":"read","arguments":{"text":"',
        8: '"}}',
    }
    responses = [
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 7, 0, 0, 8, 4, 0, 0],
        [0, 0, 0],
        [1, 0, 6, 0, 0],
        [1, 0, 0, 0],
    ]
    masks = [[1] * len(row) for row in responses]
    masks[2] = [0, 0, 1, 1]
    ids = torch.tensor([0, 3, 4, 5]).expand(len(responses), 9, 4).clone()
    scores = torch.tensor([0.6, 0.2, 0.1, 0.1]).log().expand_as(ids).clone()
    config.trainer.algorithm.ftpo = {"boundary": "repeated_tool_opener", "chosen_balance_strength": 0}
    selected = ftpo_config(config.trainer.algorithm)
    chosen, weights = select_ftpo_candidates(
        responses,
        masks,
        ids,
        scores,
        prompt_ids=[[]] * len(responses),
        decode=lambda tokens: "".join(surfaces[token] for token in tokens),
        loop=LoopCreditConfig(),
        config=selected,
        seed=42,
        eligible=[True, True, True, True, True, True, False],
    )
    assert chosen.nonzero().tolist() == [[0, 2, 1], [3, 8, 1]]
    assert weights.nonzero().tolist() == [[0, 2], [3, 8]]
    assert ids[chosen].tolist() == [3, 3]


@pytest.mark.parametrize("whitespace", ["", "\n", " \t\r\n"])
def test_ftpo_protocol_targets_second_opener_after_recorded_assistant_header(config, whitespace):
    surfaces = {
        0: "<tool_call>",
        1: "{",
        2: "<|start_header_id|>",
        3: "assistant",
        4: "<|end_header_id|>",
        5: "\n\n",
        6: whitespace,
        7: "<|start_think|>",
        8: "<|end_think|>",
        9: "tool",
    }
    header = [2, 3, 4, 5]
    responses = [
        [0, 6, 0],
        [0, 6, 0] + header + [0, 6, 0],
        [0, 6, 0],
        [7, 0, 6, 0, 8, 0, 6, 0],
        [0, 6, 0],
        [6, 0],
        [6, 0],
        [6, 0],
        [6, 0],
    ]
    masks = [[1] * len(row) for row in responses]
    masks[1] = [0] * 7 + [1] * 3
    prompts = [
        header,
        [],
        header + [7],
        header,
        [2, 9, 4, 5],
        header + [8, 0],
        header + [7, 0],
        [2, 9, 4, 5, 8, 0],
        header + [0],
    ]
    candidates = torch.tensor([0, 1]).expand(len(responses), 10, 2).clone()
    scores = torch.tensor([0.8, 0.2]).log().expand_as(candidates)
    config.trainer.algorithm.ftpo = {"boundary": "repeated_tool_opener", "chosen_balance_strength": 0}
    chosen, weights = select_ftpo_candidates(
        responses,
        masks,
        candidates,
        scores,
        prompt_ids=prompts,
        decode=lambda tokens: "".join(surfaces[token] for token in tokens),
        loop=LoopCreditConfig(),
        config=ftpo_config(config.trainer.algorithm),
        seed=42,
        eligible=[True] * len(responses),
    )
    assert chosen.nonzero().tolist() == [[0, 2, 1], [1, 9, 1], [3, 7, 1], [5, 1, 1], [8, 1, 1]]
    assert weights.nonzero().tolist() == [[0, 2], [1, 9], [3, 7], [5, 1], [8, 1]]


def _scalar_reference(logits, reference, ids, masks, rejected, weights, params):
    pref = logits.sum() * 0
    other = logits.sum() * 0
    target = logits.sum() * 0
    example_count = weights.sum()
    other_count = target_count = 0.0
    for row in range(len(logits)):
        if weights[row] == 0:
            continue
        chosen = ids[row][masks[row]].tolist()
        terms = []
        for token in chosen:
            gap = params.margin - (logits[row, token] - logits[row, rejected[row]])
            terms.append(torch.logaddexp(gap, gap.new_zeros(())) * (gap / params.margin).clamp(0, 1))
        pref = pref + weights[row] * sum(terms) / len(terms)
        for token in range(logits.shape[-1]):
            delta = logits[row, token] - reference[row, token]
            if token in chosen or token == rejected[row]:
                target = target + weights[row] * (delta.abs() - params.tau_mse_target).clamp(min=0).square()
                target_count += float(weights[row])
            else:
                other = other + weights[row] * delta.square()
                other_count += float(weights[row])
    return (
        pref / example_count
        + params.lambda_mse * other / other_count
        + params.lambda_mse_target * target / target_count
    )


@pytest.mark.parametrize("weight_scale", [0.1, 1.0])
@pytest.mark.parametrize("micro_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ftpo_value_and_gradient_match_scalar_reference_across_microbatches(config, micro_size, dtype, weight_scale):
    torch.manual_seed(9)
    logits = torch.randn(4, 3, 7, dtype=dtype).requires_grad_()
    ref = torch.randn(4, 7, dtype=dtype)
    ids = torch.tensor([1, 2, 3]).expand(4, 3, 3).clone()
    chosen = torch.zeros_like(ids, dtype=torch.bool)
    chosen[0, 1, :2] = True
    chosen[1, 0, :1] = True
    chosen[3, 2, :] = True
    weights = chosen.any(-1).float() * torch.tensor([0.3, 0.7, 0, 0.8])[:, None] * weight_scale
    rejected = torch.zeros(4, dtype=torch.long)
    targets = FTPOTargets(ids, chosen, ref)
    normalization = ftpo_counts([targets], [weights], lambda x: x)
    counts = step_counts([weights], [weights], [], [torch.zeros_like(weights)], 3, lambda x: x)
    boundary = boundary_values(logits, chosen)
    expected = _scalar_reference(
        boundary.float(),
        ref.float(),
        boundary_values(ids, chosen),
        boundary_values(chosen, chosen),
        rejected,
        weights.sum(-1),
        FTPOConfig(),
    )
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, 4, micro_size):
        chunk = slice(start, start + micro_size)
        target = FTPOTargets(ids[chunk], chosen[chunk], ref[chunk])
        inputs = FTPOInputs(boundary[chunk], target, rejected[chunk], normalization)
        batch = build_objective_micro_batch(
            action_log_probs=torch.zeros_like(weights[chunk]),
            old_action_log_probs=torch.zeros_like(weights[chunk]),
            base_action_log_probs=None,
            advantages=torch.zeros_like(weights[chunk]),
            loss_mask=weights[chunk],
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(weights[chunk]),
            think_token_weight=1,
            teacher=None,
            ftpo=inputs,
        )
        actual = (
            actual
            + compute_policy_objective(
                batch,
                loss=ftpo_policy_loss,
                counts=counts,
                config=config.trainer.algorithm,
                loss_scale=1,
                report_scale=1,
            ).optimization_loss
        )
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(actual_gradient[~chosen.any(-1)]) == 0


def test_ftpo_saturated_preferences_stop_and_reference_tether_penalizes_other_tokens():
    chosen = torch.tensor([[[False, True]]])
    ids = torch.tensor([[[0, 1]]])
    logits = torch.tensor([[0.0, 3.0, 2.0]], requires_grad=True)
    target = FTPOTargets(ids, chosen, torch.tensor([[0.0, 3.0, 0.0]]))
    counts = ftpo_counts([target], [torch.ones(1, 1)], lambda x: x)
    values, metrics = ftpo_loss(FTPOInputs(logits, target, torch.tensor([0]), counts), FTPOConfig())
    values.sum().backward()
    assert values.item() == pytest.approx(1.6)
    torch.testing.assert_close(logits.grad, torch.tensor([[0.0, 0.0, 1.6]]))
    assert metrics["ftpo/chosen_win_sum"] == 1


@pytest.mark.parametrize("micro_size", [1, 2, 4])
@pytest.mark.parametrize("mode", ["mixed", "repair_only", "ftpo_only"])
def test_joint_closure_loss_and_gradients_match_separate_means(config, micro_size, mode):
    torch.manual_seed(19)
    config.trainer.algorithm.ftpo = {"repair_sft_weight": 0.7}
    params = ftpo_config(config.trainer.algorithm)
    logits = torch.randn(4, 3, 7, requires_grad=True)
    reference = torch.randn(4, 7)
    ids = torch.tensor([1, 2]).expand(4, 3, 2).clone()
    chosen = torch.zeros_like(ids, dtype=torch.bool)
    if mode != "repair_only":
        chosen[0, 1, :] = True
        chosen[1, 0, 0] = True
    repair = torch.zeros(4, 3, dtype=torch.bool)
    if mode != "ftpo_only":
        repair[2, 1:] = True
        repair[3, 0] = True
    preference_weights = chosen.any(-1).float() * torch.tensor([0.3, 0.7, 0, 0])[:, None]
    weights = preference_weights + repair.float()
    rejected = torch.zeros(4, dtype=torch.long)
    boundary = boundary_values(logits, chosen)
    log_probs = logits.log_softmax(-1)[..., 0]
    targets = FTPOTargets(ids, chosen, reference, repair)
    normalization = ftpo_counts([targets], [weights], lambda value: value)
    counts = step_counts([weights], [weights], [], [torch.zeros_like(weights)], 3, lambda value: value)
    expected = logits.sum() * 0
    if mode != "repair_only":
        expected = expected + _scalar_reference(
            boundary,
            reference,
            boundary_values(ids, chosen),
            boundary_values(chosen, chosen),
            rejected,
            preference_weights.sum(-1),
            params,
        )
    if mode != "ftpo_only":
        expected = expected - 0.7 * log_probs[repair].mean()
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual = logits.new_zeros(())
    for start in range(0, 4, micro_size):
        chunk = slice(start, start + micro_size)
        ftpo = FTPOInputs(
            boundary[chunk],
            FTPOTargets(ids[chunk], chosen[chunk], reference[chunk], repair[chunk]),
            rejected[chunk],
            normalization,
        )
        batch = build_objective_micro_batch(
            action_log_probs=log_probs[chunk],
            old_action_log_probs=torch.zeros_like(weights[chunk]),
            base_action_log_probs=None,
            advantages=torch.zeros_like(weights[chunk]),
            loss_mask=weights[chunk],
            rollout_logprobs=None,
            response_span_tags=None,
            token_entropy=torch.zeros_like(weights[chunk]),
            think_token_weight=1,
            teacher=None,
            ftpo=ftpo,
        )
        actual = (
            actual
            + compute_policy_objective(
                batch,
                loss=ftpo_policy_loss,
                counts=counts,
                config=config.trainer.algorithm,
                loss_scale=1,
                report_scale=1,
            ).optimization_loss
        )
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(actual_gradient[~(chosen.any(-1) | repair)]) == 0


def test_full_weight_ftpo_update_and_resume_match_uninterrupted_training(tmp_path):
    torch.manual_seed(7)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_dropout=0,
        )
    )
    reference = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    sequences = torch.tensor([[4, 5, 0, 0, 0, 0]])
    chosen = torch.zeros(1, 4, 2, dtype=torch.bool)
    chosen[0, 1, :] = True
    ids = torch.tensor([1, 2]).expand(1, 4, 2)
    with torch.no_grad():
        ref_logits = boundary_values(reference(sequences).logits[:, -5:-1], chosen)
    targets = FTPOTargets(ids, chosen, ref_logits)
    counts = ftpo_counts([targets], [chosen.any(-1).float()], lambda x: x)

    def update(policy, optim):
        optim.zero_grad()
        logits = boundary_values(policy(sequences).logits[:, -5:-1], chosen)
        values, _ = ftpo_loss(FTPOInputs(logits, targets, torch.tensor([0]), counts), FTPOConfig())
        values.sum().backward()
        optim.step()
        return logits.detach()

    before = update(model, optimizer)
    checkpoint = tmp_path / "state.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    update(model, optimizer)
    resumed = deepcopy(reference)
    resumed.requires_grad_(True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    state = torch.load(checkpoint, weights_only=True)
    resumed.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    update(resumed, resumed_optimizer)
    for actual, expected in zip(resumed.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with torch.no_grad():
        after = boundary_values(model(sequences).logits[:, -5:-1], chosen)
    assert (after[0, 1:3] - after[0, 0]).mean() > (before[0, 1:3] - before[0, 0]).mean()
    assert not torch.equal(model.model.embed_tokens.weight, reference.model.embed_tokens.weight)


def test_ftpo_recipe_accepts_greedy_capture_and_rejects_unsupported_geometry(config):
    validate_cfg(config)
    assert config.generator.engine_init_kwargs.logprobs_mode == "raw_logprobs"
    for path in (
        "trainer.policy.megatron_config.tensor_model_parallel_size",
        "trainer.ref.megatron_config.context_parallel_size",
        "trainer.use_sample_packing",
    ):
        changed = deepcopy(config)
        OmegaConf.update(changed, path, True if path.endswith("packing") else 2)
        with pytest.raises(ValueError, match="FTPO"):
            validate_ftpo(changed)


@pytest.mark.parametrize("boundary", ["tail_loop", "repeated_tool_opener", "mixed"])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_ftpo_rollout_payload_survives_padding_serialization_and_microbatching(config, tmp_path, boundary, auxiliary):
    config.trainer.algorithm.ftpo = {"boundary": boundary, "repair_sft_weight": 0.7}
    if auxiliary:
        config.trainer.algorithm.policy_loss_type = "regular"
        config.trainer.algorithm.advantage_estimator = "rloo_n"
        config.trainer.algorithm.ftpo = {"boundary": boundary, "auxiliary_weight": 0.25, "repair_sft_weight": 0.7}
        config.generator.sampling_params.temperature = 1
        validate_ftpo(config)
    words = (
        ["Wait", "So", "Therefore", "Done", "[PAD]"]
        if boundary != "repeated_tool_opener"
        else ["<tool_call>", "{", "<|end_think|>", "Done", "[PAD]"]
    )
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = config
    trainer.global_step = 0
    trainer.all_metrics = {}
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({word: index for index, word in enumerate(words)})),
        pad_token="[PAD]",
    )
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=2))])
    trainer.ref_model = trainer.critic_model = None
    responses = [[0, 1] * 4, [2, 3], [0] * 5]
    if boundary == "repeated_tool_opener":
        responses[0] = [2] + [0] * 7
    trajectory = {
        "prompt_token_ids": [[2, 3], [3], [1, 2, 3]],
        "response_ids": responses,
        "loss_masks": [[1] * len(row) for row in responses],
        "rewards": [[0.0] * len(row) for row in responses],
        "student_topk_indices": [np.tile([0, 1, 2, 3], (len(row), 1)) for row in responses],
        "behavior_topk_logprobs": [
            np.tile(np.log([0.5, 0.3, 0.15, 0.05]), (len(row), 1)).astype(np.float32) for row in responses
        ],
        "is_last_step": [True, True, True],
        "exclude_from_baseline": [False, False, True],
        "verification_results": [VerificationResult.verified(0)] * 3,
    }
    batch = trainer.convert_to_training_input(trajectory, ["a", "b", "c"])
    assert batch.batch_size == 4
    structural_weights = batch["ftpo_weights"] if auxiliary else batch["loss_mask"]
    assert structural_weights.nonzero().tolist() == [[0, 2]]
    if boundary == "mixed":
        assert trainer.all_metrics["ftpo/routes/tail_loop/rows"] == 3
        assert trainer.all_metrics["ftpo/routes/tail_loop/eligible_rows"] == 2
        assert trainer.all_metrics["ftpo/routes/tail_loop/selected_boundaries"] == 1
        assert trainer.all_metrics["ftpo/routes/tail_loop/target_weight"] == float(structural_weights.sum())
        assert trainer.all_metrics["ftpo/routes/repeated_tool_opener/selected_boundaries"] == 0
        assert trainer.all_metrics["ftpo/routes/verified_closure_and_continuation/repair_tokens"] == 0
    if auxiliary:
        assert batch["loss_mask"][0].sum() == 8
        assert batch["loss_mask"][1].sum() == 2
    # The model sees compacted tokens; boundary logits must still refer to the
    # prefix just before the second copy of the loop, despite unequal prompts.
    compact_logits = torch.arange(4 * 10 * 5).reshape(4, 10, 5).float()
    boundary_logits = compact_boundary_logits(compact_logits, batch["attention_mask"], batch["ftpo_chosen_mask"])
    torch.testing.assert_close(boundary_logits[0], compact_logits[0, 3])
    batch["ftpo_reference_logits"] = boundary_logits
    for key in ("action_log_probs", "base_action_log_probs", "values", "returns", "advantages"):
        batch[key] = torch.zeros_like(batch["loss_mask"])
    path = tmp_path / "batch.pt"
    torch.save(batch, path)
    restored = torch.load(path, weights_only=False)
    experiences = list(TrainingBatchIterator(restored, sample_batch_size=1))
    assert [bool(experience.ftpo.chosen_mask.any()) for experience in experiences] == [True, False, False, False]
    torch.testing.assert_close(experiences[0].ftpo.reference_logits, boundary_logits[:1])
    expected_candidates = [1] if boundary == "repeated_tool_opener" else [1, 2, 3]
    assert experiences[0].ftpo.candidate_ids[experiences[0].ftpo.chosen_mask].tolist() == expected_candidates
    if auxiliary:
        torch.testing.assert_close(experiences[0].ftpo.to("cpu").weights, structural_weights[:1])
        counts = ftpo_counts(
            [experience.ftpo for experience in experiences],
            [experience.loss_mask for experience in experiences],
            lambda value: value,
        )
        assert counts.examples == float(structural_weights.sum())


@pytest.mark.asyncio
async def test_ftpo_empty_rollout_batch_skips_training(config):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = config
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer.context = None
    # No actors exist: an empty batch must not dispatch a forward or optimizer step.
    source = DriverBatchSource(trainer, trainer.context)
    source.prepare_from_batch(
        {"ftpo_chosen_mask": torch.zeros((2, 3, 2), dtype=torch.bool), "loss_mask": torch.zeros((2, 3))}
    )
    status = await trainer._run_training(source)
    assert status["policy_update_steps"] == 0
    assert trainer.all_metrics["ftpo/empty_batch"] == 1


@pytest.mark.parametrize("terminal_eos", [False, True])
@pytest.mark.parametrize("repair_response", [" ", ' ,"other":2'])
@pytest.mark.parametrize(
    ("boundary", "auxiliary", "task_available"),
    [
        ("verified_closure", False, False),
        ("verified_closure_and_continuation", False, False),
        ("mixed", False, False),
        ("mixed", True, True),
        ("mixed", True, False),
    ],
)
def test_verified_closure_trainer_preserves_rollouts_and_trains_verified_repairs(
    config, tmp_path, terminal_eos, repair_response, boundary, auxiliary, task_available
):
    config.trainer.algorithm.ftpo = {
        "boundary": boundary,
        "chosen_balance_strength": 0,
        "repair_sft_weight": 0.7,
        "auxiliary_weight": 0.25 if auxiliary else None,
    }
    if auxiliary:
        config.trainer.algorithm.policy_loss_type = "regular"
        config.trainer.algorithm.advantage_estimator = "rloo_n"
        config.generator.sampling_params.temperature = 1
        validate_ftpo(config)
    else:
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
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = config
    trainer.global_step = 0
    trainer.all_metrics = {}
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer.tokenizer = tokenizer
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=2))])
    trainer.ref_model = trainer.critic_model = None
    prefix = '[{"a":1'
    prompt = tokenizer.encode("assistant " + prefix, add_special_tokens=False)
    responses = [tokenizer.encode(text, add_special_tokens=False) for text in (" ]x", repair_response, " ]x", " ]x")]
    if terminal_eos:
        responses[1].append(tokenizer.eos_token_id)
    original_responses = deepcopy(responses)
    candidates = [tokenizer.encode(text, add_special_tokens=False)[0] for text in ("]", "}", "x", "\n")]
    verification = VerificationResult.verified(
        0,
        diagnostics={
            "stop_reason": "stop" if terminal_eos else "length",
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
    if auxiliary:
        if task_available:
            trajectory["verification_results"][2] = VerificationResult.verified(1)
            trajectory["rewards"][2][-1] = 1.0
        else:
            # The rollout projection excludes unavailable grading before conversion.
            trajectory["loss_masks"][2] = [0] * len(responses[2])
            trajectory["exclude_from_baseline"][2] = True
        trajectory["rollout_logprobs"] = [np.full(len(row), -0.8, dtype=np.float32) for row in responses]
    # A wholly unavailable group is rejected before conversion. Exercise an
    # unavailable repetition within an otherwise admitted prompt group.
    uids = ["a", "b", "b" if auxiliary and not task_available else "c", "d"]
    batch = trainer.convert_to_training_input(trajectory, uids, rollout_staleness=[2, 2, 2, 2])
    assert batch.metadata["response_length"] == batch["response_mask"].shape[1]
    target_weights = batch["ftpo_weights"] if auxiliary else batch["loss_mask"]
    assert target_weights.nonzero().tolist() == [[0, 1], [1, 1], [1, 2]]
    assert batch["ftpo_repair_mask"].nonzero().tolist() == [[1, 1], [1, 2]]
    assert trajectory["response_ids"] == original_responses
    repaired = batch["sequences"][1][batch["attention_mask"][1].bool()].tolist()
    assert tokenizer.decode(repaired, clean_up_tokenization_spaces=False) == 'assistant [{"a":1 }]'
    if auxiliary:
        assert batch["loss_mask"].sum() == (len(responses[2]) if task_available else 0)
        assert batch["loss_mask"][2].sum() == (len(responses[2]) if task_available else 0)
        assert batch["rewards"].sum() == int(task_available)
        assert batch["rewards"][2, len(responses[2]) - 1] == int(task_available)
        assert batch.metadata["exclude_from_baseline"][2] == (not task_available)
        torch.testing.assert_close(
            batch["rollout_logprobs"][2, : len(responses[2])], torch.full((len(responses[2]),), -0.8)
        )
        unchanged = batch["sequences"][2][batch["attention_mask"][2].bool()].tolist()
        assert unchanged == prompt + original_responses[2]
        # The trained row keeps its lease staleness through the closure repair.
        mismatch = mismatch_ratio_metrics(
            batch["rollout_logprobs"], batch["rollout_logprobs"], batch["loss_mask"], batch["rollout_staleness"]
        )
        assert mismatch["policy/mismatch/staleness2/selected_tokens"] == batch["loss_mask"].sum()
    else:
        assert batch["rollout_logprobs"] is None
    assert batch["student_topk_indices"][batch["ftpo_chosen_mask"]].tolist() == [candidates[1]]
    assert trainer.all_metrics["ftpo/selected_boundaries"] == 1
    if boundary == "mixed":
        prefix = "ftpo/routes/verified_closure_and_continuation"
        assert trainer.all_metrics[f"{prefix}/rows"] == 3
        assert trainer.all_metrics[f"{prefix}/eligible_rows"] == 2
        assert trainer.all_metrics[f"{prefix}/selected_boundaries"] == 1
        assert trainer.all_metrics[f"{prefix}/repair_tokens"] == 2
        assert trainer.all_metrics[f"{prefix}/target_weight"] == float(target_weights.sum())
        assert trainer.all_metrics["ftpo/routes/tail_loop/selected_boundaries"] == 0
    batch["ftpo_reference_logits"] = torch.zeros(4, len(tokenizer))
    for key in ("action_log_probs", "base_action_log_probs", "values", "returns", "advantages"):
        batch[key] = torch.zeros_like(batch["loss_mask"])
    path = tmp_path / "closure-batch.pt"
    torch.save(batch, path)
    restored = torch.load(path, weights_only=False)
    experiences = list(TrainingBatchIterator(restored, sample_batch_size=1))
    assert [int(experience.ftpo.repair_mask.sum()) for experience in experiences] == [0, 2, 0, 0]
    torch.testing.assert_close(experiences[1].ftpo.to("cpu").repair_mask, batch["ftpo_repair_mask"][1:2])
