"""Behavior evidence must survive collation, serialization and microbatching."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.objective.score_centering import (
    collate_score_centering,
    collate_score_probe,
    hypothetical_score_probe,
    score_centering_diagnostics,
)
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.objective.losses import PolicyLossInputs, ppo_policy_loss
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics


def test_score_evidence_survives_collation_serialization_and_microbatching(tmp_path):
    trajectory = {
        "loss_masks": [[1, 1], [1]],
        "student_topk_indices": [np.array([[3, 4], [5, 6]]), np.array([[7, 8]])],
        "behavior_topk_logprobs": [np.array([[-0.8, -1.8], [-0.9, -1.9]]), np.array([[-1.0, -2.0]])],
    }
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    ids, behavior = collate_score_centering(
        trajectory, [[3, 5], [7]], mask, 2, sampled_logprobs=torch.tensor([[-0.8, -0.9], [-1.0, 0]])
    )
    torch.testing.assert_close(ids[mask.bool()], torch.from_numpy(np.vstack(trajectory["student_topk_indices"])))
    torch.testing.assert_close(
        behavior[mask.bool()],
        torch.from_numpy(np.vstack(trajectory["behavior_topk_logprobs"])).float(),
        atol=0,
        rtol=0,
    )
    assert (ids[~mask.bool()] == -1).all()
    assert torch.isnan(behavior[~mask.bool()]).all()
    old = behavior + 0.3
    payload = {
        "sequences": torch.tensor([[1, 3, 5], [2, 7, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "response_mask": mask.bool(),
        "loss_mask": mask,
        "score_topk_indices": ids,
        "score_old_logprobs": old,
        "score_behavior_logprobs": behavior,
    }
    payload.update(
        {
            key: torch.zeros_like(mask)
            for key in ("action_log_probs", "base_action_log_probs", "values", "returns", "advantages")
        }
    )
    batch = TrainingInputBatch(payload)
    batch.metadata = {"response_length": 2}
    path = tmp_path / "batch.pt"
    torch.save(batch, path)
    restored = torch.load(path, weights_only=False)
    experiences = list(TrainingBatchIterator(restored, sample_batch_size=1))
    assert len(experiences) == 2
    for row, experience in enumerate(experiences):
        for name, expected in (
            ("score_topk_indices", ids),
            ("score_old_logprobs", old),
            ("score_behavior_logprobs", behavior),
        ):
            torch.testing.assert_close(
                getattr(experience, name), expected[row : row + 1], atol=0, rtol=0, equal_nan=True
            )


def test_sampled_loss_diagnostics_exclude_masked_tokens_and_bound_the_sample():
    head = torch.tensor([[[0.3, 0.5], [torch.nan, torch.nan], [0.1, 0.4], [0.4, 0.5]]]).log()
    current = torch.tensor([[-0.5, torch.nan, -1.0, -9.0]], requires_grad=True)
    behavior = torch.tensor([[-0.7, torch.nan, -0.6, -3.0]])
    correction = torch.tensor([[-2.0, torch.nan, 1.0, 100.0]], requires_grad=True)
    metrics = score_centering_diagnostics(
        head, head, head, current, behavior, correction, torch.tensor([[1, 0, 1, 1]]), 2
    )
    assert metrics["score_centering/sampled_tokens"] == 2
    assert metrics["score_centering/sampled_loss_mean"] == pytest.approx(-0.5)
    assert metrics["score_centering/sampled_loss_abs_mean"] == pytest.approx(1.5)
    assert metrics["score_centering/sampled_current_retained_mass"] == pytest.approx(0.65)
    assert metrics["score_centering/sampled_behavior_omitted_mass"] == pytest.approx(0.35)
    assert metrics["score_centering/sampled_current_behavior_log_ratio_mean"] == pytest.approx(-0.1)
    assert metrics["score_centering/sampled_current_behavior_log_ratio_abs_mean"] == pytest.approx(0.3)


def test_hypothetical_probe_reports_nonzero_tis_correction_and_source_strata():
    head = torch.tensor([[[0.6, 0.2]] * 4]).log()
    behavior_head = torch.tensor([[[0.1, 0.1]] * 4]).log()
    advantages = torch.tensor([[2.0, -1.0, 1.0, 1000.0]])
    valid = torch.tensor([[True, True, True, False]])
    sampled = torch.full((1, 4), 0.6).log()
    behavior_sampled = torch.full((1, 4), 0.1).log()
    metrics = hypothetical_score_probe(
        head,
        behavior_head,
        sampled,
        behavior_sampled,
        advantages,
        valid,
        torch.tensor([[7, 4, -1, 0]]),
        response_positions=torch.tensor([[0, 1024, 4095, 0]]),
        response_lengths=torch.tensor([4096]),
        training_step=7,
        loss_tokens=1000,
        tis_cap=2.0,
        eps_clip_low=0.2,
        eps_clip_high=0.2,
    )
    # At p=o, the head's effective mass is min(p, cap*q). The modeled tail
    # contributes alpha*p with alpha=min(1, cap*q_tail/p_tail)=1 here.
    reference = (
        advantages[0, :3]
        * (
            (torch.minimum(torch.tensor([0.6, 0.2]), 2 * torch.tensor([0.1, 0.1])) - torch.tensor([0.6, 0.2]))
            * torch.tensor([0.6, 0.2]).log()
        ).sum()
    )
    assert metrics["score_centering_probe/hypothetical_loss_mean"] == pytest.approx(float(reference.mean()), abs=1e-7)
    assert metrics["score_centering_probe/hypothetical_loss_abs_mean"] == pytest.approx(
        float(reference.abs().mean()), abs=1e-7
    )
    assert metrics["score_centering_probe/current_retained_mass_mean"] == pytest.approx(0.8)
    assert metrics["score_centering_probe/behavior_omitted_mass_mean"] == pytest.approx(0.8)
    assert metrics["score_centering_probe/fresh_source_tokens"] == 1
    assert metrics["score_centering_probe/stale_source_tokens"] == 1
    assert metrics["score_centering_probe/unknown_source_tokens"] == 1
    assert metrics["score_centering_probe/source_age_max"] == 3
    assert metrics["score_centering_probe/coverage"] == 0.003
    assert metrics["score_centering_probe/sampled_tis_clipped_fraction"] == 1.0
    assert metrics["score_centering_probe/current_equals_old"] == 1.0
    assert metrics["score_centering_probe/long_responses"] == 1
    assert metrics["score_centering_probe/long_response_tokens"] == 3
    assert metrics["score_centering_probe/first256_tokens"] == 1
    assert metrics["score_centering_probe/last256_tokens"] == 1
    for source in ("fresh", "stale", "unknown"):
        assert metrics[f"score_centering_probe/long_response/{source}_tokens"] == 1
    flat_rewards = hypothetical_score_probe(
        head,
        behavior_head,
        sampled,
        behavior_sampled,
        torch.zeros_like(advantages),
        valid,
        torch.tensor([[7, 4, -1, 0]]),
        response_positions=torch.tensor([[0, 1024, 4095, 0]]),
        response_lengths=torch.tensor([4096]),
        training_step=7,
        loss_tokens=1000,
        tis_cap=2.0,
        eps_clip_low=0.2,
        eps_clip_high=0.2,
    )
    assert flat_rewards["score_centering_probe/hypothetical_loss_abs_mean"] == 0
    assert flat_rewards["score_centering_probe/zero_advantage_fraction"] == 1
    assert (
        flat_rewards["score_centering_probe/unit_advantage_loss_abs_mean"]
        == metrics["score_centering_probe/unit_advantage_loss_abs_mean"]
    )
    assert flat_rewards["score_centering_probe/unit_advantage_loss_abs_mean"] > 0


def test_mismatch_keeps_unknown_ages_and_long_response_tails_separate():
    response = torch.ones((3, 4096), dtype=torch.bool)
    response[0, 2:] = False
    mask = torch.zeros_like(response)
    mask[0, :2] = True
    mask[1, [0, 1024, 4095]] = True
    mask[2, [0, 4095]] = True
    delta = torch.full(response.shape, torch.nan, dtype=torch.float64)
    delta[0, :2] = 0.1
    delta[1, [0, 1024, 4095]] = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64)
    delta[2, 4095] = -2.0
    ages = torch.full(response.shape, -1)
    ages[0] = 0
    ages[1] = 3
    metrics = mismatch_ratio_metrics(delta, torch.zeros_like(delta), mask, ages, response)
    prefix = "policy/mismatch/"
    assert metrics[prefix + "pooled/selected_tokens"] == 7
    assert metrics[prefix + "pooled/finite_tokens"] == 6
    assert metrics[prefix + "staleness0/selected_tokens"] == 2
    assert metrics[prefix + "staleness0/log_ratio_abs_mean"] == pytest.approx(0.1)
    assert metrics[prefix + "unknown/selected_tokens"] == 2
    assert metrics[prefix + "unknown/finite_tokens"] == 1
    assert metrics[prefix + "unknown/response_ge2048/log_ratio_abs_mean"] == 2.0
    assert metrics[prefix + "staleness3/response_ge2048/selected_tokens"] == 3
    assert metrics[prefix + "staleness3/response_ge2048/log_ratio_abs_mean"] == pytest.approx(0.4)
    assert metrics[prefix + "staleness3/pos_last256/selected_tokens"] == 1
    assert metrics[prefix + "staleness0/response_ge2048/selected_tokens"] == 0


def test_compact_probe_collation_keeps_only_requested_valid_positions_and_missing_rows():
    trajectory = {
        "loss_masks": [[1, 0, 1, 1], [1, 1], [0]],
        "score_probe_positions": [np.array([0, 3]), None, np.empty(0, dtype=np.int32)],
        "score_probe_indices": [np.array([[3, 7], [6, 8]], dtype=np.int32), None, None],
        "score_probe_logprobs": [np.array([[-0.8, -1.8], [-0.9, -1.9]], dtype=np.float32), None, None],
    }
    output = collate_score_probe(
        trajectory,
        [[3, 4, 5, 6], [9, 10], [11]],
        2,
        sampled_logprobs=torch.tensor([[-0.8, 0, -1.0, -0.9], [-1.0, -2.0, 0, 0], [0, 0, 0, 0]]),
    )
    assert output["score_topk_indices"].shape == (3, 2, 2)
    torch.testing.assert_close(output["score_probe_positions"], torch.tensor([[0, 3], [0, 0], [0, 0]]))
    torch.testing.assert_close(output["score_topk_indices"][0], torch.tensor([[3, 7], [6, 8]], dtype=torch.int32))
    assert (output["score_topk_indices"][1:] == -1).all()
    assert torch.isnan(output["score_behavior_logprobs"][1:]).all()
    trajectory["loss_masks"][0][0] = 0
    filtered = collate_score_probe(
        trajectory,
        [[3, 4, 5, 6], [9, 10], [11]],
        2,
        sampled_logprobs=torch.tensor([[-0.8, 0, -1.0, -0.9], [-1.0, -2.0, 0, 0], [0, 0, 0, 0]]),
    )
    torch.testing.assert_close(filtered["score_probe_positions"], torch.tensor([[3], [0], [0]]))
    torch.testing.assert_close(filtered["score_topk_indices"][0], torch.tensor([[6, 8]], dtype=torch.int32))


def test_sc_off_optimizer_is_identical_after_hypothetical_probe_consumes_its_evidence():
    config = OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "off_policy_correction": "custom",
                    "off_policy_correction_rules": [{"kind": "token", "action": "truncate", "high": 2.0}],
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                }
            }
        }
    )
    results = []
    for enabled in (False, True):
        parameters = torch.nn.Parameter(torch.tensor([0.6, 0.2, 0.2]).log())
        optimizer = torch.optim.SGD([parameters], lr=0.1, momentum=0.9)
        logprobs = parameters.log_softmax(-1)
        response = torch.tensor([[0, 1, 2, 0]])
        old = logprobs.detach()[response]
        rollout = torch.tensor([0.1, 0.1, 0.8]).log()[response]
        batch = TrainingInputBatch(
            {
                "action_log_probs": old.clone(),
                "rollout_logprobs": rollout,
                "advantages": torch.tensor([[2.0, -1.0, 0.0, 1.0]]),
                "loss_mask": torch.tensor([[1, 1, 0, 1]]),
                "response_mask": torch.ones((1, 4), dtype=torch.bool),
                "correction_weights": (old - rollout).exp().clamp(max=2.0),
            }
        )
        trainer = object.__new__(RayPPOTrainer)
        trainer.cfg, trainer.all_metrics, trainer._last_score_probe_time, trainer.global_step = config, {}, None, 4
        trainer._last_score_probe_step = None
        if enabled:
            batch.update(
                {
                    "score_probe_positions": torch.tensor([[0, 3]]),
                    "score_probe_source_steps": torch.tensor([[4, 1]]),
                    "score_topk_indices": torch.tensor([[[0, 1], [0, 1]]]),
                    "score_old_logprobs": logprobs.detach()[:2].expand(1, 2, 2).clone(),
                    "score_behavior_logprobs": torch.tensor([0.1, 0.1]).log().expand(1, 2, 2).clone(),
                }
            )
            trainer._finish_score_probe(batch)
            assert trainer.all_metrics["score_centering_probe/hypothetical_loss_abs_mean"] > 0
            assert not any(key.startswith("score_") for key in batch)
        loss = ppo_policy_loss(
            PolicyLossInputs(
                logprobs[response],
                batch["action_log_probs"],
                batch["rollout_logprobs"],
                batch["advantages"],
                batch["loss_mask"],
            ),
            config.trainer.algorithm,
        ).values
        loss = (loss * batch["correction_weights"] * batch["loss_mask"]).sum() / batch["loss_mask"].sum()
        loss.backward()
        optimizer.step()
        results.append(
            (
                parameters.detach().clone(),
                parameters.grad.clone(),
                optimizer.state[parameters]["momentum_buffer"].clone(),
            )
        )
    for baseline, observed in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(observed, baseline, atol=0, rtol=0)
