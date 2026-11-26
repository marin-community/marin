"""An unanswered infrastructure timeout is neutral in both loss and RLOO."""

from dataclasses import replace

import numpy as np
import torch
from omegaconf import OmegaConf
from taskcompendium.grading_result import GradeResult, Outcome
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.rollouts.task_projections import _training_batch, _training_decision
from skyrl_train.rollouts.training_record import TrainingRollout
from skyrl_train.utils.advantage_estimators import compute_rloo_n_outcome_advantage
from skyrl_train.utils.harbor_errors import DEFAULT_ERROR_HANDLING_CONFIG


def test_zero_token_unavailable_row_has_no_loss_or_baseline_contribution():
    win = TrainingRollout("task", (), (1,), (1,), (1,), (-0.5,), GradeResult(Outcome.GRADED, 1., passed=True), "stop")
    loss = replace(win, grade=GradeResult(Outcome.GRADED, 0., passed=False))
    unavailable = replace(win, response_token_ids=(), loss_mask=(), logprobs=(),
        grade=GradeResult(Outcome.UNAVAILABLE, None, "Turn deadline expired before the first response"),
        stop_reason="total_turn_timeout")
    rows = [win, loss, unavailable]
    decisions = [_training_decision(row, DEFAULT_ERROR_HANDLING_CONFIG, logprobs_required=True) for row in rows]
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]")))
    batch = _training_batch(rows, decisions, True, OmegaConf.create({"apply_overlong_filtering": False}), tokenizer, ["fixture"] * 3, [{}, {}, {}])
    assert batch["loss_masks"] == [[1], [1], [0]]
    assert batch["exclude_from_baseline"] == [False, False, True]
    assert batch["verification_results"][2].status.value == "unavailable"
    advantage, _ = compute_rloo_n_outcome_advantage(
        torch.tensor(batch["rewards"])[:, None], torch.tensor(batch["loss_masks"]), np.array(["task"] * 3),
        exclude_from_baseline=np.array(batch["exclude_from_baseline"]),
        group_advantage_invariant=GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=3, minimum_group_size=2),
    )
    torch.testing.assert_close(advantage, torch.tensor([[1.], [-1.], [0.]]))
