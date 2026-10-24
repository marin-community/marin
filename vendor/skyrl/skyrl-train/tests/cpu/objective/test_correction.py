import pytest
import torch

from skyrl_train.config.objective_spec import load_correction
from skyrl_train.objective.correction import compute_correction


@pytest.mark.parametrize("preset", ["tis", "outlier_mask"])
def test_mixed_teacher_tokens_do_not_receive_student_correction_or_poison_sequence_filter(preset):
    rollout = torch.tensor([[-5.0, -10.0, float("nan")]])
    old = torch.tensor([[-5.0, -1.0, float("nan")]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    sources = torch.tensor([[0, 1, 0]])
    result = compute_correction(old, rollout, mask, load_correction(preset), behavior_sources=sources)
    torch.testing.assert_close(result.weights, torch.tensor([[1.0, 1.0, 0.0]]))
