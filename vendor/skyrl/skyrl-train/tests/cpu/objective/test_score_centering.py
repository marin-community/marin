"""Behavior evidence survives the learner boundary."""

import numpy as np
import torch

from skyrl_train.objective.score_centering import collate_score_centering
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch


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
