import pytest
import torch

from skyrl_train.config.objective_spec import validate_dpo
from skyrl_train.config.utils import get_default_config
from skyrl_train.response_loss_mask import apply_response_loss_mask, exclude_response_token_ids
from skyrl_train.training_batch import TrainingInputBatch


def test_response_mask_preserves_context_and_restricts_policy_and_teacher_gradients():
    sequence = torch.tensor([[99, 7, 10, 7, 11], [99, 7, 12, 0, 0]])
    batch = TrainingInputBatch(
        {
            "sequences": sequence.clone(),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
            "loss_mask": torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            "ftpo_weights": torch.tensor([[0.5, 0.0, 0.5], [2.0, 0.0, 0.0]]),
            "distillation_loss_weights": torch.tensor([[0.5, 0.0, 0.5], [2.0, 0.0, 0.0]]),
        }
    )
    attention = batch["attention_mask"].clone()
    responses = batch["response_mask"].clone()
    # Annotation keeps only the first token of each response; a prior tool mask
    # cannot be revived. The second annotation empties the second row entirely.
    apply_response_loss_mask(batch, torch.tensor([[True, True, False], [False, True, True]]))
    policy = torch.ones((2, 3), requires_grad=True)
    teacher = torch.ones((2, 3), requires_grad=True)
    (policy.mul(batch["loss_mask"]).sum() + teacher.mul(batch["distillation_loss_weights"]).sum()).backward()
    torch.testing.assert_close(policy.grad, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(teacher.grad, torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    torch.testing.assert_close(batch["ftpo_weights"], torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert torch.equal(batch["sequences"], sequence)
    assert torch.equal(batch["attention_mask"], attention)
    assert torch.equal(batch["response_mask"], responses)


def test_token_exclusions_only_apply_to_response_coordinates():
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[7, 3, 7, 4], [7, 7, 4, 0]]),
            "loss_mask": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        }
    )
    metrics = exclude_response_token_ids(batch, [7])
    torch.testing.assert_close(batch["loss_mask"], torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    assert metrics["response_mask/tokens_after"] == 3
    assert batch["sequences"].tolist() == [[7, 3, 7, 4], [7, 7, 4, 0]]


def test_response_mask_rejects_broadcastable_but_misaligned_annotations():
    batch = TrainingInputBatch({"loss_mask": torch.ones(2, 3)})
    with pytest.raises(ValueError, match="shape"):
        apply_response_loss_mask(batch, torch.tensor([[True, False, True]]))
    torch.testing.assert_close(batch["loss_mask"], torch.ones(2, 3))


def test_dpo_rejects_token_exclusions_before_training():
    config = get_default_config()
    config.trainer.algorithm.excluded_response_token_ids = [7]
    with pytest.raises(ValueError, match="empty one side"):
        validate_dpo(config)
