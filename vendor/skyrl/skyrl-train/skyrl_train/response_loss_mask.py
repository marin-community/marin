"""Restrict response loss coordinates while preserving full attention context."""

from collections.abc import Sequence

import torch

from skyrl_train.training_batch import TrainingInputBatch


def apply_response_loss_mask(batch: TrainingInputBatch, keep: torch.Tensor) -> dict[str, float]:
    """Intersect a response-aligned boolean annotation with existing loss weights.

    Coordinates match loss_mask, including right padding and tool observations.
    True never restores a coordinate that was already excluded. Token IDs,
    attention masks, response masks, and teacher score evidence remain intact.
    """
    before = batch["loss_mask"]
    if keep.dtype != torch.bool or keep.shape != before.shape or keep.device != before.device:
        raise ValueError("Response loss mask must be boolean and match loss_mask shape and device")
    teacher_weights = batch.get("distillation_loss_weights")
    if teacher_weights is not None and teacher_weights.shape != keep.shape:
        raise ValueError("Teacher loss weights must align with the response loss mask")
    batch["loss_mask"] = before * keep
    if teacher_weights is not None:
        batch["distillation_loss_weights"] = teacher_weights * keep
    return {
        "response_mask/tokens_before": float((before > 0).sum()),
        "response_mask/tokens_after": float((batch["loss_mask"] > 0).sum()),
        "response_mask/nonempty_rows": float((batch["loss_mask"] > 0).any(dim=-1).sum()),
    }


def exclude_response_token_ids(batch: TrainingInputBatch, token_ids: Sequence[int]) -> dict[str, float]:
    """Exclude selected token IDs from learner losses, retaining them in the sequence."""
    if not token_ids:
        return {}
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids):
        raise ValueError("Excluded response token IDs must be nonnegative integers")
    width = batch["loss_mask"].shape[1]
    if width == 0 or batch["sequences"].shape[1] < width:
        raise ValueError("Response loss mask must align with the sequence suffix")
    response_ids = batch["sequences"][:, -width:]
    excluded = torch.tensor(token_ids, dtype=response_ids.dtype, device=response_ids.device)
    return apply_response_loss_mask(batch, ~torch.isin(response_ids, excluded))
