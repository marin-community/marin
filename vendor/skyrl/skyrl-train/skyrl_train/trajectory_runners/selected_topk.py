"""Align generated behavior-policy candidates with admitted training tokens."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from skyrl_train.distillation import INVALID_TOPK_INDEX


class AlignedStudentTopK(NamedTuple):
    indices: np.ndarray
    topk_logprobs: np.ndarray


def align_student_topk(
    response_ids: Sequence[int],
    loss_mask: Sequence[int],
    generated_ids: Sequence[int],
    candidate_ids: Sequence[Sequence[int]] | None,
    candidate_scores: Sequence[Sequence[float]] | None,
) -> AlignedStudentTopK | None:
    """Return aligned rows, or no evidence if candidates cannot be trusted."""
    if (candidate_ids is None) != (candidate_scores is None):
        raise ValueError("student top-K IDs and behavior scores must be provided together")
    if candidate_ids is None:
        return None
    if len(response_ids) != len(loss_mask):
        raise ValueError("student top-K alignment requires response IDs and loss mask to have equal length")
    if len(generated_ids) != len(candidate_ids) or len(candidate_scores) != len(candidate_ids):
        raise ValueError("student top-K candidate rows must align with generated IDs")
    if len(candidate_ids) == 0:
        return None
    if list(generated_ids) != [token_id for token_id, keep in zip(response_ids, loss_mask, strict=True) if keep]:
        return None
    width = len(candidate_ids[0])
    if width == 0:
        raise ValueError("student top-K candidate width must be positive")
    if any(
        len(ids) != width or len(scores) != width for ids, scores in zip(candidate_ids, candidate_scores, strict=True)
    ):
        raise ValueError("student top-K candidate widths must agree across tokens")
    candidate_ids_array = np.asarray(candidate_ids, dtype=np.int64)
    id_dtype = (
        np.int32
        if candidate_ids_array.min() >= np.iinfo(np.int32).min and candidate_ids_array.max() <= np.iinfo(np.int32).max
        else np.int64
    )
    aligned_ids = np.full((len(response_ids), width), INVALID_TOPK_INDEX, dtype=id_dtype)
    aligned_scores = np.zeros((len(response_ids), width), dtype=np.float32)
    generated_positions = np.flatnonzero(loss_mask)
    aligned_ids[generated_positions] = candidate_ids_array
    aligned_scores[generated_positions] = np.asarray(candidate_scores, dtype=np.float32)
    return AlignedStudentTopK(aligned_ids, aligned_scores)
