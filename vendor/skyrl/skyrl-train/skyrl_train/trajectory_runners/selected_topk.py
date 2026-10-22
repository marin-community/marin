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
    candidate_ids_array = np.asarray(candidate_ids)
    candidate_scores_array = np.asarray(candidate_scores, dtype=np.float32)
    if candidate_ids_array.ndim != 2 or candidate_scores_array.shape != candidate_ids_array.shape:
        raise ValueError("student top-K candidate widths must agree across tokens")
    width = candidate_ids_array.shape[1]
    if not width:
        raise ValueError("student top-K candidate width must be positive")
    if not np.issubdtype(candidate_ids_array.dtype, np.integer):
        candidate_ids_array = candidate_ids_array.astype(np.int64)
    id_dtype = (
        np.int32
        if candidate_ids_array.min() >= np.iinfo(np.int32).min and candidate_ids_array.max() <= np.iinfo(np.int32).max
        else np.int64
    )
    # A single-turn response usually already has aligned contiguous numeric
    # heads. Their consumers read them; only masked/tool positions need padding.
    if len(response_ids) == len(generated_ids):
        return AlignedStudentTopK(
            np.ascontiguousarray(candidate_ids_array, dtype=id_dtype),
            np.ascontiguousarray(candidate_scores_array),
        )
    aligned_ids = np.full((len(response_ids), width), INVALID_TOPK_INDEX, dtype=id_dtype)
    aligned_scores = np.zeros((len(response_ids), width), dtype=np.float32)
    generated_positions = np.flatnonzero(loss_mask)
    aligned_ids[generated_positions] = candidate_ids_array
    aligned_scores[generated_positions] = candidate_scores_array
    return AlignedStudentTopK(aligned_ids, aligned_scores)
