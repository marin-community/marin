from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import torch

from skyrl_train.training_batch import TrainingInputBatch


class BatchInputPhase(StrEnum):
    FORWARD = "forward"
    TRAIN = "train"


@dataclass(frozen=True)
class BatchDigest:
    tensors: tuple
    metadata: tuple
    routes: tuple | None


def digest(batch: TrainingInputBatch) -> BatchDigest:
    """Hash the consumed input bytes, shapes, dtypes and route geometry."""
    tensors = tuple(
        (
            key,
            None
            if value is None
            else (
                str(value.dtype),
                tuple(value.shape),
                sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest(),
            ),
        )
        for key, value in sorted(batch.items())
    )
    routes = batch.routed_expert_rows
    return BatchDigest(
        tensors=tensors,
        metadata=tuple((key, batch.metadata[key]) for key in ("response_length", "global_step")),
        routes=None
        if routes is None
        else (
            routes.response_len,
            routes.num_experts,
            tuple(
                (str(row.dtype), tuple(row.shape), sha256(row.tobytes(order="C")).hexdigest()) for row in routes.rows
            ),
        ),
    )
