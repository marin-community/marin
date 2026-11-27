"""Project finalized SkyRL batches onto the NumPy learner process boundary.

SkyRL owns admission, RLOO-N, reward shaping and loop credit. This module must
run after ``finalize_advantages_for_training`` and before device partitioning.
Weights describe one complete optimizer mini-batch, including all its temporal
microbatches and data ranks. The JAX process only names/shards these arrays.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from skyrl_train.config.validation import validate_levanter_algorithm
from skyrl_train.objective.reduction import policy_data_weights
from skyrl_train.training_batch import TrainingInputBatch


@dataclass(frozen=True)
class LevanterBatchPayload:
    arrays: dict[str, np.ndarray | None]
    response_length: int
    global_step: int | None

    @property
    def batch_size(self) -> int:
        tokens = self.arrays["tokens"]
        assert tokens is not None
        return tokens.shape[0]

    def as_dict(self) -> dict[str, Any]:
        """Return a process payload without importing SkyRL in the consumer."""
        return {
            "arrays": self.arrays,
            "response_length": self.response_length,
            "global_step": self.global_step,
        }


def _array(tensor: torch.Tensor, dtype: torch.dtype) -> np.ndarray:
    # Copy even CPU tensors: later batch mutation must not alter frozen scores.
    return tensor.detach().to(device="cpu", dtype=dtype).numpy().copy()


def forward_payload(data: TrainingInputBatch) -> LevanterBatchPayload:
    """Preserve left-padded prompt/right-padded response scoring coordinates."""
    if data.routed_experts is not None:
        raise ValueError("Levanter quick-loop bridge does not support router replay")
    metadata = data.metadata or {}
    response_length = int(metadata["response_length"])
    tokens = data["sequences"]
    attention = data["attention_mask"]
    if tokens.ndim != 2 or tokens.shape != attention.shape:
        raise ValueError("sequences and attention_mask must have the same [batch, position] shape")
    if tokens.shape[0] < 1 or not 0 < response_length < tokens.shape[1]:
        raise ValueError("batch must be nonempty with a nonempty prompt and response width")
    if not torch.all((attention == 0) | (attention == 1)):
        raise ValueError("attention_mask must be binary")
    response_mask = attention[:, -response_length:]
    if data.get("response_mask") is not None and not torch.equal(data["response_mask"], response_mask):
        raise ValueError("response_mask must equal the final response_length attention positions")
    position_ids = attention.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention == 0, 0)
    return LevanterBatchPayload(
        arrays={
            "tokens": _array(tokens, torch.int32),
            "attention_mask": _array(attention, torch.bool),
            "position_ids": _array(position_ids, torch.int32),
            "response_mask": _array(response_mask, torch.bool),
        },
        response_length=response_length,
        global_step=metadata.get("global_step"),
    )


def policy_payload(data: TrainingInputBatch, algorithm: DictConfig) -> LevanterBatchPayload:
    """Consume already finalized credit without recomputing any advantages.

    Call once per *global optimizer mini-batch*, before slicing for execution.
    Physical response positions include tool observations and padded clones;
    only loss_mask determines which positions contribute to the objective.
    """
    validate_levanter_algorithm(algorithm)
    if data.get("rewards") is not None or data.get("loop_advantages") is not None:
        raise ValueError("Levanter bridge requires finalized advantages (drop reward and loop-credit inputs first)")
    if data.get("correction_weights") is not None:
        raise ValueError("Levanter quick-loop bridge does not support correction_weights")
    unsupported = (
        "teacher_action_log_probs",
        "teacher_topk_indices",
        "student_topk_indices",
        "ftpo_weights",
        "pair_roles",
    )
    if any(data.get(key) is not None for key in unsupported):
        raise ValueError("Levanter quick-loop bridge does not support teacher, FTPO or preference targets")
    payload = forward_payload(data)
    shape = (payload.batch_size, payload.response_length)
    for name in ("loss_mask", "advantages", "action_log_probs"):
        if data[name].shape != shape:
            raise ValueError(f"{name} must have response-aligned shape {shape}")
    loss_mask = data["loss_mask"].float()
    if not torch.all(torch.isfinite(loss_mask) & (loss_mask >= 0)):
        raise ValueError("loss_mask must be finite and nonnegative")
    if torch.any((loss_mask > 0) & (data["response_mask"] == 0)):
        raise ValueError("loss_mask cannot select response padding")
    think_weight = float(algorithm.get("think_token_weight", 1.0))
    if not np.isfinite(think_weight) or think_weight < 0:
        raise ValueError("think_token_weight must be finite and nonnegative")
    weights = policy_data_weights(loss_mask, data.get("response_span_tags"), think_weight)
    denominator = weights.sum(dtype=torch.float64).item() or 1.0
    valid = loss_mask > 0

    def sanitized(name: str) -> np.ndarray:
        values = torch.where(valid, data[name], 0)
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite at trainable positions")
        return _array(values, torch.float32)

    return LevanterBatchPayload(
        arrays=payload.arrays
        | {
            "advantages": sanitized("advantages"),
            "old_logprobs": sanitized("action_log_probs"),
            "reference_logprobs": None,
            "policy_weights": _array(weights / denominator, torch.float32),
            "kl_weights": np.zeros(shape, dtype=np.float32),
            "loss_mask": _array(loss_mask, torch.float32),
        },
        response_length=payload.response_length,
        global_step=payload.global_step,
    )
