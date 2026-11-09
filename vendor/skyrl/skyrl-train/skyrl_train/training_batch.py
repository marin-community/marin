"""Training batch containers and iteration."""

import io
import math
import pickle
from typing import Any, Dict, Generic, Iterator, List, Optional, Protocol, Sequence, TypeVar, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Integer

from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.dataset.replay_buffer import Experience
from skyrl_train.ftpo import FTPOTargets
from skyrl_train.distillation import distillation_input_from_tensors

DictType = TypeVar("DictType")
ROUTED_EXPERTS_KEY = "rollout_routed_experts"


class RouterReplayPayload(Protocol):
    """Storage facts used by dispatch and worker arrival logs."""

    @property
    def nbytes(self) -> int: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def shape(self) -> Sequence[int]: ...


def per_data_parallel_batch_size(mini_batch_size: int, samples_per_prompt: int, data_parallel_size: int) -> int:
    """Return a role's per-data-parallel-rank mini-batch size."""
    if data_parallel_size < 1:
        raise ValueError(f"data_parallel_size must be positive, got {data_parallel_size}")
    return mini_batch_size * samples_per_prompt // data_parallel_size


def gradient_accumulation_steps(per_rank_mini_batch_size: int, micro_batch_size: int) -> int:
    """Return the number of microbatches in one optimizer update."""
    if micro_batch_size < 1:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    steps, remainder = divmod(per_rank_mini_batch_size, micro_batch_size)
    if steps < 1 or remainder:
        raise ValueError(
            f"per-rank mini-batch size {per_rank_mini_batch_size} must be a positive multiple "
            f"of micro-batch size {micro_batch_size}"
        )
    return steps


# NOTE (sumanthrh): This is inspired by `TensorDict` but is much simpler.
class TensorBatch(dict, Generic[DictType]):
    """Base class for training batches

    This defines a generic container for a batch of training data (inputs or outputs).
    Consists of tensors and metadata. Non-tensor training payloads belong on a specific batch type.
    """

    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_size = None
        self._device = None
        self._check_consistency()

    def select(self, keys: List[str], metadata_keys: Optional[List[str]] = None) -> "TensorBatch[DictType]":
        """Select a subset of the data batch.

        Args:
            keys: The keys to select
            metadata_keys: The metadata keys to select

        Returns:
            A new `TensorBatch` object with the selected keys and metadata
        """
        selected_batch_data = {}
        for key in keys:
            selected_batch_data[key] = self[key]
        selected_metadata = {}
        if metadata_keys is None:
            selected_metadata = self.metadata
        else:
            selected_metadata = {}
            for key in metadata_keys:
                selected_metadata[key] = self.metadata[key]
        new_batch = self.__class__(selected_batch_data)
        new_batch.metadata = selected_metadata
        return new_batch

    def _check_consistency(self):
        """Check consistency of all present fields"""
        present = [(key, value) for key, value in self.items() if value is not None]
        if not present:
            return
        first = present[0][1]
        if not isinstance(first, torch.Tensor):
            raise ValueError(f"Field {present[0][0]} must be a tensor, got {type(first)}")
        self._batch_size = len(first)
        self._device = first.device
        for key, value in present:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Field {key} must be a tensor, got {type(value)}")
            if len(value) != self._batch_size:
                raise ValueError(f"Batch size mismatch in {key}")
            if value.device != self._device:
                raise ValueError(f"Device mismatch in {key}. Expected {self._device}, got {value.device}")

    def __getitem__(self, index) -> "TensorBatch[DictType]":
        if isinstance(index, slice):
            return self.slice(index.start, index.stop, index.step)
        elif isinstance(index, int):
            return self.slice(index, index + 1)
        else:
            return super().__getitem__(index)

    def __setitem__(self, key: str, value: Optional[torch.Tensor]) -> None:
        if value is None:
            super().__setitem__(key, value)
            return

        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Field {key} must be a tensor, got {type(value)}")

        if hasattr(self, "_batch_size") and self._batch_size is not None and len(value) != self._batch_size:
            raise ValueError(
                f"Batch size mismatch in {key}. Expected tensor to be of size {self._batch_size}, got {len(value)}."
            )

        super().__setitem__(key, value)

        if hasattr(self, "_batch_size") and self._batch_size is None:
            self._batch_size = len(value)

    def to(
        self, device: torch.device = None, dtype: torch.dtype = None, *, non_blocking: bool = False
    ) -> "TensorBatch":
        """Move tensors to device and/or cast to dtype.

        Args:
            device: The device to move the tensors to
            dtype: The dtype to cast the tensors to
            non_blocking: Whether the operation should be non-blocking
        """
        for key, value in self.items():
            if value is None:
                continue
            assert isinstance(value, torch.Tensor), f"Field {key} must be a tensor, got {type(value)}"
            self[key] = value.to(device, dtype, non_blocking=non_blocking)
        self._device = next((value.device for value in self.values() if value is not None), None)
        return self

    @property
    def batch_size(self) -> int:
        """Batch size for the tensors"""
        return self._batch_size

    @property
    def device(self) -> torch.device:
        """Get the device for the tensors"""
        return self._device

    def __getstate__(self):
        """Serialize the `TensorBatch` object for pickle protocol"""
        if self._device is not None:
            assert self._device == torch.device("cpu"), "Tensors must be on CPU before serialization"
        batch_dict = {}
        for key, value in self.items():
            value_to_save = value
            if isinstance(value, torch.Tensor):
                logical_bytes = value.numel() * value.element_size()
                if value.storage_offset() != 0 or value.untyped_storage().nbytes() > logical_bytes:
                    value_to_save = value.clone(memory_format=torch.contiguous_format)
            buffer = io.BytesIO()
            torch.save(value_to_save, buffer)
            batch_dict[key] = buffer.getvalue()

        return {
            "batch_dict": batch_dict,
            "batch_size": self._batch_size,
            "device": self._device,
            "metadata": self.metadata,
        }

    def __reduce_ex__(self, _protocol: int):
        """Serialize only the compact custom state, not the inherited dict payload."""
        return self.__class__, (), self.__getstate__()

    def __setstate__(self, state):
        """Deserialize the `TensorBatch` object and load it into memory"""
        for key, value in state["batch_dict"].items():
            buffer = io.BytesIO(value)
            self[key] = torch.load(buffer)

        self._batch_size = state["batch_size"]
        self._device = state["device"]
        self.metadata = state["metadata"]
        self._check_consistency()
        return self

    def repeat(self, repeats: int):
        """Repeat entries in the data batch a specified number of times.

        This is similar to `torch.repeat` (and `numpy.tile`). `metadata` is not repeated.

        Args:
            repeats: The number of times to repeat the data batch

        Returns:
            A new `TensorBatch` object with the data repeated
        """
        new_batch = {}
        for key, value in self.items():
            if value is None:
                new_batch[key] = value
            else:
                assert isinstance(value, torch.Tensor), f"Field {key} must be a tensor, got {type(value)}"
                new_batch[key] = value.repeat((repeats,) + (1,) * (value.ndim - 1))
        new_batch = self.__class__(new_batch)
        new_batch.metadata = self.metadata
        return new_batch

    def repeat_interleave(self, repeats: int):
        """Repeat entries in the data batch a specified number of times.

        This is similar to `torch.repeat_interleave` (and `numpy.repeat`). `metadata` is not repeated.

        Args:
            repeats: The number of times to repeat the data batch

        Returns:
            A new `TensorBatch` object with the data repeated
        """
        new_batch = {}
        for key, value in self.items():
            if value is None:
                new_batch[key] = value
            else:
                assert isinstance(value, torch.Tensor), f"Field {key} must be a tensor, got {type(value)}"
                new_batch[key] = value.repeat_interleave(repeats, dim=0)
        new_batch = self.__class__(new_batch)
        new_batch.metadata = self.metadata
        return new_batch

    def chunk(self, chunk_size: int) -> List["TensorBatch[DictType]"]:
        """Split into smaller chunks"""
        return [self.slice(i, i + chunk_size) for i in range(0, self.batch_size, chunk_size)]

    def slice(self, start: int, end: int, step: int = 1) -> "TensorBatch[DictType]":
        """Slice the data batch.

        Args:
            start: The start index
            end: The end index
            step: The step size

        Returns:
            A new `TensorBatch` object with the view of the specified slice.
        """
        slice_obj = slice(start, end, step)
        sliced_data = {}
        for key, value in self.items():
            sliced_data[key] = None if value is None else value[slice_obj]
        sliced_batch = self.__class__(sliced_data)
        sliced_batch.metadata = self.metadata
        return sliced_batch

    def save(self, path: str):
        """Save the data to a pickle file"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """Load the data from a pickle file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def cat(cls, shards: List["TensorBatch[DictType]"]) -> "TensorBatch[DictType]":
        """Concatenate shards.

        Args:
            shards: The list of `TensorBatch` objects to cat

        Returns:
            A new `TensorBatch` object with the concatenated data
        """
        cat_data = {}
        assert len(shards) > 0, "Cannot cat an empty list of shards"
        for key, value in shards[0].items():
            if value is not None:
                cat_data[key] = torch.cat([shard[key] for shard in shards])
            else:
                # `None` values are not cat'd
                cat_data[key] = value
        metadata = shards[0].metadata
        cat_batch = cls(cat_data)
        cat_batch.metadata = metadata
        return cat_batch

    def __len__(self) -> int:
        """Length of the batch.

        Note that this is the same as the batch size rather than the number of keys in the batch.
        """
        return self._batch_size

    def __eq__(self, other: Any) -> bool:
        """Check if two `TensorBatch` objects are equal"""
        if not isinstance(other, TensorBatch):
            return False
        if self.metadata != other.metadata:
            return False
        if len(self) != len(other):
            return False
        if len(self.items()) != len(other.items()):
            return False
        for k, v in self.items():
            if k not in other:
                return False
            if v is None or other[k] is None:
                if v is not other[k]:
                    return False
            elif not torch.equal(v, other[k]):
                return False
        return True

    def __str__(self) -> str:
        """String representation of the `TensorBatch` object"""
        return f"TensorBatch(batch_size={self.batch_size}, device={self.device}, metadata={self.metadata}), items={self.items()}"

    def __repr__(self) -> str:
        """String representation of the `TensorBatch` object"""
        return self.__str__()


class TrainingInput(TypedDict, total=False):
    """Schema for training input batch"""

    sequences: Integer[torch.Tensor, "batch_size seq_len"]
    attention_mask: Integer[torch.Tensor, "batch_size seq_len"]
    loss_mask: Integer[torch.Tensor, "batch_size seq_len"]
    response_mask: Integer[torch.Tensor, "batch_size seq_len"]
    action_log_probs: Float[torch.Tensor, "batch_size seq_len"]
    base_action_log_probs: Float[torch.Tensor, "batch_size seq_len"]
    values: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    returns: Float[torch.Tensor, "batch_size seq_len"]
    advantages: Float[torch.Tensor, "batch_size seq_len"]
    kl: Float[torch.Tensor, "batch_size seq_len"]
    rewards: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    rollout_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    correction_weights: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    score_topk_indices: Optional[Integer[torch.Tensor, "batch_size seq_len top_k"]]
    score_old_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len top_k"]]
    score_behavior_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len top_k"]]
    # Policy steps each response token is behind at consumption.
    rollout_staleness: Optional[Integer[torch.Tensor, "batch_size seq_len"]]
    teacher_action_log_probs: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    teacher_topk_indices: Optional[Integer[torch.Tensor, "batch_size seq_len top_k"]]
    teacher_topk_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len top_k"]]
    teacher_retained_mass: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    student_topk_indices: Optional[Integer[torch.Tensor, "batch_size seq_len top_k"]]
    behavior_topk_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len top_k"]]
    teacher_on_student_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len top_k"]]
    ftpo_chosen_mask: Optional[Integer[torch.Tensor, "batch_size seq_len top_k"]]
    ftpo_reference_logits: Optional[Float[torch.Tensor, "batch_size vocab"]]
    ftpo_repair_mask: Optional[Integer[torch.Tensor, "batch_size seq_len"]]
    teacher_valid_mask: Optional[Integer[torch.Tensor, "batch_size seq_len"]]
    distillation_loss_weights: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    # Dense replay targets are used by diagnostic input batches. Generated compact
    # rows live in TrainingInputBatch.routed_expert_rows until worker materialization.
    rollout_routed_experts: Optional[Integer[torch.Tensor, "batch_size seq_len L K"]]
    # Loop-behavior reward shaping (Stage B / F5): per-token additive shaping
    # channel, SEPARATE from `rewards` (the RLOO-N outcome term). Default all-zeros
    # and present ONLY when trainer.algorithm.enable_token_reward_channel is True,
    # so the flag-off TrainingInputBatch keyset is byte-identical to today. The
    # combiner that ADDS this into the advantage is registered in Stage C; Stage B
    # only makes it flow as zeros (no-op).
    token_level_shaping: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    # Negative per-token loop credit, applied after the configured advantage
    # estimator. It never enters outcome-reward group statistics or returns.
    loop_advantages: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    # Loop-behavior reward shaping (Stage B / F4): per-token span tags
    # {OTHER=0, THINK=1, ACTION=2, EDIT=3}, aligned 1:1 with the response tokens
    # (same exact-token-id layout TIS uses). Present only when the channel is on.
    response_span_tags: Optional[Integer[torch.Tensor, "batch_size seq_len"]]
    # DPO pairing evidence: +1 chosen / -1 rejected per row, present only when the
    # static preference-pair runner emitted the batch as adjacent pairs.
    pair_roles: Optional[Float[torch.Tensor, "batch_size"]]  # noqa: F821


class TrainingInputBatch(TensorBatch[TrainingInput]):
    """Training tensors with optional compact route rows kept outside the tensor dictionary."""

    def __init__(self, *args, routed_expert_rows: RoutedExpertRows | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.routed_expert_rows = routed_expert_rows
        self._validate_route_rows()

    def _validate_route_rows(self) -> None:
        rows = self.routed_expert_rows
        if rows is None:
            return
        if len(rows) != self.batch_size:
            raise ValueError("compact route row count must match the training batch")
        if self.get(ROUTED_EXPERTS_KEY) is not None:
            raise ValueError("route targets cannot be both compact rows and a dense tensor")

    @property
    def routed_experts(self) -> RouterReplayPayload | None:
        if self.routed_expert_rows is not None:
            return self.routed_expert_rows
        return self.get(ROUTED_EXPERTS_KEY)

    def routed_experts_tensor(self) -> torch.Tensor | None:
        if self.routed_expert_rows is not None:
            return self.routed_expert_rows.materialize(self.device)
        return self.get(ROUTED_EXPERTS_KEY)

    def select(self, keys: List[str], metadata_keys: Optional[List[str]] = None) -> "TrainingInputBatch":
        if ROUTED_EXPERTS_KEY in keys and ROUTED_EXPERTS_KEY not in self and self.routed_expert_rows is None:
            raise KeyError(ROUTED_EXPERTS_KEY)
        tensor_keys = [key for key in keys if key != ROUTED_EXPERTS_KEY or key in self]
        selected = super().select(tensor_keys, metadata_keys)
        if ROUTED_EXPERTS_KEY in keys:
            selected.routed_expert_rows = self.routed_expert_rows
        return selected

    def slice(self, start: int, end: int, step: int = 1) -> "TrainingInputBatch":
        sliced = super().slice(start, end, step)
        if self.routed_expert_rows is not None:
            sliced.routed_expert_rows = self.routed_expert_rows[slice(start, end, step)]
        return sliced

    def __getstate__(self):
        state = super().__getstate__()
        state["routed_expert_rows"] = self.routed_expert_rows
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.routed_expert_rows = state["routed_expert_rows"]
        self._validate_route_rows()
        return self

    def repeat(self, repeats: int) -> "TrainingInputBatch":
        repeated = super().repeat(repeats)
        if self.routed_expert_rows is not None:
            rows = self.routed_expert_rows
            repeated.routed_expert_rows = RoutedExpertRows(rows.rows * repeats, rows.response_len, rows.num_experts)
        return repeated

    def repeat_interleave(self, repeats: int) -> "TrainingInputBatch":
        repeated = super().repeat_interleave(repeats)
        if self.routed_expert_rows is not None:
            rows = self.routed_expert_rows
            repeated.routed_expert_rows = RoutedExpertRows(
                tuple(row for row in rows.rows for _ in range(repeats)), rows.response_len, rows.num_experts
            )
        return repeated

    @classmethod
    def cat(cls, shards: List["TrainingInputBatch"]) -> "TrainingInputBatch":
        has_rows = [shard.routed_expert_rows is not None for shard in shards]
        if any(has_rows) and not all(has_rows):
            raise ValueError("cannot concatenate training batches with mixed route representations")
        combined = super().cat(shards)
        if has_rows[0]:
            combined.routed_expert_rows = RoutedExpertRows.cat([shard.routed_expert_rows for shard in shards])
        return combined

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TrainingInputBatch) or not super().__eq__(other):
            return False
        left, right = self.routed_expert_rows, other.routed_expert_rows
        if left is None or right is None:
            return left is right
        return (
            left.response_len == right.response_len
            and left.num_experts == right.num_experts
            and len(left.rows) == len(right.rows)
            and all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(left.rows, right.rows, strict=True))
        )


class TrainingBatchIterator(Iterator[Experience]):
    """Yield reusable ``Experience`` microbatches from a training batch."""

    def __init__(self, data: TrainingInputBatch, sample_batch_size: int):
        if sample_batch_size < 1:
            raise ValueError(f"sample_batch_size must be positive, got {sample_batch_size}")
        self._chunks = data.chunk(sample_batch_size)
        self._iterator = iter(self._chunks)
        self._length = math.ceil(data.batch_size / sample_batch_size)

    def __len__(self) -> int:
        return self._length

    def chunks(self, start: int, stop: int) -> list[TrainingInputBatch]:
        """Return the source microbatches in an accumulation window."""
        return self._chunks[start:stop]

    def __iter__(self) -> "TrainingBatchIterator":
        return self

    def __next__(self) -> Experience:
        try:
            return self._experience(next(self._iterator))
        except StopIteration:
            self._iterator = iter(self._chunks)
            raise

    @staticmethod
    def _experience(batch: TrainingInputBatch) -> Experience:
        return Experience(
            sequences=batch["sequences"],
            action_log_probs=batch["action_log_probs"],
            base_action_log_probs=batch["base_action_log_probs"],
            values=batch["values"],
            returns=batch["returns"],
            advantages=batch["advantages"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            action_mask=batch["response_mask"],
            num_actions=batch.metadata["response_length"],
            rollout_logprobs=batch.get("rollout_logprobs"),
            correction_weights=batch.get("correction_weights"),
            score_topk_indices=batch.get("score_topk_indices"),
            score_old_logprobs=batch.get("score_old_logprobs"),
            score_behavior_logprobs=batch.get("score_behavior_logprobs"),
            distillation=None
            if "ftpo_chosen_mask" in batch
            else distillation_input_from_tensors(
                teacher_action_log_probs=batch.get("teacher_action_log_probs"),
                teacher_topk_indices=batch.get("teacher_topk_indices"),
                teacher_topk_logprobs=batch.get("teacher_topk_logprobs"),
                teacher_retained_mass=batch.get("teacher_retained_mass"),
                valid_mask=batch.get("teacher_valid_mask"),
                loss_weights=batch.get("distillation_loss_weights"),
                student_topk_indices=batch.get("student_topk_indices"),
                behavior_topk_logprobs=batch.get("behavior_topk_logprobs"),
                teacher_on_student_logprobs=batch.get("teacher_on_student_logprobs"),
            ),
            ftpo=FTPOTargets(
                batch["student_topk_indices"],
                batch["ftpo_chosen_mask"] & (batch.get("ftpo_weights", batch["loss_mask"]) > 0).unsqueeze(-1),
                batch["ftpo_reference_logits"],
                batch.get("ftpo_repair_mask"),
                batch.get("ftpo_weights"),
            )
            if "ftpo_chosen_mask" in batch
            else None,
            rollout_routed_experts=batch.routed_experts_tensor(),
            response_span_tags=batch.get("response_span_tags"),
            pair_roles=batch.get("pair_roles"),
            info={},
            metadata=batch.metadata,
        )


class TrainingOutputBatch(TensorBatch[Dict[str, torch.Tensor]]):
    """Training output data"""

    pass
