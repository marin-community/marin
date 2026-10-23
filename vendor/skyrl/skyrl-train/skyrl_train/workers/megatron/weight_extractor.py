"""Megatron-to-Hugging Face weight extraction."""

from collections.abc import Iterator

import torch

from skyrl_train.weight_sync.base import WeightChunk
from skyrl_train.weight_sync.weight_extractor import WeightExtractor, weight_sync_dtype


def mapping_hf_names(mapping) -> set[str]:
    """Return the Hugging Face names owned by one bridge conversion mapping."""
    names = mapping.hf_param
    if isinstance(names, dict):
        names = names.values()
    elif isinstance(names, str):
        names = (names,)
    return {str(name) for name in names}


class BucketedMegatronWeightExtractor(WeightExtractor):
    """Pack completed Hugging Face exports into transport-sized chunks.

    Megatron Bridge conversions may accumulate several conversion tasks before emitting one
    Hugging Face tensor. The bridge receives the complete conversion stream in one invocation;
    chunking the emitted tensors cannot split grouped expert exports.
    """

    def __init__(self, bridge, actor_module, model_type: str, bucket_size_threshold_GB: float = 1.0):
        self.bridge = bridge
        self.actor_module = actor_module
        self.model_type = model_type
        self.bucket_size_threshold_bytes = int(bucket_size_threshold_GB * 1024**3)

    def _wire_tensor(self, name: str, tensor: torch.Tensor, dtype: torch.dtype, device) -> torch.Tensor:
        return tensor.to(device=device, dtype=weight_sync_dtype(self.model_type, name, dtype), non_blocking=True)

    @staticmethod
    def _chunk(entries: list[tuple[str, torch.Tensor]]) -> WeightChunk:
        return WeightChunk(
            names=[name for name, _ in entries],
            dtypes=[str(tensor.dtype) for _, tensor in entries],
            shapes=[list(tensor.shape) for _, tensor in entries],
            tensors=[tensor for _, tensor in entries],
        )

    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Export every weight, pack completed tensors, and reject incomplete groups."""

        device = torch.cuda.current_device()

        conversion_tasks = self.bridge.get_conversion_tasks(self.actor_module)
        expected_grouped_names = set()
        for task in conversion_tasks:
            if getattr(task.mapping, "is_grouped_export", False):
                expected_grouped_names.update(mapping_hf_names(task.mapping))
        emitted_grouped_names = set()
        exported_weights = self.bridge.export_hf_weights(
            self.actor_module,
            show_progress=False,
            conversion_tasks=conversion_tasks,
        )

        entries: list[tuple[str, torch.Tensor]] = []
        bucket_bytes = 0
        for name, tensor in exported_weights:
            if name in expected_grouped_names:
                emitted_grouped_names.add(name)
            tensor = self._wire_tensor(name, tensor, dtype, device)
            tensor_bytes = tensor.numel() * tensor.element_size()
            uses_special_dtype = tensor.dtype != dtype

            if entries and (
                uses_special_dtype
                or tensor.dtype != entries[0][1].dtype
                or bucket_bytes + tensor_bytes > self.bucket_size_threshold_bytes
            ):
                yield self._chunk(entries)
                entries = []
                bucket_bytes = 0

            if uses_special_dtype:
                yield self._chunk([(name, tensor)])
                continue

            entries.append((name, tensor))
            bucket_bytes += tensor_bytes
            if bucket_bytes >= self.bucket_size_threshold_bytes:
                yield self._chunk(entries)
                entries = []
                bucket_bytes = 0

        if entries:
            yield self._chunk(entries)

        missing_grouped_names = expected_grouped_names - emitted_grouped_names
        if missing_grouped_names:
            missing = ", ".join(sorted(missing_grouped_names))
            raise RuntimeError(f"Megatron Bridge omitted grouped weight exports: {missing}")
