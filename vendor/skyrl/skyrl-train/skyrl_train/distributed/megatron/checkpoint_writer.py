"""Megatron checkpoint storage shared by BF16 parameters and FP32 masters."""

from dataclasses import replace
import re

from fsspec import AbstractFileSystem
from loguru import logger
import torch
from torch.distributed.checkpoint import SavePlan
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.storage import WriteResult

from skyrl_train.io.torch_distributed_checkpoint import StreamingFsspecWriter


_MASTER_WEIGHT_KEY = re.compile(r"(?:chained_\d+\.)?optimizer\.state\.(?:param|fp32_param)\.(.+)")


def _master_weight_aliases(plans: list[SavePlan]) -> dict[str, str]:
    tensors = {
        item.index.fqn: item.tensor_data for plan in plans for item in plan.items if item.tensor_data is not None
    }
    aliases = {}
    for key, master in tensors.items():
        match = _MASTER_WEIGHT_KEY.fullmatch(key)
        if match is None or master.properties.dtype != torch.float32:
            continue
        model_key = match.group(1)
        model = tensors.get(model_key)
        if model is None or model.properties.dtype != torch.bfloat16 or model.size != master.size:
            continue
        if model_key in aliases:
            raise ValueError(f"Multiple FP32 masters for model tensor {model_key}")
        aliases[model_key] = key
    return aliases


class MasterWeightsWriter(StreamingFsspecWriter):
    """Share Megatron BF16 model records with their synchronized FP32 masters.

    DCP readers cast the FP32 record into each destination's dtype. Keeping the
    BF16 logical entries also supports model-only loads without an optimizer.
    """

    def __init__(self, path: str, *, filesystem: AbstractFileSystem) -> None:
        super().__init__(path, filesystem=filesystem)
        self._model_aliases: dict[str, str] = {}

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        # Fully parallel saving can assign a model tensor and its master to
        # different ranks, so aliases must be resolved from the global plans.
        self._model_aliases = _master_weight_aliases(plans)
        plans = [
            replace(plan, items=[item for item in plan.items if item.index.fqn not in self._model_aliases])
            for plan in plans
        ]
        logger.info("DCP sharing {} BF16 model tensors with FP32 master records", len(self._model_aliases))
        return super().prepare_global_plan(plans)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        for model_key, master_key in self._model_aliases.items():
            model = metadata.state_dict_metadata[model_key]
            master = metadata.state_dict_metadata[master_key]
            assert isinstance(model, TensorStorageMetadata) and isinstance(master, TensorStorageMetadata)
            # The source may have a different shard layout. Its chunks and
            # storage indices must agree for resharding and partial reads.
            metadata.state_dict_metadata[model_key] = replace(master, properties=model.properties)

        master_aliases = {master: model for model, master in self._model_aliases.items()}
        alias_results = [
            replace(result, index=replace(result.index, fqn=master_aliases[result.index.fqn]))
            for rank_results in results
            for result in rank_results
            if result.index.fqn in master_aliases
        ]
        super().finish(metadata, [*results, alias_results])
