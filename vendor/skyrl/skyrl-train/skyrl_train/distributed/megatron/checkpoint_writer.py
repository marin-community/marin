"""Megatron checkpoint storage shared by BF16 parameters and FP32 masters."""

from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future as ThreadFuture, ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
import re
from threading import Condition, Event

from fsspec import AbstractFileSystem
from loguru import logger
import torch
from torch.distributed.checkpoint import SavePlan, SavePlanner
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.planner import WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

from skyrl_train.io.torch_distributed_checkpoint import DEFAULT_TENSOR_COPY_AHEAD_BYTES, StreamingFsspecWriter


_MASTER_WEIGHT_KEY = re.compile(r"(?:chained_\d+\.)?optimizer\.state\.(?:param|fp32_param)\.(.+)")


def model_key_for_master(key: str) -> str | None:
    match = _MASTER_WEIGHT_KEY.fullmatch(key)
    return match.group(1) if match is not None else None


def _master_weight_aliases(plans: list[SavePlan]) -> dict[str, str]:
    tensors = {
        item.index.fqn: item.tensor_data for plan in plans for item in plan.items if item.tensor_data is not None
    }
    aliases = {}
    for key, master in tensors.items():
        model_key = model_key_for_master(key)
        if model_key is None or master.properties.dtype != torch.float32:
            continue
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
    Model-only loads read the full FP32 source record before casting to BF16.
    """

    def __init__(
        self,
        path: str,
        *,
        filesystem: AbstractFileSystem,
        tensor_copy_ahead_bytes: int = DEFAULT_TENSOR_COPY_AHEAD_BYTES,
    ) -> None:
        super().__init__(path, filesystem=filesystem, tensor_copy_ahead_bytes=tensor_copy_ahead_bytes)
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


class StagedMasterWeightsWriter(MasterWeightsWriter):
    """Copy the rank's final write plan to host memory before streaming it."""

    def __init__(self, path: str, *, filesystem: AbstractFileSystem, max_staged_bytes: int) -> None:
        super().__init__(path, filesystem=filesystem, tensor_copy_ahead_bytes=max_staged_bytes)
        self.staged = Event()

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]:
        # The global plan has already removed replicas and aliased BF16 records.
        ready: deque[tuple[WriteItem, torch.Tensor | BytesIO, int]] = deque()
        condition = Condition()
        in_flight = 0
        stopped = False

        def stage() -> None:
            nonlocal in_flight
            peak = total = 0
            for item in plan.items:
                data = planner.resolve_data(item)
                size = data.nbytes if isinstance(data, torch.Tensor) else data.getbuffer().nbytes
                with condition:
                    # A DCP record is indivisible. An oversized record takes the buffer alone.
                    condition.wait_for(
                        lambda: stopped or not in_flight or in_flight + size <= self.tensor_copy_ahead_bytes
                    )
                    if stopped:
                        return
                    in_flight += size
                    peak = max(peak, in_flight)
                if isinstance(data, torch.Tensor):
                    data = data.detach().to(device="cpu", copy=True, memory_format=torch.contiguous_format)
                else:
                    data = BytesIO(data.getvalue())
                with condition:
                    ready.append((item, data, size))
                    del data
                    condition.notify_all()
                total += size
            self.staged.set()
            logger.info("Checkpoint staged {} bytes on rank {}, peak {} bytes", total, self.rank, peak)

        def items(staging: ThreadFuture[None]) -> Iterator[tuple[WriteItem, torch.Tensor | BytesIO]]:
            nonlocal in_flight
            for _ in plan.items:
                with condition:
                    condition.wait_for(lambda: ready or staging.done())
                    if not ready:
                        staging.result()
                        raise RuntimeError("Checkpoint staging ended before the write plan was buffered")
                    item, data, size = ready.popleft()
                yield item, data
                del data
                with condition:
                    in_flight -= size
                    condition.notify_all()

        def wake_writer(_future: ThreadFuture[None]) -> None:
            with condition:
                condition.notify_all()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-stage") as executor:
            staging = executor.submit(stage)
            staging.add_done_callback(wake_writer)
            try:
                return self._write_items(plan, items(staging))
            finally:
                with condition:
                    stopped = True
                    condition.notify_all()
