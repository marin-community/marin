"""One complete DCP record at a time in each worker."""

from dataclasses import replace
import io
from pathlib import PurePosixPath
import time

from loguru import logger
from torch.distributed.checkpoint import FileSystemReader, LoadPlan, LoadPlanner
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import Metadata
from torch.futures import Future


class RecordCheckpointReader(FileSystemReader):
    """Buffer one S3 record and release its decoded tensor before the next read."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.storage_read_seconds = 0.0
        self.decode_copy_seconds = 0.0

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool, *args, **kwargs) -> None:
        super().set_up_storage_reader(metadata, is_coordinator, *args, **kwargs)
        for storage in self.storage_data.values():
            if not isinstance(storage, _StorageInfo) or storage.transform_descriptors:
                raise ValueError("Megatron restore requires untransformed torch_dist DCP records")
            path = PurePosixPath(storage.relative_path)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".distcp":
                raise ValueError(f"Unsupported DCP record path: {storage.relative_path}")

    def _slice_file(self, file, storage: _StorageInfo):
        # One bounded range read avoids many tiny S3 requests from torch.load's
        # zip-file seeks. Only this record's buffer is retained during decoding.
        started = time.monotonic()
        buffer = io.BytesIO(super()._slice_file(file, storage).read())
        self.storage_read_seconds += time.monotonic() - started
        return buffer

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        serialized_bytes = 0
        for item in plan.items:
            started = time.monotonic()
            storage_before = self.storage_read_seconds
            # Returning from the parent releases its last decoded tensor before
            # the next record is read. Workers can read independently.
            super().read_data(replace(plan, items=[item]), planner).wait()
            self.decode_copy_seconds += time.monotonic() - started - (self.storage_read_seconds - storage_before)
            serialized_bytes += self.storage_data[item.storage_index].length
        logger.info(
            "DCP records={} serialized_bytes={} storage_read={:.3f}s decode_copy={:.3f}s",
            len(plan.items),
            serialized_bytes,
            self.storage_read_seconds,
            self.decode_copy_seconds,
        )
        future: Future[None] = Future()
        future.set_result(None)
        return future
