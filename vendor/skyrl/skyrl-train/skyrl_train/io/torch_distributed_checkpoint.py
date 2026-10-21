from collections.abc import Buffer, Generator
from contextlib import contextmanager
import io
import os
from typing import cast

from fsspec import AbstractFileSystem
from loguru import logger
import torch
from torch.distributed.checkpoint import FileSystemWriter, SavePlan, SavePlanner
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.filesystem import (
    DEFAULT_SUFFIX,
    _item_size,
    _OverlappingCpuLoader,
    _SerialCpuLoader,
    _write_item,
)
from torch.distributed.checkpoint.planner import WriteItem, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

from marinskyrl.remote_io import (
    CommittableStream,
    OutputStream,
    S3_MULTIPART_CONCURRENCY,
    S3_MULTIPART_PART_BYTES,
    S3MultipartWriteStream,
    create_output_stream,
    manage_output_stream,
)


DEFAULT_TENSOR_COPY_AHEAD_BYTES = 2**30


class _DeferredWriteErrorStream(CommittableStream):
    """Preserve Python write errors that torch.save would otherwise replace."""

    def __init__(self, stream: S3MultipartWriteStream) -> None:
        self.stream = stream
        self.path = stream.path
        self.closed = False
        self._position = 0
        self._write_error: BaseException | None = None

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def flush(self) -> None:
        if self.closed:
            raise ValueError("flush of closed checkpoint stream")
        if self._write_error is None:
            self.stream.flush()

    def write(self, payload: Buffer) -> int:
        if self.closed:
            raise ValueError("write to closed checkpoint stream")
        payload_bytes = len(memoryview(payload).cast("B"))
        if self._write_error is None:
            try:
                self.stream.write(payload)
            except BaseException as error:
                # torch.save replaces exceptions from a Python write callback with
                # an `unexpected pos` assertion while finalizing its zip stream.
                # Accept the remaining callbacks and raise the storage error at commit.
                self._write_error = error
        self._position += payload_bytes
        return payload_bytes

    def commit(self) -> None:
        if self.closed:
            raise ValueError("commit of closed checkpoint stream")
        if self._write_error is not None:
            error = self._write_error
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"Failed to abort multipart upload for {self.path}: {cleanup_error}")
            raise error
        self.stream.commit()
        self.closed = True

    def close(self) -> None:
        if self.closed:
            return
        self.stream.close()
        self.closed = True


class _AbortableFsspecFileSystem(FsspecFileSystem):
    def __init__(self, filesystem: AbstractFileSystem) -> None:
        super().__init__()
        self.fs = filesystem

    def init_path(self, path: str | os.PathLike, **_kwargs) -> str | os.PathLike:
        return path

    @contextmanager
    def create_stream(
        self,
        path: str | os.PathLike,
        mode: str,
    ) -> Generator[io.IOBase | OutputStream, None, None]:
        if self.fs is None:
            raise AssertionError("filesystem has not been initialized")

        object_path = os.fspath(path)
        if mode == "wb":
            stream = create_output_stream(self.fs, object_path)
            if object_path.endswith(DEFAULT_SUFFIX) and isinstance(stream, S3MultipartWriteStream):
                stream = _DeferredWriteErrorStream(stream)
            with manage_output_stream(stream, object_path) as managed:
                yield managed
            return

        with self.fs.open(object_path, mode) as stream:
            yield cast(io.IOBase, stream)


class StreamingFsspecWriter(FileSystemWriter):
    """Stream one aggregated DCP object per rank with bounded tensor copy-ahead."""

    def __init__(
        self,
        path: str,
        *,
        filesystem: AbstractFileSystem,
        tensor_copy_ahead_bytes: int = DEFAULT_TENSOR_COPY_AHEAD_BYTES,
    ) -> None:
        if tensor_copy_ahead_bytes <= 0:
            raise ValueError("tensor_copy_ahead_bytes must be positive")
        super().__init__(
            path,
            single_file_per_rank=True,
            sync_files=False,
            thread_count=1,
            per_thread_copy_ahead=tensor_copy_ahead_bytes,
        )
        self.fs = _AbortableFsspecFileSystem(filesystem)
        self.path = self.fs.init_path(path)
        self.tensor_copy_ahead_bytes = tensor_copy_ahead_bytes

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        plan = super().prepare_local_plan(plan)
        logger.info(
            "DCP direct-write plan rank={} items={} files=1 copy_ahead_bytes={} multipart_part_bytes={} "
            "multipart_concurrency={} max_staged_bytes={}",
            self.rank,
            len(plan.items),
            self.tensor_copy_ahead_bytes,
            S3_MULTIPART_PART_BYTES,
            S3_MULTIPART_CONCURRENCY,
            self.tensor_copy_ahead_bytes + S3_MULTIPART_PART_BYTES * (S3_MULTIPART_CONCURRENCY + 1),
        )
        return plan

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]:
        """Write a rank shard without retaining serialized tensors until close."""
        storage_plan = plan.storage_data
        if storage_plan is None:
            raise AssertionError("DCP storage plan is missing its rank prefix")
        file_name = f"{storage_plan.prefix}0{DEFAULT_SUFFIX}"
        path = self.fs.concat_path(self.path, file_name)

        tensor_items = [item for item in plan.items if item.type != WriteItemType.BYTE_IO]
        if torch.cuda.is_available():
            loader = _OverlappingCpuLoader(
                planner.resolve_data,
                inflight_threshhold=self.tensor_copy_ahead_bytes,
            )
        else:
            loader = _SerialCpuLoader(planner.resolve_data)
        for item in tensor_items:
            loader.add(_item_size(item), item)
        loader.start_loading()

        results: list[WriteResult] = []
        with self.fs.create_stream(path, "wb") as stream:
            for item in plan.items:
                if item.type == WriteItemType.BYTE_IO:
                    results.append(
                        _write_item(
                            self.transforms,
                            stream,
                            planner.resolve_data(item),
                            item,
                            file_name,
                            self.serialization_format,
                        )
                    )

            # PyTorch's stock one-file writer fills a safetensors dictionary even
            # for torch.save, retaining every GPU-to-CPU copy until the rank closes.
            # Emit each result immediately so only the loader's bounded window lives.
            for tensor, item_object in loader.values():
                item = cast(WriteItem, item_object)
                if not tensor.is_cpu:
                    raise AssertionError("DCP tensor must be on CPU before serialization")
                results.append(
                    _write_item(
                        self.transforms,
                        stream,
                        tensor,
                        item,
                        file_name,
                        self.serialization_format,
                    )
                )

        future: Future[list[WriteResult]] = Future()
        future.set_result(results)
        return future
