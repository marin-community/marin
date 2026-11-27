from collections.abc import Buffer, Generator, Iterable
from contextlib import contextmanager
import io
import os
from typing import cast

from fsspec import AbstractFileSystem
from loguru import logger
import torch
from torch.distributed.checkpoint import FileSystemWriter, SavePlan
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.filesystem import (
    DEFAULT_SUFFIX,
    _write_item,
)
from torch.distributed.checkpoint.planner import WriteItem
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
    """Stream assigned records to one aggregated DCP object per rank."""

    def __init__(
        self,
        path: str,
        *,
        filesystem: AbstractFileSystem,
        max_staged_bytes: int,
    ) -> None:
        if max_staged_bytes <= 0:
            raise ValueError("max_staged_bytes must be positive")
        super().__init__(
            path,
            single_file_per_rank=True,
            sync_files=False,
            thread_count=1,
        )
        self.fs = _AbortableFsspecFileSystem(filesystem)
        self.path = self.fs.init_path(path)
        self.max_staged_bytes = max_staged_bytes

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        plan = super().prepare_local_plan(plan)
        logger.info(
            "DCP direct-write plan rank={} items={} host_staging_budget_bytes={} multipart_part_bytes={} "
            "multipart_concurrency={}",
            self.rank,
            len(plan.items),
            self.max_staged_bytes,
            S3_MULTIPART_PART_BYTES,
            S3_MULTIPART_CONCURRENCY,
        )
        return plan

    def _write_items(
        self, plan: SavePlan, items: Iterable[tuple[WriteItem, torch.Tensor | io.BytesIO]]
    ) -> Future[list[WriteResult]]:
        storage_plan = plan.storage_data
        if storage_plan is None:
            raise AssertionError("DCP storage plan is missing its rank prefix")
        file_name = f"{storage_plan.prefix}0{DEFAULT_SUFFIX}"
        path = self.fs.concat_path(self.path, file_name)
        results: list[WriteResult] = []
        with self.fs.create_stream(path, "wb") as stream:
            # Emit each record immediately. Retaining the records until close
            # would make the host buffer grow to the full rank shard.
            for item, data in items:
                if isinstance(data, torch.Tensor) and not data.is_cpu:
                    raise AssertionError("DCP tensor must be on CPU before serialization")
                results.append(
                    _write_item(
                        self.transforms,
                        stream,
                        data,
                        item,
                        file_name,
                        self.serialization_format,
                    )
                )
                del data

        future: Future[list[WriteResult]] = Future()
        future.set_result(results)
        return future
