from collections.abc import Buffer, Generator
from contextlib import contextmanager
import io
import os
from typing import cast

from fsspec import AbstractFileSystem
from torch.distributed.checkpoint._fsspec_filesystem import FileSystem as FsspecFileSystem
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX

from marinskyrl.remote_io import (
    CommittableStream,
    OutputStream,
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


class AbortableFsspecFileSystem(FsspecFileSystem):
    """Open DCP object streams with commit and abort semantics."""

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
