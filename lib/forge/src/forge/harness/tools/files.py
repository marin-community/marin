# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from tarfile import TarError
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.resources.env import Env, ReadFile, WriteFile


@dataclass(frozen=True, slots=True)
class ReadArgs:
    path: str
    base64: bool = False
    offset: int = 0
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class WriteArgs:
    path: str
    content: str
    base64: bool = False


@contextmanager
def tool_path_errors(path: str) -> Iterator[None]:
    """Translate user-selected path failures at the tool boundary."""
    try:
        yield
    except IsADirectoryError as error:
        raise ValueError(f"Cannot access {path}: path is a directory") from error
    except (FileNotFoundError, KeyError) as error:
        raise ValueError(f"File not found: {path}") from error
    except OSError as error:
        raise ValueError(f"Cannot access {path}: {error}") from error
    except TarError as error:
        raise ValueError(f"Cannot read {path}: archive extraction failed: {error}") from error


async def read_tool_file(ctx: Context, path: str, *, offset: int = 0, limit: int | None = None) -> bytes:
    with tool_path_errors(path):
        return await ctx.emit(ReadFile(PurePosixPath(path), offset=offset, limit=limit))


@dataclass(frozen=True, slots=True)
class ReadFileTool(ValidatedTool[ReadArgs]):
    name = "read_file"
    description = (
        "Read a byte range as UTF-8 text or base64. offset and limit count file bytes, before decoding; "
        "use base64 for exact binary slices."
    )
    args_type = ReadArgs

    @override
    async def run(self, ctx: Context, args: ReadArgs, env: Env) -> ToolResult:
        try:
            if args.offset < 0 or (args.limit is not None and args.limit < 0):
                return ToolResult("offset and limit must be nonnegative")
            data = await read_tool_file(ctx, args.path, offset=args.offset, limit=args.limit)
        except ValueError as error:
            return ToolResult(str(error))
        text = base64.b64encode(data).decode() if args.base64 else data.decode(errors="replace")
        return ToolResult(text)


@dataclass(frozen=True, slots=True)
class WriteFileTool(ValidatedTool[WriteArgs]):
    name = "write_file"
    description = "Write UTF-8 text, or base64-decoded binary data, to a file."
    args_type = WriteArgs

    @override
    async def run(self, ctx: Context, args: WriteArgs, env: Env) -> ToolResult:
        try:
            data = base64.b64decode(args.content, validate=True) if args.base64 else args.content.encode()
        except ValueError as error:
            return ToolResult(f"Invalid base64: {error}")
        try:
            with tool_path_errors(args.path):
                await ctx.emit(WriteFile(PurePosixPath(args.path), data))
        except ValueError as error:
            return ToolResult(str(error))
        return ToolResult("File written")
