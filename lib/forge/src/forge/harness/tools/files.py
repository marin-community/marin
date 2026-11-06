# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
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


async def read_tool_file(ctx: Context, path: str) -> bytes:
    """Translate user-selected path failures at the tool boundary."""
    try:
        return await ctx.emit(ReadFile(PurePosixPath(path)))
    except IsADirectoryError as error:
        raise ValueError(f"Cannot read {path}: path is a directory") from error
    except (FileNotFoundError, KeyError) as error:
        raise ValueError(f"File not found: {path}") from error
    except TarError as error:
        raise ValueError(f"Cannot read {path}: archive extraction failed: {error}") from error


@dataclass(frozen=True, slots=True)
class ReadFileTool(ValidatedTool[ReadArgs]):
    name = "read_file"
    description = "Read UTF-8 text or base64. Use offset and limit to select a slice of the returned characters."
    args_type = ReadArgs

    @override
    async def run(self, ctx: Context, args: ReadArgs, env: Env) -> ToolResult:
        try:
            if args.offset < 0 or (args.limit is not None and args.limit < 0):
                return ToolResult("offset and limit must be nonnegative")
            data = await read_tool_file(ctx, args.path)
        except ValueError as error:
            return ToolResult(str(error))
        text = base64.b64encode(data).decode() if args.base64 else data.decode(errors="replace")
        end = None if args.limit is None else args.offset + args.limit
        return ToolResult(text[args.offset : end])


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
        await ctx.emit(WriteFile(PurePosixPath(args.path), data))
        return ToolResult("File written")
