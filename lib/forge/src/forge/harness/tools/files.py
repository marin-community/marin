# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.resources.env import ReadFile, WriteFile


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


@dataclass(frozen=True, slots=True)
class ReadFileTool(ValidatedTool[ReadArgs]):
    name = "read_file"
    description = (
        "Read a byte range as UTF-8 text or base64. offset and limit count file bytes, before decoding; "
        "use base64 for exact binary slices."
    )
    args_type = ReadArgs

    @override
    async def run(self, ctx: Context, args: ReadArgs) -> ToolResult:
        if args.offset < 0 or (args.limit is not None and args.limit < 0):
            return ToolResult("offset and limit must be nonnegative")
        data = await ctx.emit(ReadFile(PurePosixPath(args.path), offset=args.offset, limit=args.limit))
        return ToolResult(base64.b64encode(data).decode() if args.base64 else data.decode(errors="replace"))


@dataclass(frozen=True, slots=True)
class WriteFileTool(ValidatedTool[WriteArgs]):
    name = "write_file"
    description = "Write UTF-8 text, or base64-decoded binary data, to a file."
    args_type = WriteArgs

    @override
    async def run(self, ctx: Context, args: WriteArgs) -> ToolResult:
        try:
            data = base64.b64decode(args.content, validate=True) if args.base64 else args.content.encode()
        except ValueError as error:
            return ToolResult(f"Invalid base64: {error}")
        await ctx.emit(WriteFile(PurePosixPath(args.path), data))
        return ToolResult("File written")
