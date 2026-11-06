# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.resources.env import Env, ReadFile, WriteFile


@dataclass(frozen=True, slots=True)
class ReadArgs:
    path: str
    base64: bool = False


@dataclass(frozen=True, slots=True)
class WriteArgs:
    path: str
    content: str
    base64: bool = False


@dataclass(frozen=True, slots=True)
class ReadFileTool(ValidatedTool[ReadArgs]):
    name = "read_file"
    description = "Read a file as UTF-8 text, or base64 for binary data."
    args_type = ReadArgs

    @override
    async def run(self, ctx: Context, args: ReadArgs, env: Env) -> ToolResult:
        try:
            data = await ctx.emit(ReadFile(PurePosixPath(args.path)))
        except (FileNotFoundError, KeyError) as error:
            return ToolResult(f"File not found: {error}")
        return ToolResult(base64.b64encode(data).decode() if args.base64 else data.decode(errors="replace"))


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
