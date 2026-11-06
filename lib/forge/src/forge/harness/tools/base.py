# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate model arguments before a tool performs effects."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import ClassVar, override

from pydantic import TypeAdapter, ValidationError

from forge.effects.context import Context
from forge.harness.protocol import FinishRequest, ReturnRequest, SubmitRequest, Tool, ToolResult
from forge.model.llm import ToolSchema
from forge.resources.env import Env

type Outcome = ToolResult | SubmitRequest | ReturnRequest[object] | FinishRequest


class ValidatedTool[A](Tool[object]):
    name: ClassVar[str]
    description: ClassVar[str]
    args_type: type[A]

    @property
    @override
    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, TypeAdapter(self.args_type).json_schema())

    @override
    async def __call__(self, ctx: Context, args: Mapping[str, object], env: Env) -> Outcome:
        try:
            parsed = TypeAdapter(self.args_type).validate_python(dict(args))
        except ValidationError as error:
            return ToolResult(f"Invalid arguments: {error}")
        return await self.run(ctx, parsed, env)

    @abstractmethod
    async def run(self, ctx: Context, args: A, env: Env) -> Outcome: ...
