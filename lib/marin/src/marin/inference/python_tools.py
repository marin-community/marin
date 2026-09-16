# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and execute Python tools authored in the serving dashboard."""

import ast
import asyncio
import contextlib
import inspect
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

MAX_PYTHON_TOOL_SOURCE_BYTES = 64 * 1024
_MAX_TOOL_NAME_LENGTH = 64
_TOOL_NAME_PATTERN = re.compile(rf"^[A-Za-z0-9_-]{{1,{_MAX_TOOL_NAME_LENGTH}}}$")
_SUPPORTED_PARAMETER_KINDS = frozenset(
    {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
)


@dataclass(frozen=True)
class PythonTool:
    """A typed Python function with runtime argument and result validation."""

    name: str
    function: Callable[..., object]
    arguments_model: type[BaseModel]
    result_adapter: TypeAdapter[Any]

    def validate_arguments(self, arguments: object) -> dict[str, object]:
        parsed = self.arguments_model.model_validate(arguments)
        return {name: getattr(parsed, name) for name in self.arguments_model.model_fields}

    def serialize_result(self, result: object) -> bytes:
        return self.result_adapter.dump_json(self.result_adapter.validate_python(result))


def python_tools_from_source(source: str) -> tuple[PythonTool, ...]:
    """Compile top-level typed function definitions from dashboard-authored source."""
    if len(source.encode()) > MAX_PYTHON_TOOL_SOURCE_BYTES:
        raise ValueError(f"Python tool source exceeds {MAX_PYTHON_TOOL_SOURCE_BYTES} bytes")

    module = ast.parse(source, filename="<chat-python-tools>")
    definitions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for statement in module.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError("Python tool source may contain only top-level function definitions")
        if statement.decorator_list:
            raise ValueError(f"Python tool function {statement.name!r} may not use decorators")
        definitions.append(statement)
    if not definitions:
        raise ValueError("Python tool source must define at least one function")

    namespace: dict[str, object] = {"__name__": "__chat_python_tools__"}
    exec(compile(module, "<chat-python-tools>", "exec"), namespace)
    functions = tuple(namespace[definition.name] for definition in definitions)
    tools = tuple(_python_tool(function) for function in functions)
    names = [tool.name for tool in tools]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(f"Python tool names must be unique; repeated: {', '.join(duplicates)}")
    return tools


def _python_tool(function: object) -> PythonTool:
    if not inspect.isfunction(function):
        raise ValueError("Python tools must be functions")
    name = function.__name__
    if not _TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Python tool {name!r} must have a 1-{_MAX_TOOL_NAME_LENGTH} character name containing only "
            "letters, numbers, '_' or '-'"
        )

    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    unsupported = [parameter.name for parameter in parameters if parameter.kind not in _SUPPORTED_PARAMETER_KINDS]
    if unsupported:
        raise ValueError(f"Python tool {name!r} has unsupported parameters: {', '.join(unsupported)}")

    try:
        annotations = get_type_hints(function, include_extras=True)
    except (NameError, TypeError) as exc:
        raise ValueError(f"Could not resolve type annotations for Python tool {name!r}: {exc}") from exc
    missing_annotations = [parameter.name for parameter in parameters if parameter.name not in annotations]
    if missing_annotations:
        raise ValueError(f"Python tool {name!r} must annotate: {', '.join(missing_annotations)}")
    if "return" not in annotations:
        raise ValueError(f"Python tool {name!r} must annotate its return value")

    fields: dict[str, tuple[Any, Any]] = {}
    for parameter in parameters:
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotations[parameter.name], default)
    arguments_model = create_model(
        f"{name}_arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return PythonTool(
        name=name,
        function=function,
        arguments_model=arguments_model,
        result_adapter=TypeAdapter(annotations["return"]),
    )


async def _invoke_tool(source: str, name: str, arguments: object) -> bytes:
    with contextlib.redirect_stdout(sys.stderr):
        tools = {tool.name: tool for tool in python_tools_from_source(source)}
        tool = tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown Python tool {name!r}")
        validated = tool.validate_arguments(arguments)
        result = tool.function(**validated)
        if inspect.isawaitable(result):
            result = await result
        return tool.serialize_result(result)


def _main() -> int:
    try:
        payload = json.load(sys.stdin)
        source = payload["source"]
        name = payload["name"]
        arguments = payload["arguments"]
        if not isinstance(source, str) or not isinstance(name, str):
            raise ValueError("Tool source and name must be strings")
        result = asyncio.run(_invoke_tool(source, name, arguments))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
