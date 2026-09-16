# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and execute Python tools authored in the serving dashboard."""

import ast
import asyncio
import contextlib
import dataclasses
import inspect
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

MAX_PYTHON_TOOL_SOURCE_BYTES = 64 * 1024
_PYTHON_TOOL_FILENAME = "<chat-python-tools>"
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


@dataclass(frozen=True)
class PythonToolRequest:
    """One serialized request from the dashboard to a tool subprocess."""

    source: str
    name: str
    arguments: dict[str, object]

    def to_json_bytes(self) -> bytes:
        return json.dumps(dataclasses.asdict(self)).encode()

    @classmethod
    def from_payload(cls, payload: object, *, name: str | None = None) -> "PythonToolRequest":
        if not isinstance(payload, dict):
            raise ValueError("Tool request must be an object")
        source = payload.get("source")
        resolved_name = payload.get("name") if name is None else name
        arguments = payload.get("arguments")
        if not isinstance(source, str) or not isinstance(resolved_name, str) or not isinstance(arguments, dict):
            raise ValueError("Tool request requires string source and name, and object arguments")
        if len(source.encode()) > MAX_PYTHON_TOOL_SOURCE_BYTES:
            raise PythonToolSourceTooLarge(f"Python tool source exceeds {MAX_PYTHON_TOOL_SOURCE_BYTES} bytes")
        return cls(source=source, name=resolved_name, arguments=arguments)


class PythonToolSourceTooLarge(ValueError):
    """The submitted source exceeds the dashboard execution limit."""


def python_tools_from_source(source: str) -> tuple[PythonTool, ...]:
    """Compile top-level typed function definitions from dashboard-authored source."""
    module = ast.parse(source, filename=_PYTHON_TOOL_FILENAME)
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
    exec(compile(module, _PYTHON_TOOL_FILENAME, "exec"), namespace)
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


async def _invoke_tool(request: PythonToolRequest) -> bytes:
    with contextlib.redirect_stdout(sys.stderr):
        tools = {tool.name: tool for tool in python_tools_from_source(request.source)}
        tool = tools.get(request.name)
        if tool is None:
            raise ValueError(f"Unknown Python tool {request.name!r}")
        validated = tool.validate_arguments(request.arguments)
        result = tool.function(**validated)
        if inspect.isawaitable(result):
            result = await result
        return tool.serialize_result(result)


def _main() -> None:
    request = PythonToolRequest.from_payload(json.load(sys.stdin))
    result = asyncio.run(_invoke_tool(request))
    sys.stdout.buffer.write(result)


if __name__ == "__main__":
    _main()
