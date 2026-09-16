# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expose typed Python functions as OpenAI-compatible chat tools."""

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SUPPORTED_PARAMETER_KINDS = frozenset(
    {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
)


@dataclass(frozen=True)
class OpenAIFunctionDefinition:
    """Function fields in an OpenAI tool definition."""

    name: str
    parameters: dict[str, object]
    description: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "parameters": self.parameters}
        if self.description is not None:
            result["description"] = self.description
        return result


@dataclass(frozen=True)
class OpenAIToolDefinition:
    """OpenAI tool definition serialized into model requests."""

    function: OpenAIFunctionDefinition
    type: Literal["function"] = "function"

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "function": self.function.to_dict()}


@dataclass(frozen=True)
class PythonTool:
    """A validated Python callable and its model-facing function definition."""

    name: str
    function: Callable[..., object]
    arguments_model: type[BaseModel]
    result_adapter: TypeAdapter[Any]
    definition: OpenAIToolDefinition

    def validate_arguments(self, arguments: object) -> dict[str, object]:
        """Validate JSON arguments and retain the annotated Python value types."""
        parsed = self.arguments_model.model_validate(arguments)
        return {name: getattr(parsed, name) for name in self.arguments_model.model_fields}

    def serialize_result(self, result: object) -> bytes:
        """Validate and JSON-encode one tool result for the next model turn."""
        return self.result_adapter.dump_json(self.result_adapter.validate_python(result))


def python_tools(functions: tuple[Callable[..., object], ...]) -> tuple[PythonTool, ...]:
    """Build tool schemas and argument validators for Python functions."""
    tools = tuple(_python_tool(function) for function in functions)
    names = [tool.name for tool in tools]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(f"Tool function names must be unique; repeated: {', '.join(duplicates)}")
    return tools


def _python_tool(function: Callable[..., object]) -> PythonTool:
    name = getattr(function, "__name__", "")
    if not _TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Tool function {function!r} must have a 1-64 character name containing only letters, numbers, '_' or '-'"
        )

    signature = inspect.signature(function)
    parameters = tuple(signature.parameters.values())
    unsupported = [parameter.name for parameter in parameters if parameter.kind not in _SUPPORTED_PARAMETER_KINDS]
    if unsupported:
        raise ValueError(f"Tool function {name!r} has unsupported parameters: {', '.join(unsupported)}")

    try:
        annotations = get_type_hints(function, include_extras=True)
    except (NameError, TypeError) as exc:
        raise ValueError(f"Could not resolve type annotations for tool function {name!r}: {exc}") from exc

    missing_annotations = [parameter.name for parameter in parameters if parameter.name not in annotations]
    if missing_annotations:
        raise ValueError(
            f"Tool function {name!r} must annotate every parameter; missing: {', '.join(missing_annotations)}"
        )

    fields: dict[str, tuple[Any, Any]] = {}
    for parameter in parameters:
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotations[parameter.name], default)
    arguments_model = create_model(
        f"{name}_arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    parameters = arguments_model.model_json_schema()
    parameters.pop("title", None)

    description = inspect.getdoc(function)
    return PythonTool(
        name=name,
        function=function,
        arguments_model=arguments_model,
        result_adapter=TypeAdapter(annotations.get("return", Any)),
        definition=OpenAIToolDefinition(
            function=OpenAIFunctionDefinition(name=name, parameters=parameters, description=description)
        ),
    )
