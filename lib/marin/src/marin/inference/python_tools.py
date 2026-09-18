# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and execute Python tools authored in the serving dashboard."""

import ast
import dataclasses
import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import shellsim
from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

MAX_PYTHON_TOOL_SOURCE_BYTES = 64 * 1024
_PYTHON_TOOL_FILENAME = "<chat-python-tools>"
_SHELLSIM_PROGRAM_PATH = "/work/marin-tool.py"
_SHELLSIM_RESULT_PATH = "/work/.marin-tool-result.json"
_SHELLSIM_CPU_LIMIT = 10_000_000
_SHELLSIM_MEMORY_LIMIT = 64 * 1024 * 1024
_SHELLSIM_DISK_LIMIT = 4 * 1024 * 1024
_SHELLSIM_OUTPUT_LIMIT = 1024 * 1024
_SCALAR_ANNOTATIONS: dict[str, Any] = {
    "bool": bool,
    "float": float,
    "int": int,
    "object": object,
    "str": str,
}


class PythonToolOperation(StrEnum):
    """Operations supported by the isolated Python-tool worker."""

    DEFINITIONS = "definitions"
    INVOKE = "invoke"


@dataclass(frozen=True)
class PythonTool:
    """A typed Python function contract derived without executing its source."""

    name: str
    description: str | None
    arguments_model: type[BaseModel]
    result_adapter: TypeAdapter[Any]

    def definition(self) -> dict[str, object]:
        parameters = self.arguments_model.model_json_schema()
        parameters.pop("title", None)
        function: dict[str, object] = {
            "name": self.name,
            "parameters": parameters,
        }
        if self.description:
            function["description"] = self.description
        return {"type": "function", "function": function}

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


@dataclass(frozen=True)
class PythonToolDefinitionsRequest:
    """Python source submitted for conversion to model-facing tool definitions."""

    source: str

    def to_json_bytes(self) -> bytes:
        return json.dumps(dataclasses.asdict(self)).encode()

    @classmethod
    def from_payload(cls, payload: object) -> "PythonToolDefinitionsRequest":
        if not isinstance(payload, dict) or not isinstance(payload.get("source"), str):
            raise ValueError("Tool definitions request requires string source")
        source = payload["source"]
        if len(source.encode()) > MAX_PYTHON_TOOL_SOURCE_BYTES:
            raise PythonToolSourceTooLarge(f"Python tool source exceeds {MAX_PYTHON_TOOL_SOURCE_BYTES} bytes")
        return cls(source=source)


class PythonToolSourceTooLarge(ValueError):
    """The submitted source exceeds the dashboard execution limit."""


def python_tools_from_source(source: str) -> tuple[PythonTool, ...]:
    """Parse top-level typed tool contracts without executing dashboard-authored source."""
    module = ast.parse(source, filename=_PYTHON_TOOL_FILENAME)
    definitions: list[ast.FunctionDef] = []
    for statement in module.body:
        if isinstance(statement, ast.AsyncFunctionDef):
            raise ValueError(f"ShellSim Python tool {statement.name!r} must be synchronous")
        if not isinstance(statement, ast.FunctionDef):
            raise ValueError("Python tool source may contain only top-level function definitions")
        if statement.decorator_list:
            raise ValueError(f"Python tool function {statement.name!r} may not use decorators")
        definitions.append(statement)
    if not definitions:
        raise ValueError("Python tool source must define at least one function")

    tools = tuple(_python_tool(definition) for definition in definitions)
    names = [tool.name for tool in tools]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(f"Python tool names must be unique; repeated: {', '.join(duplicates)}")
    return tools


def python_tool_definitions(source: str) -> list[dict[str, object]]:
    return [tool.definition() for tool in python_tools_from_source(source)]


def _python_tool(definition: ast.FunctionDef) -> PythonTool:
    name = definition.name
    positional_only = [parameter.arg for parameter in definition.args.posonlyargs]
    keyword_only = [parameter.arg for parameter in definition.args.kwonlyargs]
    variadic = [parameter.arg for parameter in (definition.args.vararg, definition.args.kwarg) if parameter is not None]
    unsupported = positional_only + keyword_only + variadic
    if unsupported:
        raise ValueError(f"ShellSim Python tool {name!r} has unsupported parameters: {', '.join(unsupported)}")

    parameters = definition.args.args
    missing_annotations = [parameter.arg for parameter in parameters if parameter.annotation is None]
    if missing_annotations:
        raise ValueError(f"Python tool {name!r} must annotate: {', '.join(missing_annotations)}")
    if definition.returns is None:
        raise ValueError(f"Python tool {name!r} must annotate its return value")

    defaults = [None] * (len(parameters) - len(definition.args.defaults)) + list(definition.args.defaults)
    fields: dict[str, tuple[Any, Any]] = {}
    for parameter, default_node in zip(parameters, defaults, strict=True):
        default = ... if default_node is None else _literal_default(name, parameter.arg, default_node)
        assert parameter.annotation is not None
        fields[parameter.arg] = (_annotation_type(parameter.annotation), default)
    arguments_model = create_model(
        f"{name}_arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return PythonTool(
        name=name,
        description=ast.get_docstring(definition, clean=True),
        arguments_model=arguments_model,
        result_adapter=TypeAdapter(_annotation_type(definition.returns)),
    )


def _annotation_type(annotation: ast.expr) -> Any:
    if isinstance(annotation, ast.Name) and annotation.id in _SCALAR_ANNOTATIONS:
        return _SCALAR_ANNOTATIONS[annotation.id]
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return type(None)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_type(annotation.left) | _annotation_type(annotation.right)
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        if annotation.value.id == "list":
            return list[_annotation_type(annotation.slice)]
        if annotation.value.id == "dict" and isinstance(annotation.slice, ast.Tuple) and len(annotation.slice.elts) == 2:
            key_node, value_node = annotation.slice.elts
            key_type = _annotation_type(key_node)
            if key_type is not str:
                raise ValueError("Python tool dictionaries must use string keys")
            return dict[str, _annotation_type(value_node)]
    raise ValueError(f"Unsupported Python tool annotation: {ast.unparse(annotation)}")


def _literal_default(tool_name: str, parameter_name: str, node: ast.expr) -> object:
    try:
        value = ast.literal_eval(node)
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"ShellSim Python tool {tool_name!r} parameter {parameter_name!r} must use a JSON-literal default"
        ) from exc
    return value


def _shellsim_source(request: PythonToolRequest, tool: PythonTool, validated: dict[str, object]) -> str:
    arguments = ", ".join(f"{name}=_marin_arguments[{name!r}]" for name in tool.arguments_model.model_fields)
    arguments_json = json.dumps(json.dumps(validated))
    return f"""import json as _marin_json

_marin_open = open

{request.source}

_marin_arguments = _marin_json.loads({arguments_json})
_marin_result = {request.name}({arguments})
with _marin_open({_SHELLSIM_RESULT_PATH!r}, "w") as _marin_output:
    _marin_output.write(_marin_json.dumps(_marin_result))
"""


def _invoke_tool(request: PythonToolRequest) -> bytes:
    tools = {tool.name: tool for tool in python_tools_from_source(request.source)}
    tool = tools.get(request.name)
    if tool is None:
        raise ValueError(f"Unknown Python tool {request.name!r}")
    validated = tool.validate_arguments(request.arguments)

    environment = shellsim.Environment(
        cpu=_SHELLSIM_CPU_LIMIT,
        memory=_SHELLSIM_MEMORY_LIMIT,
        disk=_SHELLSIM_DISK_LIMIT,
        output=_SHELLSIM_OUTPUT_LIMIT,
    )
    environment.write_file(_SHELLSIM_PROGRAM_PATH, _shellsim_source(request, tool, validated))
    result = environment.run(f"python3.14 {_SHELLSIM_PROGRAM_PATH}")
    if result.stdout:
        sys.stderr.buffer.write(result.stdout)
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
    if result.returncode != 0:
        reason = f" ({result.stop_reason})" if result.stop_reason else ""
        raise ValueError(f"ShellSim Python tool exited with status {result.returncode}{reason}")
    return tool.serialize_result(json.loads(environment.read_file(_SHELLSIM_RESULT_PATH)))


def _main() -> None:
    if len(sys.argv) != 2:
        raise ValueError(f"Expected one Python tool operation, got {sys.argv[1:]!r}")
    operation = PythonToolOperation(sys.argv[1])
    match operation:
        case PythonToolOperation.DEFINITIONS:
            request = PythonToolDefinitionsRequest.from_payload(json.load(sys.stdin))
            definitions = python_tool_definitions(request.source)
            sys.stdout.buffer.write(json.dumps(definitions).encode())
        case PythonToolOperation.INVOKE:
            request = PythonToolRequest.from_payload(json.load(sys.stdin))
            result = _invoke_tool(request)
            sys.stdout.buffer.write(result)


if __name__ == "__main__":
    _main()
