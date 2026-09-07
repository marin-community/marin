# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine-readable operations exported by checked-in Marina applications."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from marina.apps import RegisteredApi

MARINA_EXTENSION = "x-marina"
JSON_MEDIA_TYPE = "application/json"
HTTP_METHODS = frozenset({"delete", "get", "head", "options", "patch", "post", "put", "trace"})
OPERATION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class OperationRisk(StrEnum):
    """The maximum side effect an operation may have."""

    READ = "read"
    WRITE = "write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class OperationExtension(BaseModel):
    """Marina metadata attached to an OpenAPI operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: bool = False
    risk: OperationRisk | None = None

    @model_validator(mode="after")
    def require_risk_for_agent_operation(self) -> OperationExtension:
        if self.agent and self.risk is None:
            raise ValueError("agent-visible operations must declare risk")
        return self


class OperationDescriptor(BaseModel):
    """One app operation in the stable registry wire format."""

    model_config = ConfigDict(frozen=True)

    id: str
    app: str
    method: str
    path: str
    summary: str
    description: str
    risk: OperationRisk
    input_schema: dict[str, object]
    output_schema: dict[str, object]


class OperationList(BaseModel):
    operations: list[OperationDescriptor]


def operation_extension(risk: OperationRisk) -> dict[str, object]:
    """Return FastAPI ``openapi_extra`` for an agent-visible operation."""
    extension = OperationExtension(agent=True, risk=risk)
    return {MARINA_EXTENSION: extension.model_dump(mode="json", exclude_none=True)}


def _object(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _schemas(document: Mapping[str, object]) -> Mapping[str, object]:
    components = _object(document.get("components", {}), "OpenAPI components")
    return _object(components.get("schemas", {}), "OpenAPI component schemas")


def _expanded_schema(
    value: object,
    schemas: Mapping[str, object],
    resolving: frozenset[str] = frozenset(),
) -> object:
    if isinstance(value, list):
        return [_expanded_schema(item, schemas, resolving) for item in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
        name = reference.rsplit("/", 1)[-1]
        if name in resolving:
            return dict(value)
        target = _object(schemas.get(name), f"OpenAPI schema {name!r}")
        expanded = _expanded_schema(target, schemas, resolving | {name})
        assert isinstance(expanded, dict)
        return {**expanded, **{key: item for key, item in value.items() if key != "$ref"}}
    return {key: _expanded_schema(item, schemas, resolving) for key, item in value.items()}


def _parameters(
    path_item: Mapping[str, object],
    operation: Mapping[str, object],
) -> list[Mapping[str, object]]:
    path_parameters = path_item.get("parameters", [])
    operation_parameters = operation.get("parameters", [])
    if not isinstance(path_parameters, list) or not isinstance(operation_parameters, list):
        raise ValueError("OpenAPI parameters must be arrays")
    return [_object(value, "OpenAPI parameter") for value in (*path_parameters, *operation_parameters)]


def _input_schema(
    path_item: Mapping[str, object],
    operation: Mapping[str, object],
    schemas: Mapping[str, object],
) -> dict[str, object]:
    properties: dict[str, object] = {}
    required: list[str] = []
    for parameter in _parameters(path_item, operation):
        location = parameter.get("in")
        if location not in {"path", "query"}:
            raise ValueError(f"unsupported OpenAPI parameter location {location!r}")
        name = parameter.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("OpenAPI parameter name must be a non-empty string")
        if name in properties:
            raise ValueError(f"duplicate OpenAPI input {name!r}")
        schema = _object(parameter.get("schema"), f"OpenAPI parameter {name!r} schema")
        expanded = _expanded_schema(schema, schemas)
        assert isinstance(expanded, dict)
        description = parameter.get("description")
        if isinstance(description, str) and description:
            expanded.setdefault("description", description)
        properties[name] = expanded
        if parameter.get("required") is True:
            required.append(name)

    request_body = operation.get("requestBody")
    if request_body is not None:
        body = _object(request_body, "OpenAPI request body")
        content = _object(body.get("content"), "OpenAPI request body content")
        media = _object(content.get(JSON_MEDIA_TYPE), f"OpenAPI {JSON_MEDIA_TYPE} request body")
        expanded = _expanded_schema(media.get("schema"), schemas)
        body_schema = _object(expanded, f"OpenAPI {JSON_MEDIA_TYPE} request schema")
        body_properties = body_schema.get("properties")
        if body_schema.get("type") != "object" or not isinstance(body_properties, dict):
            raise ValueError("agent-visible JSON request bodies must be objects with named properties")
        for name, schema in body_properties.items():
            if name in properties:
                raise ValueError(f"duplicate OpenAPI input {name!r}")
            properties[name] = schema
        body_required = body_schema.get("required", [])
        if not isinstance(body_required, list) or any(not isinstance(name, str) for name in body_required):
            raise ValueError("OpenAPI request schema required must be an array of strings")
        required.extend(body_required)

    result: dict[str, object] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


def _output_schema(operation: Mapping[str, object], schemas: Mapping[str, object]) -> dict[str, object]:
    responses = _object(operation.get("responses"), "OpenAPI responses")
    for status, value in responses.items():
        if not str(status).startswith("2"):
            continue
        response = _object(value, f"OpenAPI response {status}")
        content = _object(response.get("content"), f"OpenAPI response {status} content")
        media = _object(content.get(JSON_MEDIA_TYPE), f"OpenAPI response {status} {JSON_MEDIA_TYPE}")
        expanded = _expanded_schema(media.get("schema"), schemas)
        return dict(_object(expanded, f"OpenAPI response {status} schema"))
    raise ValueError(f"agent-visible operations must declare a {JSON_MEDIA_TYPE} success response")


def operation_catalog(apis: Mapping[str, RegisteredApi]) -> tuple[OperationDescriptor, ...]:
    """Return every agent-visible application operation in stable id order."""
    result: list[OperationDescriptor] = []
    ids: set[str] = set()
    for app, registered in sorted(apis.items()):
        document = registered.openapi
        schemas = _schemas(document)
        paths = _object(document.get("paths"), f"OpenAPI paths for app {app!r}")
        for path, raw_path_item in paths.items():
            path_item = _object(raw_path_item, f"OpenAPI path {path!r}")
            for method, raw_operation in path_item.items():
                if method not in HTTP_METHODS:
                    continue
                operation = _object(raw_operation, f"OpenAPI operation {method.upper()} {path}")
                raw_extension = operation.get(MARINA_EXTENSION)
                if raw_extension is None:
                    continue
                extension = OperationExtension.model_validate(raw_extension)
                if not extension.agent:
                    continue
                assert extension.risk is not None
                operation_id = operation.get("operationId")
                if not isinstance(operation_id, str) or OPERATION_ID_PATTERN.fullmatch(operation_id) is None:
                    raise ValueError(
                        f"agent-visible operation {method.upper()} {path} needs an operationId matching "
                        f"{OPERATION_ID_PATTERN.pattern}"
                    )
                qualified_id = f"{app}.{operation_id}"
                if qualified_id in ids:
                    raise ValueError(f"duplicate Marina operation id {qualified_id!r}")
                ids.add(qualified_id)
                summary = operation.get("summary")
                description = operation.get("description")
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError(f"agent-visible operation {qualified_id!r} needs a summary")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(f"agent-visible operation {qualified_id!r} needs a description")
                result.append(
                    OperationDescriptor(
                        id=qualified_id,
                        app=app,
                        method=method.upper(),
                        path=str(path),
                        summary=summary,
                        description=description,
                        risk=extension.risk,
                        input_schema=_input_schema(path_item, operation, schemas),
                        output_schema=_output_schema(operation, schemas),
                    )
                )
    return tuple(sorted(result, key=lambda item: item.id))
