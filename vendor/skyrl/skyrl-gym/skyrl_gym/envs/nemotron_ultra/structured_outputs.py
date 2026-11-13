# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured-output verifier ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import csv
import io
import json
import tomllib
from typing import Any

import xmltodict
import yaml
from openapi_schema_validator import validate as validate_openapi


def _coerce_xml(data: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(schema, dict) or "type" not in schema:
        return data
    schema_type = schema["type"]
    if schema_type == "object" and isinstance(data, dict):
        properties = schema.get("properties", {})
        return {key: _coerce_xml(value, properties[key]) if key in properties else value for key, value in data.items()}
    if schema_type == "array":
        if isinstance(data, dict) and len(data) == 1:
            data = next(iter(data.values()))
        if not isinstance(data, list):
            data = [data] if data is not None else []
        return [_coerce_xml(item, schema.get("items", {})) for item in data]
    if data is None and schema_type == "string":
        return ""
    if isinstance(data, str):
        try:
            if schema_type == "integer":
                return int(data)
            if schema_type == "number":
                return float(data)
            if schema_type == "boolean" and data.lower() in {"true", "1", "false", "0"}:
                return data.lower() in {"true", "1"}
        except (ValueError, AttributeError):
            pass
    return data


def _coerce_csv_scalar(value: str, target_type: Any) -> Any:
    if isinstance(target_type, list):
        if (value is None or value == "") and "null" in target_type:
            return None
        for item_type in target_type:
            if item_type == "null":
                continue
            result = _coerce_csv_scalar(value, item_type)
            if not isinstance(result, str) or item_type == "string":
                return result
        return value
    if value is None or value == "":
        return value
    try:
        if target_type == "integer":
            return int(value)
        if target_type == "number":
            return float(value)
        if target_type == "boolean" and value.lower() in {"true", "1", "false", "0"}:
            return value.lower() in {"true", "1"}
    except (ValueError, AttributeError):
        pass
    return value


def _coerce_csv(rows: list[dict[str, str]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    properties = schema.get("items", schema).get("properties", {})
    return [
        {key: _coerce_csv_scalar(value, properties.get(key, {}).get("type", "string")) for key, value in row.items()}
        for row in rows
    ]


def _parse(schema_type: str, content: str) -> Any:
    if schema_type == "json":
        return json.loads(content)
    if schema_type == "yaml":
        return yaml.safe_load(content)
    if schema_type == "xml":
        return xmltodict.parse(content)
    if schema_type == "toml":
        return tomllib.loads(content)
    if schema_type == "csv":
        return list(csv.DictReader(io.StringIO(content)))
    raise NotImplementedError(f"Unsupported schema type {schema_type!r}")


def _tool_payload(record: dict[str, Any], assistant_message: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return None, "missing_tool_call", "No function_call item found in assistant response"
    if len(calls) > 1:
        return None, "multiple_tool_calls", f"Expected exactly one function_call, got {len(calls)}"
    function = calls[0].get("function") or {}
    if record.get("tool_name") and function.get("name") != record["tool_name"]:
        return None, "wrong_tool_name", f"Expected tool {record['tool_name']!r}, got {function.get('name')!r}"
    arguments = function.get("arguments")
    try:
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError as error:
        return None, "tool_arguments_parse_error", f"{type(error).__name__}: {str(error)[:200]}"
    if not isinstance(arguments, dict):
        return None, "tool_arguments_parse_error", f"Unsupported arguments type: {type(arguments).__name__}"
    payload_key = record.get("tool_payload_key")
    if payload_key:
        if payload_key not in arguments:
            return None, "missing_tool_payload_key", f"Missing tool payload key: {payload_key}"
        arguments = arguments[payload_key]
    return arguments, None, None


def grade_structured_output(
    text: str,
    record: dict[str, Any],
    assistant_message: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Validate text or tool arguments against the row's supplied OpenAPI schema."""
    try:
        schema = json.loads(record["schema_str"])
    except Exception as error:
        return 0.0, {"error_type": "schema_error", "error_message": str(error)[:200]}

    if record.get("response_mode", "text") == "tool_call":
        value, error_type, message = _tool_payload(record, assistant_message)
        if error_type is not None:
            return 0.0, {"error_type": error_type, "error_message": message}
    else:
        if not text.strip():
            return 0.0, {"error_type": "empty_response", "error_message": "No assistant response text"}
        schema_type = str(record.get("schema_type", "json")).lower()
        try:
            value = _parse(schema_type, text)
        except Exception as error:
            return 0.0, {"error_type": "parse_error", "error_message": f"{type(error).__name__}: {str(error)[:200]}"}
        if schema_type == "xml":
            value = _coerce_xml(value, schema)
        if schema_type == "csv":
            value = _coerce_csv(value, schema)
    try:
        validate_openapi(value, schema)
    except Exception as error:
        return 0.0, {"error_type": "validation_error", "error_message": f"{type(error).__name__}: {str(error)[:200]}"}
    return 1.0, {"error_type": None, "error_message": None}
