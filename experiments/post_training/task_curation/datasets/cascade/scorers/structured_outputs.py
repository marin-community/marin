# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import io
import json
import tomllib
from enum import StrEnum
from typing import Any, Dict, Optional, Tuple
import xmltodict
import yaml
from openapi_schema_validator import validate as validate_against_schema_openapi


class SchemaType(StrEnum):
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TOML = "toml"
    CSV = "csv"

class StructuredOutputsResourcesServer:
    def __init__(self, config):
        self.config = config

    def parse_content(self, schema_type: SchemaType, content: str):
        match schema_type.lower():
            case SchemaType.JSON:
                parsed = json.loads(content)
            case SchemaType.YAML:
                parsed = yaml.safe_load(content)
            case SchemaType.XML:
                parsed = xmltodict.parse(content)
            case SchemaType.TOML:
                parsed = tomllib.loads(content)
            case SchemaType.CSV:
                parsed = list(csv.DictReader(io.StringIO(content)))
            case _:
                parsed = None
        return parsed
    def strictify_schema(self, schema: Dict[str, Any]):
        """Make a schema strict as per OpenAPI guidelines"""
        if isinstance(schema, Dict):
            if "properties" in schema:
                schema["required"] = list(schema["properties"])
                schema["additionalProperties"] = False
            for k, v in schema.items():
                self.strictify_schema(v)
    def coerce_xml_types(self, data: Any, schema: Dict[str, Any]) -> Any:
        """Recursively coerce xmltodict string values to match the JSON schema types.

        xmltodict.parse() returns all leaf values as strings. This method walks the
        parsed data alongside the schema and converts values where possible.
        On conversion failure the original value is returned so that schema
        validation can report the error.
        """
        if not isinstance(schema, dict) or "type" not in schema:
            return data

        schema_type = schema["type"]

        if schema_type == "object" and isinstance(data, dict):
            properties = schema.get("properties", {})
            coerced = {}
            for key, value in data.items():
                if key in properties:
                    coerced[key] = self.coerce_xml_types(value, properties[key])
                else:
                    coerced[key] = value
            return coerced

        if schema_type == "array":
            items_schema = schema.get("items", {})
            # xmltodict represents repeated child elements as {"tagName": [values]},
            # e.g. <skills><string>a</string><string>b</string></skills> becomes
            # {"string": ["a", "b"]}. For single elements, xmltodict gives
            # {"string": "python"} instead of a list. In both cases, unwrap the
            # single-key dict since we're at an array schema position -- a dict here
            # is always the xmltodict wrapping artifact, not a meaningful structure.
            if isinstance(data, dict) and len(data) == 1:
                data = next(iter(data.values()))
            if not isinstance(data, list):
                data = [data] if data is not None else []
            return [self.coerce_xml_types(item, items_schema) for item in data]

        # xmltodict returns None for empty tags like <field/> or <field></field>.
        # Coerce to "" only for string types (parity with JSON/YAML where "" is valid).
        # Non-string types (integer, boolean, etc.) intentionally left as None so
        # they fail validation -- 0 and False are meaningful values, not "empty".
        if data is None and schema_type == "string":
            return ""

        if isinstance(data, str):
            try:
                if schema_type == "integer":
                    return int(data)
                if schema_type == "number":
                    return float(data)
                if schema_type == "boolean":
                    lower = data.lower()
                    if lower in ("true", "1"):
                        return True
                    if lower in ("false", "0"):
                        return False
            except (ValueError, AttributeError):
                pass

        return data
    def coerce_csv_types(self, rows: List[Dict[str, str]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Coerce CSV string values to match JSON Schema types.

        csv.DictReader returns all values as strings.  Walk rows alongside the
        items schema and convert values where possible.  On conversion failure
        the original string is kept so that schema validation reports the error.
        """
        items_schema = schema.get("items", schema)
        properties = items_schema.get("properties", {})

        coerced_rows = []
        for row in rows:
            coerced: Dict[str, Any] = {}
            for key, value in row.items():
                prop_schema = properties.get(key, {})
                prop_type = prop_schema.get("type", "string")
                coerced[key] = self._coerce_csv_scalar(value, prop_type)
            coerced_rows.append(coerced)
        return coerced_rows
    def _coerce_csv_scalar(self, value: str, target_type) -> Any:
        """Coerce a single CSV string to the target JSON Schema type."""
        if isinstance(target_type, list):
            if (value is None or value == "") and "null" in target_type:
                return None
            for t in target_type:
                if t == "null":
                    continue
                result = self._coerce_csv_scalar(value, t)
                if not isinstance(result, str) or t == "string":
                    return result
            return value

        if value is None or value == "":
            return value

        try:
            if target_type == "integer":
                return int(value)
            if target_type == "number":
                return float(value)
            if target_type == "boolean":
                lower = value.lower()
                if lower in ("true", "1"):
                    return True
                if lower in ("false", "0"):
                    return False
        except (ValueError, AttributeError):
            pass
        return value
    def evaluate_structured_object_response(
        self, schema_str: str, response_obj: Any
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Returns (reward, error_type, error_message) for already-parsed tool-call arguments."""
        try:
            schema = json.loads(schema_str)
        except Exception as e:
            return 0.0, "schema_error", str(e)[:200]

        self.strictify_schema(schema)

        try:
            validate_against_schema_openapi(response_obj, schema)
            return 1.0, None, None
        except Exception as e:
            return 0.0, "validation_error", f"{type(e).__name__}: {str(e)[:200]}"
    def evaluate_structured_output_response(
        self, schema_type: SchemaType, schema_str: str, response_text: str
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Returns (reward, error_type, error_message)."""
        if not response_text or not response_text.strip():
            return 0.0, "empty_response", "No assistant response text"

        try:
            schema = json.loads(schema_str)
        except Exception as e:
            return 0.0, "schema_error", str(e)[:200]

        self.strictify_schema(schema)

        try:
            response_obj = self.parse_content(schema_type, response_text)
        except Exception as e:
            return 0.0, "parse_error", f"{type(e).__name__}: {str(e)[:200]}"

        try:
            if schema_type == SchemaType.XML and self.config.xml_coerce_types:
                response_obj = self.coerce_xml_types(response_obj, schema)
            if schema_type == SchemaType.CSV:
                response_obj = self.coerce_csv_types(response_obj, schema)
            validate_against_schema_openapi(response_obj, schema)
            return 1.0, None, None
        except Exception as e:
            return 0.0, "validation_error", f"{type(e).__name__}: {str(e)[:200]}"
