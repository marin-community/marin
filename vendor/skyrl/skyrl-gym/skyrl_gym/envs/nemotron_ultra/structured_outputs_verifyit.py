"""Client framing for the pinned OpenAPI structured-output contract."""

from __future__ import annotations

import json
from typing import Any

from jsonschema.validators import validates, validator_for
from openapi_schema_validator import OAS32Validator
from referencing import Registry

from .structured_outputs import _coerce_csv, _coerce_xml, _parse, _tool_payload

_DIALECT = "urn:marin:skyrl:openapi-0.9-local-only"
_ORIGINAL_DIALECT = "x-marin-original-schema-dialect"


def _original_schema(schema: dict[str, Any]) -> dict[str, Any]:
    original = dict(schema)
    dialect = original.pop(_ORIGINAL_DIALECT)
    if dialect is None:
        original.pop("$schema", None)
    else:
        original["$schema"] = dialect
    return original


@validates(_DIALECT)
class LocalOpenAPIValidator:
    """Register source-owned validator construction with the existing schema mode."""

    META_SCHEMA = dict(OAS32Validator.META_SCHEMA, **{"$id": _DIALECT})
    ID_OF = staticmethod(OAS32Validator.ID_OF)

    @staticmethod
    def check_schema(schema: dict[str, Any]) -> None:
        OAS32Validator.check_schema(_original_schema(schema))

    def __new__(cls, schema: dict[str, Any]):
        original = _original_schema(schema)
        return OAS32Validator(original, registry=Registry()).evolve(schema=original)


def grade_structured_output_verifyit(
    text: str, record: dict[str, Any], assistant_message: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Extract using source helpers; delegate schema correctness to verifyit."""
    try:
        from verifyit.grade import InvalidTask, Status
        from verifyit.modes.grade_json_schema import grade_json_schema_candidate

    except ImportError:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": "verifyit schema dependency unavailable",
        }

    try:
        schema = json.loads(record["schema_str"])
        if not isinstance(schema, dict) or _ORIGINAL_DIALECT in schema:
            raise ValueError("Expected a schema object without reserved client fields")
        if "$schema" in schema:
            selected = validator_for(schema, default=None)
            if selected is None or selected.__module__ not in {
                "jsonschema.validators",
                "openapi_schema_validator.validators",
            }:
                raise ValueError("Unsupported schema dialect")
        OAS32Validator.check_schema(schema)
    except Exception as error:
        return 0.0, {"error_type": "schema_error", "error_message": str(error)[:200]}

    try:
        if record.get("response_mode", "text") == "tool_call":
            value, error_type, message = _tool_payload(record, assistant_message)
            if error_type is not None:
                return 0.0, {"error_type": error_type, "error_message": message}
        else:
            if not text.strip():
                return 0.0, {
                    "error_type": "empty_response",
                    "error_message": "No assistant response text",
                }
            schema_type = str(record.get("schema_type", "json")).lower()
            try:
                value = _parse(schema_type, text)
            except Exception as error:
                return 0.0, {
                    "error_type": "parse_error",
                    "error_message": f"{type(error).__name__}: {str(error)[:200]}",
                }
            if schema_type == "xml":
                value = _coerce_xml(value, schema)
            if schema_type == "csv":
                value = _coerce_csv(value, schema)
        framed = dict(schema)
        framed[_ORIGINAL_DIALECT] = framed.get("$schema")
        framed["$schema"] = _DIALECT
        verdict = grade_json_schema_candidate(framed, value)
        if verdict.status is not Status.SCORED:
            return 0.0, {
                "error_type": "verification_error",
                "error_message": "Unscored schema verdict",
            }
        if verdict.reward == 1.0:
            return 1.0, {"error_type": None, "error_message": None}
        return 0.0, {
            "error_type": "validation_error",
            "error_message": str(verdict.detail.get("error", verdict.detail.get("reason")))[:200],
        }
    except InvalidTask:
        return 0.0, {
            "error_type": "schema_error",
            "error_message": "Invalid verifier schema",
        }
    except Exception as error:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": f"{type(error).__name__}: {str(error)[:200]}",
        }
