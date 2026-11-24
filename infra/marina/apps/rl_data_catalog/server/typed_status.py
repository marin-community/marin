# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Refuse served catalog rows whose typed status fields leave their enums.

The catalog load validates declarations on construction; this is the serve-time
guard on the JSONB the selected load carries. A row whose Status, PendingStage,
ExclusionReason, Use or Method value falls outside its enum is
served-catalog drift, and the projection refuses it instead of rendering free
text on the Atlas pages.
"""

from enum import StrEnum
from typing import Any, TypeVar

from finestore.rl.curation import SourceUse
from finestore.rl.status_taxonomy import ExclusionReason, PendingStage, SourceMethod, SourceStatus

EnumT = TypeVar("EnumT", bound=StrEnum)


def _enum_value(cls: type[EnumT], value: Any, field: str, source_id: str) -> str:
    try:
        return cls(value).value
    except ValueError:
        raise ValueError(f"Untyped {field} value {value!r} on {source_id} is outside {cls.__name__}") from None


def require_typed_served_declaration(declaration: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize one served declaration's typed status fields against the enums.

    Every typed field the Atlas renders must be an enum member; the guard returns
    the declaration with canonical values and raises on anything else. Absent and
    null fields pass: stage presence on Pending rows is the load contract, not a
    render concern.
    """
    result = dict(declaration)
    source_id = result.get("id", "<unnamed>")
    if result.get("status") is not None:
        result["status"] = _enum_value(SourceStatus, result["status"], "status", source_id)
    if result.get("stage") is not None:
        result["stage"] = _enum_value(PendingStage, result["stage"], "stage", source_id)
    if result.get("intended_use") is not None:
        result["intended_use"] = _enum_value(SourceUse, result["intended_use"], "use", source_id)
    if result.get("method_tags") is not None:
        result["method_tags"] = [_enum_value(SourceMethod, tag, "method", source_id) for tag in result["method_tags"]]
    result["exclusion_claims"] = [
        {**claim, "reason": _enum_value(ExclusionReason, claim.get("reason"), "reason", source_id)}
        for claim in result.get("exclusion_claims") or []
    ]
    return result
