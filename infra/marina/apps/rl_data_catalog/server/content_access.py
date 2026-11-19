# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Withhold task content without changing a source's stored data tier or admission."""

from typing import Any

from finestore.rl.licenses import LicenseDecisionKind, LicenseReceipt
from finestore.rl.status_taxonomy import ExclusionReason

from .licensing import declaration_license_ban
from .restrictions import (
    declaration_restriction,
    is_private,
    private_content_marker,
    private_declaration_projection,
)

LICENSE_CONTENT_MARKER = "Content withheld: this source has a license exclusion."


def license_content_excluded(declaration: dict[str, Any]) -> bool:
    """Read canonical bans and explicit license decisions, never general status."""
    if declaration_license_ban(declaration) is not None:
        return True
    if any(
        claim.get("reason") in (ExclusionReason.LICENSE_BAN.value, ExclusionReason.LICENSE_MISSING.value)
        for claim in declaration.get("exclusion_claims") or []
    ):
        return True
    raw = declaration.get("license_receipt")
    if raw is None:
        return False
    receipt = LicenseReceipt.model_validate(raw)
    return any(decision.kind is LicenseDecisionKind.BAN or not decision.approved for decision in receipt.decisions)


def content_is_withheld(declaration: dict[str, Any]) -> bool:
    return is_private(declaration) or license_content_excluded(declaration)


def content_marker(declaration: dict[str, Any]) -> dict[str, Any]:
    """Describe withheld content while preserving the independent private-data tier."""
    if is_private(declaration):
        return private_content_marker()
    if not license_content_excluded(declaration):
        raise ValueError("Content marker requested for an unrestricted source")
    return {
        "data_restriction": declaration_restriction(declaration).value,
        "content_withheld": True,
        "withheld_reason": "license-excluded",
        "content": None,
        "marker": LICENSE_CONTENT_MARKER,
    }


def withheld_declaration_projection(declaration: dict[str, Any]) -> dict[str, Any]:
    """Keep source identity/count metadata and the license reason, never task prose."""
    projected = private_declaration_projection(declaration)
    if raw_receipt := declaration.get("license_receipt"):
        receipt = LicenseReceipt.model_validate(raw_receipt).model_dump(mode="json")
        for evidence in receipt["evidence"]:
            evidence["quoted_lines"] = []
        for source in [receipt["resolved_source"], *receipt["wrapper_chain"]]:
            if source is not None:
                source["declaration"]["quoted_lines"] = []
        projected["license_receipt"] = receipt
    if is_private(declaration):
        return projected
    projected.update(content_withheld=True, withheld_reason="license-excluded", notes=LICENSE_CONTENT_MARKER)
    if ban := declaration_license_ban(declaration):
        projected["license_ban"] = ban.model_dump(mode="json")
    return projected


def fingerprint_metadata(fingerprint: dict[str, Any] | None) -> dict[str, Any] | None:
    if fingerprint is None:
        return None
    return {**fingerprint, "sample": []}


def backend_metadata(capabilities: dict[str, Any] | None) -> dict[str, Any] | None:
    if capabilities is None:
        return None
    return {
        "source_id": capabilities.get("source_id"),
        "backends": {name: {"support": fit["support"]} for name, fit in (capabilities.get("backends") or {}).items()},
    }
