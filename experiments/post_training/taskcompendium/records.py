# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Catalog, ledger, and public-row projections for TaskCompendium ingestion."""

from enum import StrEnum

from taskcompendium.models import TaskSpec
from taskcompendium.submission import PlainText, render_instruction


class Disposition(StrEnum):
    """One input row's terminal ingestion decision."""

    IMPORTED = "imported"
    DUPLICATE = "duplicate"
    OUT_OF_SCOPE = "out-of-scope"
    REJECTED = "rejected"


def catalog_record(
    specification: TaskSpec,
    *,
    source: str,
    path: str,
    route: str,
    mode: str,
    family: str,
    converter: str,
    template_id: str,
    archive_sha256: str,
    source_metadata_json: str,
) -> dict:
    """Return one private catalog row with ordered source tags and TaskSpec JSON."""
    return {
        "id": specification.id,
        "source": source,
        "path": path,
        "route": route,
        "mode": mode,
        "family": family,
        "converter": converter,
        "template_id": template_id,
        "tags": list(specification.tags),
        "archive_sha256": archive_sha256,
        "source_metadata_json": source_metadata_json,
        "specification_json": specification.model_dump_json(),
    }


def public_task_record(
    specification: TaskSpec,
    *,
    family: str,
) -> dict:
    """Return the typed PublicTask-v1 field allowlist, without verifier material."""
    return {
        "record_version": 1,
        "id": specification.id,
        "context": specification.context.model_dump(mode="json"),
        "environment_requirements": specification.environment_requirements.model_dump(mode="json"),
        "tool_providers": {key: value.model_dump(mode="json") for key, value in specification.tool_providers.items()},
        "final_tools": [tool.model_dump(mode="json") for tool in specification.final_tools],
        "answer_type": specification.answer_type.value,
        "source": specification.source.model_dump(mode="json"),
        "submission_instruction": render_instruction(specification, PlainText(id="plain")),
        "tags": list(specification.tags),
        "source_category": family,
    }


def ledger_record(
    *,
    input_split: str,
    input_file: str,
    input_row: int,
    source: str | None,
    path: str | None,
    route: str,
    mode: str | None,
    family: str | None,
    converter: str | None,
    template_id: str | None,
    tags: list[str],
    input_object_pin: str,
    archive_sha256: str | None,
    disposition: Disposition,
    imported_id: str | None = None,
    reason: str | None = None,
) -> dict:
    """Return a stable ledger row for exactly one source Parquet row."""
    return {
        "input_split": input_split,
        "input_file": input_file,
        "input_row": input_row,
        "source": source,
        "path": path,
        "route": route,
        "mode": mode,
        "family": family,
        "converter": converter,
        "template_id": template_id,
        "tags": list(tags),
        "input_object_pin": input_object_pin,
        "archive_sha256": archive_sha256,
        "disposition": disposition.value,
        "imported_id": imported_id,
        "reason": reason,
    }
