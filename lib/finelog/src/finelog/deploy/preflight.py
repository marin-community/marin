# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decide a finelog deploy before it touches a host.

A finelog image registers ``log`` and ``telemetry_v1`` for itself, and
``RegisterTable`` merges each against whatever that deployment's catalog holds.
A merge the catalog rejects wedges the namespace for as long as the image is
deployed: the server listens, ``/health`` stays green, and every write fails.

This module captures the registered side from a running server, runs the
decision *inside the candidate image* (``docker run`` against the pinned
digest), and turns the verdicts into a rollout gate. The image's built-in schema
and the merge rules that judge it both live in the binary about to ship; nothing
here reimplements them.

Scope is the two server-owned namespaces. A namespace a client registers
(``iris.worker``, zephyr's tables) is reported as unchecked.
"""

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from finelog.client.log_client import LOG_NAMESPACE
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.schema import Schema

# The namespaces a finelog image registers for itself, and so the only ones it
# can decide. Everything else in a catalog belongs to the client that registered
# it.
SERVER_OWNED_NAMESPACES = (LOG_NAMESPACE, "telemetry_v1")


def schema_to_catalog_json(schema: Schema) -> dict[str, object]:
    """Encode ``schema`` in the catalog JSON form ``check-schema`` parses.

    That is `schema_from_json` in `rust/src/store/schema.rs`, not the proto wire
    form: column types are enum NAMES, and the grouped-extrema keys are the
    store's (``json_column`` / ``json_key``) rather than the proto's
    (``group_json_column`` / ``group_json_key``).
    """
    return {
        "key_column": schema.key_column,
        "columns": [
            {
                "name": column.name,
                "type": stats_pb2.ColumnType.Name(column.type),
                "nullable": column.nullable,
                "index": {
                    "trigram": column.trigram_index,
                    "exact_values": list(column.exact_values),
                    "value_counts": column.value_counts,
                },
            }
            for column in schema.columns
        ],
        "projections": [
            {
                "name": projection.name,
                "predicate_column": projection.predicate_column,
                "predicate_values": list(projection.predicate_values),
                "columns": list(projection.columns),
            }
            for projection in schema.projections
        ],
        "grouped_extrema": [
            {
                "filter_column": config.filter_column,
                "json_column": config.group_json_column,
                "json_key": config.group_json_key,
                "extrema_column": config.extrema_column,
            }
            for config in schema.grouped_extrema
        ],
    }


class SchemaSource(StrEnum):
    """Where a document's registered schemas came from."""

    # A deployment's own catalog. Only this is evidence about that deployment.
    CATALOG = "catalog"
    # A finelog-server build's own schemas. These agree with any binary whose
    # schemas have not changed since, including one that conflicts with what a
    # catalog holds, so merging against them decides nothing.
    BINARY = "binary"


def registered_schema_document(
    *,
    deployment: str,
    namespaces: Mapping[str, Schema],
    captured_at: str,
    source: SchemaSource,
    captured_from: str,
) -> dict[str, object]:
    """Build the document ``check-schema`` reads and the deploy golden records.

    ``source`` decides whether the document is evidence about ``deployment``;
    ``captured_from`` says the same in prose. The server reads only
    ``namespaces``.
    """
    return {
        "deployment": deployment,
        "captured_at": captured_at,
        "source": source.value,
        "captured_from": captured_from,
        "namespaces": {name: schema_to_catalog_json(schema) for name, schema in sorted(namespaces.items())},
    }


def document_source(document: Mapping[str, object]) -> SchemaSource:
    """The provenance ``document`` records. A document with none never came from a catalog."""
    return SchemaSource(document.get("source", SchemaSource.BINARY))


def render_document(document: Mapping[str, object]) -> str:
    """Serialize a document for a checked-in golden: stable order, one trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


class Outcome(StrEnum):
    """What the pre-flight decided for one deployment."""

    PASS = "pass"
    FAIL = "fail"
    # Nothing this deployment's catalog holds was reachable: an unreachable
    # server and no golden captured from it.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PreflightResult:
    """One deployment's decision, and where its registered side came from."""

    deployment: str
    outcome: Outcome
    source: str
    report: str


def blocks_rollout(results: Sequence[PreflightResult]) -> bool:
    """Whether any deployment's schemas would fail to register."""
    return any(result.outcome is Outcome.FAIL for result in results)


def summarize(results: Sequence[PreflightResult]) -> str:
    """Render every deployment's decision, failures last so they end the output."""
    ordered = sorted(results, key=lambda result: (result.outcome is Outcome.FAIL, result.deployment))
    lines: list[str] = []
    for result in ordered:
        lines.append(f"== {result.deployment} ({result.outcome.value}, registered schema from {result.source}) ==")
        lines.append(result.report.rstrip("\n"))
    failed = [result.deployment for result in results if result.outcome is Outcome.FAIL]
    unknown = [result.deployment for result in results if result.outcome is Outcome.UNKNOWN]
    if unknown:
        lines.append(f"UNDECIDED, no catalog to decide against: {', '.join(sorted(unknown))}")
    lines.append(
        f"PREFLIGHT FAIL: {', '.join(sorted(failed))}"
        if failed
        else "PREFLIGHT PASS: every deployment accepts this image's schemas"
    )
    return "\n".join(lines) + "\n"


def check_image(image: str, document: Mapping[str, object], *, docker: str = "docker") -> tuple[bool, str]:
    """Run ``image``'s own pre-flight over ``document``; return (passed, report).

    ``passed`` is false when any server-owned namespace in ``document`` would
    fail to register; ``report`` is the image's per-namespace decision, ready to
    print.
    """
    result = subprocess.run(
        # `--network=none`: the subcommand reads stdin and nothing else.
        [docker, "run", "--rm", "-i", "--network=none", image, "finelog-server", "check-schema", "-"],
        input=render_document(document),
        capture_output=True,
        text=True,
        check=False,
    )
    report = result.stdout + result.stderr
    return result.returncode == 0, report


def load_golden(path: Path) -> dict[str, object] | None:
    """Read a recorded deploy golden, or ``None`` when none has been recorded."""
    if not path.is_file():
        return None
    return json.loads(path.read_text())
