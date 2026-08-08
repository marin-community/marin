# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decide a finelog deploy before it touches a host.

A finelog image registers two namespaces for itself — the privileged ``log``
namespace and ``telemetry_v1`` — and ``RegisterTable`` merges each against
whatever that deployment's catalog already holds. A merge the catalog rejects
wedges the namespace for as long as the image is deployed: the server listens,
``/health`` stays green, and every write to the namespace fails.

The decision needs only two schemas, and both are cheap to reach: the one the
candidate image requests, and the one the catalog holds. This module captures
the second from a running server, hands both to the first, and turns the
verdicts into a rollout gate.

The check itself runs *inside the candidate image* (``docker run`` against the
pinned digest), because the image's built-in schema and the merge rules that
judge it both live in the binary that is about to ship. Nothing here reimplements
those rules.

Scope is the two server-owned namespaces. Namespaces a client registers
(``iris.worker``, zephyr's tables) belong to that client's schema, and the
report names them as unchecked rather than passing over them silently.
"""

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.schema import Schema

# The registered side is written in the catalog JSON sidecar form the server
# already parses (`schema_from_json` in `rust/src/store/schema.rs`), not the
# proto wire form: the reader is the server binary, so the document speaks the
# store's own vocabulary. Column types are proto enum NAMES, which survive
# renumbering.


def schema_to_catalog_json(schema: Schema) -> dict[str, object]:
    """Encode ``schema`` in the catalog JSON form ``check-schema`` parses.

    The grouped-extrema keys are the store's (``json_column`` / ``json_key``),
    not the proto's (``group_json_column`` / ``group_json_key``); the wire names
    the roles more explicitly than the struct does.
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


def registered_schema_document(
    *,
    deployment: str,
    namespaces: Mapping[str, Schema],
    captured_at: str,
    captured_from: str,
) -> dict[str, object]:
    """Build the document ``check-schema`` reads and the deploy golden records.

    ``captured_from`` says which catalog these schemas came from — a golden is
    only as good as its provenance, and the server ignores everything but
    ``namespaces``, so the metadata exists for the reader of the diff.
    """
    return {
        "deployment": deployment,
        "captured_at": captured_at,
        "captured_from": captured_from,
        "namespaces": {name: schema_to_catalog_json(schema) for name, schema in sorted(namespaces.items())},
    }


def render_document(document: Mapping[str, object]) -> str:
    """Serialize a document for a checked-in golden: stable order, one trailing newline."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


class Outcome(StrEnum):
    """What the pre-flight decided for one deployment."""

    PASS = "pass"
    FAIL = "fail"
    # No registered schema was reachable — a live server that did not answer and
    # no recorded golden. A first deploy has no catalog to conflict with, so this
    # does not block a rollout; it is reported loudly instead.
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
        lines.append(
            f"no registered schema for: {', '.join(sorted(unknown))} (nothing to conflict with, or unreachable)"
        )
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
        # `--network=none`: the subcommand touches nothing outside its stdin, and
        # this makes that structural rather than a claim.
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
