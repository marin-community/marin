# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curriculum load: import the reviewed curriculum release and select it atomically.

One maintained module replaces the per-release ``*-curriculum-load-worker-v1.py``
copies (see INVENTORY.md). The r39 worker's shape is preserved — SHA-bound handoff,
text-scan and coverage references, resume cursor, atomic selection and readback
verification — but every release-identifying value arrives as typed
:class:`CurriculumLoadInputs`, and the dry run checks the typed target IDs against
the manifest before any write (DESIGN.md §8b).
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.history_staging import (
    DEFAULT_APPLET_ID,
    DEFAULT_SERVICE_URL,
    Query,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.release_stages.paced_query import RetryControl, open_paced_query
from infra.marina.apps.rl_data_catalog.load_curriculum import CATALOG_TEXT_CODEC, load_curriculum
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference, manifest_bytes

CURRICULUM_TABLES = ("sources", "tasks", "statistics", "confidence")


@dataclass(frozen=True)
class CurriculumLoadInputs:
    """Typed inputs for one curriculum-load run; every value is a graph output."""

    release_id: str
    prior_curriculum_load_id: str
    catalog_prior_load_id: str
    catalog_target_load_id: str
    source_revision: str
    receipt_root: str
    handoff: ArtifactRef
    text_scan: ArtifactRef
    expected_source_count: int
    expected_inventory_count: int
    expected_aligned_sources: int
    expected_task_count: int
    expected_rebound_source: str
    expected_rebound_tasks: int
    expected_confidence_rows: int
    service_url: str = DEFAULT_SERVICE_URL
    applet_id: str = DEFAULT_APPLET_ID

    def validate(self) -> None:
        """Refuse malformed inputs before any query or read runs."""
        for name in ("prior_curriculum_load_id", "catalog_prior_load_id", "catalog_target_load_id"):
            value = getattr(self, name)
            if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a 32-hex load ID, got {value!r}")
        if not self.release_id or not self.source_revision:
            raise ValueError("release_id and source_revision are required")
        if self.expected_source_count <= 0 or self.expected_task_count <= 0:
            raise ValueError("expected counts must be positive")


def read_ref(ref: ArtifactRef) -> dict:
    """Read and digest-verify one sealed JSON reference."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"sealed reference digest mismatch: {ref.uri}")
    return json.loads(payload)


def verify_handoff(inputs: CurriculumLoadInputs, handoff: dict) -> None:
    """The composition handoff must describe a complete, unchanged composition."""
    # The compose handoff's vocabulary: ``source_count`` counts the aligned (matched)
    # sources, ``inventory_count`` the catalog sources the plan declares.
    if handoff["source_count"] != inputs.expected_aligned_sources:
        raise ValueError(f"handoff aligns {handoff['source_count']} sources, expected {inputs.expected_aligned_sources}")
    if handoff["inventory_count"] not in {inputs.expected_inventory_count} & {inputs.expected_source_count}:
        raise ValueError(
            f"handoff inventories {handoff['inventory_count']} sources, expected {inputs.expected_inventory_count}"
            f" (plan: {inputs.expected_source_count})"
        )
    if handoff["task_count"] != inputs.expected_task_count:
        raise ValueError(f"handoff carries {handoff['task_count']} tasks, expected {inputs.expected_task_count}")
    unchanged = (
        handoff["all_retained_assignment_values_verified"],
        handoff["all_statistics_and_confidence_values_unchanged"],
        handoff["all_task_ids_unchanged"],
        handoff["all_public_inputs_unchanged"],
    )
    if not all(unchanged):
        raise ValueError(f"handoff reports changed composition values: {unchanged}")
    rebound_matches = handoff["rebound_source"] == inputs.expected_rebound_source and (
        handoff["rebound_tasks"] == inputs.expected_rebound_tasks
    )
    if not rebound_matches:
        raise ValueError(
            f"handoff rebound {handoff['rebound_source']!r} ({handoff['rebound_tasks']} tasks);"
            f" expected {inputs.expected_rebound_source!r} ({inputs.expected_rebound_tasks})"
        )
    if handoff["confidence_validated_rows"] != inputs.expected_confidence_rows:
        raise ValueError(
            f"handoff validated {handoff['confidence_validated_rows']} confidence rows,"
            f" expected {inputs.expected_confidence_rows}"
        )


def verify_text_scan(
    inputs: CurriculumLoadInputs, handoff: dict, handoff_ref: ArtifactRef, scan: dict, coverage: dict
) -> None:
    """The text scan must be a passing scan of exactly this handoff and coverage."""
    if scan["status"] != "PASS":
        raise ValueError(f"text scan status is {scan['status']!r}, expected PASS")
    if scan["handoff"] != handoff_ref.model_dump(mode="json"):
        raise ValueError("text scan covers a different handoff than this release")
    if scan["codec"] != CATALOG_TEXT_CODEC:
        raise ValueError(f"text scan codec {scan['codec']!r} differs from the loader codec {CATALOG_TEXT_CODEC!r}")
    if scan["artifacts"]["release"]["uri"] != handoff["release"]["uri"]:
        raise ValueError("text scan release artifact disagrees with the handoff")
    if scan["artifacts"]["coverage"]["uri"] != handoff["coverage"]["uri"]:
        raise ValueError("text scan coverage artifact disagrees with the handoff")
    if scan["artifacts"]["confidence"] != ArtifactRef(**handoff["confidence"]).model_dump(mode="json"):
        raise ValueError("text scan confidence artifact disagrees with the handoff")
    if scan["task_count"] != handoff["task_count"] or scan["source_count"] != handoff["source_count"]:
        raise ValueError("text scan counts disagree with the handoff")
    if scan["sample_task_count"] != coverage["sample_task_count"]:
        raise ValueError("text scan sample task count disagrees with the coverage")


def run_curriculum_load(
    inputs: CurriculumLoadInputs,
    *,
    dry_run: bool,
    query: Query | None = None,
    query_factory: Callable[[], Query] | None = None,
    loader: Callable[..., dict] | None = None,
    feeds: tuple[Path, ...] | None = None,
) -> dict:
    """Run one curriculum load; ``dry_run`` validates with zero SQL (DESIGN.md §8b)."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    alerts.stage_started(inputs.release_id, "curriculum_load", {"mode": "dry-run" if dry_run else "write"}, alert_feeds)
    try:
        inputs.validate()
        handoff_ref = inputs.handoff
        handoff = read_ref(handoff_ref)
        verify_handoff(inputs, handoff)
        coverage = read_ref(ArtifactRef(**handoff["coverage"]))
        scan_ref = ManifestReference(uri=inputs.text_scan.uri, sha256=inputs.text_scan.sha256)
        scan = json.loads(manifest_bytes(scan_ref))
        verify_text_scan(inputs, handoff, handoff_ref, scan, coverage)

        if dry_run:
            proof = {
                "kind": "curriculum-load-dry-run",
                "release_id": inputs.release_id,
                "prior_curriculum_load_id": inputs.prior_curriculum_load_id,
                "handoff": {"uri": inputs.handoff.uri, "sha256": inputs.handoff.sha256},
                "source_count": handoff["source_count"],
                "tasks": handoff["task_count"],
                "sql_writes": 0,
            }
            alerts.stage_finished(
                inputs.release_id,
                "curriculum_load",
                {"mode": "dry-run", "sources": handoff["source_count"]},
                alert_feeds,
            )
            return proof
        return _run_write(inputs, handoff, coverage, scan_ref, query, query_factory, loader, alert_feeds)
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "curriculum_load", {"error": str(error)[:500]}, alert_feeds)
        raise


def _run_write(inputs, handoff, coverage, scan_ref, query, query_factory, loader, alert_feeds):
    control = RetryControl()
    query = query if query is not None else open_paced_query(inputs.service_url, inputs.applet_id, control)
    query_factory = (
        query_factory
        if query_factory is not None
        else (lambda: open_paced_query(inputs.service_url, inputs.applet_id, control))
    )
    loader = loader if loader is not None else load_curriculum
    catalog_selection = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    allowed = ([{"load_id": inputs.catalog_prior_load_id}], [{"load_id": inputs.catalog_target_load_id}])
    if catalog_selection not in allowed:
        raise ValueError(f"another catalog load was selected: {catalog_selection!r}")
    curriculum_current = query("SELECT load_id,previous_load_id FROM curriculum_current WHERE name='latest'", {})["rows"]
    resume = curriculum_current[0]["load_id"] if curriculum_current else None
    if resume is not None and resume != inputs.prior_curriculum_load_id:
        cursor = query("SELECT load_id FROM curriculum_loads WHERE load_id=:load", {"load": resume})["rows"]
        if len(cursor) > 1:
            raise ValueError("ambiguous curriculum resume cursor")

    result = loader(
        handoff["release"]["uri"],
        handoff["coverage"]["uri"],
        query,
        confidence=ManifestReference.model_validate(handoff["confidence"]),
        text_scan=scan_ref,
        resume_load_id=resume,
    )
    if result["counts"]["tasks"] != coverage["sample_task_count"]:
        raise ValueError(f"curriculum load staged {result['counts']['tasks']} tasks, expected the sample count")

    selected = query("SELECT load_id,previous_load_id FROM curriculum_current WHERE name='latest'", {})["rows"]
    if selected[0]["previous_load_id"] != inputs.prior_curriculum_load_id:
        raise ValueError(
            f"curriculum selection chains from {selected[0]['previous_load_id']!r}, expected the prior load"
        )
    readback = {
        table: query(
            f"SELECT count(*) AS n FROM curriculum_{table} WHERE load_id=:load", {"load": selected[0]["load_id"]}
        )["rows"][0]["n"]
        for table in CURRICULUM_TABLES
    }
    aligned = query(
        "SELECT COUNT(*) AS n FROM curriculum_sources WHERE load_id=:load AND alignment->>'basis'='matched'",
        {"load": selected[0]["load_id"]},
    )["rows"][0]["n"]
    if readback["sources"] != inputs.expected_inventory_count or aligned != inputs.expected_aligned_sources:
        raise ValueError(f"curriculum readback {readback} with {aligned} matched sources disagrees with the handoff")
    if readback["tasks"] != coverage["sample_task_count"]:
        raise ValueError("curriculum readback task count disagrees with the coverage")
    catalog_selection_end = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    allowed_end = ([{"load_id": inputs.catalog_prior_load_id}], [{"load_id": inputs.catalog_target_load_id}])
    if catalog_selection_end not in allowed_end:
        raise ValueError(f"catalog selection moved during the curriculum load: {catalog_selection_end!r}")

    proof = {
        "kind": "curriculum-load",
        "release_id": inputs.release_id,
        "source_revision": inputs.source_revision,
        "handoff": {"uri": inputs.handoff.uri, "sha256": inputs.handoff.sha256},
        "prior_curriculum_load_id": inputs.prior_curriculum_load_id,
        "curriculum_load_id": selected[0]["load_id"],
        "readback": readback,
        "matched_sources": aligned,
        "result": result,
        "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    configure_coreweave_s3()
    receipt = _write_receipt(inputs, proof)
    alerts.stage_finished(
        inputs.release_id,
        "curriculum_load",
        {"mode": "write", "curriculum_load_id": selected[0]["load_id"], **receipt},
        alert_feeds,
    )
    return {**proof, "receipt": receipt}


def _write_receipt(inputs: CurriculumLoadInputs, proof: dict) -> dict:
    path = StoragePath(f"{inputs.receipt_root}/curriculum-load/receipt.json")
    if path.exists():
        raise ValueError(
            f"curriculum-load receipt already exists at {path}; bump the release prefix instead of overwriting"
        )
    raw = (json.dumps(proof, indent=2) + "\n").encode()
    path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError(f"receipt readback mismatch at {path}")
    return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON CurriculumLoadInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["handoff"] = ArtifactRef(**document["handoff"])
    document["text_scan"] = ArtifactRef(**document["text_scan"])
    inputs = CurriculumLoadInputs(**document)
    proof = run_curriculum_load(inputs, dry_run=args.mode == "dry-run")
    label = "CURRICULUM-LOAD-DRY-RUN" if args.mode == "dry-run" else "CURRICULUM-LOAD-SELECTED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
