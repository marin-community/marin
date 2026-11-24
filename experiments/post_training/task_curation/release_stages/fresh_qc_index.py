# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fresh-QC-index stage: the graph's producer of the release's staged QC panel index.

The release's fresh QC index is a graph output bound to the release's target load and
snapshot (ruling 2026-11-24), not a hand-produced pin. One panel entry per browse-staged
source: the panel's replayable staged document is exported from the prior load's review
tables (the assessments, populations and items rows ride only on staged documents, and
the raw-review and batch rows are re-staged from them), compressed
byte-deterministically under the release prefix, and sealed into the index the catalog
load replays. The exported document carries the browse-copied review receipt verbatim,
so the load's pointer verification binds the replay to what the browse verified.
"""

import argparse
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages import paced_query
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

INDEX_KIND_SUFFIX = "fresh-qc-index"
STAGED_DIR = "staged-v1"


@dataclass(frozen=True)
class FreshQcIndexInputs:
    """Typed inputs for the release's fresh QC index; every value is a graph output."""

    release_id: str
    prior_load_id: str
    target_load_id: str
    snapshot_id: str
    source_revision: str
    output_root: str
    service_url: str
    applet_id: str
    declarations: ArtifactRef
    qc_completion: ArtifactRef

    def validate(self) -> None:
        """Refuse malformed inputs before anything runs."""
        if not self.release_id or not self.prior_load_id or not self.target_load_id or not self.snapshot_id:
            raise ValueError("release_id, load ids and snapshot are required")
        if not self.source_revision or not self.output_root:
            raise ValueError("source_revision and output_root are required")
        if not self.service_url or not self.applet_id:
            raise ValueError("the served applet endpoint is required")


def _read_declarations(ref: ArtifactRef) -> dict[str, dict]:
    """Digest-verify the frozen declarations document the panels bind."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"declarations digest mismatch: {ref.uri}")
    document = json.loads(payload)
    if not isinstance(document, dict) or not document:
        raise ValueError("declarations document is empty or malformed")
    return document


def _completion_rows(ref: ArtifactRef) -> list[dict]:
    """The browse completion's staged panels."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"browse completion digest mismatch: {ref.uri}")
    completion = json.loads(payload)
    if completion.get("status") != "PASS":
        raise ValueError(f"the browse completion is {completion.get('status')!r}; the index needs a PASS")
    rows = completion.get("results") or []
    if not rows:
        raise ValueError("the browse completion carries no staged panels")
    return rows


def _export_panel(inputs: FreshQcIndexInputs, row: dict, declarations: dict[str, dict], query, *, write: bool) -> dict:
    """Export one panel's replayable staged document from the prior load's review tables."""
    from infra.marina.apps.rl_data_catalog.server.evidence_content import (  # noqa: PLC0415 — the maintained codec
        unpack_content,
    )
    from infra.marina.apps.rl_data_catalog.server.restrictions import (  # noqa: PLC0415 — the maintained restriction
        declaration_restriction,
    )

    source_id = row["source_id"]
    declaration = declarations.get(source_id)
    if declaration is None:
        raise ValueError(f"no frozen declaration for staged panel source: {source_id}")
    key = row["pipeline_key"]
    bound = {"load": inputs.prior_load_id, "key": key}
    assessments = query(
        "SELECT pipeline_key,source_id,scope,model,rubric,curation,curation_uri,traces"
        " FROM atlas_assessments WHERE load_id=:load AND pipeline_key=:key",
        bound,
    )["rows"]
    if len(assessments) != 1:
        raise ValueError(f"{len(assessments)} assessments for the staged panel {source_id}")
    populations = query(
        "SELECT pipeline_key,cohort,population,sample_count"
        " FROM atlas_populations WHERE load_id=:load AND pipeline_key=:key ORDER BY cohort",
        bound,
    )["rows"]
    items = query(
        "SELECT pipeline_key,task_id,cohort,status,verdict,explanation,evidence_ids,packet_uri,has_task,"
        "omissions,exclusion_reason,confidence,reference_status,defects,provenance,replaced_task_sha256,"
        "repair_advisories FROM atlas_items WHERE load_id=:load AND pipeline_key=:key ORDER BY task_id",
        bound,
    )["rows"]
    raw_rows = query(
        "SELECT pipeline_key,task_id,record,reviewer"
        " FROM atlas_raw_reviews WHERE load_id=:load AND pipeline_key=:key ORDER BY task_id",
        bound,
    )["rows"]
    batch_rows = query(
        "SELECT pipeline_key,batch_key,record,chunk_count"
        " FROM atlas_review_batches WHERE load_id=:load AND pipeline_key=:key ORDER BY batch_key",
        bound,
    )["rows"]
    chunk_rows = query(
        "SELECT batch_key,chunk_index,payload"
        " FROM atlas_review_batch_chunks WHERE load_id=:load AND pipeline_key=:key ORDER BY batch_key,chunk_index",
        bound,
    )["rows"]
    links = query(
        "SELECT pipeline_key,task_id,batch_key"
        " FROM atlas_review_batch_tasks WHERE load_id=:load AND pipeline_key=:key ORDER BY task_id",
        bound,
    )["rows"]
    chunks_by_batch: dict[str, list[str]] = {}
    for chunk in chunk_rows:
        chunks_by_batch.setdefault(chunk["batch_key"], []).append(chunk["payload"])
    batches = []
    for batch in batch_rows:
        record = dict(batch["record"])
        if batch["chunk_count"]:
            record["payload"] = "".join(chunks_by_batch.get(batch["batch_key"], []))
            record.pop("payload_chunks", None)
        batches.append(
            {
                "pipeline_key": batch["pipeline_key"],
                "batch_key": batch["batch_key"],
                "record": record,
                "chunk_count": batch["chunk_count"],
            }
        )
    raw = [{**r, "record": unpack_content(r["record"])} for r in raw_rows]
    if len(raw) != len(items):
        raise ValueError(f"the staged panel export for {source_id} covers {len(items)} items and {len(raw)} raw rows")
    staged = {
        "data_restriction": declaration_restriction(declaration).value,
        "assessment": assessments[0],
        "populations": populations,
        "items": items,
        "raw": raw,
        "batches": batches,
        "batch_links": links,
        "receipt": row["original_receipt"],
    }
    payload = gzip.compress(
        json.dumps(staged, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(), mtime=0
    )
    identity = hashlib.sha256(json.dumps(row["original_receipt"], sort_keys=True).encode()).hexdigest()
    reference = {"uri": row["original_receipt"]["artifact"]["uri"], "sha256": identity}
    if write:
        path = StoragePath(f"{inputs.output_root}/{STAGED_DIR}/{identity}.json.gz")
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"staged panel {identity} exists with different content; use a new output root")
        else:
            path.write_bytes(payload, auto_mkdir=True)
        if path.read_bytes() != payload:
            raise ValueError(f"staged panel readback mismatch: {identity}")
        reference = {"uri": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
    return {
        "source_id": source_id,
        "identity": identity,
        "status": "complete",
        "records": len(items),
        "bytes": len(payload),
        "reference": reference,
    }


def run_fresh_qc_index(
    inputs: FreshQcIndexInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
) -> dict:
    """Seal the release's fresh QC index; ``dry_run`` verifies every panel with zero writes."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "fresh_qc_index", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        declarations = _read_declarations(inputs.declarations)
        rows = _completion_rows(inputs.qc_completion)
        query = paced_query.open_paced_query(inputs.service_url, inputs.applet_id, paced_query.RetryControl())
        index_rows = [_export_panel(inputs, row, declarations, query, write=not dry_run) for row in rows]
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                "fresh_qc_index",
                {"mode": mode, "panels": len(index_rows)},
                alert_feeds,
            )
            return {
                "kind": f"atlas-{inputs.release_id}-{INDEX_KIND_SUFFIX}",
                "release_id": inputs.release_id,
                "load_id": inputs.target_load_id,
                "snapshot_id": inputs.snapshot_id,
                "panels": len(index_rows),
                "dry_run": True,
                "sql_writes": 0,
            }
        document = {
            "kind": f"atlas-{inputs.release_id}-{INDEX_KIND_SUFFIX}",
            "release_id": inputs.release_id,
            "load_id": inputs.target_load_id,
            "snapshot_id": inputs.snapshot_id,
            "source_revision": inputs.source_revision,
            "basis": (
                "The browse-staged SAMPLE QC panels at their frozen declarations' selected review evidence, "
                f"staged by this release at load {inputs.target_load_id}"
            ),
            "receipts": index_rows,
            "provenance": {
                "browse_completion": {"uri": inputs.qc_completion.uri, "sha256": inputs.qc_completion.sha256},
                "declarations": {"uri": inputs.declarations.uri, "sha256": inputs.declarations.sha256},
            },
        }
        raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        path = StoragePath(f"{inputs.output_root}/fresh-qc-index.json")
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError(f"fresh QC index at {path} exists with different content; use a new output root")
        else:
            path.write_bytes(raw, auto_mkdir=True)
        if path.read_bytes() != raw:
            raise ValueError("fresh QC index readback mismatch")
        reference = {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        alerts.stage_finished(
            inputs.release_id,
            "fresh_qc_index",
            {"mode": mode, "panels": len(index_rows), **reference},
            alert_feeds,
        )
        return {
            "kind": document["kind"],
            "release_id": inputs.release_id,
            "load_id": inputs.target_load_id,
            "snapshot_id": inputs.snapshot_id,
            "panels": len(index_rows),
            "index": reference,
            "sql_writes": 0,
        }
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "fresh_qc_index", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON FreshQcIndexInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    inputs = FreshQcIndexInputs(**json.loads(args.inputs.read_text()))
    proof = run_fresh_qc_index(inputs, dry_run=args.mode == "dry-run")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
