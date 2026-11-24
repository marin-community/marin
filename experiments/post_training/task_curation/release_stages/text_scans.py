# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Text scans: preflight the composed curriculum through the catalog text codec.

One maintained module replaces the per-release regional text-scan workers and their
merger (see INVENTORY.md). Every release-identifying value arrives as typed
:class:`TextScanInputs`. Each regional scan resolves the composed handoff's release
and coverage documents through digest-verified references and walks every parquet
row of its region's assignment files through the catalog text codec: values needing
transport encoding are encoded, affected rows must serialize as JSON and round-trip
losslessly, and identity fields needing encoding are flagged unsupported. The merge
binds the regional receipts to one handoff and one codec, checks every release and
sample path was scanned exactly once, checks the unioned row counts against the
release and coverage totals, and seals the preflight receipt only from a full pass.

A dry run scans a bounded real slice (``max_files``) into ``slice_output_root`` and
seals nothing under the main output root; a dry-run merge verifies everything and
writes nothing. CLI: ``--inputs <json> --step scan --region <label>`` or
``--step merge``, with ``--mode dry-run|write``; the merge step reads an optional
``receipts`` list of ``{uri, sha256}`` references from the inputs document.
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.census import seal
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog.server import catalog_text
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

SCAN_STAGE = "text_scans"
MERGE_STAGE = "text_scans_merge"
REGIONAL_RECEIPT_SUFFIX = "regional-text-scan"
MERGED_RECEIPT_SUFFIX = "curriculum-catalog-text-preflight"
FULL_PHASE = "full"
SAMPLE_PHASE = "sample"
# Identity fields whose values must never need transport encoding; any hit is unsupported.
IDENTITIES = frozenset(
    "source_id task_id source_key content_hash task_hash node_id subject_id parent_id ancestor_id capability_id".split()
)
SCHEMA = pa.schema(
    [(name, pa.string()) for name in ("source_id", "task_id", "field", "kind", "sha256")]
    + [("row_index", pa.int64()), ("bytes", pa.int64())]
    + [(name, pa.bool_()) for name in ("nul", "non_text_control", "reserved_prefix", "identity_field")]
)
_SCAN_BATCH_ROWS = 256
_DIAGNOSTICS_FLUSH_ROWS = 10000


@dataclass(frozen=True)
class TextScanInputs:
    """Typed inputs for one text-scan release stage; every value is a graph output."""

    release_id: str
    source_revision: str
    handoff: ArtifactRef
    output_root: str
    slice_output_root: str | None
    regions: tuple[str, ...]
    region_prefixes: dict[str, str]
    max_files: int = 0

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision or not self.output_root:
            raise ValueError("release_id, source_revision and output_root are required")
        if not self.region_prefixes or set(self.region_prefixes) != set(self.regions):
            raise ValueError("region_prefixes must cover exactly the typed regions")
        for region, prefix in self.region_prefixes.items():
            if not region or not prefix:
                raise ValueError(f"region {region!r} needs a non-empty label and prefix")
            if not prefix.endswith("/"):
                raise ValueError(f"region {region!r} prefix must end with '/' so the pin check is unambiguous")
        if self.max_files < 0:
            raise ValueError(f"max_files must be zero (all files) or positive, got {self.max_files}")


def codec_sha256() -> str:
    """The digest of the local catalog text codec module; every receipt binds it."""
    return hashlib.sha256(Path(catalog_text.__file__).read_bytes()).hexdigest()


def read_reference(reference: dict) -> dict:
    """Digest-verify one sealed {uri, sha256} reference and parse its JSON."""
    payload = StoragePath(reference["uri"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"sealed reference digest mismatch: {reference['uri']}")
    return json.loads(payload)


def read_handoff(inputs: TextScanInputs) -> tuple[dict, dict, dict, dict]:
    """Digest-verify the composed handoff and its release/coverage/confidence documents."""
    handoff = read_reference({"uri": inputs.handoff.uri, "sha256": inputs.handoff.sha256})
    if handoff.get("release_id") != inputs.release_id:
        raise ValueError(f"composed handoff binds release {handoff.get('release_id')!r}, expected {inputs.release_id!r}")
    documents = [read_reference(handoff[key]) for key in ("release", "coverage", "confidence")]
    return handoff, *documents


def read_receipt(reference: ArtifactRef) -> dict:
    """Digest-verify one sealed regional receipt and parse it."""
    return read_reference({"uri": reference.uri, "sha256": reference.sha256})


def problems(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (RFC-6901 path, value) for every string or bytes value the codec must encode."""
    if isinstance(value, (str, bytes)):
        if catalog_text.needs_encoding(value):
            yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            key_path = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if isinstance(key, (str, bytes)) and catalog_text.needs_encoding(key):
                yield key_path + "/@key", key
            yield from problems(item, key_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from problems(item, path + "/" + str(index))


def region_files(paths: list, inputs: TextScanInputs, region: str) -> list[tuple[int, str]]:
    """The composed paths assigned to one region, as (document index, path) pairs.

    Every file must fall under exactly one typed region prefix; files bound for other
    regions are skipped here and covered by their own regional scan.
    """
    assigned: list[tuple[int, str]] = []
    for index, path in enumerate(paths):
        matches = [label for label, prefix in inputs.region_prefixes.items() if path.startswith(prefix)]
        if not matches:
            raise ValueError(f"assignment path {path} is outside every configured region prefix")
        if len(matches) > 1:
            raise ValueError(f"assignment path {path} matches several region prefixes: {matches}")
        if matches[0] == region:
            assigned.append((index, path))
    return assigned


def diagnostic_row(sid: str, task_id: Any, row_index: int, field: str, value: Any, identity: bool) -> dict:
    """One diagnostics parquet row describing a text field the codec had to encode."""
    encoded_value = catalog_text.EncodedCatalogText.encode(value)
    raw = value if isinstance(value, bytes) else value.encode("utf-8", "surrogatepass")
    return {
        "source_id": sid,
        "task_id": task_id,
        "row_index": row_index,
        "field": field,
        "kind": encoded_value.kind.value,
        "sha256": encoded_value.sha256,
        "bytes": len(raw),
        "nul": b"\0" in value if isinstance(value, bytes) else "\0" in value,
        "non_text_control": isinstance(value, bytes) or bool(catalog_text.NON_TEXT.search(value)),
        "reserved_prefix": isinstance(value, str) and value.startswith(catalog_text.PREFIX),
        "identity_field": identity,
    }


def aggregate_results(results: list[dict]) -> tuple[dict, dict, Counter, int]:
    """Union per-file scan results into per-phase row/bad counters, field counts, unsupported total."""
    totals: dict[str, Counter[str]] = {FULL_PHASE: Counter(), SAMPLE_PHASE: Counter()}
    bad: dict[str, Counter[str]] = {FULL_PHASE: Counter(), SAMPLE_PHASE: Counter()}
    fields: Counter[tuple[str, str]] = Counter()
    unsupported = 0
    for result in results:
        totals[result["phase"]].update(result["rows"])
        bad[result["phase"]].update(result["bad_rows"])
        unsupported += result["unsupported_identity_fields"]
        for field in result["fields"]:
            fields[(field["source_id"], field["field"])] += field["count"]
    return totals, bad, fields, unsupported


def scan_file(job: tuple[str, int, str, str]) -> dict:
    """Scan one assignment parquet file through the codec; the maintained reference port."""
    configure_coreweave_s3()
    pa.set_cpu_count(1)
    phase, index, path, output_root = job
    source_rows: Counter[str] = Counter()
    source_bad: Counter[str] = Counter()
    field_counts: Counter[tuple[str, str]] = Counter()
    diagnostics: list[dict] = []
    pending: list[dict] = []
    writer: pq.ParquetWriter | None = None
    output: Any = None
    unsupported = 0
    row_index = -1

    with ExitStack() as outputs:

        def flush() -> None:
            nonlocal writer, output, pending
            if not pending:
                return
            if writer is None:
                output = StoragePath(output_root) / f"diagnostics/{phase}-{index:05d}.parquet"
                assert not output.exists(), str(output)
                output_stream = outputs.enter_context(output.open("wb"))
                writer = outputs.enter_context(pq.ParquetWriter(output_stream, SCHEMA, compression="zstd"))
            writer.write_table(pa.Table.from_pylist(pending, schema=SCHEMA))
            pending = []

        with StoragePath(path).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=_SCAN_BATCH_ROWS):
                for row in batch.to_pylist():
                    row_index += 1
                    sid = canonical_source_id(row["source_id"])
                    source_rows[sid] += 1
                    found = list(problems(row))
                    encoded = catalog_text.encode_catalog_value(row)
                    json.dumps(encoded, ensure_ascii=False, allow_nan=False)
                    if not found:
                        continue
                    assert catalog_text.decode_catalog_value(encoded) == row, (sid, row["task_id"])
                    source_bad[sid] += 1
                    for field, value in found:
                        identity = field.rsplit("/", 1)[-1] in IDENTITIES
                        unsupported += identity
                        field_counts[(sid, field)] += 1
                        pending.append(diagnostic_row(sid, row["task_id"], row_index, field, value, identity))
                    if len(pending) >= _DIAGNOSTICS_FLUSH_ROWS:
                        flush()
        flush()
    if writer is not None:
        diagnostics.append({"uri": str(output), "sha256": file_sha256(output)})
    return {
        "phase": phase,
        "input": path,
        "rows": dict(source_rows),
        "bad_rows": dict(source_bad),
        "fields": [{"source_id": key[0], "field": key[1], "count": count} for key, count in field_counts.items()],
        "unsupported_identity_fields": unsupported,
        "diagnostics": diagnostics,
    }


def regional_scan(inputs: TextScanInputs, region: str, *, dry_run: bool, feeds: tuple | None = None) -> dict:
    """Scan one region's assignment files and seal its regional text-scan receipt.

    ``dry_run`` scans a bounded real slice (``max_files``) into ``slice_output_root``
    and seals nothing under the main output root.
    """
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, SCAN_STAGE, {"mode": mode, "region": region}, alert_feeds)
    try:
        inputs.validate()
        if region not in inputs.region_prefixes:
            raise ValueError(f"region {region!r} is not one of the typed regions {sorted(inputs.region_prefixes)}")
        if dry_run and not inputs.max_files:
            raise ValueError("dry-run regional scan requires a positive max_files bound: the preflight slices")
        if dry_run and inputs.slice_output_root is None:
            raise ValueError("dry-run regional scan requires slice_output_root in the typed inputs")
        if not dry_run and inputs.max_files:
            raise ValueError("write-mode regional scan covers every file; use dry-run for a max_files slice")
        configure_coreweave_s3()
        _, release, coverage, _ = read_handoff(inputs)
        root = inputs.slice_output_root if dry_run else inputs.output_root
        scan_root = f"{root}/text-scan-{region}"
        jobs = [(FULL_PHASE, index, path, scan_root) for index, path in region_files(release["paths"], inputs, region)]
        jobs += [
            (SAMPLE_PHASE, index, path, scan_root)
            for index, path in region_files(coverage["sample_paths"], inputs, region)
        ]
        if not jobs:
            raise ValueError(f"no assignment files selected for region {region!r}")
        if inputs.max_files:
            jobs = jobs[: inputs.max_files]
        results = [scan_file(job) for job in jobs]
        totals, bad, fields, unsupported = aggregate_results(results)
        receipt = {
            "kind": f"atlas-{inputs.release_id}-{REGIONAL_RECEIPT_SUFFIX}",
            "release_id": inputs.release_id,
            "region": region,
            "codec": catalog_text.CATALOG_TEXT_CODEC,
            "codec_sha256": codec_sha256(),
            "source_revision": inputs.source_revision,
            "handoff": {"uri": inputs.handoff.uri, "sha256": inputs.handoff.sha256},
            "files_scanned": sorted(result["input"] for result in results),
            "rows": dict(totals[FULL_PHASE] + totals[SAMPLE_PHASE]),
            "bad_rows": dict(bad[FULL_PHASE] + bad[SAMPLE_PHASE]),
            "fields": [{"source_id": key[0], "field": key[1], "count": count} for key, count in fields.items()],
            "unsupported_identity_fields": unsupported,
            "file_results": results,
            "all_rows_encoded_and_json_serialized": True,
            "affected_rows_lossless_roundtrip": True,
            "sql_writes": 0,
            "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        sealed = seal(f"{scan_root}/regional-scan-receipt.json", receipt)
        proof = {**receipt, "dry_run": dry_run, "receipt": {"uri": sealed.uri, "sha256": sealed.sha256}}
        evidence = {
            "mode": mode,
            "region": region,
            "files": len(results),
            "rows": sum(counter.total() for counter in totals.values()),
        }
        alerts.stage_finished(inputs.release_id, SCAN_STAGE, {**evidence, **proof["receipt"]}, alert_feeds)
        return proof
    except Exception as error:
        detail = {"mode": mode, "region": region, "error": str(error)[:500]}
        alerts.stage_failed(inputs.release_id, SCAN_STAGE, detail, alert_feeds)
        raise


def merge_region_receipts(
    inputs: TextScanInputs, receipts: tuple[ArtifactRef, ...], *, dry_run: bool, feeds: tuple | None = None
) -> dict:
    """Merge the regional receipts into the sealed catalog-text preflight receipt.

    ``dry_run`` performs every verification with zero writes.
    """
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, MERGE_STAGE, {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        if not receipts:
            raise ValueError("merge requires the regional receipt references")
        configure_coreweave_s3()
        handoff, release, coverage, confidence = read_handoff(inputs)
        documents = {"release": release, "coverage": coverage, "confidence": confidence}
        loaded = [read_receipt(reference) for reference in receipts]
        regions = [receipt["region"] for receipt in loaded]
        if len(set(regions)) != len(regions):
            raise ValueError(f"two regional receipts claim the same region: {regions}")
        if set(regions) != set(inputs.region_prefixes):
            raise ValueError(f"regional receipts cover {sorted(regions)}, expected {sorted(inputs.region_prefixes)}")
        revisions = {receipt["source_revision"] for receipt in loaded}
        if revisions != {inputs.source_revision}:
            raise ValueError(f"regional receipts pin revisions {revisions}, expected {inputs.source_revision!r}")
        module_digest = codec_sha256()
        codecs = {receipt["codec_sha256"] for receipt in loaded}
        if codecs != {module_digest}:
            raise ValueError(f"regional receipts carry codec digests {codecs}, local module digests {module_digest}")
        handoff_refs = {json.dumps(receipt["handoff"], sort_keys=True) for receipt in loaded}
        expected_ref = json.dumps({"uri": inputs.handoff.uri, "sha256": inputs.handoff.sha256}, sort_keys=True)
        if handoff_refs != {expected_ref}:
            raise ValueError("regional receipts bind different handoff references than the typed inputs")
        file_results = [result for receipt in loaded for result in receipt["file_results"]]
        for label, scanned, expected in (
            (FULL_PHASE, [r["input"] for r in file_results if r["phase"] == FULL_PHASE], release["paths"]),
            (SAMPLE_PHASE, [r["input"] for r in file_results if r["phase"] == SAMPLE_PHASE], coverage["sample_paths"]),
        ):
            if len(set(scanned)) != len(scanned):
                raise ValueError(f"the regional scans covered a {label} file more than once")
            if sorted(scanned) != sorted(expected):
                raise ValueError(f"the scanned {label} files differ from the sealed document")
        totals, bad, fields, unsupported = aggregate_results(file_results)
        sources = {row["source_id"]: row["task_count"] for row in release["sources"]}
        full_rows = dict(totals[FULL_PHASE])
        if full_rows != sources:
            raise ValueError(f"unioned full-assignment rows {full_rows} disagree with the release sources {sources}")
        if sum(full_rows.values()) != release["task_count"]:
            raise ValueError(
                f"unioned full rows total {sum(full_rows.values())}, release declares {release['task_count']}"
            )
        if sum(totals[SAMPLE_PHASE].values()) != coverage["sample_task_count"]:
            raise ValueError(
                f"unioned sample rows total {sum(totals[SAMPLE_PHASE].values())},"
                f" coverage declares {coverage['sample_task_count']}"
            )
        if unsupported:
            raise ValueError(f"{unsupported} unsupported identity fields across the regional scans")
        manifest_fields = []
        for kind, document in documents.items():
            encoded = catalog_text.encode_catalog_value(document)
            json.dumps(encoded, ensure_ascii=False, allow_nan=False)
            if catalog_text.decode_catalog_value(encoded) != document:
                raise ValueError(f"the sealed {kind} document does not survive the catalog text codec")
            if json.loads(json.dumps(document, ensure_ascii=False, allow_nan=False)) != document:
                raise ValueError(f"the sealed {kind} document does not round-trip through json")
            for field, value in problems(document):
                encoded_value = catalog_text.EncodedCatalogText.encode(value)
                manifest_fields.append(
                    {"document": kind, "field": field, "sha256": encoded_value.sha256, "kind": encoded_value.kind.value}
                )
        if manifest_fields:
            raise ValueError(f"the sealed documents carry {len(manifest_fields)} unsupported text fields")
        report = {
            "kind": f"atlas-{inputs.release_id}-{MERGED_RECEIPT_SUFFIX}",
            "release_id": inputs.release_id,
            "status": "PASS" if unsupported == 0 and not manifest_fields else "FAIL",
            "codec": catalog_text.CATALOG_TEXT_CODEC,
            "codec_sha256": module_digest,
            "source_revision": inputs.source_revision,
            "handoff": {"uri": inputs.handoff.uri, "sha256": inputs.handoff.sha256},
            "artifacts": {key: handoff[key] for key in documents},
            "regions": [
                {"region": receipt["region"], "receipt": {"uri": reference.uri, "sha256": reference.sha256}}
                for reference, receipt in zip(receipts, loaded, strict=True)
            ],
            "source_count": len(sources),
            "task_count": sum(totals[FULL_PHASE].values()),
            "sample_task_count": sum(totals[SAMPLE_PHASE].values()),
            "affected_rows": {"full": sum(bad[FULL_PHASE].values()), "sample": sum(bad[SAMPLE_PHASE].values())},
            "affected_sources": {"full": dict(bad[FULL_PHASE]), "sample": dict(bad[SAMPLE_PHASE])},
            "field_counts": [
                {"source_id": key[0], "field": key[1], "count": count} for key, count in sorted(fields.items())
            ],
            "manifest_fields": manifest_fields,
            "unsupported_identity_fields": unsupported,
            "file_results": file_results,
            "all_rows_encoded_and_json_serialized": True,
            "affected_rows_lossless_roundtrip": True,
            "sql_writes": 0,
            "vectors_recomputed": 0,
            "scores_recomputed": 0,
            "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        if report["status"] != "PASS":
            raise ValueError("regional text scans did not pass; refusing to seal the merged receipt")
        if dry_run:
            alerts.stage_finished(
                inputs.release_id,
                MERGE_STAGE,
                {"mode": mode, "status": report["status"], "task_count": report["task_count"]},
                alert_feeds,
            )
            return {**report, "dry_run": True}
        sealed = seal(f"{inputs.output_root}/scan-receipt.json", report)
        proof = {**report, "dry_run": False, "receipt": {"uri": sealed.uri, "sha256": sealed.sha256}}
        evidence = {"mode": mode, "status": report["status"], "task_count": report["task_count"]}
        alerts.stage_finished(inputs.release_id, MERGE_STAGE, {**evidence, **proof["receipt"]}, alert_feeds)
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, MERGE_STAGE, {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON TextScanInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    parser.add_argument("--step", choices=("scan", "merge"), required=True)
    parser.add_argument("--region", help="Region label to scan (required with --step scan)")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    receipts = tuple(ArtifactRef(**reference) for reference in document.pop("receipts", []))
    inputs = TextScanInputs(**document)
    if args.step == "scan":
        if not args.region:
            parser.error("--step scan requires --region")
        proof = regional_scan(inputs, args.region, dry_run=args.mode == "dry-run")
    else:
        proof = merge_region_receipts(inputs, receipts, dry_run=args.mode == "dry-run")
    label = "TEXT-SCAN-PREFLIGHT" if args.mode == "dry-run" else "TEXT-SCAN-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
