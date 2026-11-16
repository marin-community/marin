# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind verified, actually modified tool exports to frozen candidate input rows."""

import argparse
import hashlib
import json
import logging
import shutil
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fray.types import ResourceConfig
from levanter.store.jagged_array import JaggedArrayStore
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.sft_new_mix.exp3_build import configure_storage, tokenizer_for

EXPORT_FILES = ("input_ids.i32", "loss_mask.u8", "messages.json", "text.txt", "tools.json")


@dataclass(frozen=True)
class ExportKey:
    source_shard: str
    source_row: int
    input_ids_sha256: str
    archive_sha256: str
    record_prefix: str


def export_key(row: dict) -> ExportKey:
    return ExportKey(
        source_shard=row["source_shard"],
        source_row=int(row["source_row"]),
        input_ids_sha256=row["input_ids_sha256"],
        archive_sha256=row["archive_sha256"],
        record_prefix=row["record_prefix"],
    )


def implementation_identity() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cache
def equivalent_tokenizer_json(target_uri: str, target_sha256: str, export_uri: str, export_sha256: str) -> str:
    """Require exact parsed JSON equivalence, retaining both distinct file pins."""
    values = []
    for uri, sha256 in ((target_uri, target_sha256), (export_uri, export_sha256)):
        content = StoragePath(uri).read_bytes()
        if hashlib.sha256(content).hexdigest() != sha256:
            raise ValueError("Tokenizer file SHA256 mismatch")
        values.append(json.loads(content))
    if values[0] != values[1]:
        raise ValueError("Tool export tokenizer is not JSON-equivalent to the candidate tokenizer")
    return hashlib.sha256(json.dumps(values[0], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_records(
    snapshots: list[tuple[bytes, int]], inputs: set[tuple[str, str]], excluded_exports: set[ExportKey]
) -> tuple[list[dict], dict]:
    """Require actual-call evidence, rejecting conflicts at equal explicit priority."""
    selected = {}
    counts = Counter()
    superseded = []
    for content, priority in snapshots:
        snapshot_sha256 = hashlib.sha256(content).hexdigest()
        for line in content.splitlines():
            row = json.loads(line)
            counts["manifest_rows"] += 1
            if (row["source"], row["source_shard"]) not in inputs:
                counts["not_in_frozen_candidate_inputs"] += 1
                continue
            if export_key(row) in excluded_exports:
                counts["explicitly_withheld_or_superseded_export"] += 1
                continue
            if row["status"] != "verified" or row["tool_calls_modified"] is not True:
                raise ValueError("Overlay snapshot contains an unverified or unmodified row")
            if row["diff"]["modified"] is not True or not row["diff"]["changes"]:
                raise ValueError("Verified overlay lacks an actual ordered tool-call difference")
            if row["diff"]["before_sha256"] == row["diff"]["after_sha256"]:
                raise ValueError("No-op call hash cannot replace an original trace")
            key = (row["source_shard"], int(row["source_row"]))
            candidate = {**row, "snapshot_sha256": snapshot_sha256, "priority": priority}
            previous = selected.get(key)
            if previous is None:
                selected[key] = candidate
                continue
            if previous["input_ids_sha256"] != row["input_ids_sha256"]:
                raise ValueError("Overlay snapshots disagree on the original source row")
            old_payload = previous["export_manifest"]["files"]["input_ids.i32"]
            new_payload = row["export_manifest"]["files"]["input_ids.i32"]
            if old_payload == new_payload:
                counts["identical_duplicate_exports"] += 1
                continue
            if previous["priority"] == priority:
                raise ValueError("Conflicting tool exports require an explicit priority decision")
            winner, loser = (candidate, previous) if priority > previous["priority"] else (previous, candidate)
            selected[key] = winner
            superseded.append(
                {
                    "source_shard": key[0],
                    "source_row": key[1],
                    "selected_snapshot_sha256": winner["snapshot_sha256"],
                    "superseded_snapshot_sha256": loser["snapshot_sha256"],
                    "selected_priority": winner["priority"],
                    "superseded_priority": loser["priority"],
                }
            )
    counts["selected_verified_modified_rows"] = len(selected)
    return list(selected.values()), {"counts": dict(counts), "superseded": superseded}


def materialize_archive(job: dict) -> dict:
    """Verify compressed archive, every selected member and original cache row."""
    configure_storage()
    if job["implementation_sha256"] != implementation_identity():
        raise ValueError("Tool payload verifier implementation changed")
    root = StoragePath(job["output_prefix"]) / "archives" / job["archive_sha256"]
    receipt_path = root / "receipt.json"
    job_sha256 = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["job_sha256"] != job_sha256:
            raise ValueError("Archive selection changed; freeze a new overlay output version")
        return {"receipt_uri": str(receipt_path), "records": len(receipt["records"])}
    tokenizer = tokenizer_for(job["tokenizer_uri"], job["tokenizer_sha256"])
    originals = {}
    by_source = defaultdict(list)
    for row in job["records"]:
        by_source[row["source_shard"]].append(row)
    for source_shard, rows in by_source.items():
        store = JaggedArrayStore.open(prefix_join(source_shard, "input_ids"), mode="r", dtype=np.int32)
        indices = [int(row["source_row"]) for row in rows]
        for row, tokens in zip(rows, store.get_batch_sync(indices), strict=True):
            original_sha256 = hashlib.sha256(tokens.astype("<i4", copy=False).tobytes()).hexdigest()
            if original_sha256 != row["input_ids_sha256"]:
                raise ValueError("Tool overlay does not match the frozen original token row")
            originals[(source_shard, int(row["source_row"]))] = original_sha256
    selected = []
    with TemporaryDirectory(prefix="exp3-verified-tools-") as scratch:
        archive_path = Path(scratch) / "payload.tar.gz"
        with StoragePath(job["archive_uri"]).open("rb") as source, archive_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        with archive_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != job["archive_sha256"]:
                raise ValueError("Tool archive SHA256 mismatch")
        with tarfile.open(archive_path, "r:gz") as archive:
            for row in job["records"]:
                exported = row["export_manifest"]
                export_sha256 = exported["tokenizer_sha256"]
                export_pin = job["export_tokenizer_pins"][export_sha256]
                canonical_tokenizer_sha256 = None
                if export_sha256 != job["tokenizer_sha256"]:
                    canonical_tokenizer_sha256 = equivalent_tokenizer_json(
                        job["tokenizer_uri"], job["tokenizer_sha256"], export_pin["uri"], export_sha256
                    )
                payloads = {}
                for name in EXPORT_FILES:
                    member = archive.getmember(row["record_prefix"] + "export/" + name)
                    if not member.isfile():
                        raise ValueError("Tool export member is not a regular file")
                    with archive.extractfile(member) as stream:
                        payload = stream.read()
                    if hashlib.sha256(payload).hexdigest() != exported["files"][name]:
                        raise ValueError("Tool export member SHA256 mismatch")
                    payloads[name] = payload
                tokens = np.frombuffer(payloads["input_ids.i32"], dtype="<i4")
                if len(tokens) != exported["tokens"] or len(payloads["loss_mask.u8"]) != len(tokens):
                    raise ValueError("Tool export token or target-mask length mismatch")
                encoded = np.asarray(
                    tokenizer.encode(payloads["text.txt"].decode(), add_special_tokens=False).ids, dtype=np.int32
                )
                if not np.array_equal(tokens, encoded):
                    raise ValueError("Verified tool text does not reproduce its exported token IDs")
                payload_path = StoragePath(job["output_prefix"]) / "payloads" / exported["files"]["input_ids.i32"]
                (payload_path / "input_ids.i32").write_bytes(payloads["input_ids.i32"])
                selected.append(
                    {
                        **row,
                        "payload_uri": str(payload_path / "input_ids.i32"),
                        "verified_original_sha256": originals[(row["source_shard"], int(row["source_row"]))],
                        "tokens": len(tokens),
                        "tokenizer_json_equivalence_sha256": canonical_tokenizer_sha256,
                        "tokenizer_verification": (
                            "identical_file_sha256" if canonical_tokenizer_sha256 is None else "identical_parsed_json"
                        ),
                    }
                )
    receipt = {
        "complete": True,
        "job_sha256": job_sha256,
        "archive_uri": job["archive_uri"],
        "archive_sha256": job["archive_sha256"],
        "records": selected,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return {"receipt_uri": str(receipt_path), "records": len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_storage()
    selection = json.loads(StoragePath(args.selection).read_text())
    with ZephyrContext(
        name="exp3-verified-tool-payloads",
        max_workers=args.workers,
        resources=ResourceConfig(cpu=2, ram="8g", disk="20g"),
        chunk_storage_prefix=prefix_join(selection["output_prefix"], "zephyr"),
        max_execution_retries=2,
    ) as context:
        result = context.execute(Dataset.from_list(selection["jobs"]).map(materialize_archive))
    records = []
    receipts = []
    for value in result.results:
        content = StoragePath(value["receipt_uri"]).read_bytes()
        receipt = json.loads(content)
        records.extend(receipt["records"])
        receipts.append({"uri": value["receipt_uri"], "sha256": hashlib.sha256(content).hexdigest()})
    (StoragePath(selection["output_prefix"]) / "manifest.json").write_text(
        json.dumps(
            {
                "selection_uri": args.selection,
                "selection_sha256": hashlib.sha256(StoragePath(args.selection).read_bytes()).hexdigest(),
                "records": records,
                "archive_receipts": receipts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
