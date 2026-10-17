# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit final #9984 records using pinned active snapshots and native-writer provenance.

This retains the historical audit, including diagnostic re-extraction from normalized
output. That diagnostic is not an ae362881 evaluator replay.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq
from finestore.reader import ReadView
from rigging.filesystem.factory import filesystem as get_filesystem
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.second_expert.constants import CANONICAL, CANONICAL_SHA256, CAPTURE, SUBSET_SHA256

EVALS = "s3://marin-us-east-02a/marin/evals"
# Record IDs and HEAD commits from the final issue9984 manifest, not superseded streams.
RECORDS = {
    "b_mmlu": "20261007-172014-gate-student-rlvr-step92-mmlu-pro-220c",
    "rs_crux": "20261012-020330-gate-mixed-rs-lr1e-06-step727-cruxeval-55eb",
    "rs_mmlu": "20261012-020330-gate-mixed-rs-lr1e-06-step727-mmlu-pro-19c9",
    "mopd_crux": "20261011-075144-gate-mopd2-2ep-step100-cruxeval-ced7",
    "mopd_mmlu": "20261011-075144-gate-mopd2-2ep-step100-mmlu-pro-f728",
    "merge_new": "20261012-041309-gate-merge-ta-a0.5-b0.5-math500-8bf7",
    "merge_old": "20261010-042342-gate-merge-ta-a0.5-b0.5-math500-eb56",
}

SNAPSHOTS = {
    "b_mmlu": "55b16b9998c3428babdc2d0031d0a387",
    "rs_crux": "5d8963afd94841338ea94e8a2407e4d4",
    "rs_mmlu": "c6cf231489b34cb3a3c72a6790e1f084",
    "mopd_crux": "e5be83f10b014f05b62ed548851fd1f6",
    "mopd_mmlu": "946025cec9964374b833da37b12ba541",
    "merge_new": "ab38b691daac44c6badc29539edd5626",
    "merge_old": "3710ca07515743f7b5b4cc2497d02c65",
}


def require_snapshot(view: ReadView, expected: str) -> None:
    """Refuse to silently audit a different evaluation attempt after HEAD changes."""
    if view.token is None or view.token.commit_id != expected:
        raise ValueError(f"Snapshot changed: expected {expected}, got {view.token}")


def extract_letter(text: str) -> str | None:
    for pattern in (r"answer is \(?([A-J])\)?", r"Answer:\s*([A-J])"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    matches = re.findall(r"\b[A-J]\b", text)
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Fresh local path or object URI for the audit JSON")
    args = parser.parse_args()
    destination = StoragePath(args.output)
    if destination.exists():
        raise ValueError("Use a fresh audit output; historical reports are immutable")
    filesystem = get_filesystem("s3")
    canonical_bytes = (StoragePath(CANONICAL) / "canonical.jsonl").read_bytes()
    subset_bytes = (StoragePath(CANONICAL) / "subset-groups-1600-seed20261009.json").read_bytes()
    assert hashlib.sha256(canonical_bytes).hexdigest() == CANONICAL_SHA256
    assert hashlib.sha256(subset_bytes).hexdigest() == SUBSET_SHA256
    canonical = {row["custom_id"]: row for row in map(json.loads, canonical_bytes.splitlines())}
    subset = set(json.loads(subset_bytes)["custom_ids"])
    prompt_ids = defaultdict(list)
    question_prompts = defaultdict(list)
    for line in (StoragePath(CAPTURE) / "requests.jsonl").read_bytes().splitlines():
        request = json.loads(line)
        messages = request["b_request"]["messages"]
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        prompt_ids[prompt].append(request["custom_id"])
        question = prompt.split("Question:\n", 1)[1].rsplit("\nOptions:\n", 1)[0]
        question_prompts[question].append((prompt, request["custom_id"]))
    assert sum(map(len, prompt_ids.values())) == len(canonical) == 12032
    assert len(subset) == 1606 and subset <= canonical.keys()
    reports: dict[str, dict] = {}
    for label, run_id in RECORDS.items():
        root = StoragePath(EVALS) / run_id
        record = json.loads((root / "record.json").read_text())
        print(
            "RECORD",
            label,
            json.dumps(
                {
                    key: value
                    for key, value in record.items()
                    if key in ("status", "error", "metrics", "results", "scores")
                }
            )[:12000],
            flush=True,
        )
        view = ReadView(str(root / "results"))
        require_snapshot(view, SNAPSHOTS[label])
        wanted = [
            "task",
            "doc_id",
            "trial_id",
            "filter",
            "correct",
            "_writer",
            "prompt_text",
            "extracted",
            "doc",
            "output",
            "grading",
        ]
        table = view.scan("samples", columns=wanted)
        assert table is not None, (label, "no active sample table", str(view.token))
        writers = defaultdict(list)
        for row in table.to_pylist():
            writers[str(row["_writer"])].append(row)
        print("SNAPSHOT", label, str(view.token), {writer: len(rows) for writer, rows in writers.items()}, flush=True)
        candidates = list(writers.items())
        expected = 12032 if label.endswith("mmlu") else 1600 if label.endswith("crux") else 500
        if len(candidates) != 1 or len(candidates[0][1]) != expected:
            reports[label] = {
                "record": str(root),
                "unresolved": "active writer counts require provenance reconciliation",
                "snapshot": str(view.token),
                "writers": {writer: len(rows) for writer, rows in writers.items()},
            }
            print("UNRESOLVED", json.dumps(reports[label]), flush=True)
            continue
        writer, rows = candidates[0]
        markers = Counter()
        for row in rows:
            for marker in (
                "AgentTimeoutError",
                "ClientConnectorError",
                "ClientResponseError",
                "ServerDisconnectedError",
            ):
                if marker in str(row.get("output")) or marker in str(row.get("grading")):
                    markers[marker] += 1
        report: dict = {
            "record": str(root),
            "snapshot": str(view.token),
            "writer": writer,
            "active_sample_files": [shard.path for shard in view.list_shards("samples")],
            "infra_marker_rows": dict(markers),
            "rows": len(rows),
            "record_metrics": record.get("metrics"),
        }
        blobs = view.scan("blobs", columns=["name", "_writer"])
        report["source_blobs"] = blobs.to_pylist() if blobs is not None else []
        native_writers = {blob["_writer"] for blob in report["source_blobs"] if "/native/" in blob["name"]}
        assert len(native_writers) == 1, (label, native_writers)
        native_writer = next(iter(native_writers))
        native_rows = {}
        native_files = []
        for path in sorted(filesystem.find(str(root / "results" / "data" / "samples"))):
            if not path.endswith(".parquet"):
                continue
            with filesystem.open(path, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                columns = [name for name in wanted if name in parquet.schema_arrow.names]
                physical_rows = parquet.read(columns=columns).to_pylist()
            selected = [row for row in physical_rows if row["_writer"] == native_writer]
            if not selected:
                continue
            native_files.append(path)
            for row in selected:
                key = tuple(str(row.get(name)) for name in ("task", "doc_id", "trial_id", "filter"))
                if key in native_rows:
                    assert native_rows[key] == row, (label, "conflicting same-writer duplicate", key)
                native_rows[key] = row
        assert len(native_rows) == expected, (label, native_writer, len(native_rows))
        rows = list(native_rows.values())
        report["native_writer_from_active_blobs"] = native_writer
        report["native_writer_files"] = native_files
        report["normalized_infra_marker_rows"] = report.pop("infra_marker_rows")
        report["infra_marker_rows"] = {
            marker: sum(marker in str(row.get("output")) or marker in str(row.get("grading")) for row in rows)
            for marker in ("AgentTimeoutError", "ClientConnectorError", "ClientResponseError", "ServerDisconnectedError")
        }
        if label.endswith("crux"):
            groups = defaultdict(list)
            for row in rows:
                prompt = row["prompt_text"]
                if prompt.startswith("You will be given a function f and an output"):
                    direction = "input"
                else:
                    assert prompt.startswith("You are given a Python function and an assertion"), prompt[:100]
                    direction = "output"
                groups[direction].append(row)
            for direction, group in groups.items():
                assert len(group) == len({str(row["doc_id"]) for row in group}) == 800
                report[direction] = {
                    "correct": sum(bool(row["correct"]) for row in group),
                    "denominator": len(group),
                    "unset_correct": sum(row["correct"] is None for row in group),
                }
        elif label.endswith("mmlu"):
            by_identifiers = defaultdict(list)
            for row in rows:
                doc = json.loads(row["doc"]) if isinstance(row["doc"], str) else row["doc"]
                options = doc["options"]
                options_block = "\n".join(
                    f"{letter}. {option}" for letter, option in zip("ABCDEFGHIJ", options, strict=False)
                )
                exact_suffix = (
                    f"Question:\n{doc['question']}\nOptions:\n{options_block}\nAnswer: Let's think step by step."
                )
                identifiers = sorted(
                    identifier
                    for prompt, identifier in question_prompts[doc["question"]]
                    if prompt.endswith(exact_suffix)
                )
                assert identifiers, (label, row["doc_id"], "no exact question/options block")
                by_identifiers[tuple(identifiers)].append(row)
            scores = {}
            disagreements = Counter()
            for identifiers, group in by_identifiers.items():
                assert len(group) == len(identifiers)
                assert (
                    len(
                        {
                            (canonical[identifier]["canonical_gold"], canonical[identifier]["group"])
                            for identifier in identifiers
                        }
                    )
                    == 1
                )
                for identifier, row in zip(identifiers, sorted(group, key=lambda row: int(row["doc_id"])), strict=True):
                    correct = row["extracted"] == canonical[identifier]["canonical_gold"]
                    scores[identifier] = correct
                    disagreements["stored_correct"] += bool(row["correct"]) != correct
                    if row.get("output") is not None:
                        disagreements["output_reextraction"] += extract_letter(row["output"]) != row["extracted"]
            assert len(scores) == len(rows)
            report["disagreements"] = dict(disagreements)
            for slice_name, identifiers in (
                ("subset", subset),
                ("held_out", canonical.keys() - subset),
                ("all", canonical.keys()),
            ):
                report[slice_name] = {
                    "correct": sum(scores[identifier] for identifier in identifiers),
                    "denominator": len(identifiers),
                    "groups": len({canonical[identifier]["group"] for identifier in identifiers}),
                }
        else:
            assert len({str(row["doc_id"]) for row in rows}) == 500
            report["correct"] = sum(bool(row["correct"]) for row in rows)
        reports[label] = report
        print("AUDIT", json.dumps(report, sort_keys=True), flush=True)
    destination.write_text(json.dumps(reports, sort_keys=True, indent=2))
    print("AUDIT_COMPLETE", str(destination), flush=True)


if __name__ == "__main__":
    main()
