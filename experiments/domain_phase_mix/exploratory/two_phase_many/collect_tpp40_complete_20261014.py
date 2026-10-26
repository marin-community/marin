# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "numpy", "pandas"]
# ///
"""Collect the complete TPP40 scalar evidence with pinned checkpoint identities.

Run from the pinned Marin workspace with uv run --no-sync python. Only small
metadata and metric objects are read; this script cannot launch evaluations.
Completed row receipts are hash-checked and reused on interruption or replay.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from collect_tpp40_surrogate_snapshot_20260909 import ASSIGNMENT, BASE, SPECS, STANDALONE, digest, write_json
from rigging.filesystem.buckets import filesystem_for

REPO = Path(__file__).resolve().parents[4]
INVENTORY = REPO / ".experiments/artifacts/status_refresh_20261006"
OUTPUT = BASE / "tpp40_complete_20261014"
MAX_OBJECT_BYTES = 4_000_000
CHECKPOINTS = (21855, 27335)
DEPLOYMENT_FIELDS = {"tpu_type", "tpu_region", "tpu_zone"}


@dataclass(frozen=True)
class RemoteObject:
    name: str
    uri: str
    expected_sha256: str | None


def input_sources() -> dict[str, Path]:
    return {
        "run_specs.json": SPECS,
        "assignment.json": ASSIGNMENT,
        "buckets.csv": STANDALONE / "data/buckets.csv",
        "objectives.csv": STANDALONE / "data/objectives.csv",
        "completion_inventory.json": INVENTORY / "records.json",
        "east5_evaluations.json": INVENTORY / "east5_evaluations.json",
        "europe_evaluations.json": INVENTORY / "europe_evaluations.json",
    }


def prepare(output: Path) -> tuple[list[dict], list[dict], dict[str, dict]]:
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, path in input_sources().items():
        target, content = inputs / name, path.read_bytes()
        if target.exists():
            assert target.read_bytes() == content, f"Input changed: {path}"
        else:
            target.write_bytes(content)
    specs = json.loads((inputs / "run_specs.json").read_text())
    records = [r for r in json.loads((inputs / "completion_inventory.json").read_text()) if r["group"] == "tpp40"]
    records.sort(key=lambda row: row["order"])
    assert [r["order"] for r in records] == list(range(280))
    assert [s["run_order"] for s in specs] == list(range(280))
    evaluations = {
        row["path"]: row
        for region in ("east5", "europe")
        for row in json.loads((inputs / f"{region}_evaluations.json").read_text())
    }
    assignment = json.loads((inputs / "assignment.json").read_text())
    assigned: dict[int, str] = {}
    for region in ("east5", "europe"):
        for item in assignment["run_order_args"][region].split(","):
            first, *last = item.split("-")
            for order in range(int(first), int(last[0]) + 1 if last else int(first) + 1):
                assert order not in assigned
                assigned[order] = region
    assigned.update({order: "east5" for order in assignment["observed"]["east5_final"]})
    assert len(assigned) == 280
    for row, spec in zip(records, specs, strict=True):
        assert row["region"] == assigned[row["order"]]
        assert row["path"].startswith(assignment[f"{row['region']}_root"] + "/")
        assert row["name"][:-7] == spec["run_name"]
        assert row["status"] == "SUCCESS" and not row["issues"]
        assert len(row["final_table9"]) == len(row["boundary_table9"]) == 1
    return records, specs, evaluations


def remote_objects(row: dict, evaluations: dict[str, dict]) -> list[RemoteObject]:
    root = row["path"]
    objects = [
        RemoteObject("training_status.txt", root + "/.executor_status", None),
        RemoteObject("training_info.json", root + "/.executor_info", None),
        RemoteObject("eval_metrics.jsonl", root + "/checkpoints/eval_metrics.jsonl", row["metrics_sha256"]),
        RemoteObject("phase0_validation.json", row["boundary_validation"]["path"], row["boundary_validation"]["sha256"]),
    ]
    for label, step in zip(("phase0", "endpoint"), CHECKPOINTS, strict=True):
        objects.append(
            RemoteObject(
                f"{label}_metadata.json",
                root + f"/checkpoints/step-{step}/metadata.json",
                row["checkpoints"][str(step)]["sha256"],
            )
        )
        key = "boundary_table9" if label == "phase0" else "final_table9"
        path = row[key][0]["path"]
        objects.append(RemoteObject(f"{label}_table9.json", path, evaluations[path]["sha256"]))
    return objects


def validate_row(folder: Path, row: dict, spec: dict) -> dict:
    assert (folder / "training_status.txt").read_text().strip() == "SUCCESS"
    info = json.loads((folder / "training_info.json").read_text())
    config = info["config"]
    assert config["output_path"] == row["path"]
    persisted = config["run_spec"]
    expected_region = {"east5": "us-east5", "europe": "europe-west4"}[row["region"]]
    assert persisted["tpu_region"] == expected_region
    assert persisted["tpu_zone"].startswith(expected_region + "-")
    assert {k: v for k, v in persisted.items() if k not in DEPLOYMENT_FIELDS} == {
        k: v for k, v in spec.items() if k not in DEPLOYMENT_FIELDS
    }, f"Scientific spec mismatch: {row['order']}"
    for label, step in zip(("phase0", "endpoint"), CHECKPOINTS, strict=True):
        metadata = json.loads((folder / f"{label}_metadata.json").read_text())
        assert metadata["step"] == step and not metadata["is_temporary"]
    boundary = json.loads((folder / "phase0_validation.json").read_text())
    assert boundary["run_order"] == row["order"]
    assert boundary["training_output_path"] == row["path"]
    assert boundary["checkpoint_path"] == row["path"] + "/checkpoints/step-21855"
    assert boundary["checkpoint_step"] == 21855
    assert boundary["checkpoint_metadata_sha256"] == digest((folder / "phase0_metadata.json").read_bytes())
    for key in ("data_seed", "trainer_seed"):
        assert boundary[key] == spec[key]
    assert len(boundary["validation_caches"]) == 23
    assert all(math.isfinite(boundary["metrics"][f"eval/{name}/bpb"]) for name in boundary["validation_caches"])
    metrics = [json.loads(line) for line in (folder / "eval_metrics.jsonl").read_text().splitlines() if line.strip()]
    endpoints = [m for m in metrics if m.get("step") == 27335]
    assert endpoints
    scientific = [
        {k: v for k, v in m.items() if k.startswith("eval/") and k.endswith(("bpb", "loss"))} for m in endpoints
    ]
    assert all(m == scientific[0] for m in scientific), f"Conflicting endpoint records: {row['order']}"
    assert all(math.isfinite(v) for m in scientific for v in m.values())
    assert all(
        math.isfinite(v) for k, v in boundary["metrics"].items() if k.startswith("eval/") and k.endswith(("bpb", "loss"))
    )
    uncheatable = {
        k: v
        for k, v in scientific[0].items()
        if k.startswith("eval/uncheatable_eval/") and k.endswith("/bpb") and k.count("/") == 3
    }
    assert len(uncheatable) == 7 and all(math.isfinite(v) for v in uncheatable.values())
    for label, step in zip(("phase0", "endpoint"), CHECKPOINTS, strict=True):
        result = json.loads((folder / f"{label}_table9.json").read_text())
        assert result["provenance"]["swarm_run_name"] == spec["run_name"]
        assert result["provenance"]["source_run_name"] == spec["source_run_name"]
        if label == "endpoint":
            assert result["checkpoint_path"].rstrip("/") == row["path"] + f"/hf/step-{step}"
        else:
            assert result["provenance"]["source_checkpoint"] == boundary["checkpoint_path"]
            expected = row["boundary_validation"]["path"].rsplit("/", 1)[0] + f"/hf/step-{step}"
            assert result["checkpoint_path"].rstrip("/") == expected
        assert len(result["table9_components"]) == 51
        assert len(result["task_bpb"]) == len(result["num_instances"]) == 104
        assert set(result["task_bpb"]) == set(result["num_instances"])
        assert all(math.isfinite(v) for v in result["table9_components"].values())
        assert all(math.isfinite(v) for v in result["task_bpb"].values())
        assert all(v > 0 for v in result["num_instances"].values())
        assert math.isclose(
            sum(result["table9_components"].values()) / 51, result["table9_macro_bpb"], rel_tol=0, abs_tol=1e-10
        )
    return {
        "final_metric_records": len(endpoints),
        "native_history_records": len(metrics),
        "endpoint_uncheatable_components": len(uncheatable),
        "phase0_native_components": 23,
        "endpoint_native_bpb_components": len([k for k in scientific[0] if k.endswith("/bpb") and k.count("/") == 3]),
        "table9_components_per_checkpoint": 51,
        "table9_tasks_per_checkpoint": 104,
        "persisted_deployment": {k: persisted[k] for k in sorted(DEPLOYMENT_FIELDS)},
    }


def collect_row(output: Path, row: dict, spec: dict, evaluations: dict[str, dict]) -> dict:
    folder = output / "raw" / f"{row['order']:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "receipt.json"
    objects = remote_objects(row, evaluations)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["order"] == row["order"] and receipt["training_root"] == row["path"]
        for obj in objects:
            saved = receipt["files"][obj.name]
            assert saved["uri"] == obj.uri
            actual = digest((folder / obj.name).read_bytes())
            assert actual == saved["sha256"]
            assert obj.expected_sha256 is None or actual == obj.expected_sha256
        assert validate_row(folder, row, spec) == receipt["validation"]
        return receipt
    files: dict[str, Any] = {}
    for obj in objects:
        target = folder / obj.name
        if target.exists() and obj.expected_sha256 is not None and digest(target.read_bytes()) == obj.expected_sha256:
            content = target.read_bytes()
        else:
            fs, path = filesystem_for(obj.uri)
            content = fs.cat_file(path, end=MAX_OBJECT_BYTES)
            assert len(content) < MAX_OBJECT_BYTES, f"Oversized metadata/metrics object: {obj.uri}"
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        actual = digest(content)
        assert obj.expected_sha256 is None or actual == obj.expected_sha256, f"Changed remote result: {obj.uri}"
        files[obj.name] = {"uri": obj.uri, "sha256": actual, "bytes": len(content)}
    receipt = {
        "order": row["order"],
        "region": row["region"],
        "training_root": row["path"],
        "collected_utc": datetime.now(UTC).isoformat(),
        "files": files,
        "validation": validate_row(folder, row, spec),
    }
    write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    records, specs, evaluations = prepare(args.output)
    started = datetime.now(UTC).isoformat()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_row, args.output, row, specs[row["order"]], evaluations) for row in records]
        receipts = []
        for future in futures:
            receipts.append(future.result())
            if len(receipts) % 20 == 0:
                print(json.dumps({"verified_rows": len(receipts), "expected_rows": 280}), flush=True)
    files = sorted(p for p in (args.output / "raw").rglob("*") if p.is_file())
    summary = {
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "rows": len(receipts),
        "regions": {r: sum(v["region"] == r for v in receipts) for r in ("east5", "europe")},
        "remote_objects": sum(len(r["files"]) for r in receipts),
        "raw_object_bytes": sum(f["bytes"] for r in receipts for f in r["files"].values()),
        "sources": {str(p): digest(p.read_bytes()) for p in input_sources().values()},
        "sha256": {str(p.relative_to(args.output)): digest(p.read_bytes()) for p in files},
        "lm_jobs": 0,
        "surrogate_fits": 0,
    }
    complete = args.output / "collection.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        for key in ("rows", "regions", "remote_objects", "raw_object_bytes", "sources", "sha256"):
            assert previous[key] == summary[key], f"Changed completed collection: {key}"
    else:
        write_json(complete, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"sources", "sha256"}}, indent=2))


if __name__ == "__main__":
    main()
