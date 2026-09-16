# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["fsspec==2026.1.0", "gcsfs==2026.1.0", "pandas==2.2.2", "numpy==2.3.5"]
# ///
"""Freeze small, completed TPP40 endpoint metrics without launching remote work."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent / "reference_outputs"
OUTPUT = BASE / "tpp40_frontier_gap_20260909"
STANDALONE = Path("/Users/calvinxu/Projects/Work/Marin/mixture-selection")
SPECS = BASE / "delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815/launch_dry_run/run_specs.json"
ASSIGNMENT = BASE / "delphi_tpp40_multiregion_assignment_20260830/assignment_v2.json"
FINAL_STEP = 27335
RUN_PATTERN = re.compile(r"fit_(\d{3})_.+-[a-f0-9]{6}$")


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    complete = OUTPUT / "snapshot.json"
    if complete.exists():
        snapshot = json.loads(complete.read_text())
        for name, expected in snapshot["sha256"].items():
            assert digest((OUTPUT / name).read_bytes()) == expected, f"changed snapshot: {name}"
        print(json.dumps(snapshot["coverage"], indent=2))
        return
    started = datetime.now(UTC).isoformat()
    inputs = OUTPUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    sources = {
        "run_specs.json": SPECS,
        "assignment.json": ASSIGNMENT,
        "mariner.py": STANDALONE / "mixture_selection.py",
        "buckets.csv": STANDALONE / "data/buckets.csv",
        "objectives.csv": STANDALONE / "data/objectives.csv",
        "historical_3e18_anchors.csv": STANDALONE / "data/anchors.csv",
    }
    for name, source in sources.items():
        target = inputs / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes(), f"source changed during collection: {source}"
        else:
            target.write_bytes(source.read_bytes())
    specs = json.loads((inputs / "run_specs.json").read_text())
    assignment = json.loads((inputs / "assignment.json").read_text())
    fs = fsspec.filesystem("gs")
    paths = []
    for region in ("east5", "europe"):
        for directory in fs.ls(assignment[f"{region}_root"]):
            match = RUN_PATTERN.fullmatch(directory.rsplit("/", 1)[-1])
            if match:
                order = int(match.group(1))
                assert directory.rsplit("/", 1)[-1][:-7] == specs[order]["run_name"]
                paths.append((region, order, directory))

    def collect(item: tuple[str, int, str]) -> dict:
        region, order, directory = item
        destination = OUTPUT / "raw" / region / f"{order:03}"
        record_path = destination / "record.json"
        if record_path.exists():
            return json.loads(record_path.read_text())
        destination.mkdir(parents=True, exist_ok=True)
        status_path = directory + "/.executor_status"
        try:
            status = fs.cat(status_path).decode().strip()
        except FileNotFoundError:
            status = "MISSING"
        record = dict(region=region, order=order, run_dir="gs://" + directory, status=status)
        if status == "SUCCESS":
            metric_path = directory + "/checkpoints/eval_metrics.jsonl"
            raw = fs.cat(metric_path)
            assert len(raw) < 2_000_000, "unexpectedly large metric file"
            metrics = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            endpoint = [row for row in metrics if row.get("step") == FINAL_STEP]
            assert endpoint, (directory, "missing final endpoint")
            scientific = [
                {key: value for key, value in row.items() if key.startswith("eval/uncheatable_eval/")}
                for row in endpoint
            ]
            assert all(row == scientific[0] for row in scientific), (directory, "conflicting final evaluations")
            record["identical_final_metric_records"] = len(endpoint)
            metadata = fs.cat(directory + f"/checkpoints/step-{FINAL_STEP}/metadata.json")
            checkpoint = json.loads(metadata)
            assert checkpoint["step"] == FINAL_STEP, (directory, checkpoint)
            (destination / "eval_metrics.jsonl").write_bytes(raw)
            (destination / "metadata.json").write_bytes(metadata)
            write_json(destination / "endpoint.json", endpoint[0])
            record.update(
                metrics_uri="gs://" + metric_path, metrics_sha256=digest(raw), metadata_sha256=digest(metadata)
            )
        write_json(record_path, record)
        return record

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(collect, paths))
    pd.DataFrame(records).to_csv(OUTPUT / "regional_status.csv", index=False)
    done = [row for row in records if row["status"] == "SUCCESS"]
    assert len({row["order"] for row in done}) == len(done), "duplicate successful region aliases need reconciliation"
    by_order = {row["order"]: row for row in done}
    outcomes, coverage = [], []
    for spec in specs:
        order = spec["run_order"]
        w0, w1 = spec["phase_weights"]["phase_0"], spec["phase_weights"]["phase_1"]
        tied = all(abs(w0[k] - w1[k]) < 1e-12 for k in w0)
        row = dict(
            order=order,
            run_name=spec["run_name"],
            source_run=spec["source_run_name"],
            source=spec["panel_source"],
            tied=tied,
            completed=order in by_order,
        )
        if order in by_order:
            record = by_order[order]
            row["region"] = record["region"]
            endpoint = json.loads((OUTPUT / "raw" / record["region"] / f"{order:03}" / "endpoint.json").read_text())
            outcomes.append({**row, **endpoint})
        coverage.append(row)
    frame = pd.DataFrame(outcomes).sort_values("order")
    assert frame.filter(regex=r"^eval/uncheatable_eval/.*/bpb$").shape[1] == 7
    assert np.isfinite(frame.filter(regex=r"^eval/uncheatable_eval/.*bpb$").to_numpy(float)).all()
    frame.to_csv(inputs / "outcomes.csv", index=False)
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(inputs / "coverage.csv", index=False)
    counts = coverage_frame.groupby(["source", "tied"]).agg(designed=("order", "size"), completed=("completed", "sum"))
    summary = {
        "completed": len(done),
        "designed": len(specs),
        "completed_tied": int(frame.tied.sum()),
        "completed_asymmetric": int((~frame.tied).sum()),
        "by_source": counts.reset_index().to_dict("records"),
        "by_region": frame.region.value_counts().to_dict(),
    }
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "snapshot.json")
    write_json(
        complete,
        {
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(),
            "coverage": summary,
            "source_paths": {k: str(v) for k, v in sources.items()},
            "sha256": {str(path.relative_to(OUTPUT)): digest(path.read_bytes()) for path in files},
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
