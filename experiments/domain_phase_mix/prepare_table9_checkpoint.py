# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze both accuracy suites for a newly completed regional Qwen3 checkpoint.

Reads object metadata and small JSON files, never model weights. This does not
launch evaluation or change training. Repeating identical preparation is safe;
an existing output directory cannot silently be assigned different plans.
"""

import argparse
import hashlib
import json
from pathlib import Path

import fsspec

from experiments.domain_phase_mix import evaluate_mariner_ladder_accuracy as overlap
from experiments.domain_phase_mix import evaluate_table9_accuracy as backfill


def checkpoint_row(name: str, method: str, uri: str, step: int, metadata_uri: str, experiment: str) -> dict:
    if step <= 0 or not uri.endswith(f"/hf/step-{step}"):
        raise ValueError("Expected an explicitly selected HF export step")
    if not all(path.startswith(backfill.PREFIX + "/") for path in (uri, metadata_uri)):
        raise ValueError("Checkpoint and training metadata must be in east5")
    payload = overlap.read_bytes(metadata_uri)
    metadata = json.loads(payload)
    if metadata["step"] != step or metadata["is_temporary"]:
        raise ValueError("Training checkpoint must be permanent and match the selected export step")
    fs, path = fsspec.core.url_to_fs(uri)
    files = {}
    for obj in fs.ls(path, detail=True):
        if obj["type"] != "file":
            continue
        filename = obj["name"].removeprefix(path.rstrip("/") + "/")
        if "/" in filename:
            raise ValueError("Unexpected nested checkpoint object")
        files[filename] = {key: obj[key] for key in ("size", "generation", "crc32c")}
    overlap.checkpoint_model_config(uri)
    return {
        "name": name,
        "method": method,
        "checkpoint_uri": uri,
        "checkpoint_step": step,
        "checkpoint_files": files,
        "source_training_experiment": experiment,
        "training_metadata": {"uri": metadata_uri, "sha256": hashlib.sha256(payload).hexdigest(), "data": metadata},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--training-metadata", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--overlap-template", type=Path, required=True)
    parser.add_argument("--backfill-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = checkpoint_row(args.name, args.method, args.checkpoint, args.step, args.training_metadata, args.experiment)
    first = json.loads(args.overlap_template.read_text()) | {
        "rows": [row],
        "source_hashes": overlap.code_pins(),
        "uv_lock_sha256": overlap.code_pins()["uv.lock"],
        "runtime_versions": overlap.runtime_versions(),
        "lm_eval_revision": overlap.lm_eval_revision(),
    }
    second = json.loads(args.backfill_template.read_text()) | {
        "rows": [row],
        "source_pins": backfill.source_pins(),
        "runtime_versions": overlap.runtime_versions(),
        "lm_eval_revision": overlap.lm_eval_revision(),
    }
    overlap.validate_plan(first)
    overlap.validate_inputs(first)
    backfill.validate_plan(second)
    plans = {"overlap_plan.json": first, "backfill_plan.json": second}
    payloads = {name: json.dumps(plan, indent=2) + "\n" for name, plan in plans.items()}
    for name, text in payloads.items():
        path = args.output / name
        if path.exists() and path.read_text() != text:
            raise ValueError(f"Refusing to replace an already frozen plan: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, text in payloads.items():
        (args.output / name).write_text(text)
        print(f"{name}: {overlap.plan_sha256(plans[name])}")


if __name__ == "__main__":
    main()
