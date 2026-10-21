# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build arms from real generation traces and re-check every row independently.

Runs build.py for each requested arm into a local directory with the real pool (the version build.py pins) and the
replay slice, re-decodes every row with validate_local.check_arm (trained spans equal the glm turns, no loss on
padding or past content, held-out status matches the arm), uploads each arm prefix under --output, and prints one
report line per arm with the build's wall time and peak memory. Every output prefix gets a DRY_RUN_NOT_FOR_TRAINING
marker: validation rows never feed an arm.

Usage:
    python -m experiments.post_training.sft_shootout_rows.validate_pilot --traces <rebuilt.jsonl> ... \\
        --arms plain rs-multi --pool <tasks.jsonl> --pool-manifest <manifest.json> --replay <prefix> --output <prefix>
"""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.validate_local import check_arm
from experiments.post_training.sweep_data.prepare_agentic import audit_hashes, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=["plain", "rs-multi"])
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--builder-commit", default="unrecorded (dry run)", help="recorded in each dry-run manifest as builder_commit"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        (work / "tokenizer").mkdir()
        tokenizer = load_tokenizer(work / "tokenizer", audit_hashes())
        local_traces = []
        for index, uri in enumerate(args.traces):
            path = work / f"traces-{index}.jsonl"
            path.write_bytes(StoragePath(uri).read_bytes())
            local_traces.append(str(path))
        pool_rows = [json.loads(line) for line in StoragePath(args.pool).read_text().splitlines()]
        heldout = {row["task_id"] for row in pool_rows if row["heldout"]}
        by_id = {}
        for path in local_traces:
            for line in Path(path).read_text().splitlines():
                trace = json.loads(line)
                if trace.get("conversation", 0) == 0:
                    by_id.setdefault(trace["trial_id"], trace)
        plain = None
        for arm in args.arms:
            out = work / arm
            command = [
                sys.executable,
                "-m",
                "experiments.post_training.sft_shootout_rows.build",
                "--arm",
                arm,
                "--traces",
                *local_traces,
                "--pool",
                args.pool,
                "--pool-manifest",
                args.pool_manifest,
                "--replay",
                args.replay,
                "--output",
                str(out),
                "--builder-commit",
                args.builder_commit,
            ]
            if arm in ("rs-multi", "relay"):
                command += ["--plain", str(plain)]
            started = time.monotonic()
            subprocess.run(command, check=True)
            build_seconds = round(time.monotonic() - started, 1)
            # Peak RSS of the largest child so far (KiB on Linux); arms run in order, so later arms report at least this.
            peak_child_mib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
            if arm == "plain":
                plain = out
            report = check_arm(tokenizer, out, by_id, heldout)
            manifest = json.loads((out / "manifest.json").read_text())
            report |= {
                "candidates": manifest["candidates"],
                "B": manifest["B_supervised_tokens"],
                "replay": manifest["replay"],
                "build_seconds": build_seconds,
                "peak_child_rss_mib": peak_child_mib,
            }
            target = StoragePath(args.output) / arm
            # A dry-run prefix never gets the builder's _SUCCESS: it marks a complete production prefix.
            for path in sorted(p for p in out.iterdir() if p.name != "_SUCCESS"):
                (target / path.name).write_bytes(path.read_bytes())
                uploaded = hashlib.sha256((target / path.name).read_bytes()).hexdigest()
                if uploaded != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"Upload mismatch {arm}/{path.name}")
            (target / "DRY_RUN_NOT_FOR_TRAINING").write_text(
                "validation output of validate_pilot.py; never feed an arm\n"
            )
            print("PILOT_VALIDATION " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
