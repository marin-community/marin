# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build arms from real #207 pilot traces and re-check every row independently (task #211).

Runs build.py for each requested arm into a local directory with the real pool v4 and the #202 replay slice, re-decodes
every row with validate_local.check_arm (trained spans equal the glm turns, no loss on padding or past content,
held-out status matches the arm), uploads each arm prefix under --output, and prints one report line per arm.

Usage:
    python -m experiments.post_training.sft_shootout_rows.validate_pilot --traces <rebuilt.jsonl> ... \\
        --arms plain rs-multi --pool <tasks.jsonl> --pool-manifest <manifest.json> --replay <prefix> --output <prefix>
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.validate_local import check_arm
from experiments.post_training.sweep_data.prepare_agentic import STUDENT_MODEL, STUDENT_REVISION, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=["plain", "rs-multi"])
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        audit = hf_hub_download(STUDENT_MODEL, "provenance/campaign3/audit.json", revision=STUDENT_REVISION)
        expected = {name: entry["sha256"] for name, entry in json.loads(Path(audit).read_text())["files"].items()}
        (work / "tokenizer").mkdir()
        tokenizer = load_tokenizer(work / "tokenizer", expected)
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
            ]
            if arm in ("rs-multi", "relay"):
                command += ["--plain", str(plain)]
            subprocess.run(command, check=True)
            if arm == "plain":
                plain = out
            report = check_arm(tokenizer, out, by_id, heldout)
            manifest = json.loads((out / "manifest.json").read_text())
            report |= {
                "candidates": manifest["candidates"],
                "B": manifest["B_supervised_tokens"],
                "replay": manifest["replay"],
            }
            target = StoragePath(args.output) / arm
            for path in sorted(out.iterdir()):
                (target / path.name).write_bytes(path.read_bytes())
                uploaded = hashlib.sha256((target / path.name).read_bytes()).hexdigest()
                if uploaded != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"Upload mismatch {arm}/{path.name}")
            print("PILOT_VALIDATION " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
