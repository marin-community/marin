# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run all 4 writers on the STRONG-only filtered spec, write per-judge
output to `cross_tier_rubrics_v2_<judge>_with_strong_only_edits.jsonl`.

Pre-req: `build_strong_filtered_spec.py` to produce
`openai_model_spec_strong_r1_only.jsonl`.

Usage:
    source .env && uv run --with openai --with google-genai python \\
        experiments/posttrain/run_strong_filtered_propagation.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_strong_filtered_propagation")

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
SPEC = WORKTREE / "experiments/posttrain/specs/openai_model_spec_strong_r1_only.jsonl"

JUDGES = [
    {"label": "flash", "script": "experiments/posttrain/write_cross_tier_rubrics_v2.py", "workers": 4},
    {"label": "gpt51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_gpt51.py", "workers": 4},
    {"label": "pro", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_pro.py", "workers": 2},
    {"label": "glm51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_glm51.py", "workers": 4},
]


def run_judge(judge):
    label = judge["label"]
    output = STAGE3 / f"cross_tier_rubrics_v2_{label}_with_strong_only_edits.jsonl"
    cmd = [
        "uv",
        "run",
        "--with",
        "openai",
        "--with",
        "google-genai",
        "python",
        judge["script"],
        "--spec-path",
        str(SPEC),
        "--output",
        str(output),
        "--max-workers",
        str(judge["workers"]),
    ]
    logger.info("running %s", label)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(WORKTREE), capture_output=False, text=True)
    elapsed = time.time() - t0
    if proc.returncode == 0 and output.exists():
        rows = [json.loads(line) for line in output.open() if line.strip()]
        schema = sum(1 for r in rows if r["diag"].get("schema_ok"))
        logger.info("%s: rows=%d schema_ok=%d/%d elapsed=%.0fs", label, len(rows), schema, len(rows), elapsed)
        return True
    logger.error("%s failed rc=%d", label, proc.returncode)
    return False


def main():
    if not SPEC.exists():
        raise SystemExit(f"missing {SPEC}; run build_strong_filtered_spec.py first")
    results = [run_judge(j) for j in JUDGES]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
