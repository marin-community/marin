# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run each judge against a spec forked with COMPILER-PROPOSED edits.

This validates the LM compiler primitive beyond target_statement_id match —
do the compiler's NEW_EXAMPLES actually propagate when applied?

Output: `cross_tier_rubrics_v2_<judge>_with_compiler_edits.jsonl` per judge.

Usage:
    source .env && uv run --with openai --with google-genai python \\
        experiments/posttrain/run_compiler_edits_propagation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_compiler_edits_propagation")

WORKTREE = Path(__file__).resolve().parents[2]
LM_COMPILER_EDITS = WORKTREE / "experiments/posttrain/lm_compiler_proposed_edits"
BASE_SPEC = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
SPEC_DIR = WORKTREE / "experiments/posttrain/specs"
OUT_DIR = WORKTREE / "experiments/posttrain/stage3_output"

JUDGES = [
    {"label": "flash", "script": "experiments/posttrain/write_cross_tier_rubrics_v2.py", "workers": 4},
    {"label": "gpt51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_gpt51.py", "workers": 4},
    {"label": "pro", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_pro.py", "workers": 2},
    {"label": "glm51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_glm51.py", "workers": 4},
]


def fork_spec_with_compiler_edits(judge: str) -> Path:
    base = [json.loads(l) for l in open(BASE_SPEC) if l.strip()]
    by_id = {r["id"]: r for r in base}

    edits_dir = LM_COMPILER_EDITS / judge
    n = 0
    for f in sorted(edits_dir.glob("*.json")):
        e = json.loads(f.read_text())
        target = e.get("target_statement_id")
        if target not in by_id:
            logger.warning("[%s] skipping edit %s — target %s not in spec", judge, e.get("edit_id"), target)
            continue
        new_example = dict(e["new_example"])
        new_example["_origin"] = f"compiler/{judge}/{e.get('edit_id')}"
        meta = by_id[target].setdefault("metadata", {})
        examples = meta.setdefault("examples", [])
        examples.append(new_example)
        n += 1

    forked = SPEC_DIR / f"openai_model_spec_{judge}_compiler_edits.jsonl"
    forked.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in base) + "\n")
    logger.info("[%s] forked spec with %d compiler edits → %s", judge, n, forked.name)
    return forked


def run_judge(judge: dict) -> dict:
    label = judge["label"]
    forked = fork_spec_with_compiler_edits(label)
    output = OUT_DIR / f"cross_tier_rubrics_v2_{label}_with_compiler_edits.jsonl"
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
        str(forked),
        "--output",
        str(output),
        "--max-workers",
        str(judge["workers"]),
    ]
    logger.info("running %s: %s", label, " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(WORKTREE), capture_output=False, text=True)
    elapsed = time.time() - t0
    success = proc.returncode == 0 and output.exists()
    if success:
        rows = [json.loads(l) for l in open(output) if l.strip()]
        schema = sum(1 for r in rows if r["diag"].get("schema_ok"))
        stats = {"rows": len(rows), "schema_ok": f"{schema}/{len(rows)}", "elapsed_s": round(elapsed, 1)}
    else:
        stats = {"rows": 0, "schema_ok": "0/0", "elapsed_s": round(elapsed, 1)}
        logger.error("%s failed: rc=%d", label, proc.returncode)
    return {"label": label, "output": str(output), "success": success, "stats": stats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judges", nargs="+", default=[j["label"] for j in JUDGES])
    args = parser.parse_args()
    judges = [j for j in JUDGES if j["label"] in args.judges]
    results = [run_judge(j) for j in judges]
    for r in results:
        logger.info("%s: %s", r["label"], r["stats"])
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
