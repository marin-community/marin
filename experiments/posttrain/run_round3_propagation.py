# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-3 propagation: each judge regenerates rubrics on a spec forked
with its R1+R2+R3 edits cumulatively.

Tests whether R3 edits propagate at the diminishing-returns rate (we saw
R2 dropped to ~22% from R1's 66%) or worse.

Output: `cross_tier_rubrics_v2_<judge>_with_r1r2r3_edits.jsonl` per judge.

Usage:
    source .env && uv run --with openai --with google-genai python \\
        experiments/posttrain/run_round3_propagation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_round3_propagation")

WORKTREE = Path(__file__).resolve().parents[2]
LM_JUDGE_EDITS = WORKTREE / "experiments/posttrain/lm_judge_edits"
BASE_SPEC = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
SPEC_DIR = WORKTREE / "experiments/posttrain/specs"
OUT_DIR = WORKTREE / "experiments/posttrain/stage3_output"
REPORT = OUT_DIR / "round3_propagation_report.md"

# Skipping glm51 since round-3 agent didn't run for it (r1r2 had max_tokens issue)
JUDGES = [
    {"label": "flash", "script": "experiments/posttrain/write_cross_tier_rubrics_v2.py", "workers": 4},
    {"label": "gpt51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_gpt51.py", "workers": 4},
    {"label": "pro", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_pro.py", "workers": 2},
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")


def fork_spec_r1r2r3(judge: str) -> tuple[Path, list]:
    """Build a spec forked with R1+R2+R3 edits applied cumulatively.

    Note: Uses _origin metadata field instead of bracketed description prefix
    to avoid the leakage bug where prefix appeared in rubric citations.
    """
    base = load_jsonl(BASE_SPEC)
    by_id = {r["id"]: r for r in base}

    applied: list[tuple[str, str, str]] = []
    for round_dir in ["proposed_edits", "round2_proposed_edits", "round3_proposed_edits"]:
        edits_dir = LM_JUDGE_EDITS / judge / round_dir
        if not edits_dir.exists():
            continue
        for f in sorted(edits_dir.glob("*.json")):
            e = json.loads(f.read_text())
            target = e["target_statement_id"]
            if target not in by_id:
                logger.warning("[%s] skipping edit %s — target %s not in spec", judge, e["edit_id"], target)
                continue
            new_example = dict(e["new_example"])
            # FIX: don't pollute description with traceability prefix; use _origin metadata
            new_example["_origin"] = f"{judge}/{round_dir}/{e['edit_id']}"
            meta = by_id[target].setdefault("metadata", {})
            examples = meta.setdefault("examples", [])
            examples.append(new_example)
            applied.append((round_dir, e["edit_id"], target))

    forked = SPEC_DIR / f"openai_model_spec_{judge}_r1r2r3_edits.jsonl"
    write_jsonl(forked, base)
    logger.info("[%s] forked spec: %d cumulative edits → %s", judge, len(applied), forked.name)
    return forked, applied


def run_judge(judge: dict, smoke: bool = False) -> dict:
    label = judge["label"]
    forked, applied = fork_spec_r1r2r3(label)
    output_path = OUT_DIR / f"cross_tier_rubrics_v2_{label}_with_r1r2r3_edits.jsonl"
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
        str(output_path),
        "--max-workers",
        str(judge["workers"]),
    ]
    if smoke:
        cmd.append("--smoke")
    logger.info("running %s: %s", label, " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(WORKTREE), capture_output=False, text=True)
    elapsed = time.time() - t0
    success = proc.returncode == 0 and output_path.exists()
    if success:
        rows = load_jsonl(output_path)
        schema = sum(1 for r in rows if r["diag"].get("schema_ok"))
        stats = {"rows": len(rows), "schema_ok": f"{schema}/{len(rows)}", "elapsed_s": round(elapsed, 1)}
    else:
        stats = {"rows": 0, "schema_ok": "0/0", "elapsed_s": round(elapsed, 1)}
        logger.error("%s failed: rc=%d", label, proc.returncode)
    return {"label": label, "n_edits": len(applied), "output": str(output_path), "success": success, "stats": stats}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--judges", nargs="+", default=[j["label"] for j in JUDGES])
    args = parser.parse_args()

    judges_to_run = [j for j in JUDGES if j["label"] in args.judges]
    logger.info("running round-3 propagation on %d judges", len(judges_to_run))
    results = [run_judge(j, smoke=args.smoke) for j in judges_to_run]

    lines = [
        "# Round-3 propagation run report (R1+R2+R3 cumulative)",
        "",
        "Each judge regenerated rubrics against a spec forked with R1+R2+R3 edits applied.",
        "Note: glm51 skipped (no R3 review; R1+R2 hit max_tokens issue).",
        "**FIX in this run**: edit traceability stored in `_origin` metadata field",
        "instead of bracketed description prefix (per gpt51 R3 finding of path leakage).",
        "",
    ]
    lines.append("## Results")
    lines.append("")
    lines.append("| judge | r1+r2+r3 edits | rows | schema_ok | elapsed |")
    lines.append("|---|---:|---:|---|---:|")
    for r in results:
        s = r["stats"]
        status = f"{s['rows']} | {s['schema_ok']}" if r["success"] else "**FAILED**"
        lines.append(f"| {r['label']} | {r['n_edits']} | {status} | {s['elapsed_s']}s |")
    REPORT.write_text("\n".join(lines))
    logger.info("wrote report")
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
