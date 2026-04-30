# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run all 4 judges against the union spec (all 29 round-1 edits applied).

For Experiment 3 (cross-judge edit benefit): does each judge produce
better rubrics when given access to edits proposed by ALL judges, not
just its own?

Sequential per-judge to avoid Google API rate-limit conflicts between
Flash and Pro.

Usage:
    source .env && uv run --with openai --with google-genai python \\
        experiments/posttrain/run_union_spec_propagation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_union_spec_propagation")

WORKTREE = Path(__file__).resolve().parents[2]
UNION_SPEC = WORKTREE / "experiments/posttrain/specs/openai_model_spec_union_round1_edits.jsonl"
OUT_DIR = WORKTREE / "experiments/posttrain/stage3_output"
REPORT = OUT_DIR / "union_spec_propagation_report.md"

JUDGES = [
    {"label": "flash", "script": "experiments/posttrain/write_cross_tier_rubrics_v2.py", "workers": 4},
    {"label": "gpt51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_gpt51.py", "workers": 4},
    {"label": "pro", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_pro.py", "workers": 2},
    {"label": "glm51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_glm51.py", "workers": 4},
]


def compute_stats(jsonl_path: Path) -> dict:
    if not jsonl_path.exists():
        return {"rows": 0, "schema_ok": "0/0"}
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    n = len(rows)
    schema_ok = sum(1 for r in rows if r["diag"].get("schema_ok"))
    return {"rows": n, "schema_ok": f"{schema_ok}/{n}"}


def run_judge(judge: dict, smoke: bool = False) -> dict:
    label = judge["label"]
    output_path = OUT_DIR / f"cross_tier_rubrics_v2_{label}_with_union_edits.jsonl"
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
        str(UNION_SPEC),
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
    stats = compute_stats(output_path) if success else {"rows": 0, "schema_ok": "0/0"}
    stats["elapsed_s"] = round(elapsed, 1)
    if success:
        logger.info("%s done in %.1fs: %s", label, elapsed, stats)
    else:
        logger.error("%s failed: rc=%d", label, proc.returncode)
    return {"label": label, "output": str(output_path), "success": success, "stats": stats}


def write_report(results: list[dict], smoke: bool) -> None:
    lines = [
        "# Union-spec propagation run report",
        "",
        f"All 4 judges ran against the union spec (`{UNION_SPEC.name}`) which",
        "contains the BASE spec + all 29 round-1 proposed edits applied.",
        "",
        "Tests whether each judge benefits from having access to other judges'",
        "proposed edits, not just its own self-edits.",
        "",
    ]
    if smoke:
        lines.append("**SMOKE MODE** — only first cross-tier record processed per judge.")
        lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| judge | rows | schema_ok | elapsed |")
    lines.append("|---|---:|---|---:|")
    for r in results:
        if not r["success"]:
            lines.append(f"| {r['label']} | - | **FAILED** | {r['stats']['elapsed_s']}s |")
        else:
            s = r["stats"]
            lines.append(f"| {r['label']} | {s['rows']} | {s['schema_ok']} | {s['elapsed_s']}s |")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for r in results:
        if r["success"]:
            lines.append(f"- **{r['label']}**: `{Path(r['output']).relative_to(WORKTREE)}`")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("Compare each judge's `with_union_edits` rubrics to its `with_self_edits`")
    lines.append("baseline to measure whether the additional 21-23 cross-judge edits help.")
    REPORT.write_text("\n".join(lines))
    logger.info("wrote report to %s", REPORT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--judges", nargs="+", default=[j["label"] for j in JUDGES])
    args = parser.parse_args()

    if not UNION_SPEC.exists():
        logger.error("union spec not found at %s — run union spec build first", UNION_SPEC)
        return 1

    judges_to_run = [j for j in JUDGES if j["label"] in args.judges]
    if not judges_to_run:
        logger.error("no valid judges in %s", args.judges)
        return 1

    logger.info("running union-spec propagation on %d judges", len(judges_to_run))
    results = [run_judge(j, smoke=args.smoke) for j in judges_to_run]
    write_report(results, smoke=args.smoke)
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
