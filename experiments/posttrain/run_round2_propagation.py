# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-2 propagation: each judge regenerates rubrics on a spec forked
with BOTH its round-1 + round-2 proposed edits applied (cumulatively).

Tests whether iterative spec refinement converges. After round-1 edits
propagated 66% strong, round-2 found 25 additional pathologies. If
applying round-1 + round-2 produces stable rubrics, we have evidence the
edit-and-regen loop converges.

Output: `cross_tier_rubrics_v2_<judge>_with_r1r2_edits.jsonl` per judge.

Usage:
    source .env && uv run --with openai --with google-genai python \\
        experiments/posttrain/run_round2_propagation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_round2_propagation")

WORKTREE = Path(__file__).resolve().parents[2]
LM_JUDGE_EDITS = WORKTREE / "experiments/posttrain/lm_judge_edits"
BASE_SPEC = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
SPEC_DIR = WORKTREE / "experiments/posttrain/specs"
OUT_DIR = WORKTREE / "experiments/posttrain/stage3_output"
REPORT = OUT_DIR / "round2_propagation_report.md"

JUDGES = [
    {"label": "flash", "script": "experiments/posttrain/write_cross_tier_rubrics_v2.py", "workers": 4},
    {"label": "gpt51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_gpt51.py", "workers": 4},
    {"label": "pro", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_pro.py", "workers": 2},
    {"label": "glm51", "script": "experiments/posttrain/write_cross_tier_rubrics_v2_glm51.py", "workers": 4},
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")


def load_round_edits(judge: str, round_dir: str) -> list[dict]:
    edits_dir = LM_JUDGE_EDITS / judge / round_dir
    if not edits_dir.exists():
        return []
    edits = []
    for f in sorted(edits_dir.glob("*.json")):
        edits.append(json.loads(f.read_text()))
    return edits


def fork_spec_cumulative(judge: str) -> tuple[Path, list[tuple[str, str, str]]]:
    """Apply round-1 + round-2 edits cumulatively to base spec."""
    base = load_jsonl(BASE_SPEC)
    by_id = {r["id"]: r for r in base}

    applied: list[tuple[str, str, str]] = []
    # Round-1 first (preserve order)
    for r in ["proposed_edits", "round2_proposed_edits"]:
        edits = load_round_edits(judge, r)
        for e in edits:
            target = e["target_statement_id"]
            if target not in by_id:
                logger.warning("[%s] skipping edit %s — target %s not in spec", judge, e["edit_id"], target)
                continue
            new_example = dict(e["new_example"])
            original_desc = new_example.get("description", "") or ""
            new_example["description"] = f"[{judge}/{r}/{e['edit_id']}] {original_desc}"
            meta = by_id[target].setdefault("metadata", {})
            examples = meta.setdefault("examples", [])
            examples.append(new_example)
            applied.append((r, e["edit_id"], target))

    forked = SPEC_DIR / f"openai_model_spec_{judge}_r1r2_edits.jsonl"
    write_jsonl(forked, base)
    return forked, applied


def compute_stats(jsonl_path: Path) -> dict:
    if not jsonl_path.exists():
        return {"rows": 0, "schema_ok": "0/0"}
    rows = load_jsonl(jsonl_path)
    n = len(rows)
    schema_ok = sum(1 for r in rows if r["diag"].get("schema_ok"))
    return {"rows": n, "schema_ok": f"{schema_ok}/{n}"}


def run_judge(judge: dict, smoke: bool = False) -> dict:
    label = judge["label"]
    forked, applied = fork_spec_cumulative(label)
    logger.info("[%s] forked spec: %d cumulative edits applied → %s", label, len(applied), forked.name)

    output_path = OUT_DIR / f"cross_tier_rubrics_v2_{label}_with_r1r2_edits.jsonl"
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
    stats = compute_stats(output_path) if success else {"rows": 0, "schema_ok": "0/0"}
    stats["elapsed_s"] = round(elapsed, 1)
    if success:
        logger.info("%s done in %.1fs: %s", label, elapsed, stats)
    else:
        logger.error("%s failed: rc=%d", label, proc.returncode)
    return {
        "label": label,
        "n_edits": len(applied),
        "applied_edits": applied,
        "forked_spec": str(forked),
        "output": str(output_path),
        "success": success,
        "stats": stats,
    }


def write_report(results: list[dict], smoke: bool) -> None:
    lines = [
        "# Round-2 (cumulative r1+r2) propagation run report",
        "",
        "Each judge ran rubric regeneration against a spec with BOTH its round-1",
        "AND round-2 proposed edits applied cumulatively. Tests convergence of",
        "the edit-and-regen loop.",
        "",
    ]
    if smoke:
        lines.append("**SMOKE MODE** — only first cross-tier record processed per judge.")
        lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| judge | r1+r2 edits | rows | schema_ok | elapsed |")
    lines.append("|---|---:|---:|---|---:|")
    for r in results:
        if not r["success"]:
            lines.append(f"| {r['label']} | {r['n_edits']} | - | **FAILED** | {r['stats']['elapsed_s']}s |")
        else:
            s = r["stats"]
            lines.append(f"| {r['label']} | {r['n_edits']} | {s['rows']} | {s['schema_ok']} | {s['elapsed_s']}s |")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for r in results:
        if r["success"]:
            lines.append(f"- **{r['label']}**: `{Path(r['output']).relative_to(WORKTREE)}`")
            lines.append(f"  - forked spec: `{Path(r['forked_spec']).relative_to(WORKTREE)}`")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("Compare each judge's `with_r1r2_edits` rubrics to:")
    lines.append("- baseline (no edits): cumulative effect of 2 rounds")
    lines.append("- `with_self_edits` (round-1 only): marginal effect of round-2")
    REPORT.write_text("\n".join(lines))
    logger.info("wrote report to %s", REPORT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--judges", nargs="+", default=[j["label"] for j in JUDGES])
    args = parser.parse_args()

    judges_to_run = [j for j in JUDGES if j["label"] in args.judges]
    if not judges_to_run:
        logger.error("no valid judges in %s", args.judges)
        return 1

    logger.info("running round-2 (cumulative r1+r2) propagation on %d judges", len(judges_to_run))
    results = [run_judge(j, smoke=args.smoke) for j in judges_to_run]
    write_report(results, smoke=args.smoke)
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
