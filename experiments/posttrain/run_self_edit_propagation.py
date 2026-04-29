# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-judge self-edit propagation runner.

For each LM judge (flash / gpt51 / pro / glm51):
1. Load that judge's proposed spec edits from
   `experiments/posttrain/lm_judge_edits/<judge>/proposed_edits/*.json`.
2. Validate edits (parsable, target_statement_id exists, only add_example
   channel, new_example has all 4 required fields).
3. Fork the OpenAI Model Spec by appending each edit's `new_example` to
   the target statement's `metadata.examples` array. The new example's
   `description` is prefixed with `[<judge>/<edit_id>]` so it stays
   traceable post-fork.
4. Save the forked spec at
   `experiments/posttrain/specs/openai_model_spec_<judge>_self_edits.jsonl`.
5. Run that judge's writer (with --spec-path pointing at the forked spec)
   to regenerate the 22 cross-tier rubrics against the forked spec.
6. Output to
   `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_<judge>_with_self_edits.jsonl`.

**Each judge sees ONLY its own forked spec.** No cross-contamination.

By default runs all 4 judges sequentially (~10-15 min wall, ~$0.73 spend).
Use `--judges flash gpt51` to subset, or `--fork-only` to fork specs
without running writers (useful for inspection).

Usage:
    source .env && uv run --with openai --with google-genai python \\
        experiments/posttrain/run_self_edit_propagation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_self_edit_propagation")

WORKTREE = Path(__file__).resolve().parents[2]
LM_JUDGE_EDITS = WORKTREE / "experiments/posttrain/lm_judge_edits"
BASE_SPEC = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
SPEC_DIR = WORKTREE / "experiments/posttrain/specs"
OUT_DIR = WORKTREE / "experiments/posttrain/stage3_output"
REPORT = OUT_DIR / "self_edit_propagation_report.md"

JUDGE_CONFIG: dict[str, dict] = {
    "flash": {
        "writer_script": "experiments/posttrain/write_cross_tier_rubrics_v2.py",
        "max_workers": 4,
    },
    "gpt51": {
        "writer_script": "experiments/posttrain/write_cross_tier_rubrics_v2_gpt51.py",
        "max_workers": 4,
    },
    "pro": {
        "writer_script": "experiments/posttrain/write_cross_tier_rubrics_v2_pro.py",
        "max_workers": 2,
    },
    "glm51": {
        "writer_script": "experiments/posttrain/write_cross_tier_rubrics_v2_glm51.py",
        "max_workers": 4,
    },
}

REQUIRED_EDIT_KEYS = {"edit_id", "target_statement_id", "edit_channel", "new_example"}
REQUIRED_NEW_EXAMPLE_KEYS = {"description", "user_query", "good_response", "bad_response"}


def load_spec_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_spec_records(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n")


def load_edits(judge: str) -> list[dict]:
    edits_dir = LM_JUDGE_EDITS / judge / "proposed_edits"
    if not edits_dir.exists():
        return []
    edits = []
    for f in sorted(edits_dir.glob("*.json")):
        try:
            edits.append(json.loads(f.read_text()))
        except json.JSONDecodeError as exc:
            logger.error("failed to parse %s: %s", f, exc)
            sys.exit(1)
    return edits


def validate_edits(judge: str, edits: list[dict], spec_by_id: dict[str, dict]) -> None:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for e in edits:
        eid = e.get("edit_id", "?")
        if eid in seen_ids:
            errors.append(f"{eid}: duplicate edit_id within {judge}")
        seen_ids.add(eid)
        for k in REQUIRED_EDIT_KEYS:
            if k not in e:
                errors.append(f"{eid}: missing top-level key {k!r}")
        target = e.get("target_statement_id")
        if target and target not in spec_by_id:
            errors.append(f"{eid}: target_statement_id {target!r} not found in spec")
        ch = e.get("edit_channel")
        if ch and ch != "add_example":
            errors.append(f"{eid}: only edit_channel='add_example' supported, got {ch!r}")
        ne = e.get("new_example") or {}
        for k in REQUIRED_NEW_EXAMPLE_KEYS:
            if k not in ne:
                errors.append(f"{eid}: missing new_example.{k}")
    if errors:
        logger.error("validation errors for judge %s:", judge)
        for err in errors:
            logger.error("  %s", err)
        sys.exit(1)


def fork_spec(judge: str, edits: list[dict]) -> Path:
    """Build a per-judge forked spec by appending each edit's new_example
    to the target statement's metadata.examples list."""
    base = load_spec_records(BASE_SPEC)
    by_id = {r["id"]: r for r in base}

    applied: list[tuple[str, str]] = []
    for e in edits:
        target = e["target_statement_id"]
        new_example = dict(e["new_example"])
        # Trace tag in description for post-hoc identification
        original_desc = new_example.get("description", "")
        new_example["description"] = f"[{judge}/{e['edit_id']}] {original_desc}"
        stmt = by_id[target]
        meta = stmt.setdefault("metadata", {})
        examples = meta.setdefault("examples", [])
        examples.append(new_example)
        applied.append((e["edit_id"], target))

    forked_path = SPEC_DIR / f"openai_model_spec_{judge}_self_edits.jsonl"
    write_spec_records(forked_path, base)
    logger.info("forked spec for %s: %d edits applied → %s", judge, len(applied), forked_path.name)
    for eid, target in applied:
        logger.info("  + %s → %s", eid, target)
    return forked_path


def compute_stats(jsonl_path: Path) -> dict:
    if not jsonl_path.exists():
        return {"rows": 0, "schema_ok": "0/0"}
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    n = len(rows)
    schema_ok = sum(1 for r in rows if r["diag"].get("schema_ok"))
    return {"rows": n, "schema_ok": f"{schema_ok}/{n}"}


def run_writer(judge: str, forked_spec: Path, smoke: bool = False) -> tuple[bool, Path, dict]:
    config = JUDGE_CONFIG[judge]
    output_path = OUT_DIR / f"cross_tier_rubrics_v2_{judge}_with_self_edits.jsonl"
    cmd = [
        "uv",
        "run",
        "--with",
        "openai",
        "--with",
        "google-genai",
        "python",
        config["writer_script"],
        "--spec-path",
        str(forked_spec),
        "--output",
        str(output_path),
        "--max-workers",
        str(config["max_workers"]),
    ]
    if smoke:
        cmd.append("--smoke")
    logger.info("running %s: %s", judge, " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(WORKTREE), capture_output=False, text=True)
    elapsed = time.time() - t0
    success = proc.returncode == 0 and output_path.exists()
    if not success:
        logger.error("%s failed: rc=%d (output_exists=%s)", judge, proc.returncode, output_path.exists())
        return False, output_path, {"elapsed_s": round(elapsed, 1), "rows": 0, "schema_ok": "0/0"}
    stats = compute_stats(output_path)
    stats["elapsed_s"] = round(elapsed, 1)
    logger.info("%s done in %.1fs: %s", judge, elapsed, stats)
    return True, output_path, stats


def write_report(results: dict[str, dict], smoke: bool = False) -> None:
    lines = [
        "# Self-edit propagation run report",
        "",
        "Each LM judge re-ran cross-tier rubric generation on a spec forked",
        "with that judge's own proposed edits applied. Each judge sees only",
        "its own forked spec (no cross-judge contamination).",
        "",
    ]
    if smoke:
        lines.append("**SMOKE MODE** — only the first cross-tier record was processed per judge.")
        lines.append("")
    lines.append("## Per-judge results")
    lines.append("")
    lines.append("| judge | edits applied | forked spec | output | rows | schema_ok | elapsed |")
    lines.append("|---|---:|---|---|---:|---|---:|")
    for judge, r in results.items():
        if not r.get("success"):
            lines.append(
                f"| {judge} | {r.get('n_edits', '-')} | {Path(r.get('forked_spec', '-')).name} "
                f"| **FAILED** | - | - | {r.get('stats', {}).get('elapsed_s', '-')}s |"
            )
            continue
        s = r["stats"]
        lines.append(
            f"| {judge} | {r['n_edits']} | {Path(r['forked_spec']).name} | "
            f"{Path(r['output']).name} | {s['rows']} | {s['schema_ok']} | {s['elapsed_s']}s |"
        )
    lines.append("")
    lines.append("## Source edits")
    lines.append("")
    for judge, r in results.items():
        lines.append(
            f"- **{judge}**: {r.get('n_edits', '?')} edits from "
            f"`experiments/posttrain/lm_judge_edits/{judge}/proposed_edits/`"
        )
    lines.append("")
    lines.append("## Output rubric files")
    lines.append("")
    for judge, r in results.items():
        if r.get("output"):
            lines.append(f"- **{judge}**: `{Path(r['output']).relative_to(WORKTREE)}`")
    lines.append("")
    lines.append("## Forked specs")
    lines.append("")
    for judge, r in results.items():
        if r.get("forked_spec"):
            lines.append(f"- **{judge}**: `{Path(r['forked_spec']).relative_to(WORKTREE)}`")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("Compare each judge's `with_self_edits` rubrics against the original")
    lines.append("`cross_tier_rubrics_v2_<judge>.jsonl` to measure whether the proposed")
    lines.append("edits actually moved the rubrics in the predicted direction.")
    REPORT.write_text("\n".join(lines))
    logger.info("wrote report to %s", REPORT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judges", nargs="+", default=list(JUDGE_CONFIG.keys()), help="Which judges to run (default: all 4)."
    )
    parser.add_argument("--fork-only", action="store_true", help="Only fork specs; do not run writers.")
    parser.add_argument(
        "--smoke", action="store_true", help="Run each writer on 1 record only (for fast end-to-end test)."
    )
    args = parser.parse_args()

    invalid = [j for j in args.judges if j not in JUDGE_CONFIG]
    if invalid:
        logger.error("unknown judges: %s; valid: %s", invalid, list(JUDGE_CONFIG))
        return 1

    base = load_spec_records(BASE_SPEC)
    spec_by_id = {r["id"]: r for r in base}
    logger.info("loaded base spec from %s (%d statements)", BASE_SPEC.name, len(base))

    results: dict[str, dict] = {}

    # Step 1: validate + fork all specs first (fast, no API calls)
    for judge in args.judges:
        edits = load_edits(judge)
        logger.info("[%s] loaded %d edits", judge, len(edits))
        if not edits:
            logger.warning("[%s] no edits found, skipping", judge)
            results[judge] = {"n_edits": 0, "skipped": True}
            continue
        validate_edits(judge, edits, spec_by_id)
        forked_spec = fork_spec(judge, edits)
        results[judge] = {
            "n_edits": len(edits),
            "forked_spec": str(forked_spec),
        }

    if args.fork_only:
        logger.info("--fork-only set; not running writers")
        return 0

    # Step 2: run each judge's writer pointed at its forked spec
    for judge in args.judges:
        if results[judge].get("skipped") or "forked_spec" not in results[judge]:
            continue
        forked_spec = Path(results[judge]["forked_spec"])
        success, output_path, stats = run_writer(judge, forked_spec, smoke=args.smoke)
        results[judge].update(
            {
                "success": success,
                "output": str(output_path),
                "stats": stats,
            }
        )

    write_report(results, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
