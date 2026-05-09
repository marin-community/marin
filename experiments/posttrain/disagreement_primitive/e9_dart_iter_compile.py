"""DART iterative validation — Phase 5 compile-with-history.

For Round N>1: re-call all 3 compilers (GPT, Gemini, Claude) with the original
DART step-3 prompt PLUS an explicit edit-history block showing what was adopted
in prior rounds and the empirical result. This is the test of compiler
self-correction.

Inputs read per statement:
  - history.json (1 to N-1 entries)
  - rubric_v{N}.json (current state — = v1 + adopted edits from rounds 1..N-1)
  - spec_v{N}.txt (current state)
  - per_judgment_iter_round_{N-1}.jsonl + baseline (combined to compute new
    poison cell rankings under the v_N state)

Outputs per statement:
  - dart_iteration/{sid}/round_{N}_compile/{compiler}.json — raw diagnosis output
  - Will be consumed by e9_dart_iter_synthesize.py --round {N} for majority vote

Only fires for statements where verdict == "improving" in round N-1. Skips
"converged" and "stuck" statements automatically.

Usage:
    .venv/bin/python e9_dart_iter_compile.py --round 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import SPEC_PATH, get_examples, render_anchors, render_examples
from e9_dart_compiler import COMPILER_SYSTEM

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def render_history_block(history: list[dict], current_round: int) -> str:
    """Build the edit-history block for the compiler prompt."""
    parts = ["=== EDIT HISTORY FOR THIS STATEMENT ===", ""]
    parts.append(
        "The rubric and spec text shown above already incorporate the edits below. "
        "The poison cells shown after this section are computed under the CURRENT state, not the baseline.\n"
    )
    for entry in history:
        rn = entry["round"]
        parts.append(f"Round {rn}:")
        parts.append(f"  Majority diagnosis: {entry.get('round_diagnosis_majority', '?')} "
                     f"(tier: {entry.get('round_diagnosis_tier', '?')})")
        ru = entry.get("rubric_edits_adopted") or []
        sp = entry.get("spec_edits_adopted") or []
        parts.append(f"  Rubric edits adopted: {len(ru)}")
        for e in ru[:6]:
            parts.append(f"    anchor {e.get('anchor')}: {(e.get('old') or '')[:80]!r} → {(e.get('new') or '')[:80]!r}")
        parts.append(f"  Spec edits adopted: {len(sp)}")
        for e in sp[:4]:
            parts.append(f"    {(e.get('old_phrase') or '')[:80]!r} → {(e.get('new_phrase') or '')[:80]!r}")
        ab = entry.get("alpha_before_round")
        aa = entry.get("alpha_after_round")
        da = entry.get("delta_alpha")
        pwv = entry.get("delta_pwv_top10_pct_drop")
        ab_s = f"{ab:.3f}" if ab is not None else "?"
        aa_s = f"{aa:.3f}" if aa is not None else "?"
        da_s = f"{da:+.3f}" if da is not None else "?"
        pwv_s = f"{pwv*100:.0f}%" if pwv is not None else "?"
        parts.append(f"  Empirical: α {ab_s} → {aa_s} (Δ={da_s}); top-10 Δpwv dropped {pwv_s}")
        parts.append(f"  Status: {entry.get('verdict', '?')}\n")
    parts.append("Given this history:")
    parts.append("- If α gain is decelerating, propose a different KIND of edit (e.g., switch from rubric to spec).")
    parts.append("- If the same disagreements persist on the new poison cells, declare irreducible.")
    parts.append("- If your prior edit moved α the wrong way, propose a reversal.")
    parts.append("- Otherwise, refine the edit.")
    if current_round >= 3:
        parts.append("- THIS IS THE FINAL ROUND. If convergence is not imminent, declare irreducible.")
    parts.append("")
    return "\n".join(parts)


def compute_poison_cells_under_current_state(sid: str, round_n: int) -> tuple[list[dict], list[dict], int, int]:
    """Compute top-K bare-poison and rubric-poison cells under the v_N state.

    Reads per_judgment_iter_round_{N-1}.jsonl + per_judgment_opposite.jsonl
    (for variant_A bare-condition) and ranks cells.

    Returns (bare_cells, rubric_cells, bare_pwv_total, rubric_pwv_total).
    """
    iter_path = ITER_DIR / f"per_judgment_iter_round_{round_n - 1}.jsonl"
    if not iter_path.exists():
        raise SystemExit(f"missing {iter_path}")
    iter_rows = load_jsonl(iter_path)
    baseline = [r for r in load_jsonl(DIR / "per_judgment_opposite.jsonl")
                if r.get("statement_id") == sid]
    iter_rows = [r for r in iter_rows if r.get("statement_id") == sid]

    # bare condition — always from baseline variant_A
    bare = [r for r in baseline if r.get("condition") == "variant_A"]
    # rubric+spec condition — from iter rows at C3 (full v2)
    rubric = [r for r in iter_rows if r.get("condition") == "C3"]

    def cell_pwv(rows):
        by_cell = defaultdict(dict)
        for r in rows:
            if r.get("score") is None:
                continue
            ck = (r["statement_id"], r["scenario_idx"], r["generator"])
            by_cell[ck][r["judge"]] = r["score"]
        out = []
        for ck, scores in by_cell.items():
            ss = [s for s in scores.values() if s is not None]
            v = 0.0
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    v += (ss[i] - ss[j]) ** 2
            out.append((ck, v, scores))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    bare_ranked = cell_pwv(bare)
    rubric_ranked = cell_pwv(rubric)
    bare_pwv_total = sum(v for _, v, _ in bare_ranked)
    rubric_pwv_total = sum(v for _, v, _ in rubric_ranked)
    return (bare_ranked[:10], rubric_ranked[:10], bare_pwv_total, rubric_pwv_total)


def compile_for_statement(sid: str, round_n: int):
    """Run all 3 compilers for one statement."""
    sid_dir = ITER_DIR / sid
    history = json.loads((sid_dir / "history.json").read_text())
    rubric_v_current = json.loads((sid_dir / f"rubric_v{round_n}.json").read_text())
    spec_v_current = (sid_dir / f"spec_v{round_n}.txt").read_text()

    # Load spec metadata for examples
    spec_meta = next((r for r in load_jsonl(SPEC_PATH) if r["id"] == sid), None)
    if spec_meta is None:
        raise SystemExit(f"spec metadata for {sid} not found")
    examples = get_examples(spec_meta)

    bare_top, rubric_top, bare_total, rubric_total = compute_poison_cells_under_current_state(sid, round_n)

    history_block = render_history_block(history, round_n)
    out_dir = sid_dir / f"round_{round_n}_compile"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the compiler user prompt — based on e9_dart_compiler.build_user_prompt but with
    # current spec_v / rubric_v and the history block inserted.
    user_prompt = (
        f"SPEC STATEMENT (current — = v1 + adopted edits from rounds 1..{round_n - 1}):\n{spec_v_current}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"CURRENT RUBRIC:\n{render_anchors(rubric_v_current)}\n\n"
        f"{history_block}\n"
        f"BARE-POISON CELLS (Σ pwv = {bare_total:.0f}, top-K under current state):\n"
        f"{format_cells(bare_top)}\n\n"
        f"RUBRIC-POISON CELLS (Σ pwv = {rubric_total:.0f}, top-K under current state):\n"
        f"{format_cells(rubric_top)}\n\n"
        f"Diagnose. Propose edits OR declare convergence/irreducible. Use the bidirectional output schema."
    )
    return user_prompt, out_dir


def format_cells(ranked: list) -> str:
    parts = []
    for i, (ck, pwv, scores) in enumerate(ranked):
        sid, scen, gen = ck
        s = ", ".join(f"{k}={v}" for k, v in scores.items())
        parts.append(f"  #{i+1} scen={scen} gen={gen} pwv={pwv:.1f} scores=[{s}]")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--statements", default="all")
    args = ap.parse_args()

    if args.round < 2:
        raise SystemExit("compile script only meaningful for round >= 2; round 1 uses Run 1/2/3 outputs")

    statements = sorted({s.name for s in ITER_DIR.iterdir() if s.is_dir() and (s / "history.json").exists()})
    if args.statements != "all":
        wanted = set(args.statements.split(","))
        statements = [s for s in statements if s in wanted]

    eligible = []
    for sid in statements:
        history = json.loads((ITER_DIR / sid / "history.json").read_text())
        if len(history) < args.round - 1:
            continue
        last_verdict = history[-1].get("verdict")
        if last_verdict == "improving":
            eligible.append(sid)
    print(f"Eligible (verdict=improving from round {args.round - 1}): {len(eligible)}")
    print(f"  {eligible}")

    # For now, this script writes the prompts but does NOT call APIs — that needs the
    # compiler-call infrastructure which lives in the existing e9_dart_compiler*.py files.
    # We dump prompts here so the orchestrator can submit them via those scripts.
    for sid in eligible:
        prompt, out_dir = compile_for_statement(sid, args.round)
        (out_dir / "user_prompt.txt").write_text(prompt)
        print(f"  wrote {out_dir}/user_prompt.txt ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
