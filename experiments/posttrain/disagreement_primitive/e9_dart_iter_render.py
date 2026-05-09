"""DART iterative validation — final report renderer.

Reads all per-statement history.json files + round summaries and produces
.agents/logbooks/dart_run_004_iterative.md with α tables, per-statement
verdicts, and the escalation queue.

Usage:
    .venv/bin/python e9_dart_iter_render.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"
OUT = Path(".agents/logbooks/dart_run_004_iterative.md")
ESCALATION_OUT = ITER_DIR / "escalation_queue.md"


def main():
    statements = sorted(s.name for s in ITER_DIR.iterdir()
                        if s.is_dir() and (s / "history.json").exists())
    parts = ["# DART Run 4 — Iterative validation report", ""]
    parts.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    parts.append(f"Statements: {len(statements)}")
    parts.append("")

    # Build cross-statement summary
    summary_rows = []
    for sid in statements:
        history = json.loads((ITER_DIR / sid / "history.json").read_text())
        last = history[-1]
        summary_rows.append({
            "sid": sid,
            "rounds": len(history),
            "alpha_baseline": history[0].get("alpha_before_round"),
            "alpha_final": last.get("alpha_after_round"),
            "delta_alpha_total": (
                (last.get("alpha_after_round") or 0) - (history[0].get("alpha_before_round") or 0)
                if (last.get("alpha_after_round") is not None and history[0].get("alpha_before_round") is not None)
                else None
            ),
            "verdict_final": last.get("verdict"),
            "rubric_edits_total": sum(len(h.get("rubric_edits_adopted") or []) for h in history),
            "spec_edits_total": sum(len(h.get("spec_edits_adopted") or []) for h in history),
        })

    parts.append("## Per-statement summary\n")
    parts.append("| statement | rounds | α baseline | α final | Δα total | rubric edits | spec edits | verdict |")
    parts.append("|---|--:|--:|--:|--:|--:|--:|---|")
    for s in summary_rows:
        ab = f"{s['alpha_baseline']:.3f}" if s['alpha_baseline'] is not None else "?"
        af = f"{s['alpha_final']:.3f}" if s['alpha_final'] is not None else "?"
        da = f"{s['delta_alpha_total']:+.3f}" if s['delta_alpha_total'] is not None else "?"
        parts.append(f"| {s['sid']} | {s['rounds']} | {ab} | {af} | {da} | {s['rubric_edits_total']} | {s['spec_edits_total']} | {s['verdict_final']} |")
    parts.append("")

    # Verdict counts
    verdict_counter = Counter(s["verdict_final"] for s in summary_rows)
    parts.append(f"\n**Verdict distribution**: {dict(verdict_counter)}")
    converged = [s for s in summary_rows if s["verdict_final"] == "converged"]
    parts.append(f"\n**Converged** ({len(converged)} statements with α ≥ 0.5):")
    for s in converged:
        parts.append(f"- `{s['sid']}` (α {s['alpha_final']:.3f}, Δ{s['delta_alpha_total']:+.3f}, {s['rounds']} rounds)")

    improving = [s for s in summary_rows if s["verdict_final"] == "improving"]
    if improving:
        parts.append(f"\n**Still improving** ({len(improving)} statements — would have continued to next round):")
        for s in improving:
            parts.append(f"- `{s['sid']}` (α {s['alpha_final']:.3f}, Δ{s['delta_alpha_total']:+.3f})")

    stuck = [s for s in summary_rows if s["verdict_final"] == "stuck"]
    if stuck:
        parts.append(f"\n**Stuck** ({len(stuck)} statements — escalated for spec-author review):")
        for s in stuck:
            parts.append(f"- `{s['sid']}` (α {s['alpha_final']:.3f}, Δ{s['delta_alpha_total']:+.3f})")

    # Per-round breakdown
    parts.append("\n## Per-round summary\n")
    max_rounds = max(s["rounds"] for s in summary_rows)
    for rn in range(1, max_rounds + 1):
        parts.append(f"\n### Round {rn}\n")
        round_summary = ITER_DIR / f"round_{rn}_analysis_summary.json"
        if round_summary.exists():
            data = json.loads(round_summary.read_text())
            parts.append(f"Verdict counts: {data.get('verdict_counts', {})}\n")
        parts.append("| statement | α before | α after | Δα | verdict | rubric edits adopted | spec edits adopted |")
        parts.append("|---|--:|--:|--:|---|--:|--:|")
        for sid in statements:
            history = json.loads((ITER_DIR / sid / "history.json").read_text())
            if len(history) < rn:
                continue
            h = history[rn - 1]
            ab = f"{h.get('alpha_before_round'):.3f}" if h.get('alpha_before_round') is not None else "?"
            aa = f"{h.get('alpha_after_round'):.3f}" if h.get('alpha_after_round') is not None else "?"
            da = f"{h.get('delta_alpha'):+.3f}" if h.get('delta_alpha') is not None else "?"
            ru = len(h.get("rubric_edits_adopted") or [])
            sp = len(h.get("spec_edits_adopted") or [])
            parts.append(f"| {sid} | {ab} | {aa} | {da} | {h.get('verdict','?')} | {ru} | {sp} |")

    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")

    # Escalation queue: stuck statements
    esc_parts = ["# DART Escalation Queue", ""]
    esc_parts.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    esc_parts.append("")
    esc_parts.append(f"## Statements requiring spec-author triage ({len(stuck)})")
    esc_parts.append("")
    for s in stuck:
        history = json.loads((ITER_DIR / s["sid"] / "history.json").read_text())
        esc_parts.append(f"### `{s['sid']}`")
        esc_parts.append(f"- α: {s['alpha_baseline']:.3f} → {s['alpha_final']:.3f} (Δ{s['delta_alpha_total']:+.3f}, {s['rounds']} rounds attempted)")
        for h in history:
            esc_parts.append(f"  - Round {h['round']}: maj diag = {h.get('round_diagnosis_majority','?')}; "
                             f"adopted {len(h.get('rubric_edits_adopted') or [])} rubric / "
                             f"{len(h.get('spec_edits_adopted') or [])} spec edits")
        esc_parts.append("")
    ESCALATION_OUT.write_text("\n".join(esc_parts))
    print(f"wrote {ESCALATION_OUT}")


if __name__ == "__main__":
    main()
