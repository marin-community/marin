# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare control vs treatment metric JSONs and emit a verdict + comment.

Inputs are the JSON files produced by ``collect_perf_metrics.py`` for the two
legs of a zephyr perf gate.

Outputs:
  --verdict-out   JSON: {"verdict": "pass"|"warn"|"fail", "reasons": [...], ...}
  --markdown-out  The exact comment body to post on the PR (sentinel-marked,
                  ready to feed into ``post_pr_comment.py``).

Default thresholds (tunable via --thresholds <yaml>):
  hard fail when:
    * any new OOM in treatment (and not in control)
    * any new failed shard in treatment (and not in control)
    * total wall-time delta > +10%
    * any per-stage wall-time delta > +10%
  warn when:
    * total wall-time delta in (+5%, +10%]
    * any per-stage wall-time delta in (+5%, +10%]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import sys
from typing import Any

DEFAULT_THRESHOLDS = {
    "stage_warn_pct": 5.0,
    "stage_fail_pct": 10.0,
    "total_warn_pct": 5.0,
    "total_fail_pct": 10.0,
}

GATE_LABELS = {
    "1": "Gate 1 (fineweb)",
    "2": "Gate 2 (nemotron partial-slice)",
    "3": "Gate 3 (nemotron full-slice)",
}

SENTINEL = "<!-- zephyr-perf-gate -->"


@dataclasses.dataclass
class StageRow:
    name: str
    control: float | None
    treatment: float | None
    delta_pct: float | None
    verdict: str  # ✅ / ⚠ / ❌ / —


def _pct(control: float | None, treatment: float | None) -> float | None:
    if control is None or treatment is None or control == 0:
        return None
    return (treatment - control) / control * 100.0


def _stage_verdict(delta_pct: float | None, t: dict[str, float]) -> str:
    if delta_pct is None:
        return "—"
    if delta_pct > t["stage_fail_pct"]:
        return "❌"
    if delta_pct > t["stage_warn_pct"]:
        return "⚠"
    return "✅"


def compare(
    *,
    control: dict[str, Any],
    treatment: dict[str, Any],
    gate: str,
    pr: int,
    thresholds: dict[str, float],
    assessment: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    reasons_fail: list[str] = []
    reasons_warn: list[str] = []

    # OOMs / failed shards: hard fail if any new ones in treatment.
    new_ooms = max(0, (treatment.get("ooms") or 0) - (control.get("ooms") or 0))
    new_fails = max(0, (treatment.get("failed_shards") or 0) - (control.get("failed_shards") or 0))
    if new_ooms:
        reasons_fail.append(f"{new_ooms} new OOM(s) in treatment")
    if new_fails:
        reasons_fail.append(f"{new_fails} new failed shard(s) in treatment")

    # Total wall-time.
    total_delta = _pct(control.get("wall_seconds_total"), treatment.get("wall_seconds_total"))
    if total_delta is not None:
        if total_delta > thresholds["total_fail_pct"]:
            reasons_fail.append(f"total wall-time +{total_delta:.1f}%")
        elif total_delta > thresholds["total_warn_pct"]:
            reasons_warn.append(f"total wall-time +{total_delta:.1f}%")

    # Per-stage wall-times. Stages may differ between control and treatment if
    # the diff renames or splits a stage; rows for stages missing on one side
    # render with "—" and don't count toward the verdict.
    all_stages = sorted(set(control.get("stage_wall_seconds") or {}) | set(treatment.get("stage_wall_seconds") or {}))
    stage_rows: list[StageRow] = []
    for stage in all_stages:
        c = (control.get("stage_wall_seconds") or {}).get(stage)
        t = (treatment.get("stage_wall_seconds") or {}).get(stage)
        delta = _pct(c, t)
        v = _stage_verdict(delta, thresholds)
        stage_rows.append(StageRow(name=stage, control=c, treatment=t, delta_pct=delta, verdict=v))
        if delta is not None:
            if delta > thresholds["stage_fail_pct"]:
                reasons_fail.append(f"{stage} +{delta:.1f}%")
            elif delta > thresholds["stage_warn_pct"]:
                reasons_warn.append(f"{stage} +{delta:.1f}%")

    if reasons_fail:
        verdict = "fail"
    elif reasons_warn:
        verdict = "warn"
    else:
        verdict = "pass"

    summary = {
        "verdict": verdict,
        "gate": gate,
        "pr": pr,
        "reasons_fail": reasons_fail,
        "reasons_warn": reasons_warn,
        "total_wall_delta_pct": total_delta,
        "new_ooms": new_ooms,
        "new_failed_shards": new_fails,
        "stage_deltas_pct": {row.name: row.delta_pct for row in stage_rows if row.delta_pct is not None},
        "thresholds": thresholds,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    markdown = _render_markdown(
        verdict=verdict,
        gate=gate,
        control=control,
        treatment=treatment,
        stage_rows=stage_rows,
        total_delta=total_delta,
        reasons_fail=reasons_fail,
        reasons_warn=reasons_warn,
        assessment=assessment,
    )
    return summary, markdown


def _fmt_pct(p: float | None) -> str:
    if p is None:
        return "—"
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def _fmt_seconds(s: float | None) -> str:
    if s is None:
        return "—"
    if s < 90:
        return f"{s:.1f}s"
    minutes, secs = divmod(int(s), 60)
    if minutes < 90:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _render_markdown(
    *,
    verdict: str,
    gate: str,
    control: dict[str, Any],
    treatment: dict[str, Any],
    stage_rows: list[StageRow],
    total_delta: float | None,
    reasons_fail: list[str],
    reasons_warn: list[str],
    assessment: dict[str, Any] | None = None,
) -> str:
    icon = {"pass": "✅ pass", "warn": "⚠ warn", "fail": "❌ fail"}[verdict]
    gate_label = GATE_LABELS.get(gate, f"Gate {gate}")

    lines: list[str] = []
    lines.append(SENTINEL)
    lines.append(f"🤖 ## Zephyr perf gate — {gate_label}")
    lines.append("")
    lines.append(f"**Verdict:** {icon}")
    lines.append("")
    if reasons_fail:
        lines.append("**Hard fails:** " + "; ".join(reasons_fail))
    if reasons_warn:
        lines.append("**Warns:** " + "; ".join(reasons_warn))
    if reasons_fail or reasons_warn:
        lines.append("")

    if assessment:
        lines.extend(_render_assessment(assessment))
        lines.append("")

    lines.append("| | Control | Treatment |")
    lines.append("|---|---|---|")
    lines.append(f"| Iris job | `{control.get('iris_job_id') or '—'}` | `{treatment.get('iris_job_id') or '—'}` |")
    lines.append(f"| Status | {control.get('status') or '—'} | {treatment.get('status') or '—'} |")
    lines.append(f"| W&B | {_link(control.get('wandb_url'))} | {_link(treatment.get('wandb_url'))} |")
    lines.append(
        f"| Total wall-time | {_fmt_seconds(control.get('wall_seconds_total'))} "
        f"| {_fmt_seconds(treatment.get('wall_seconds_total'))} ({_fmt_pct(total_delta)}) |"
    )
    lines.append("")

    lines.append("### Stage timings")
    lines.append("")
    lines.append("| Stage | Control | Treatment | Δ | Verdict |")
    lines.append("|---|---|---|---|---|")
    for row in stage_rows:
        lines.append(
            f"| `{row.name}` | {_fmt_seconds(row.control)} | {_fmt_seconds(row.treatment)} "
            f"| {_fmt_pct(row.delta_pct)} | {row.verdict} |"
        )
    lines.append("")

    lines.append("### Workers")
    lines.append("")
    lines.append("| | Control | Treatment |")
    lines.append("|---|---|---|")
    lines.append(f"| OOMs | {control.get('ooms', 0)} | {treatment.get('ooms', 0)} |")
    lines.append(f"| Failed shards | {control.get('failed_shards', 0)} | {treatment.get('failed_shards', 0)} |")
    lines.append(
        f"| Peak worker memory (MB) | {control.get('peak_worker_memory_mb') or '—'} "
        f"| {treatment.get('peak_worker_memory_mb') or '—'} |"
    )
    lines.append("")

    counters = treatment.get("counters") or {}
    if counters:
        lines.append("<details><summary>Counters (treatment)</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(counters, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    warnings = (control.get("warnings") or []) + (treatment.get("warnings") or [])
    if warnings:
        lines.append("<details><summary>Collector warnings</summary>")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def _link(url: str | None) -> str:
    return f"[link]({url})" if url else "—"


_ASSESSMENT_DIMS = (
    ("trivial", "trivial"),
    ("shuffle", "shuffle"),
    ("memory", "memory"),
    ("cpu", "CPU"),
    ("design", "design"),
)


def _render_assessment(assessment: dict[str, Any]) -> list[str]:
    """Render the agent's pre-run impact assessment as a markdown block.

    Schema (see SKILL.md → step 2):
        {"gate": "1"|"2"|"skip", "rationale": "...",
         "per_file": {<path>: {"trivial": bool, "shuffle": bool, "memory": bool,
                               "cpu": bool, "design": bool, "summary": "..."}}}
    """
    out: list[str] = ["### Diff assessment"]
    rationale = assessment.get("rationale")
    chosen = assessment.get("gate")
    if chosen or rationale:
        bits = []
        if chosen:
            bits.append(f"chose **{chosen}**")
        if rationale:
            bits.append(rationale)
        out.append("_" + "; ".join(bits) + "_")
        out.append("")

    per_file = assessment.get("per_file") or {}
    if not per_file:
        return out

    out.append("| File | trivial | shuffle | memory | CPU | design | summary |")
    out.append("|---|---|---|---|---|---|---|")
    for path, dims in sorted(per_file.items()):
        cells = [f"`{path}`"]
        for key, _ in _ASSESSMENT_DIMS:
            cells.append("✓" if dims.get(key) else "")
        cells.append(dims.get("summary", ""))
        out.append("| " + " | ".join(cells) + " |")
    return out


def _load_thresholds(path: str | None) -> dict[str, float]:
    if not path:
        return dict(DEFAULT_THRESHOLDS)
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    merged = dict(DEFAULT_THRESHOLDS)
    merged.update({k: float(v) for k, v in raw.items()})
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--gate", required=True, choices=sorted(GATE_LABELS))
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--thresholds", help="YAML overriding default thresholds.")
    parser.add_argument(
        "--assessment",
        help="JSON file with the agent's pre-run impact assessment (see SKILL.md step 2).",
    )
    parser.add_argument("--markdown-out", required=True)
    parser.add_argument("--verdict-out", required=True)
    args = parser.parse_args()

    with open(args.control) as f:
        control = json.load(f)
    with open(args.treatment) as f:
        treatment = json.load(f)
    assessment: dict[str, Any] | None = None
    if args.assessment:
        with open(args.assessment) as f:
            assessment = json.load(f)

    thresholds = _load_thresholds(args.thresholds)
    summary, markdown = compare(
        control=control,
        treatment=treatment,
        gate=args.gate,
        pr=args.pr,
        thresholds=thresholds,
        assessment=assessment,
    )

    with open(args.verdict_out, "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.markdown_out, "w") as f:
        f.write(markdown)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
