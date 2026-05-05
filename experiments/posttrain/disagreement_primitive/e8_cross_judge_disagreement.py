# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E8 cross-judge disagreement analysis — pure analysis, no API calls.

Per Ahmed (2026-05-04): flip the axis from per-judge faithfulness (phase 1+2's
Spearman ρ within judge) to **across-judge disagreement on a fixed criterion**.
This directly captures "compliance disagreement" in the paper's framing — when
3 judges look at the same (rubric, scenario, response) and score it differently,
the *rubric* itself is ambiguous to different model priors. Same logic for spec.

For each (statement_id, scenario_idx, generator):
  - 3 variant-A scores (GPT spec-judge, Gemini spec-judge, GLM spec-judge)
  - 3 variant-B scores (GPT rubric-judge, Gemini rubric-judge, GLM rubric-judge)
  - cross-judge stdev on A = spec ambiguity for this scenario
  - cross-judge stdev on B = rubric ambiguity for this scenario

Per statement (60 paired scenarios):
  - mean cross-judge stdev on A = spec ambiguity for this statement
  - mean cross-judge stdev on B = rubric ambiguity for this statement
  - B − A = does the rubric introduce ambiguity (>0), preserve it (~0), or resolve it (<0)

Outputs:
  - e8_cross_judge.jsonl   (per-statement summary with both ambiguity metrics)
  - e8_cross_judge.md      (sortable table + flagged-statement decomposition)
"""

from __future__ import annotations
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import DIR

JUDGES = [
    ("gpt",    DIR / "e8_va_judgments.jsonl",        DIR / "e8_vb_judgments.jsonl"),
    ("gemini", DIR / "phase2_gemini/va_judgments.jsonl", DIR / "phase2_gemini/vb_judgments.jsonl"),
    ("glm",    DIR / "phase2_glm/va_judgments.jsonl",    DIR / "phase2_glm/vb_judgments.jsonl"),
]


def load_scores(path: Path) -> dict[tuple, int]:
    """key=(statement_id, scenario_idx, generator) → score (or skipped if error/None)."""
    out = {}
    if not path.exists():
        return out
    for line in path.open():
        r = json.loads(line)
        if "error" in r:
            continue
        score = r.get("score")
        if score is None:
            continue
        out[(r["statement_id"], r["scenario_idx"], r["generator"])] = score
    return out


def stdev_or_none(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return statistics.stdev(xs)


def main() -> int:
    # Load 6 score maps: 3 judges × 2 variants
    a_scores: dict[str, dict[tuple, int]] = {}
    b_scores: dict[str, dict[tuple, int]] = {}
    for judge, va_path, vb_path in JUDGES:
        a_scores[judge] = load_scores(va_path)
        b_scores[judge] = load_scores(vb_path)
        print(f"  {judge}: variant A {len(a_scores[judge])} valid scores, variant B {len(b_scores[judge])} valid scores")

    # All keys we've ever seen
    all_keys: set[tuple] = set()
    for d in list(a_scores.values()) + list(b_scores.values()):
        all_keys.update(d.keys())
    print(f"\n  total (statement, scenario, generator) keys: {len(all_keys)}")

    # Per-(statement, scenario, generator): collect 3 A-scores + 3 B-scores; compute cross-judge stdev
    per_scenario_a_stdev: dict[tuple, float] = {}
    per_scenario_b_stdev: dict[tuple, float] = {}
    coverage_a = []  # list of n_judges available per (sid, idx, gen)
    coverage_b = []
    for k in all_keys:
        a_vals = [a_scores[j].get(k) for j in ("gpt", "gemini", "glm")]
        a_vals_v = [v for v in a_vals if v is not None]
        b_vals = [b_scores[j].get(k) for j in ("gpt", "gemini", "glm")]
        b_vals_v = [v for v in b_vals if v is not None]
        coverage_a.append(len(a_vals_v))
        coverage_b.append(len(b_vals_v))
        sa = stdev_or_none(a_vals_v)
        sb = stdev_or_none(b_vals_v)
        if sa is not None:
            per_scenario_a_stdev[k] = sa
        if sb is not None:
            per_scenario_b_stdev[k] = sb

    print(f"\n  coverage variant A (judges per scenario): {dict((c, coverage_a.count(c)) for c in (0,1,2,3))}")
    print(f"  coverage variant B (judges per scenario): {dict((c, coverage_b.count(c)) for c in (0,1,2,3))}")

    # Aggregate per statement
    by_stmt: dict[str, dict] = defaultdict(lambda: {"a_stdevs": [], "b_stdevs": [], "n_a": 0, "n_b": 0})
    for k, sa in per_scenario_a_stdev.items():
        by_stmt[k[0]]["a_stdevs"].append(sa)
        by_stmt[k[0]]["n_a"] += 1
    for k, sb in per_scenario_b_stdev.items():
        by_stmt[k[0]]["b_stdevs"].append(sb)
        by_stmt[k[0]]["n_b"] += 1

    rows = []
    for sid, d in by_stmt.items():
        mean_a = sum(d["a_stdevs"]) / len(d["a_stdevs"]) if d["a_stdevs"] else None
        mean_b = sum(d["b_stdevs"]) / len(d["b_stdevs"]) if d["b_stdevs"] else None
        diff = (mean_b - mean_a) if (mean_a is not None and mean_b is not None) else None
        rows.append({
            "statement_id": sid,
            "n_scenarios_a": d["n_a"],
            "n_scenarios_b": d["n_b"],
            "mean_judge_stdev_spec":   round(mean_a, 4) if mean_a is not None else None,
            "mean_judge_stdev_rubric": round(mean_b, 4) if mean_b is not None else None,
            "rubric_minus_spec":       round(diff, 4) if diff is not None else None,
        })

    rows.sort(key=lambda r: r["statement_id"])

    out_jsonl = DIR / "e8_cross_judge.jsonl"
    with out_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  wrote {out_jsonl}")

    # Build markdown report
    md = ["# E8 cross-judge disagreement analysis", ""]
    md.append("**Per (scenario, response):** stdev across 3 judges' scores. **Per statement:** mean stdev across 60 scenarios.")
    md.append("")
    md.append("- `mean_judge_stdev_spec` = spec ambiguity (how much do judges disagree when given the spec text + examples)")
    md.append("- `mean_judge_stdev_rubric` = rubric ambiguity (how much do judges disagree when given only the rubric)")
    md.append("- `rubric_minus_spec` > 0 → rubric introduces ambiguity (translation failure mode)")
    md.append("- `rubric_minus_spec` ≈ 0 → rubric preserves the spec's ambiguity faithfully")
    md.append("- `rubric_minus_spec` < 0 → rubric resolves ambiguity (more deterministic than the spec text)")
    md.append("")

    # Distribution summaries
    a_means = [r["mean_judge_stdev_spec"]   for r in rows if r["mean_judge_stdev_spec"]   is not None]
    b_means = [r["mean_judge_stdev_rubric"] for r in rows if r["mean_judge_stdev_rubric"] is not None]
    diffs   = [r["rubric_minus_spec"]       for r in rows if r["rubric_minus_spec"]       is not None]

    def quant(xs):
        s = sorted(xs)
        return s[len(s)//4], s[len(s)//2], s[3*len(s)//4]

    md.append("## Distribution across statements")
    md.append("")
    md.append("| metric | n | min | p25 | median | p75 | max | mean |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for label, xs in (("spec ambiguity",   a_means), ("rubric ambiguity", b_means), ("rubric − spec",   diffs)):
        if not xs:
            continue
        p25, med, p75 = quant(xs)
        md.append(f"| {label} | {len(xs)} | {min(xs):.3f} | {p25:.3f} | **{med:.3f}** | {p75:.3f} | {max(xs):.3f} | {sum(xs)/len(xs):.3f} |")
    md.append("")

    # Sorted by rubric_minus_spec — top translation failures (rubric adds ambiguity)
    md.append("## Top 10 statements where the rubric ADDS ambiguity (rubric_minus_spec > 0)")
    md.append("")
    sorted_diff = sorted([r for r in rows if r["rubric_minus_spec"] is not None], key=lambda r: -r["rubric_minus_spec"])
    md.append("| statement | spec ambig | rubric ambig | rubric − spec | n_a | n_b |")
    md.append("|---|--:|--:|--:|--:|--:|")
    for r in sorted_diff[:10]:
        md.append(f"| `{r['statement_id']}` | {r['mean_judge_stdev_spec']} | {r['mean_judge_stdev_rubric']} | {r['rubric_minus_spec']:+.4f} | {r['n_scenarios_a']} | {r['n_scenarios_b']} |")
    md.append("")

    # Top statements where rubric resolves ambiguity
    md.append("## Top 10 statements where the rubric RESOLVES ambiguity (rubric_minus_spec < 0)")
    md.append("")
    md.append("| statement | spec ambig | rubric ambig | rubric − spec | n_a | n_b |")
    md.append("|---|--:|--:|--:|--:|--:|")
    for r in sorted_diff[-10:][::-1]:
        md.append(f"| `{r['statement_id']}` | {r['mean_judge_stdev_spec']} | {r['mean_judge_stdev_rubric']} | {r['rubric_minus_spec']:+.4f} | {r['n_scenarios_a']} | {r['n_scenarios_b']} |")
    md.append("")

    # Top spec-ambiguous statements (high baseline ambiguity in spec language)
    md.append("## Top 10 spec-ambiguous statements (highest mean_judge_stdev_spec — judges disagree most when reading the spec text directly)")
    md.append("")
    md.append("| statement | spec ambig | rubric ambig | rubric − spec |")
    md.append("|---|--:|--:|--:|")
    sorted_spec = sorted([r for r in rows if r["mean_judge_stdev_spec"] is not None], key=lambda r: -r["mean_judge_stdev_spec"])
    for r in sorted_spec[:10]:
        d = r["rubric_minus_spec"]
        md.append(f"| `{r['statement_id']}` | **{r['mean_judge_stdev_spec']}** | {r['mean_judge_stdev_rubric']} | {d:+.4f} |")
    md.append("")

    # Full per-statement table
    md.append("## Per-statement table (sorted by rubric_minus_spec)")
    md.append("")
    md.append("| statement | spec ambig | rubric ambig | rubric − spec | n_a | n_b |")
    md.append("|---|--:|--:|--:|--:|--:|")
    for r in sorted_diff:
        md.append(f"| `{r['statement_id']}` | {r['mean_judge_stdev_spec']} | {r['mean_judge_stdev_rubric']} | {r['rubric_minus_spec']:+.4f} | {r['n_scenarios_a']} | {r['n_scenarios_b']} |")
    md.append("")

    out_md = DIR / "e8_cross_judge.md"
    out_md.write_text("\n".join(md))
    print(f"  wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
