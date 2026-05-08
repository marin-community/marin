"""Rubric-poison cell ranking via Δpwv (per-cell pairwise variance contribution to Δα).

For each cell c (statement × scenario × generator):
  bare_pwv(c)   = Σ_{i<j} (score_i − score_j)²  over 3 judges, BARE condition
  rubric_pwv(c) = same, RUBRIC condition
  Δpwv(c)       = rubric_pwv(c) − bare_pwv(c)

Cells with the largest positive Δpwv are the cells where the rubric introduced
disagreement that didn't exist under bare. Top-K cells become the inputs for
rubric revision.

Outputs:
  .agents/logbooks/rubric_poison_<sid>.md   (one per statement)
  .agents/logbooks/rubric_poison_summary.md (cross-statement summary)

Pure stdlib + the standard load_judgments helper.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e9_build_qualitative_inputs import load_judgments, load_response_index

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")
RUBRICS_PATH = DIR / "e8_rubrics.jsonl"
OUT_DIR = Path(".agents/logbooks")

DEFAULT_HURT = [
    "refusal_style",
    "no_agenda",
    "comply_with_laws",
    "avoid_abuse",
    "sexual_content_involving_minors",
]
JUDGES = ("gpt", "gemini", "claude")


def pairwise_variance(scores: list[int]) -> int:
    return sum((scores[i] - scores[j]) ** 2
               for i in range(len(scores))
               for j in range(i + 1, len(scores)))


def outlier_judge(scores_by_judge: dict[str, int]) -> tuple[str, int] | None:
    """Returns (judge, max_pairwise_diff_to_others) or None if no clear outlier."""
    if len(scores_by_judge) != 3:
        return None
    best_judge, best_dev = None, -1
    for j in JUDGES:
        if j not in scores_by_judge: continue
        others = [scores_by_judge[other] for other in JUDGES if other != j]
        dev = max(abs(scores_by_judge[j] - o) for o in others)
        if dev > best_dev:
            best_dev = dev
            best_judge = j
    if best_dev < 2:
        return None
    return (best_judge, best_dev)


def rank_rubric_poison(by_cell: dict, sid: str) -> list[dict]:
    bare_cells = {(c[2], c[3]): jd for c, jd in by_cell.items()
                  if c[0] == sid and c[1] == "variant_A" and len(jd) == 3}
    rub_cells = {(c[2], c[3]): jd for c, jd in by_cell.items()
                 if c[0] == sid and c[1] == "rubric_plus_spec" and len(jd) == 3}

    rows = []
    for key in set(bare_cells) & set(rub_cells):
        bare_jd = bare_cells[key]
        rub_jd = rub_cells[key]
        bare_dict = {j: bare_jd[j]["score"] for j in JUDGES}
        rub_dict = {j: rub_jd[j]["score"] for j in JUDGES}
        bs = [bare_dict[j] for j in JUDGES]
        rs = [rub_dict[j] for j in JUDGES]
        rows.append({
            "scen": key[0], "gen": key[1],
            "bare_scores": bs, "rubric_scores": rs,
            "bare_pwv": pairwise_variance(bs),
            "rubric_pwv": pairwise_variance(rs),
            "delta_pwv": pairwise_variance(rs) - pairwise_variance(bs),
            "bare_judgments": {j: bare_jd[j] for j in JUDGES},
            "rubric_judgments": {j: rub_jd[j] for j in JUDGES},
            "outlier_bare": outlier_judge(bare_dict),
            "outlier_rubric": outlier_judge(rub_dict),
        })
    rows.sort(key=lambda r: -r["delta_pwv"])
    return rows


def per_anchor_frequency(rows: list[dict], top_k: int = 12) -> dict[str, Any]:
    top = rows[:top_k]
    modal = Counter()
    low = Counter()
    high = Counter()
    pairs = Counter()
    outlier_counts = Counter()
    for r in top:
        rs = r["rubric_scores"]
        c = Counter(rs)
        max_count = max(c.values())
        modal_score = min(s for s, n in c.items() if n == max_count)
        modal[modal_score] += 1
        low[min(rs)] += 1
        high[max(rs)] += 1
        pairs[(min(rs), max(rs))] += 1
        if r["outlier_rubric"]:
            outlier_counts[r["outlier_rubric"][0]] += 1
    return {
        "modal_anchor": modal, "low_anchor": low, "high_anchor": high,
        "divergence_pairs": pairs, "outlier_judge_counts": outlier_counts,
        "n_top_k": len(top),
    }


def trunc(s, n=500):
    s = (s or "").replace("\r", "")
    return s if len(s) <= n else s[:n] + "…"


def render_report(sid, rows, anchor_freq, response_idx, spec_text, rubric_anchors, top_k=12):
    parts = []
    parts.append(f"# Rubric-poison report — `{sid}`\n")
    parts.append(f"**Cells analyzed**: {len(rows)} (cells with all 3 judges scored under both conditions)")
    parts.append(f"**Top-K shown**: {top_k}\n")

    bare_total = sum(r["bare_pwv"] for r in rows)
    rub_total = sum(r["rubric_pwv"] for r in rows)
    parts.append(f"**Total bare_pwv**: {bare_total}")
    parts.append(f"**Total rubric_pwv**: {rub_total}")
    parts.append(f"**Total Δpwv (sum)**: {rub_total - bare_total:+d}\n")

    top_d_total = sum(r["delta_pwv"] for r in rows[:top_k])
    pct = 100 * top_d_total / max(1, rub_total - bare_total) if rub_total != bare_total else 0
    parts.append(f"**Top-{top_k} Δpwv share of total**: {top_d_total:+d} ({pct:.1f}%)\n")
    parts.append("> If top-K accounts for >70% of total Δpwv, fixing those cells closes the rubric paradox on this statement.\n")

    parts.append(f"## Per-anchor frequency in top-{top_k} poison cells\n")
    parts.append("| anchor | n cells where MODAL | n cells where LOW | n cells where HIGH |")
    parts.append("|---|--:|--:|--:|")
    for a in (1, 2, 3, 4, 5):
        parts.append(f"| {a} | {anchor_freq['modal_anchor'].get(a, 0)} | "
                     f"{anchor_freq['low_anchor'].get(a, 0)} | "
                     f"{anchor_freq['high_anchor'].get(a, 0)} |")
    parts.append("")
    parts.append(f"### Top divergence pairs\n")
    parts.append("| (low, high) | count |")
    parts.append("|---|--:|")
    for (lo, hi), n in anchor_freq["divergence_pairs"].most_common(8):
        parts.append(f"| ({lo}, {hi}) | {n} |")
    parts.append("")

    parts.append(f"### Outlier-judge concentration in top-{top_k}\n")
    parts.append("| judge | n cells where this judge is the outlier |")
    parts.append("|---|--:|")
    for j in JUDGES:
        parts.append(f"| {j} | {anchor_freq['outlier_judge_counts'].get(j, 0)} |")
    n_o = sum(anchor_freq["outlier_judge_counts"].values())
    if n_o > 0:
        max_j, max_n = anchor_freq["outlier_judge_counts"].most_common(1)[0]
        conc = 100 * max_n / n_o
        parts.append(f"\n**Outlier concentration**: `{max_j}` is the outlier in {conc:.0f}% of cells with a clear outlier (n={n_o}).")
        if conc > 70:
            parts.append(f"\n⚠️ **CONCENTRATION > 70%** — judge-prior leakage. Anchor language must be EMPHATIC enough to override `{max_j}`'s prior; tightening alone won't suffice.")
    parts.append("")

    parts.append("## Spec statement\n")
    parts.append("> " + spec_text.replace("\n", "\n> "))
    parts.append("\n## Rubric anchors\n")
    for k in ("1", "2", "3", "4", "5"):
        anc = rubric_anchors.get(k, {}) or {}
        parts.append(f"**Anchor {k}**: {trunc(anc.get('criterion'), 600)}\n")

    parts.append(f"## Top-{top_k} rubric-poison cells\n")
    for i, r in enumerate(rows[:top_k]):
        rm = response_idx.get((sid, r["scen"], r["gen"]), {})
        bs_pretty = ",".join(str(s) for s in r["bare_scores"])
        rs_pretty = ",".join(str(s) for s in r["rubric_scores"])
        parts.append(f"### Rank {i+1} — Δpwv = {r['delta_pwv']:+d}  (bare_pwv={r['bare_pwv']}, rubric_pwv={r['rubric_pwv']})\n")
        parts.append(f"**scen={r['scen']}, generator={r['gen']}**")
        parts.append(f"  - bare scores (gpt,gemini,claude) = ({bs_pretty})")
        parts.append(f"  - rubric scores (gpt,gemini,claude) = ({rs_pretty})")
        if r["outlier_rubric"]:
            parts.append(f"  - rubric outlier: **{r['outlier_rubric'][0]}** (deviation={r['outlier_rubric'][1]})")
        parts.append("")
        parts.append(f"**User query**: {trunc(rm.get('user_query'), 400)}")
        parts.append(f"**Assistant response**: {trunc(rm.get('response'), 500)}\n")
        parts.append("**Bare reasoning (judges agreed):**")
        for j in JUDGES:
            jr = r["bare_judgments"][j]
            parts.append(f"- _{j}_ (score={jr['score']}): {trunc(jr.get('reasoning'), 400)}")
        parts.append("\n**Rubric reasoning (judges diverged):**")
        for j in JUDGES:
            jr = r["rubric_judgments"][j]
            parts.append(f"- _{j}_ (score={jr['score']}): {trunc(jr.get('reasoning'), 400)}")
        parts.append("")

    return "\n".join(parts)


def render_summary(reports):
    parts = ["# Rubric-poison summary across analyzed statements\n"]
    parts.append("| statement | n cells | total Δpwv | top-K share | dominant outlier | top divergence pair |")
    parts.append("|---|--:|--:|--:|---|---|")
    for r in reports:
        parts.append(f"| {r['sid']} | {r['n']} | {r['total_delta_pwv']:+d} | {r['topk_share_pct']:.0f}% | "
                     f"{r['dominant_outlier_judge']} | {r['dominant_divergence_pair']} |")
    parts.append("")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--statement-ids", nargs="*", default=DEFAULT_HURT)
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()

    print("Loading judgments + responses + rubrics...")
    by_cell = load_judgments()
    response_idx = load_response_index()
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    rubrics = {r["statement_id"]: r["rubric"] for r in (json.loads(l) for l in RUBRICS_PATH.open()) if "error" not in r}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for sid in args.statement_ids:
        if sid not in spec:
            print(f"  WARN: unknown statement {sid}")
            continue
        rows = rank_rubric_poison(by_cell, sid)
        if not rows:
            print(f"  WARN: no full-3-judge cells for {sid}")
            continue
        anchor_freq = per_anchor_frequency(rows, top_k=args.top_k)
        rubric_anchors = (rubrics.get(sid, {}) or {}).get("anchors", {})
        report = render_report(sid, rows, anchor_freq, response_idx,
                               spec[sid]["text"], rubric_anchors, top_k=args.top_k)
        out = OUT_DIR / f"rubric_poison_{sid}.md"
        out.write_text(report)
        total_d = sum(r["delta_pwv"] for r in rows)
        topk_d = sum(r["delta_pwv"] for r in rows[:args.top_k])
        topk_pct = 100 * topk_d / max(1, total_d) if total_d != 0 else 0
        oj_c = anchor_freq["outlier_judge_counts"]
        oj_str = f"{oj_c.most_common(1)[0][0]} ({100*oj_c.most_common(1)[0][1]/max(1,sum(oj_c.values())):.0f}%)" if oj_c else "none"
        pair_str = (f"({anchor_freq['divergence_pairs'].most_common(1)[0][0][0]},{anchor_freq['divergence_pairs'].most_common(1)[0][0][1]}) "
                    f"×{anchor_freq['divergence_pairs'].most_common(1)[0][1]}") if anchor_freq['divergence_pairs'] else "n/a"
        summary_rows.append({
            "sid": sid, "n": len(rows), "total_delta_pwv": total_d,
            "topk_share_pct": topk_pct,
            "dominant_outlier_judge": oj_str,
            "dominant_divergence_pair": pair_str,
        })
        print(f"  {sid}: n={len(rows)}, total Δpwv={total_d:+d}, top-{args.top_k} share={topk_pct:.0f}%")

    if summary_rows:
        (OUT_DIR / "rubric_poison_summary.md").write_text(render_summary(summary_rows))
        print(f"  wrote summary -> {OUT_DIR / 'rubric_poison_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
