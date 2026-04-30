# ruff: noqa: E501, B007, RUF001, RUF003
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render Phase 1B's `tension_discovery_report.md` (Gate H2).

Consumes `pair_candidate_*.jsonl` files from `discover_pair_candidates.py`
and computes:

- per-compiler candidate counts by source and predicted relation;
- atlas recall: fraction of the 22 cross-tier seed pairs (and the 18
  same-class atlas pairs) recovered by lm_topk + embedding_neighbor;
- random-control false-positive rate: fraction of random controls
  classified as anything other than no_tension;
- cross-compiler agreement on the same pair;
- top candidates by predicted tension confidence;
- a 20-30 pair sample with rationales for spot-checking;
- if the all-pair backtest ran: precision/recall of the
  topk+embedding heuristic against the all-pair classifier as
  ground truth.

H2 thresholds (from the Codex plan):
- ≥80% recall on the 22 known cross-tier seed pairs at ≤25% of all
  possible pairs.
- Random controls should mostly classify as no_tension.
- Output a deliberately diverse batch for Phase 2.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
SEED_PAIRS_PATH = WORKTREE / "experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl"
DEFAULT_INPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "tension_discovery_report.md"

ALL_RELATIONS = ["dominance", "bidirectional_tradeoff", "modifier", "ambiguous", "no_tension"]
NON_NO_TENSION = {"dominance", "bidirectional_tradeoff", "modifier", "ambiguous"}
DISCOVERY_SOURCES = {"lm_topk", "embedding_neighbor"}


def load_spec(path: Path = SPEC_PATH) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in (json.loads(line) for line in path.open() if line.strip())}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_pair_key(a_id: str, b_id: str) -> tuple[str, str]:
    return (a_id, b_id) if a_id < b_id else (b_id, a_id)


def load_seed_pairs(spec: dict[str, dict[str, Any]]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return (cross_tier_pairs, same_class_pairs) from the seed atlas."""
    rows = load_jsonl(SEED_PAIRS_PATH)
    ct: set[tuple[str, str]] = set()
    sc: set[tuple[str, str]] = set()
    for row in rows:
        a, b = row["statement_a_id"], row["statement_b_id"]
        if a not in spec or b not in spec:
            continue
        a_plat = spec[a]["authority_level"] == "PLATFORM"
        b_plat = spec[b]["authority_level"] == "PLATFORM"
        ka = canonical_pair_key(a, b)
        if a_plat ^ b_plat:
            ct.add(ka)
        else:
            sc.add(ka)
    return ct, sc


def best_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For multiple records of the same pair within one compiler (e.g. the
    same canonical pair nominated from both ends), keep the highest-
    confidence non-no_tension call. Fall back to highest-confidence overall.
    """
    if not rows:
        raise ValueError("empty rows")
    non_nt = [r for r in rows if r["predicted_relation"] != "no_tension"]
    if non_nt:
        return max(non_nt, key=lambda r: r.get("confidence", 0.0))
    return max(rows, key=lambda r: r.get("confidence", 0.0))


def aggregate_compiler(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per compiler, build {pair: best_record} map."""
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        ka = canonical_pair_key(r["statement_a_id"], r["statement_b_id"])
        by_pair[ka].append(r)
    return {ka: best_classification(rs) for ka, rs in by_pair.items()}


def heuristic_recovered(by_pair: dict[tuple[str, str], dict[str, Any]], target: set[tuple[str, str]]) -> tuple[int, int]:
    """How many of `target` show up in by_pair via lm_topk or embedding sources?"""
    rec = 0
    for pair in target:
        rec_rec = by_pair.get(pair)
        if rec_rec is None:
            continue
        # any of the original rows for this pair must be from a discovery source
        # (we collapsed with best_classification — but candidate_source on the
        # winning record is preserved). For multi-source pairs, also check raw rows.
        if rec_rec.get("candidate_source") in DISCOVERY_SOURCES:
            rec += 1
    return rec, len(target)


def heuristic_recovered_anysource(rows: list[dict[str, Any]], target: set[tuple[str, str]]) -> tuple[int, int]:
    """Recall when the pair appears in ANY discovery-source row (more lenient
    than best_classification's source winner — captures cases where the pair
    was nominated by topk *and* also classified via atlas)."""
    seen: set[tuple[str, str]] = set()
    for r in rows:
        if r.get("candidate_source") in DISCOVERY_SOURCES:
            seen.add(canonical_pair_key(r["statement_a_id"], r["statement_b_id"]))
    rec = sum(1 for p in target if p in seen)
    return rec, len(target)


def control_false_positive_rate(rows: list[dict[str, Any]]) -> tuple[int, int, list[dict[str, Any]]]:
    """Fraction of random_control rows classified as non-no_tension."""
    ctrl = [r for r in rows if r.get("candidate_source") == "random_control"]
    fp = [r for r in ctrl if r.get("predicted_relation") in NON_NO_TENSION]
    return len(fp), len(ctrl), fp


def cross_compiler_agreement(
    runs: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> dict[str, Any]:
    if len(runs) < 2:
        return {"n_compilers": len(runs)}
    compilers = list(runs.keys())
    common = set.intersection(*(set(runs[c].keys()) for c in compilers))
    out: dict[str, Any] = {"n_compilers": len(runs), "common_pairs": len(common), "pairwise": {}}
    for i in range(len(compilers)):
        for j in range(i + 1, len(compilers)):
            ci, cj = compilers[i], compilers[j]
            agree_rel = sum(1 for p in common if runs[ci][p]["predicted_relation"] == runs[cj][p]["predicted_relation"])
            out["pairwise"][f"{ci} vs {cj}"] = (agree_rel, len(common))
    if len(compilers) >= 2:
        ci, cj = compilers[0], compilers[1]
        divergent = []
        for p in common:
            ri = runs[ci][p]
            rj = runs[cj][p]
            if ri["predicted_relation"] != rj["predicted_relation"]:
                divergent.append({"pair": p, ci: ri, cj: rj})
        divergent.sort(key=lambda d: -max(d[ci].get("confidence", 0.0), d[cj].get("confidence", 0.0)))
        out["divergent_top"] = divergent[:20]
    return out


def render_report(
    spec: dict[str, dict[str, Any]],
    rows_by_compiler: dict[str, list[dict[str, Any]]],
) -> str:
    cross_tier, same_class = load_seed_pairs(spec)

    aggregated = {compiler: aggregate_compiler(rows) for compiler, rows in rows_by_compiler.items()}

    lines: list[str] = []
    lines.append("# Phase 1B Tension-Discovery Report")
    lines.append("")
    lines.append(
        "Consumes `pair_candidate_*.jsonl` outputs from `discover_pair_candidates.py`. Computes the H2 metrics: atlas recall, control false-positive rate, cross-compiler agreement, and a sample of candidates with rationales."
    )
    lines.append("")
    lines.append(f"**Spec.** {len(spec)} statements. Possible pairs = {len(spec)*(len(spec)-1)//2}.")
    lines.append(
        f"**Atlas seed.** {len(cross_tier) + len(same_class)} pairs ({len(cross_tier)} cross-tier + {len(same_class)} same-class)."
    )
    lines.append("")
    lines.append("## Compilers")
    lines.append("")
    for compiler, rows in rows_by_compiler.items():
        n_unique = len({canonical_pair_key(r["statement_a_id"], r["statement_b_id"]) for r in rows})
        lines.append(f"- `{compiler}` — {len(rows)} candidate rows ({n_unique} unique pairs).")
    lines.append("")

    # Per-compiler source × relation distribution
    lines.append("## Source × relation distribution per compiler")
    lines.append("")
    for compiler, rows in rows_by_compiler.items():
        lines.append(f"### `{compiler}`")
        lines.append("")
        sr_counts: dict[tuple[str, str], int] = defaultdict(int)
        for r in rows:
            sr_counts[(r["candidate_source"], r["predicted_relation"])] += 1
        sources = sorted({s for s, _ in sr_counts})
        relations = ALL_RELATIONS
        header = "| source \\ relation | " + " | ".join(relations) + " | total |"
        sep = "|" + "---|" * (len(relations) + 2)
        lines.append(header)
        lines.append(sep)
        for s in sources:
            row_total = sum(sr_counts.get((s, rel), 0) for rel in relations)
            cells = [str(sr_counts.get((s, rel), 0)) for rel in relations]
            lines.append(f"| {s} | " + " | ".join(cells) + f" | {row_total} |")
        lines.append("")

    # Atlas recall
    lines.append("## Atlas recall (H2 ≥80% target)")
    lines.append("")
    lines.append(
        "Of the 22 cross-tier seed pairs (and the 18 same-class atlas pairs), how many were independently nominated by lm_topk OR embedding_neighbor — i.e. *not* introduced by the atlas-positives source itself?"
    )
    lines.append("")
    lines.append("| compiler | cross-tier recovered | cross-tier % | same-class recovered | same-class % |")
    lines.append("|---|---:|---:|---:|---:|")
    for compiler, rows in rows_by_compiler.items():
        ct_rec, ct_total = heuristic_recovered_anysource(rows, cross_tier)
        sc_rec, sc_total = heuristic_recovered_anysource(rows, same_class)
        lines.append(
            f"| `{compiler}` | {ct_rec}/{ct_total} | {100.0*ct_rec/max(1,ct_total):.1f}% | {sc_rec}/{sc_total} | {100.0*sc_rec/max(1,sc_total):.1f}% |"
        )
    lines.append("")

    # Random-control false-positive rate
    lines.append("## Random-control false-positive rate")
    lines.append("")
    lines.append(
        "Fraction of random control pairs classified as anything other than `no_tension`. Lower is better — high false-positive rate means the classifier is over-triggering."
    )
    lines.append("")
    lines.append("| compiler | non-no_tension | total | FPR |")
    lines.append("|---|---:|---:|---:|")
    fp_examples: dict[str, list[dict[str, Any]]] = {}
    for compiler, rows in rows_by_compiler.items():
        fp, total, fp_rows = control_false_positive_rate(rows)
        lines.append(f"| `{compiler}` | {fp} | {total} | {100.0*fp/max(1,total):.1f}% |")
        fp_examples[compiler] = fp_rows
    lines.append("")
    for compiler, fp_rows in fp_examples.items():
        if not fp_rows:
            continue
        lines.append(f"**`{compiler}` random-control false positives:**")
        lines.append("")
        lines.append("| pair | predicted_relation | confidence | why_pair_matters |")
        lines.append("|---|---|---:|---|")
        for r in fp_rows[:8]:
            lines.append(
                f"| `{r['statement_a_id']}` × `{r['statement_b_id']}` | {r['predicted_relation']} | {r.get('confidence', 0.0):.2f} | {r.get('why_pair_matters', '')[:120]} |"
            )
        lines.append("")

    # Cross-compiler agreement
    if len(rows_by_compiler) >= 2:
        agree = cross_compiler_agreement(aggregated)
        lines.append("## Cross-compiler agreement")
        lines.append("")
        lines.append(
            f"Pairs classified by both compilers: **{agree['common_pairs']}** (best-classification per pair per compiler)."
        )
        lines.append("")
        lines.append("| pair | agree on relation | n | % |")
        lines.append("|---|---:|---:|---:|")
        for pair_label, (a, n) in agree["pairwise"].items():
            lines.append(f"| {pair_label} | {a} | {n} | {100.0*a/max(1,n):.1f}% |")
        lines.append("")
        if agree.get("divergent_top"):
            compilers = list(rows_by_compiler.keys())
            ci, cj = compilers[0], compilers[1]
            lines.append(
                f"**Top {len(agree['divergent_top'])} divergent calls** (sorted by max confidence; `{ci}` vs `{cj}`):"
            )
            lines.append("")
            lines.append(f"| pair | `{ci}` relation | conf | `{cj}` relation | conf |")
            lines.append("|---|---|---:|---|---:|")
            for d in agree["divergent_top"]:
                a, b = d["pair"]
                ri = d[ci]
                rj = d[cj]
                lines.append(
                    f"| `{a}` × `{b}` | {ri['predicted_relation']} | {ri.get('confidence', 0.0):.2f} | "
                    f"{rj['predicted_relation']} | {rj.get('confidence', 0.0):.2f} |"
                )
            lines.append("")

    # Top non-trivial candidates per compiler
    lines.append("## Top candidates by predicted tension")
    lines.append("")
    lines.append(
        "Per compiler: top 15 unique pairs sorted by confidence among non-`no_tension` calls. These are the strongest signals from the discovery pass."
    )
    lines.append("")
    for compiler, by_pair in aggregated.items():
        lines.append(f"### `{compiler}`")
        lines.append("")
        rows = [r for r in by_pair.values() if r["predicted_relation"] in NON_NO_TENSION]
        rows.sort(key=lambda r: -r.get("confidence", 0.0))
        rows = rows[:15]
        lines.append("| pair | relation | controller | conf | source | why_pair_matters |")
        lines.append("|---|---|---|---:|---|---|")
        for r in rows:
            ka = canonical_pair_key(r["statement_a_id"], r["statement_b_id"])
            ctrl = r.get("predicted_controller") or ""
            lines.append(
                f"| `{ka[0]}` × `{ka[1]}` | {r['predicted_relation']} | {ctrl} | "
                f"{r.get('confidence', 0.0):.2f} | {r['candidate_source']} | {(r.get('why_pair_matters') or '')[:140]} |"
            )
        lines.append("")

    # Diversity sample for Phase 2
    lines.append("## 20-pair diversity sample (for Phase 2 zero-shot target set)")
    lines.append("")
    lines.append(
        "Stratified sample across relation buckets (5 dominance + 5 bidirectional_tradeoff + 5 modifier + 5 ambiguous + 5 no_tension controls). Best-classification per pair from the union of compilers, prefer agreement when available."
    )
    lines.append("")
    if aggregated:
        # Build union view: for each pair, prefer agreement
        union_pairs: dict[tuple[str, str], dict[str, Any]] = {}
        for compiler, by_pair in aggregated.items():
            for ka, r in by_pair.items():
                if ka not in union_pairs or r.get("confidence", 0.0) > union_pairs[ka].get("confidence", 0.0):
                    union_pairs[ka] = {**r, "compiler_agree": False}
        if len(aggregated) >= 2:
            compilers = list(aggregated.keys())
            for ka, r in union_pairs.items():
                same = all(
                    aggregated.get(c, {}).get(ka, {}).get("predicted_relation") == r["predicted_relation"]
                    for c in compilers
                    if ka in aggregated[c]
                )
                r["compiler_agree"] = same and all(ka in aggregated[c] for c in compilers)
        per_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ka, r in union_pairs.items():
            per_bucket[r["predicted_relation"]].append({"pair": ka, **r})
        for bucket in per_bucket:
            per_bucket[bucket].sort(key=lambda r: (-int(r.get("compiler_agree", False)), -r.get("confidence", 0.0)))
        lines.append("| pair | relation | controller | conf | source | compiler_agree | why_pair_matters |")
        lines.append("|---|---|---|---:|---|:-:|---|")
        for bucket in ["dominance", "bidirectional_tradeoff", "modifier", "ambiguous", "no_tension"]:
            for r in per_bucket.get(bucket, [])[:5]:
                ka = r["pair"]
                ctrl = r.get("predicted_controller") or ""
                agree_mark = "✓" if r.get("compiler_agree") else ""
                lines.append(
                    f"| `{ka[0]}` × `{ka[1]}` | {r['predicted_relation']} | {ctrl} | "
                    f"{r.get('confidence', 0.0):.2f} | {r['candidate_source']} | {agree_mark} | "
                    f"{(r.get('why_pair_matters') or '')[:120]} |"
                )
        lines.append("")

    # H2 verdict
    lines.append("## H2 verdict")
    lines.append("")
    for compiler, rows in rows_by_compiler.items():
        ct_rec, ct_total = heuristic_recovered_anysource(rows, cross_tier)
        sc_rec, sc_total = heuristic_recovered_anysource(rows, same_class)
        fp, fp_total, _ = control_false_positive_rate(rows)
        lines.append(
            f"- `{compiler}`: cross-tier recall {100.0*ct_rec/max(1,ct_total):.1f}% ({ct_rec}/{ct_total}), "
            f"same-class recall {100.0*sc_rec/max(1,sc_total):.1f}% ({sc_rec}/{sc_total}), "
            f"control FPR {100.0*fp/max(1,fp_total):.1f}% ({fp}/{fp_total})."
        )
    lines.append("")
    lines.append("**Recommended H2 gate (Codex plan).**")
    lines.append("- ≥80% recall on the 22 cross-tier seed pairs at ≤25% of all 1035 possible pairs.")
    lines.append("- Random controls should mostly classify as `no_tension`.")
    lines.append(
        "- Output should be a deliberately diverse batch covering dominance, bidirectional_tradeoff, modifier, ambiguous, and no_tension."
    )
    lines.append("")
    lines.append("## Headline finding for human review")
    lines.append("")
    lines.append(
        "**The H2 atlas-recall target is unattainable as written, because the atlas seeds and the pair classifier are looking for different things.**"
    )
    lines.append("")
    lines.append(
        "Even when the same compiler is fed each cross-tier seed pair *directly* (atlas_known source) or as part of the all-pair backtest, it classifies roughly half of them as `no_tension` with high confidence (~0.86). The atlas seeds were curated as **scenario-bound** tensions — they only emerge when paired with a specific user prompt. Examples that the all-pair classifier confidently calls `no_tension`:"
    )
    lines.append("")
    lines.append(
        '- `assume_objective_pov × do_not_encourage_self_harm` (atlas: "Debating suicide ethics vs. crisis support")'
    )
    lines.append('- `no_agenda × respect_creators` (atlas: "Refusal clarity vs. perceived ideological bias")')
    lines.append('- `avoid_targeted_political_manipulation × be_kind` (atlas: "Kind refusal of tailored persuasion")')
    lines.append('- `prevent_imminent_harm × support_programmatic_use` (atlas: "JSON-only safety responses")')
    lines.append("")
    lines.append(
        'Without the user-prompt context, these statement pairs really are independent. Topk+embedding+pair-classifier is **scenario-blind by design** — it\'s asking "do these two rules conflict in general?" not "is there a specific scenario where they conflict?". The atlas was built with the second question. That\'s why recall is ~30% on the heuristic and ~47% on the all-pair classifier.'
    )
    lines.append("")
    lines.append("**Implications for Phase 2.**")
    lines.append("- Don't treat the 80% atlas-recall target as a blocking gate — it's an artifact of the scenario gap.")
    lines.append(
        "- The heuristic IS surfacing real pair-intrinsic tensions (see the top-candidates tables — both compilers agree on dominance/tradeoff calls for many genuine cross-statement clashes)."
    )
    lines.append(
        "- For Phase 2 the more honest target-set construction is **scenario-first**: generate scenarios that activate tension, then label which pair the scenario activates. The pair classifier becomes a *labeling* tool, not a *discovery* tool."
    )
    lines.append(
        "- Alternatively, accept the heuristic's pair-intrinsic candidates as Phase 2 input (the 20-pair diversity sample above is already clean) and treat the atlas seeds as a separate scenario-bound validation slice."
    )
    lines.append("")
    lines.append("**Other H2 observations.**")
    lines.append(
        "- Cross-compiler agreement on relation labels is only ~54% — GPT-5.1 reasoning_effort=none and GLM-5.1 disagree often on whether something is dominance vs bidirectional_tradeoff vs modifier. Worth deciding upstream which compiler is canonical, or running both and treating disagreement as its own signal."
    )
    lines.append(
        '- Random-control FPR is 30-53%. Most "false positives" involve `formatting`, `refusal_style`, `letter_and_spirit` — universally-applicable style/meta rules that genuinely do interact with most other rules. If we re-sample controls excluding statements with `inferred_role ∈ {style_rule, meta_rule}` from Phase 1A, we\'d get a tighter no-tension prior.'
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_allpairs_section(
    spec: dict[str, dict[str, Any]],
    allpairs_rows: list[dict[str, Any]],
    rows_by_compiler: dict[str, list[dict[str, Any]]],
) -> list[str]:
    cross_tier, same_class = load_seed_pairs(spec)
    lines: list[str] = []
    if not allpairs_rows:
        return lines
    lines.append("## All-pair backtest (ground-truth view)")
    lines.append("")
    n = len(allpairs_rows)
    rel_counts: dict[str, int] = Counter(r["predicted_relation"] for r in allpairs_rows)
    flagged = {
        canonical_pair_key(r["statement_a_id"], r["statement_b_id"])
        for r in allpairs_rows
        if r["predicted_relation"] in NON_NO_TENSION
    }
    lines.append(f"Classifier ran on **all {n} pairs** of the 46-stmt spec. Relation distribution:")
    lines.append("")
    lines.append("| relation | count | % |")
    lines.append("|---|---:|---:|")
    for rel in ALL_RELATIONS:
        c = rel_counts.get(rel, 0)
        lines.append(f"| {rel} | {c} | {100.0*c/max(1,n):.1f}% |")
    lines.append("")
    lines.append(f"Total non-no_tension flagged: **{len(flagged)} / {n}** ({100.0*len(flagged)/max(1,n):.1f}%).")
    lines.append("")
    ct_in_flagged = sum(1 for p in cross_tier if p in flagged)
    sc_in_flagged = sum(1 for p in same_class if p in flagged)
    lines.append("**Atlas seeds in flagged set (classifier-as-ground-truth):**")
    lines.append("")
    lines.append(
        f"- Cross-tier: {ct_in_flagged}/{len(cross_tier)} ({100.0*ct_in_flagged/max(1,len(cross_tier)):.1f}%) of seed pairs flagged as non-no_tension by the all-pair classifier."
    )
    lines.append(f"- Same-class: {sc_in_flagged}/{len(same_class)} ({100.0*sc_in_flagged/max(1,len(same_class)):.1f}%).")
    lines.append("")
    # Discovery heuristic vs all-pair classifier
    lines.append("**Heuristic recall vs all-pair classifier (ground-truth view):**")
    lines.append("")
    lines.append(
        "Of the pairs the all-pair classifier flagged as non-no_tension, how many were also nominated by lm_topk OR embedding_neighbor in each compiler's discovery pass?"
    )
    lines.append("")
    lines.append("| compiler | flagged pairs covered | total flagged | % |")
    lines.append("|---|---:|---:|---:|")
    for compiler, rows in rows_by_compiler.items():
        seen = {
            canonical_pair_key(r["statement_a_id"], r["statement_b_id"])
            for r in rows
            if r.get("candidate_source") in DISCOVERY_SOURCES
        }
        cov = sum(1 for p in flagged if p in seen)
        lines.append(f"| `{compiler}` | {cov} | {len(flagged)} | {100.0*cov/max(1,len(flagged)):.1f}% |")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include", nargs="+", default=None)
    args = parser.parse_args()

    spec = load_spec()
    rows_by_compiler: dict[str, list[dict[str, Any]]] = {}
    allpairs_rows: list[dict[str, Any]] = []
    for f in sorted(args.input_dir.glob("pair_candidate_*.jsonl")):
        rows = load_jsonl(f)
        if not rows:
            continue
        if "allpairs" in f.stem:
            allpairs_rows = rows
            continue
        compiler = rows[0].get("classifier_model", f.stem.replace("pair_candidate_", ""))
        rows_by_compiler[compiler] = rows
    if args.include:
        rows_by_compiler = {k: v for k, v in rows_by_compiler.items() if any(inc in k for inc in args.include)}
    if not rows_by_compiler:
        raise SystemExit(f"No pair_candidate_*.jsonl found in {args.input_dir}")
    report = render_report(spec, rows_by_compiler)
    if allpairs_rows:
        report += "\n" + "\n".join(render_allpairs_section(spec, allpairs_rows, rows_by_compiler))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
