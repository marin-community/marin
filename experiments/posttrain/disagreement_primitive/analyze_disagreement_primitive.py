# ruff: noqa: E501, B007, RUF001, RUF003
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 disagreement-primitive analyzer.

Joins `oracle_response.jsonl` and `judge_panel_score.jsonl` against
`scenario_probe.jsonl` and `target_set.jsonl`, computes the four core
signals from the Codex plan:

- **Oracle satisfiability**: for each scenario, can ANY of the strong
  generators produce a response that all 3 judges score >= 7?
- **Compliance disagreement**: for each (scenario, generator) pair,
  the spread of compliance scores across the 3 judges. Expressed as
  a Fleiss-style κ on pass/fail discretization.
- **Activation disagreement**: do the 3 judges agree on the
  controlling statement for the same response?
- **Behavioral dispersion among generators**: do the 3 generators
  produce meaningfully different responses (judged by mean compliance
  spread + self-declared controlling statement variance)?

Each scenario gets a label per Codex Phase 4:
- #1 `model_behavior`: oracle satisfiable + judges agree → training
  signal, no spec repair needed
- #2 `cross_tension_needed`: real value tradeoff; cross-tension rubric
  needed
- #3 `spec_ambiguity`: judges disagree on best generator's response
- `oracle_unsatisfiable`: no generator's response clears all judges
- `scenario_bug`: low confidence + varied failure reasons → probe is bad

Renders `oracle_satisfiability_report.md`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "oracle_satisfiability_report.md"

PASS_THRESHOLD = 7.0
COMPLIANCE_SPREAD_AMBIGUITY = 3.0  # >=3-pt spread on best response → ambiguity
ACTIVATION_DISAGREEMENT_THRESHOLD = 0.66  # <2-of-3 agree on controlling statement → ambiguity


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fleiss_kappa_binary(ratings: list[list[int]]) -> float:
    """Compute Fleiss' kappa for binary ratings.

    `ratings` is a list of items; each item is a list of binary 0/1
    ratings from N raters. Returns kappa in [-1, 1].
    """
    if not ratings:
        return 0.0
    n_items = len(ratings)
    n_raters = len(ratings[0])
    if n_raters < 2:
        return 0.0
    # Per-item agreement: Pi = (sum n_ij^2 - n) / (n*(n-1))
    pis = []
    p_total = [0.0, 0.0]
    for item in ratings:
        if len(item) != n_raters:
            continue
        c0 = item.count(0)
        c1 = item.count(1)
        pi = (c0 * c0 + c1 * c1 - n_raters) / (n_raters * (n_raters - 1))
        pis.append(pi)
        p_total[0] += c0
        p_total[1] += c1
    if not pis:
        return 0.0
    p_mean = sum(pis) / len(pis)
    p_total[0] /= n_items * n_raters
    p_total[1] /= n_items * n_raters
    p_e = p_total[0] ** 2 + p_total[1] ** 2
    if p_e == 1.0:
        return 1.0  # everyone agreed unanimously on every item
    return (p_mean - p_e) / (1.0 - p_e)


def aggregate_by_oracle(
    oracles: list[dict[str, Any]],
    judges: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """For each oracle_response_id, attach all judge scores.

    Reconstructs oracle_response_id by hashing (scenario_id, generator,
    mode) — must match the formula used by both writer scripts.
    """
    import hashlib

    def oid(scenario_id: str, gen: str, mode: str) -> str:
        return f"or_{hashlib.sha1(f'{scenario_id}|{gen}|{mode}'.encode()).hexdigest()[:12]}"

    o_by_id: dict[str, dict[str, Any]] = {}
    for o in oracles:
        rid = oid(o["scenario_id"], o["generator_model"], o.get("generator_mode", "default"))
        o_by_id[rid] = o

    j_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for j in judges:
        rid = j["oracle_response_id"]
        if rid in o_by_id:
            j_by_pair[rid].append(j)

    out: dict[str, dict[str, Any]] = {}
    for rid, o in o_by_id.items():
        out[rid] = {"oracle": o, "judges": j_by_pair.get(rid, [])}
    return out


def per_scenario_view(
    scenarios: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    oracles: list[dict[str, Any]],
    judges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_by_pair = {(t["statement_a_id"], t["statement_b_id"]): t for t in targets}
    by_oracle = aggregate_by_oracle(oracles, judges)
    by_scenario: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        by_scenario[s["scenario_id"]] = {
            "scenario": s,
            "target": target_by_pair.get((s["statement_a_id"], s["statement_b_id"])),
            "oracle_responses": [],
        }
    for rid, payload in by_oracle.items():
        sid = payload["oracle"]["scenario_id"]
        if sid not in by_scenario:
            continue
        by_scenario[sid]["oracle_responses"].append(payload)
    return list(by_scenario.values())


def label_scenario(view: dict[str, Any]) -> dict[str, Any]:
    """Apply Codex Phase 4 label rules to one scenario view.

    Returns a dict with: label, oracle_satisfiable, best_generator,
    best_mean_score, best_score_spread, controlling_majority,
    activation_disagreement, compliance_disagreement, signals.
    """
    target = view["target"] or {}
    predicted_relation = target.get("predicted_relation", "ambiguous")
    bucket = target.get("bucket", predicted_relation)
    a_id = view["scenario"]["statement_a_id"]
    b_id = view["scenario"]["statement_b_id"]

    # For each oracle response: judge mean, judge min, pass/fail per judge
    per_oracle: list[dict[str, Any]] = []
    for r in view["oracle_responses"]:
        scores = [j["compliance_score"] for j in r["judges"]]
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        spread = max(scores) - min(scores)
        all_pass = all(s >= PASS_THRESHOLD for s in scores)
        ctrls = [j["controlling_statement"] for j in r["judges"]]
        ctrl_counts = Counter(ctrls)
        controlling_majority = ctrl_counts.most_common(1)[0][0] if ctrl_counts else "neither"
        activation_agree_frac = (ctrl_counts.most_common(1)[0][1] / len(ctrls)) if ctrls else 0.0
        per_oracle.append(
            {
                "generator": r["oracle"]["generator_model"],
                "self_declared_ctrl": r["oracle"].get("self_declared_controlling_statement"),
                "scores": scores,
                "mean": mean,
                "spread": spread,
                "min": min(scores),
                "max": max(scores),
                "all_pass": all_pass,
                "controlling_majority": controlling_majority,
                "activation_agree_frac": activation_agree_frac,
                "judges": r["judges"],
                "response": r["oracle"]["response"],
            }
        )

    if not per_oracle:
        return {
            "scenario_id": view["scenario"]["scenario_id"],
            "bucket": bucket,
            "predicted_relation": predicted_relation,
            "label": "oracle_unsatisfiable",
            "oracle_satisfiable": False,
            "best_generator": None,
            "best_mean_score": None,
            "best_score_spread": None,
            "controlling_majority": None,
            "activation_disagreement": None,
            "compliance_disagreement": None,
            "signals": {"missing_oracles": True},
            "n_oracles": 0,
            "per_oracle": [],
        }

    oracle_satisfiable = any(p["all_pass"] for p in per_oracle)
    best = max(per_oracle, key=lambda p: p["mean"])
    activation_disagreement = best["activation_agree_frac"] < ACTIVATION_DISAGREEMENT_THRESHOLD
    compliance_disagreement = best["spread"] >= COMPLIANCE_SPREAD_AMBIGUITY

    # Behavioral dispersion across generators (range of mean scores)
    means = [p["mean"] for p in per_oracle]
    mean_range = max(means) - min(means) if means else 0.0
    behavioral_dispersion = mean_range >= COMPLIANCE_SPREAD_AMBIGUITY

    # Label rules
    if not oracle_satisfiable:
        label = "oracle_unsatisfiable"
    elif compliance_disagreement or activation_disagreement:
        label = "spec_ambiguity"
    elif predicted_relation in ("bidirectional_tradeoff", "modifier") and oracle_satisfiable:
        # Genuine value tradeoff that the spec admits — Phase 5 may want a
        # cross-tension rubric but current rubric still operational.
        label = "cross_tension_needed" if behavioral_dispersion else "model_behavior"
    else:
        # Oracle satisfiable + judges agree → training signal.
        label = "model_behavior"

    return {
        "scenario_id": view["scenario"]["scenario_id"],
        "bucket": bucket,
        "predicted_relation": predicted_relation,
        "variant": view["scenario"]["variant"],
        "statement_a_id": a_id,
        "statement_b_id": b_id,
        "scenario_text": view["scenario"]["scenario_text"][:200],
        "label": label,
        "oracle_satisfiable": oracle_satisfiable,
        "best_generator": best["generator"],
        "best_mean_score": round(best["mean"], 2),
        "best_score_spread": best["spread"],
        "controlling_majority": best["controlling_majority"],
        "activation_disagreement": activation_disagreement,
        "compliance_disagreement": compliance_disagreement,
        "behavioral_dispersion": behavioral_dispersion,
        "mean_range": round(mean_range, 2),
        "signals": {
            "compliance_spread_best": best["spread"],
            "activation_agree_frac_best": round(best["activation_agree_frac"], 2),
            "mean_range_across_generators": round(mean_range, 2),
        },
        "n_oracles": len(per_oracle),
        "per_oracle": per_oracle,
    }


def render_report(per_scenario: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Phase 4 — Disagreement Primitive Eval")
    lines.append("")
    lines.append(
        "65 target pairs × 3 scenarios × 3 strong-oracle generators × 3 ensemble judges. Computes oracle satisfiability, compliance disagreement, activation disagreement, and per-scenario label per the Codex plan."
    )
    lines.append("")
    lines.append(f"**Pass threshold.** Compliance score >= {PASS_THRESHOLD} (out of 10) counts as a passing judgment.")
    lines.append(
        f"**Compliance disagreement.** Score spread on best generator's response >= {COMPLIANCE_SPREAD_AMBIGUITY} points."
    )
    lines.append(
        f"**Activation disagreement.** <{int(ACTIVATION_DISAGREEMENT_THRESHOLD*100)}% of judges agree on controlling statement (i.e. <2-of-3)."
    )
    lines.append("")

    n = len(per_scenario)
    label_counts = Counter(s["label"] for s in per_scenario)
    bucket_counts = Counter(s["bucket"] for s in per_scenario)

    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Total scenarios: **{n}**")
    lines.append(
        f"- Oracle satisfiable: **{sum(1 for s in per_scenario if s['oracle_satisfiable'])}/{n}** ({100.0*sum(1 for s in per_scenario if s['oracle_satisfiable'])/max(1,n):.1f}%)"
    )
    lines.append("")
    lines.append("**Label distribution (per Codex Phase 4):**")
    lines.append("")
    lines.append("| label | count | % |")
    lines.append("|---|--:|--:|")
    for lab in ["model_behavior", "cross_tension_needed", "spec_ambiguity", "oracle_unsatisfiable", "scenario_bug"]:
        c = label_counts.get(lab, 0)
        lines.append(f"| {lab} | {c} | {100.0*c/max(1,n):.1f}% |")
    lines.append("")

    # Bucket × label breakdown
    lines.append("## Label × bucket breakdown")
    lines.append("")
    by_bucket_label: dict[tuple[str, str], int] = defaultdict(int)
    for s in per_scenario:
        by_bucket_label[(s["bucket"], s["label"])] += 1
    label_cols = ["model_behavior", "cross_tension_needed", "spec_ambiguity", "oracle_unsatisfiable", "scenario_bug"]
    lines.append("| bucket | " + " | ".join(label_cols) + " | total |")
    lines.append("|" + "---|" * (len(label_cols) + 2))
    for bucket in ["dominance", "bidirectional_tradeoff", "modifier", "ambiguous", "no_tension"]:
        if bucket_counts.get(bucket, 0) == 0:
            continue
        cells = [str(by_bucket_label.get((bucket, lab), 0)) for lab in label_cols]
        total = bucket_counts.get(bucket, 0)
        lines.append(f"| {bucket} | " + " | ".join(cells) + f" | {total} |")
    lines.append("")

    # Judge ensemble metrics
    lines.append("## Judge ensemble agreement")
    lines.append("")
    # Compliance pass/fail per (oracle_response, judge): build matrix per scenario's best oracle
    pf_matrix: list[list[int]] = []
    ctrl_matches: list[int] = []
    for s in per_scenario:
        if not s["per_oracle"]:
            continue
        best = next((p for p in s["per_oracle"] if p["generator"] == s["best_generator"]), s["per_oracle"][0])
        scores = best["scores"]
        if len(scores) < 2:
            continue
        pf_matrix.append([1 if x >= PASS_THRESHOLD else 0 for x in scores])
        ctrls = [j["controlling_statement"] for j in best["judges"]]
        if len(ctrls) >= 2:
            ctrl_matches.append(1 if len(set(ctrls)) == 1 else 0)
    fk = fleiss_kappa_binary(pf_matrix)
    unanimous_ctrl = sum(ctrl_matches) / max(1, len(ctrl_matches))
    lines.append(f"- Pass/fail Fleiss κ on best-generator response across {len(pf_matrix)} scenarios: **{fk:.3f}**")
    lines.append(
        f"- Fraction of best-generator responses where all 3 judges name the same controlling statement: **{100.0*unanimous_ctrl:.1f}%** ({sum(ctrl_matches)}/{len(ctrl_matches)})"
    )
    lines.append("")

    # Behavioral dispersion across generators
    lines.append("## Behavioral dispersion across generators")
    lines.append("")
    mean_ranges = [s["mean_range"] for s in per_scenario if s["per_oracle"]]
    if mean_ranges:
        avg_range = sum(mean_ranges) / len(mean_ranges)
        gt3 = sum(1 for r in mean_ranges if r >= COMPLIANCE_SPREAD_AMBIGUITY)
        lines.append(f"- Average mean-score range across generators: **{avg_range:.2f}** points")
        lines.append(
            f"- Scenarios with >= {COMPLIANCE_SPREAD_AMBIGUITY}-point dispersion: **{gt3}/{len(mean_ranges)}** ({100.0*gt3/max(1,len(mean_ranges)):.1f}%)"
        )
    lines.append("")

    # Per-generator mean scores
    gen_scores: dict[str, list[float]] = defaultdict(list)
    for s in per_scenario:
        for p in s["per_oracle"]:
            gen_scores[p["generator"]].append(p["mean"])
    lines.append("**Per-generator mean compliance score (averaged over scenarios):**")
    lines.append("")
    lines.append("| generator | n responses | mean | min | max |")
    lines.append("|---|--:|--:|--:|--:|")
    for gen, scores in gen_scores.items():
        if not scores:
            continue
        lines.append(
            f"| `{gen}` | {len(scores)} | {sum(scores)/len(scores):.2f} | {min(scores):.2f} | {max(scores):.2f} |"
        )
    lines.append("")

    # Top oracle-unsatisfiable scenarios (interesting for spec repair)
    unsat = [s for s in per_scenario if not s["oracle_satisfiable"]]
    lines.append("## Oracle-unsatisfiable scenarios")
    lines.append("")
    if not unsat:
        lines.append("(none — every scenario had at least one generator pass all 3 judges)")
        lines.append("")
    else:
        lines.append(
            f"**{len(unsat)}** scenarios where no generator's response cleared all 3 judges. These are the strongest signals for spec ambiguity / overconstrained rubric / scenario bug."
        )
        lines.append("")
        unsat.sort(key=lambda s: -s["best_mean_score"] if s["best_mean_score"] is not None else 0)
        lines.append("| scenario_id | bucket | predicted_relation | best generator | best mean | best spread |")
        lines.append("|---|---|---|---|--:|--:|")
        for s in unsat[:20]:
            lines.append(
                f"| `{s['scenario_id']}` | {s['bucket']} | {s['predicted_relation']} | "
                f"{s['best_generator']} | {s['best_mean_score']} | {s['best_score_spread']} |"
            )
        lines.append("")

    # Top compliance-disagreement scenarios
    spec_amb = [s for s in per_scenario if s["label"] == "spec_ambiguity"]
    lines.append("## Spec-ambiguity scenarios (judge disagreement)")
    lines.append("")
    if not spec_amb:
        lines.append("(none)")
    else:
        spec_amb.sort(key=lambda s: -s["best_score_spread"] if s["best_score_spread"] is not None else 0)
        lines.append(
            f"**{len(spec_amb)}** scenarios where judges disagree on the best generator's response (compliance spread >= {COMPLIANCE_SPREAD_AMBIGUITY} OR <2/3 agree on controlling statement)."
        )
        lines.append("")
        lines.append("| scenario_id | bucket | best mean | spread | activation disagree | controlling majority |")
        lines.append("|---|---|--:|--:|:-:|---|")
        for s in spec_amb[:20]:
            ad = "✓" if s["activation_disagreement"] else ""
            lines.append(
                f"| `{s['scenario_id']}` | {s['bucket']} | {s['best_mean_score']} | "
                f"{s['best_score_spread']} | {ad} | {s['controlling_majority']} |"
            )
    lines.append("")

    # No_tension control behavior
    lines.append("## no_tension control behavior")
    lines.append("")
    nt = [s for s in per_scenario if s["bucket"] == "no_tension"]
    if nt:
        nt_unsat = sum(1 for s in nt if not s["oracle_satisfiable"])
        nt_amb = sum(1 for s in nt if s["label"] == "spec_ambiguity")
        nt_mean = sum(s["best_mean_score"] for s in nt if s["best_mean_score"] is not None) / max(1, len(nt))
        lines.append(f"- {len(nt)} no_tension scenarios.")
        lines.append(f"- Oracle-unsatisfiable on controls: **{nt_unsat}/{len(nt)}** (should be near 0).")
        lines.append(f"- Spec-ambiguity flagged on controls: **{nt_amb}/{len(nt)}** (should be near 0).")
        lines.append(f"- Mean compliance on best generator (controls): **{nt_mean:.2f}** (should be near 10).")
    lines.append("")

    # H4 verdict
    lines.append("## H4 verdict")
    lines.append("")
    lines.append("Three signals to decide which become Phase 5 materialization triggers:")
    lines.append(
        f"- **Oracle satisfiability** as a primary signal — {sum(1 for s in per_scenario if s['oracle_satisfiable'])}/{n} satisfiable. Unsat scenarios go straight to spec_repair candidates."
    )
    lines.append(
        f"- **Compliance disagreement** (Fleiss κ on pass/fail) — {fk:.3f}. Lower κ = more spec ambiguity. Worth keeping as a trigger."
    )
    lines.append(
        f"- **Activation disagreement** — {100.0 - 100.0*unanimous_ctrl:.1f}% of best-generator responses had non-unanimous controlling statement. This is where cross-tension rubrics earn their keep."
    )
    lines.append(
        "- **Behavioral dispersion** (mean-range across generators) is a candidate auxiliary signal but, per the Codex plan, should be ablated against strong-only generators before becoming a trigger."
    )
    lines.append("")
    lines.append("**Recommendation for Gate H4 → Phase 5 materialization triggers:**")
    lines.append("- Materialize for repair: `oracle_unsatisfiable` and `spec_ambiguity` labels.")
    lines.append("- Do NOT materialize: `model_behavior` labels (training signal, no spec edit).")
    lines.append(
        "- For `cross_tension_needed`: surface to the spec author as candidates for explicit cross-tension rubrics."
    )
    lines.append("- Behavioral dispersion stays as diagnostic only until a future ablation justifies it.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_INPUT_DIR / "scenario_probe.jsonl")
    parser.add_argument("--target-set", type=Path, default=DEFAULT_INPUT_DIR / "target_set.jsonl")
    parser.add_argument("--oracle-responses", type=Path, default=DEFAULT_INPUT_DIR / "oracle_response.jsonl")
    parser.add_argument("--judge-scores", type=Path, default=DEFAULT_INPUT_DIR / "judge_panel_score.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-scenario-out", type=Path, default=DEFAULT_INPUT_DIR / "per_scenario_labels.jsonl")
    args = parser.parse_args()

    scenarios = load_jsonl(args.scenarios)
    targets = load_jsonl(args.target_set)
    oracles = load_jsonl(args.oracle_responses)
    judges = load_jsonl(args.judge_scores)

    views = per_scenario_view(scenarios, targets, oracles, judges)
    labeled = [label_scenario(v) for v in views]

    args.per_scenario_out.parent.mkdir(parents=True, exist_ok=True)
    with args.per_scenario_out.open("w", encoding="utf-8") as fh:
        for s in labeled:
            row = {k: v for k, v in s.items() if k != "per_oracle"}
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    report = render_report(labeled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output} and {args.per_scenario_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
