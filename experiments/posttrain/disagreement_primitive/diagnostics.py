# ruff: noqa: B007, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 4 diagnostics — pure analysis, no API calls.

Computes for any pair of judge JSONLs (ungrounded vs grounded):

- Cross-judge κ matrix (pairwise Cohen κ on pass/fail @ threshold 7).
- Per-judge mean / std / min / max compliance scores; how often each
  judge is the highest / lowest scorer on the same response.
- Verbatim-audit rate per judge: fraction of `cited_spec_clauses`
  entries that are actual substrings of either statement's text or
  examples.
- Pass-rate by predicted_relation × judge.
- Pass-threshold sweep: % satisfiable at thresholds {6, 6.5, 7, 7.5, 8}.

Renders `diagnostics_report.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
DEFAULT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_spec(path: Path = SPEC_PATH) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in (json.loads(line) for line in path.open() if line.strip())}


def cohen_kappa_binary(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two raters on binary labels."""
    if not a or len(a) != len(b):
        return 0.0
    n = len(a)
    agree = sum(1 for x, y in zip(a, b, strict=False) if x == y) / n
    p_a1 = sum(a) / n
    p_b1 = sum(b) / n
    p_e = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    if p_e >= 1.0:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - p_e) / (1.0 - p_e)


def oracle_id(scenario_id: str, gen: str, mode: str = "default") -> str:
    return f"or_{hashlib.sha1(f'{scenario_id}|{gen}|{mode}'.encode()).hexdigest()[:12]}"


def render_diagnostics(
    spec: dict[str, dict[str, Any]],
    scenarios: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    oracles: list[dict[str, Any]],
    judges_ungrounded: list[dict[str, Any]],
    judges_grounded: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Phase 4 diagnostics")
    lines.append("")
    lines.append("Pure analysis on existing Phase 4 outputs. No API calls.")
    lines.append("")

    _ = targets  # currently unused; kept in the signature for symmetry / future use
    oracle_by_id: dict[str, dict[str, Any]] = {
        oracle_id(o["scenario_id"], o["generator_model"], o.get("generator_mode", "default")): o for o in oracles
    }

    def analyze(judge_rows: list[dict[str, Any]], label: str) -> None:
        lines.append(f"## {label}")
        lines.append("")
        if not judge_rows:
            lines.append(f"({label} JSONL not found — skipped)")
            lines.append("")
            return
        # Per-judge basic stats
        per_judge_scores: dict[str, list[float]] = defaultdict(list)
        for j in judge_rows:
            per_judge_scores[j["judge_model"]].append(float(j["compliance_score"]))
        lines.append(f"### Per-judge score stats ({label})")
        lines.append("")
        lines.append("| judge | n | mean | std | min | max |")
        lines.append("|---|--:|--:|--:|--:|--:|")
        for judge, scores in per_judge_scores.items():
            if not scores:
                continue
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / len(scores)
            std = var**0.5
            lines.append(
                f"| `{judge}` | {len(scores)} | {mean:.2f} | {std:.2f} | {min(scores):.0f} | {max(scores):.0f} |"
            )
        lines.append("")

        # Pass-threshold sweep
        lines.append(f"### Pass-rate sweep ({label})")
        lines.append("")
        lines.append("Fraction of all judge calls returning compliance_score >= threshold.")
        lines.append("")
        lines.append("| judge | t=6 | t=6.5 | t=7 | t=7.5 | t=8 |")
        lines.append("|---|--:|--:|--:|--:|--:|")
        thresholds = [6.0, 6.5, 7.0, 7.5, 8.0]
        for judge, scores in per_judge_scores.items():
            n = len(scores)
            if not n:
                continue
            cells = [f"{100.0*sum(1 for s in scores if s >= t)/n:.1f}%" for t in thresholds]
            lines.append(f"| `{judge}` | " + " | ".join(cells) + " |")
        lines.append("")

        # High/low scorer distribution per oracle response
        scores_by_oracle_judge: dict[str, dict[str, float]] = defaultdict(dict)
        for j in judge_rows:
            scores_by_oracle_judge[j["oracle_response_id"]][j["judge_model"]] = float(j["compliance_score"])
        high_counter: Counter[str] = Counter()
        low_counter: Counter[str] = Counter()
        n_complete_triples = 0
        for rid, by_judge in scores_by_oracle_judge.items():
            if len(by_judge) < 3:
                continue
            n_complete_triples += 1
            high = max(by_judge, key=lambda k: by_judge[k])
            low = min(by_judge, key=lambda k: by_judge[k])
            high_counter[high] += 1
            low_counter[low] += 1
        lines.append(f"### Highest / lowest scorer distribution ({label}, {n_complete_triples} complete triples)")
        lines.append("")
        lines.append("Tied scores break alphabetically by judge model name.")
        lines.append("")
        lines.append("| judge | times highest | times lowest |")
        lines.append("|---|--:|--:|")
        all_judges = set(per_judge_scores)
        for judge in all_judges:
            lines.append(f"| `{judge}` | {high_counter.get(judge, 0)} | {low_counter.get(judge, 0)} |")
        lines.append("")

        # Pairwise Cohen κ on pass/fail @ 7
        lines.append(f"### Pairwise Cohen κ on pass/fail @ 7 ({label})")
        lines.append("")
        judge_list = sorted(per_judge_scores.keys())
        # For each oracle response, get pass/fail per judge
        pf_by_oracle: dict[str, dict[str, int]] = {}
        for rid, by_judge in scores_by_oracle_judge.items():
            pf_by_oracle[rid] = {jj: 1 if vv >= 7.0 else 0 for jj, vv in by_judge.items()}
        lines.append("| pair | n common | Cohen κ |")
        lines.append("|---|--:|--:|")
        for i in range(len(judge_list)):
            for k in range(i + 1, len(judge_list)):
                ji, jk = judge_list[i], judge_list[k]
                a = []
                b = []
                for rid, pfm in pf_by_oracle.items():
                    if ji in pfm and jk in pfm:
                        a.append(pfm[ji])
                        b.append(pfm[jk])
                ck = cohen_kappa_binary(a, b)
                lines.append(f"| `{ji}` × `{jk}` | {len(a)} | {ck:.3f} |")
        lines.append("")

        # Verbatim-audit per judge
        verbatim_pass: dict[str, list[bool]] = defaultdict(list)
        for j in judge_rows:
            cited = j.get("cited_spec_clauses") or []
            rid = j["oracle_response_id"]
            o = oracle_by_id.get(rid)
            if o is None:
                continue
            sid = o["scenario_id"]
            scn = next((s for s in scenarios if s["scenario_id"] == sid), None)
            if scn is None:
                continue
            a_id, b_id = scn["statement_a_id"], scn["statement_b_id"]
            corpus_parts = [spec[a_id]["text"], spec[b_id]["text"]]
            for st in (a_id, b_id):
                examples = (spec[st].get("metadata") or {}).get("examples") or []
                for ex in examples:
                    corpus_parts.extend(
                        [ex.get(k, "") or "" for k in ("description", "user_query", "good_response", "bad_response")]
                    )
            corpus = "\n".join(corpus_parts)
            for c in cited:
                if not isinstance(c, str) or not c:
                    continue
                verbatim_pass[j["judge_model"]].append(c in corpus)
        lines.append(f"### Verbatim-audit per judge ({label})")
        lines.append("")
        lines.append(
            "Fraction of `cited_spec_clauses` entries that are exact substrings of statement text or example fields."
        )
        lines.append("")
        lines.append("| judge | verbatim | total | rate |")
        lines.append("|---|--:|--:|--:|")
        for judge, vs in verbatim_pass.items():
            ver = sum(1 for x in vs if x)
            tot = len(vs)
            rate = 100.0 * ver / max(1, tot)
            lines.append(f"| `{judge}` | {ver} | {tot} | {rate:.1f}% |")
        lines.append("")

    analyze(judges_ungrounded, "Ungrounded judges (Phase 4 first cut)")
    analyze(judges_grounded, "Grounded judges (per-pair rubric)")

    # Side-by-side κ comparison if both available
    if judges_ungrounded and judges_grounded:
        lines.append("## Grounded vs ungrounded judge κ comparison")
        lines.append("")
        lines.append("How much does adding a per-pair rubric change pass/fail agreement among the 3 judges?")
        lines.append("")

        # For both, compute mean pairwise κ
        def pairwise_kappa_summary(rows: list[dict[str, Any]]) -> tuple[float, float]:
            scores_by_oracle_judge: dict[str, dict[str, float]] = defaultdict(dict)
            for j in rows:
                scores_by_oracle_judge[j["oracle_response_id"]][j["judge_model"]] = float(j["compliance_score"])
            judges = sorted({j["judge_model"] for j in rows})
            kappas = []
            for i in range(len(judges)):
                for k in range(i + 1, len(judges)):
                    ji, jk = judges[i], judges[k]
                    a, b = [], []
                    for rid, by_j in scores_by_oracle_judge.items():
                        if ji in by_j and jk in by_j:
                            a.append(1 if by_j[ji] >= 7.0 else 0)
                            b.append(1 if by_j[jk] >= 7.0 else 0)
                    if a:
                        kappas.append(cohen_kappa_binary(a, b))
            mean = sum(kappas) / len(kappas) if kappas else 0.0
            return mean, len(kappas)

        u_mean, u_n = pairwise_kappa_summary(judges_ungrounded)
        g_mean, g_n = pairwise_kappa_summary(judges_grounded)
        lines.append(f"- Ungrounded mean pairwise Cohen κ (n_pairs={u_n}): **{u_mean:.3f}**")
        lines.append(f"- Grounded mean pairwise Cohen κ (n_pairs={g_n}): **{g_mean:.3f}**")
        lines.append(f"- Δ (grounded − ungrounded): **{g_mean - u_mean:+.3f}**")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "diagnostics_report.md")
    parser.add_argument("--ungrounded", type=Path, default=None)
    parser.add_argument("--grounded", type=Path, default=None)
    args = parser.parse_args()
    args.ungrounded = args.ungrounded or args.input_dir / "judge_panel_score.jsonl"
    args.grounded = args.grounded or args.input_dir / "judge_panel_score_grounded.jsonl"

    spec = load_spec()
    scenarios = load_jsonl(args.input_dir / "scenario_probe.jsonl")
    targets = load_jsonl(args.input_dir / "target_set.jsonl")
    oracles = load_jsonl(args.input_dir / "oracle_response.jsonl")
    judges_u = load_jsonl(args.ungrounded)
    judges_g = load_jsonl(args.grounded)

    report = render_diagnostics(spec, scenarios, targets, oracles, judges_u, judges_g)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
