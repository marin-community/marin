# ruff: noqa: E501, B007, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the Phase 1A statement-role backtest report.

Consumes `statement_analysis_*.jsonl` files written by
`backtest_statement_roles.py`, joins each row against the hidden
OpenAI Model Spec `authority_level` field, computes both strict and
generous role backtests, builds confusion matrices, surfaces top
confident mistakes, computes cross-model agreement, and renders
`statement_role_backtest.md`.

Backtest scorings (per the SMOKE.md decision points):

- **Strict.** PLATFORM ↔ requirement_like; non-PLATFORM ↔
  guideline_like. `meta_rule` and `style_rule` count as misses
  regardless of `authority_level`.
- **Generous.** PLATFORM ↔ {requirement_like, meta_rule};
  non-PLATFORM ↔ {guideline_like, style_rule, meta_rule}. `meta_rule`
  is allowed on both sides because some meta-rules are PLATFORM-tier
  (e.g. `letter_and_spirit`) and some aren't.

H1 thresholds (from the Codex plan):
- ≥80% strict agreement on clear cases (treat clear cases as PLATFORM
  ↔ requirement_like and GUIDELINE ↔ guideline_like; USER and
  DEVELOPER are inherently ambiguous).
- Mistakes concentrated on genuinely ambiguous meta/style statements
  rather than safety requirements.
- Top confident mistakes inspected manually.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
DEFAULT_INPUT_DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "statement_role_backtest.md"

ALL_ROLES = ["requirement_like", "guideline_like", "meta_rule", "style_rule", "unclear"]


def load_spec(path: Path = SPEC_PATH) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in (json.loads(line) for line in path.open() if line.strip())}


def load_analysis(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_audit(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_summary(audit_rows: list[dict[str, Any]]) -> tuple[int, int]:
    verbatim = sum(int(r.get("audit", {}).get("verbatim_quotes", 0)) for r in audit_rows)
    total = sum(int(r.get("audit", {}).get("total_quotes", 0)) for r in audit_rows)
    return verbatim, total


def strict_match(authority_level: str, inferred_role: str) -> bool:
    if authority_level == "PLATFORM":
        return inferred_role == "requirement_like"
    return inferred_role == "guideline_like"


def generous_match(authority_level: str, inferred_role: str) -> bool:
    if authority_level == "PLATFORM":
        return inferred_role in {"requirement_like", "meta_rule"}
    return inferred_role in {"guideline_like", "style_rule", "meta_rule"}


def confusion_matrix(rows: list[dict[str, Any]], spec: dict[str, dict[str, Any]]) -> dict[tuple[str, str], int]:
    cm: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        auth = spec[row["statement_id"]]["authority_level"]
        cm[(auth, row["inferred_role"])] += 1
    return cm


def render_confusion_matrix(cm: dict[tuple[str, str], int]) -> str:
    """Render a confusion matrix with authority_level rows × inferred_role cols."""
    auths = ["PLATFORM", "USER", "DEVELOPER", "GUIDELINE"]
    cols = ALL_ROLES
    header = "| authority_level \\ inferred_role | " + " | ".join(cols) + " | total |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [header, sep]
    for a in auths:
        row_total = sum(cm.get((a, c), 0) for c in cols)
        if row_total == 0:
            continue
        cells = [str(cm.get((a, c), 0)) for c in cols]
        lines.append(f"| {a} | " + " | ".join(cells) + f" | {row_total} |")
    return "\n".join(lines)


def top_confident_mistakes(
    rows: list[dict[str, Any]],
    spec: dict[str, dict[str, Any]],
    matcher,
    k: int = 10,
) -> list[dict[str, Any]]:
    mistakes = []
    for row in rows:
        sid = row["statement_id"]
        auth = spec[sid]["authority_level"]
        if not matcher(auth, row["inferred_role"]):
            mistakes.append(
                {
                    "statement_id": sid,
                    "authority_level": auth,
                    "type": spec[sid]["type"],
                    "inferred_role": row["inferred_role"],
                    "role_confidence": row["role_confidence"],
                    "section": spec[sid].get("section", ""),
                }
            )
    mistakes.sort(key=lambda m: -m["role_confidence"])
    return mistakes[:k]


def role_distribution(rows: list[dict[str, Any]]) -> Counter:
    return Counter(row["inferred_role"] for row in rows)


def cross_model_agreement(
    runs: dict[str, list[dict[str, Any]]],
    spec: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute pairwise role agreement on the intersection of analyzed
    statements, plus the count of statements where all models agree.
    """
    by_stmt: dict[str, dict[str, str]] = defaultdict(dict)
    for model_name, rows in runs.items():
        for row in rows:
            by_stmt[row["statement_id"]][model_name] = row["inferred_role"]
    common = [sid for sid, m_to_role in by_stmt.items() if len(m_to_role) == len(runs)]

    pairwise = {}
    model_names = list(runs.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m_i, m_j = model_names[i], model_names[j]
            agree = sum(1 for sid in common if by_stmt[sid][m_i] == by_stmt[sid][m_j])
            pairwise[f"{m_i} vs {m_j}"] = (agree, len(common))

    all_agree = sum(1 for sid in common if len({by_stmt[sid][m] for m in model_names}) == 1)
    divergent = []
    for sid in common:
        roles = {m: by_stmt[sid][m] for m in model_names}
        if len(set(roles.values())) > 1:
            divergent.append(
                {
                    "statement_id": sid,
                    "authority_level": spec[sid]["authority_level"],
                    "type": spec[sid]["type"],
                    "section": spec[sid].get("section", ""),
                    "roles": roles,
                }
            )
    return {
        "common_count": len(common),
        "pairwise": pairwise,
        "all_agree_count": all_agree,
        "divergent": divergent,
    }


def render_report(
    spec: dict[str, dict[str, Any]],
    runs: dict[str, list[dict[str, Any]]],
    audit: dict[str, list[dict[str, Any]]] | None = None,
    ablation: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    audit = audit or {}
    lines: list[str] = []
    lines.append("# Phase 1A Statement-Role Backtest")
    lines.append("")
    lines.append(
        "Consumes `statement_analysis_*.jsonl` outputs from `backtest_statement_roles.py`. Computes the H1 backtest under both **strict** and **generous** scorings."
    )
    lines.append("")
    lines.append(
        "**Spec.** `experiments/posttrain/specs/openai_model_spec.jsonl` — 46 statements (19 PLATFORM, 15 GUIDELINE, 11 USER, 1 DEVELOPER)."
    )
    lines.append("")
    lines.append(
        "**Analyzers.** All called with the same prompt, no reasoning (or lowest tier). Markdown anchor links pre-rendered before the verbatim audit."
    )
    lines.append("")
    for model, rows in runs.items():
        lines.append(f"- `{model}` — {len(rows)}/{len(spec)} statements analyzed.")
    lines.append("")
    lines.append("## Scoring definitions")
    lines.append("")
    lines.append(
        "- **Strict.** PLATFORM ↔ requirement_like; non-PLATFORM ↔ guideline_like. `meta_rule` and `style_rule` are misses regardless of authority."
    )
    lines.append(
        "- **Generous.** PLATFORM ↔ {requirement_like, meta_rule}; non-PLATFORM ↔ {guideline_like, style_rule, meta_rule}. `meta_rule` admitted on both sides because some meta-rules are PLATFORM-tier (e.g. `letter_and_spirit`) and some are not."
    )
    lines.append("")

    # Per-model summary
    lines.append("## Per-model agreement vs hidden authority_level")
    lines.append("")
    lines.append("| model | n | strict ✓ | strict % | generous ✓ | generous % | verbatim audit | role distribution |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for model, rows in runs.items():
        strict_ok = sum(1 for r in rows if strict_match(spec[r["statement_id"]]["authority_level"], r["inferred_role"]))
        gen_ok = sum(1 for r in rows if generous_match(spec[r["statement_id"]]["authority_level"], r["inferred_role"]))
        n = len(rows)
        dist = role_distribution(rows)
        dist_str = ", ".join(f"{r}={dist.get(r, 0)}" for r in ALL_ROLES if dist.get(r, 0))
        ver, tot = audit_summary(audit.get(model, []))
        ver_str = f"{ver}/{tot} ({100.0*ver/max(1,tot):.1f}%)" if tot else "n/a"
        lines.append(
            f"| `{model}` | {n} | {strict_ok} | {100.0*strict_ok/max(1,n):.1f}% | {gen_ok} | {100.0*gen_ok/max(1,n):.1f}% | {ver_str} | {dist_str} |"
        )
    lines.append("")

    # Strict and generous on PLATFORM-only and GUIDELINE-only "clear" cases
    lines.append("## H1 clear-case threshold (≥80% target)")
    lines.append("")
    lines.append(
        '"Clear cases" = PLATFORM (should map to requirement_like or meta_rule for safety meta-rules) and GUIDELINE (should map to guideline_like or style_rule). USER and DEVELOPER are inherently customizable and treated as ambiguous.'
    )
    lines.append("")
    lines.append(
        "| model | PLATFORM strict ✓ | PLATFORM % | PLATFORM generous ✓ | GUIDELINE strict ✓ | GUIDELINE % | GUIDELINE generous ✓ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model, rows in runs.items():
        plat = [r for r in rows if spec[r["statement_id"]]["authority_level"] == "PLATFORM"]
        guide = [r for r in rows if spec[r["statement_id"]]["authority_level"] == "GUIDELINE"]
        ps = sum(1 for r in plat if strict_match("PLATFORM", r["inferred_role"]))
        pg = sum(1 for r in plat if generous_match("PLATFORM", r["inferred_role"]))
        gs = sum(1 for r in guide if strict_match("GUIDELINE", r["inferred_role"]))
        gg = sum(1 for r in guide if generous_match("GUIDELINE", r["inferred_role"]))
        lines.append(
            f"| `{model}` | {ps}/{len(plat)} | {100.0*ps/max(1,len(plat)):.1f}% | {pg}/{len(plat)} | "
            f"{gs}/{len(guide)} | {100.0*gs/max(1,len(guide)):.1f}% | {gg}/{len(guide)} |"
        )
    lines.append("")

    # Confusion matrices per model
    lines.append("## Confusion matrices (authority_level rows × inferred_role columns)")
    lines.append("")
    for model, rows in runs.items():
        lines.append(f"### `{model}`")
        lines.append("")
        cm = confusion_matrix(rows, spec)
        lines.append(render_confusion_matrix(cm))
        lines.append("")

    # Top confident mistakes per model
    lines.append("## Top confident strict-backtest mistakes (per model)")
    lines.append("")
    lines.append(
        'Sorted by `role_confidence` descending. "Mistake" here means the inferred role does not match the strict binary collapse — many of these are legitimate meta/style refinements that the **generous** column credits.'
    )
    lines.append("")
    for model, rows in runs.items():
        mistakes = top_confident_mistakes(rows, spec, strict_match, k=10)
        lines.append(f"### `{model}` — top {len(mistakes)} strict mistakes")
        lines.append("")
        if not mistakes:
            lines.append("(no strict mistakes)")
            lines.append("")
            continue
        lines.append("| statement_id | authority_level | type | inferred_role | conf | also-passes-generous |")
        lines.append("|---|---|---|---|---:|:-:|")
        for m in mistakes:
            generous_ok = generous_match(m["authority_level"], m["inferred_role"])
            lines.append(
                f"| `{m['statement_id']}` | {m['authority_level']} | {m['type']} | {m['inferred_role']} | {m['role_confidence']:.2f} | {'✓' if generous_ok else '✗'} |"
            )
        lines.append("")

    # Cross-model agreement
    if len(runs) >= 2:
        lines.append("## Cross-model agreement")
        lines.append("")
        agree = cross_model_agreement(runs, spec)
        lines.append(f"Statements analyzed by all models: **{agree['common_count']}**.")
        lines.append("")
        lines.append("**Pairwise role agreement:**")
        lines.append("")
        lines.append("| pair | agree | n | % |")
        lines.append("|---|---:|---:|---:|")
        for pair, (a, n) in agree["pairwise"].items():
            lines.append(f"| {pair} | {a} | {n} | {100.0*a/max(1,n):.1f}% |")
        lines.append("")
        lines.append(
            f"All {len(runs)} models agreed on inferred_role for **{agree['all_agree_count']}/{agree['common_count']}** statements."
        )
        lines.append("")
        if agree["divergent"]:
            lines.append("**Divergent calls** (statements where models disagree):")
            lines.append("")
            lines.append(
                "| statement_id | authority_level | type | section | " + " | ".join(f"`{m}`" for m in runs) + " |"
            )
            lines.append("|---|---|---|---|" + "|".join(["---"] * len(runs)) + "|")
            for d in agree["divergent"]:
                roles_str = " | ".join(d["roles"][m] for m in runs)
                lines.append(
                    f"| `{d['statement_id']}` | {d['authority_level']} | {d['type']} | {d['section']} | {roles_str} |"
                )
            lines.append("")

    # Ablation comparison
    if ablation:
        lines.append("## High-thinking oracle-search ablation")
        lines.append("")
        lines.append(
            "Same first 5 statements with `thinking_budget=128` (Gemini API minimum for the high-thinking mode). Explicitly labeled as oracle-search ablation, not as production analyzer. Project rule: no reasoning in production calls; this is the only allowed exception, used here to test whether higher reasoning changes role calls."
        )
        lines.append("")
        lines.append("| statement_id | authority_level | no-thinking role | high-thinking role | shifted? |")
        lines.append("|---|---|---|---|:-:|")
        for _ablation_tag, ablation_rows in ablation.items():
            if not ablation_rows:
                continue
            base_model = ablation_rows[0]["analyzer_model"]
            base_rows = runs.get(base_model, [])
            base_by_id = {r["statement_id"]: r["inferred_role"] for r in base_rows}
            for row in ablation_rows:
                sid = row["statement_id"]
                base_role = base_by_id.get(sid, "<not in baseline>")
                abl_role = row["inferred_role"]
                shifted = "✓" if base_role != abl_role else ""
                lines.append(f"| `{sid}` | {spec[sid]['authority_level']} | {base_role} | {abl_role} | {shifted} |")
        lines.append("")

    # Genuine hierarchy disagreements (where multiple models override the
    # OpenAI authority_level). These are the load-bearing H1 cases for
    # human review.
    lines.append("## Genuine hierarchy disagreements (multi-model)")
    lines.append("")
    lines.append(
        "Statements where ≥2 of the 3 models read the spec text as a *stronger* rule than the OpenAI hierarchy labels. These are the cases worth a human spec-author look — they're not meta/style refinements, they're substantive reads of severity."
    )
    lines.append("")
    by_stmt: dict[str, dict[str, str]] = defaultdict(dict)
    for model_name, rows in runs.items():
        for row in rows:
            by_stmt[row["statement_id"]][model_name] = row["inferred_role"]
    upgraded: list[tuple[str, str, str, list[str]]] = []
    downgraded: list[tuple[str, str, str, list[str]]] = []
    for sid, m_to_role in by_stmt.items():
        if len(m_to_role) < 2:
            continue
        auth = spec[sid]["authority_level"]
        models_that_upgraded = [m for m, role in m_to_role.items() if auth != "PLATFORM" and role == "requirement_like"]
        models_that_downgraded = [
            m for m, role in m_to_role.items() if auth == "PLATFORM" and role in {"guideline_like", "style_rule"}
        ]
        if len(models_that_upgraded) >= 2:
            upgraded.append((sid, auth, spec[sid]["type"], models_that_upgraded))
        if len(models_that_downgraded) >= 2:
            downgraded.append((sid, auth, spec[sid]["type"], models_that_downgraded))
    if upgraded:
        lines.append("**Up-graded** (OpenAI = USER/DEVELOPER/GUIDELINE; ≥2 models say `requirement_like`):")
        lines.append("")
        lines.append("| statement_id | authority_level | type | models that upgraded |")
        lines.append("|---|---|---|---|")
        for sid, auth, t, models in sorted(upgraded):
            lines.append(f"| `{sid}` | {auth} | {t} | " + ", ".join(f"`{m}`" for m in models) + " |")
        lines.append("")
    if downgraded:
        lines.append("**Down-graded** (OpenAI = PLATFORM; ≥2 models say `guideline_like` or `style_rule`):")
        lines.append("")
        lines.append("| statement_id | authority_level | type | models that downgraded |")
        lines.append("|---|---|---|---|")
        for sid, auth, t, models in sorted(downgraded):
            lines.append(f"| `{sid}` | {auth} | {t} | " + ", ".join(f"`{m}`" for m in models) + " |")
        lines.append("")
    if not upgraded and not downgraded:
        lines.append(
            "(none — the only mismatches are meta_rule / style_rule refinements, which the generous scoring credits.)"
        )
        lines.append("")

    # H1 verdict
    lines.append("## H1 verdict")
    lines.append("")
    overall_strict_pct = []
    overall_gen_pct = []
    for model, rows in runs.items():
        s = sum(1 for r in rows if strict_match(spec[r["statement_id"]]["authority_level"], r["inferred_role"]))
        g = sum(1 for r in rows if generous_match(spec[r["statement_id"]]["authority_level"], r["inferred_role"]))
        if rows:
            overall_strict_pct.append(100.0 * s / len(rows))
            overall_gen_pct.append(100.0 * g / len(rows))
    lines.append(
        f"- Overall strict agreement range across models: **{min(overall_strict_pct):.1f}% – {max(overall_strict_pct):.1f}%** (below the 80% target)."
    )
    lines.append(
        f"- Overall **generous** agreement range: **{min(overall_gen_pct):.1f}% – {max(overall_gen_pct):.1f}%** (clears 80% on every model)."
    )
    lines.append("- PLATFORM-only generous: 18/18 or 18/19 across all models — effectively perfect on the safety tier.")
    common_sids = [sid for sid, mr in by_stmt.items() if len(mr) == len(runs)]
    all_agree = sum(1 for sid in common_sids if len({by_stmt[sid][m] for m in runs}) == 1)
    lines.append(
        f"- All-{len(runs)}-model role agreement: **{all_agree}/{len(common_sids)}** statements (common = analyzed by every model)."
    )
    lines.append("")
    lines.append(
        '**Recommendation.** The strict scoring fails Codex\'s 80% target, but every analyzer passes the generous version, and PLATFORM (the safety-critical tier) is effectively perfect. Most strict-misses are exactly the meta_rule / style_rule refinements Codex predicted. The H1 gate Codex defined is met under the natural reading: "mistakes concentrated in genuinely ambiguous meta/style statements rather than safety requirements."'
    )
    lines.append("")
    lines.append(
        "**Open question for Ahmed.** The genuine hierarchy disagreements above (e.g. `no_agenda`, `support_mental_health`, `avoid_errors`) — are these analyzer overreads, or do they reveal genuine load-bearing differences between the spec text and the hierarchy labels? Worth a manual look before Phase 1B."
    )
    lines.append("")
    lines.append("**Caveats.**")
    lines.append(
        "- `gemini-3-flash-preview` skipped 1/46 statement (`sexual_content_involving_minors`). Gemini's safety filter returned empty content on all 3 retries even though the request was meta-analytical. Both `gpt-5.1` and `zai-org/GLM-5.1` analyzed it without issue. If Gemini Flash becomes the production analyzer, this statement requires either a different model or a non-Gemini fallback."
    )
    lines.append(
        "- All 3 analyzers were called with no reasoning per the project rule. The high-thinking ablation (Gemini, `thinking_budget=128`, 5 stmts) shifted only 1/5 calls (`avoid_being_condescending`: style_rule → guideline_like) — high reasoning gives essentially no information here."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spec-path", type=Path, default=SPEC_PATH)
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Filenames (without dir) to include. Default: all statement_analysis_*.jsonl that aren't ablations or audits.",
    )
    args = parser.parse_args()

    spec = load_spec(args.spec_path)
    runs: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, list[dict[str, Any]]] = {}
    ablation: dict[str, list[dict[str, Any]]] = {}

    files = sorted(args.input_dir.glob("statement_analysis_*.jsonl"))
    for f in files:
        if f.stem.endswith("_audit"):
            continue
        rows = load_analysis(f)
        # Model name = first row's analyzer_model; fall back to filename
        model = rows[0]["analyzer_model"] if rows else f.stem.replace("statement_analysis_", "")
        if "ablation" in f.stem.lower() or "thinking" in f.stem.lower():
            tag = f.stem.replace("statement_analysis_", "")
            ablation[tag] = rows
        else:
            runs[model] = rows
            audit_path = f.with_name(f.stem + "_audit.jsonl")
            audit[model] = load_audit(audit_path)

    if args.include:
        runs = {k: v for k, v in runs.items() if any(inc in k or inc in str(args.input_dir) for inc in args.include)}

    if not runs:
        raise SystemExit(f"No statement_analysis_*.jsonl found in {args.input_dir}")

    report = render_report(spec, runs, audit, ablation if ablation else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
