# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E9 repair compiler: propose statement-text repair candidates.

This script implements Stream A/E1 candidate generation and Stream C/E3's
rich-vs-minimal input ablation. For each target statement it produces:

- 4 rich-input candidates
- 4 minimal-input candidates

Each candidate is saved as:

    repair_v0/round_<N>/<statement_id>/<candidate_id>/candidate.json
    repair_v0/round_<N>/<statement_id>/<candidate_id>/statement.jsonl

The statement JSONL is a single edited statement object; verification scripts
build phantom full specs from it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import GPT, parse_json
from e9_repair_common import (
    E1_TARGETS,
    PHASE4_PATHS,
    REPAIR_DIR,
    VAR_A_PATHS,
    baseline_profile,
    case_key,
    case_key_string,
    load_all_score_maps,
    load_spec_by_id,
    response_cases,
    select_compiler_cases,
    statement_candidate_dir,
    statement_examples_text,
    summarize_judgment,
    write_candidate_statement,
)

COMPILER_SYSTEM = """You are a model-spec repair compiler. Your job is to propose small wording repairs for ONE statement in a model specification.

The statement has been flagged because LM judges disagree on how to score the same responses. You will propose candidate rewrites that make the statement easier to apply consistently while preserving the original spec author's apparent intent.

Hard constraints:
- Preserve the statement's meaning. Do not make a normative policy change.
- Do not add new prohibitions, permissions, exceptions, domains, or priority rules that were not already implied by the statement.
- Prefer clarifying vague phrases, preserving qualifiers, and making implicit thresholds operational.
- Keep the statement general enough to remain a spec statement, not a case list.
- Do not edit examples in this experiment; rewrite statement text only.
- Use GPT-5.1 with no reasoning; therefore make the output explicit and schema-compliant.

Output JSON only:
{
  "candidates": [
    {
      "label": "<short snake_case label>",
      "rewritten_text": "<full replacement statement text>",
      "phrases_changed": ["<exact old phrase>", "..."],
      "rationale": "<2-4 sentences explaining what ambiguity this resolves and why meaning is preserved>",
      "predicted_delta_kappa": <number between -1 and 1>,
      "risk_notes": "<what could regress or why this might fail>",
      "meaning_preservation_notes": "<why this is a clarification rather than a policy change>"
    }
  ]
}

Return exactly the requested number of candidates. No markdown.
"""

RICH_USER_TEMPLATE = """STATEMENT_ID: {statement_id}
ROUND: {round_id}
CASE_SPLIT_STRATEGY: {case_strategy}

ORIGINAL STATEMENT:
{statement_text}

ORIGINAL EXAMPLES:
{examples}

BASELINE KAPPA PROFILE ON ALL 60 RESPONSE CASES:
{baseline_profile_json}

COMPILER INPUT CASES:
The following are the 10 compiler-input response cases for this statement, selected according to CASE_SPLIT_STRATEGY. The remaining 50 cases are held out and will be used to reject overfit edits.

{case_bundle_json}

PREVIOUS ROUND FAILURE SUMMARY:
{previous_failure_summary}

ROUND-SPECIFIC GUIDANCE:
{round_guidance}

Use the per-judge rationales and quotes to infer exactly which wording is being read differently. Pay special attention to:
- same spec phrase, different scores -> ambiguous phrase
- different spec phrases -> relevance/scope ambiguity
- rubric_spec_tension=true -> judge noticed tension between source text and rubric

Produce {n_candidates} meaning-preserving rewrite candidates.
"""

MINIMAL_USER_TEMPLATE = """STATEMENT_ID: {statement_id}
ROUND: {round_id}
CASE_SPLIT_STRATEGY: {case_strategy}

ORIGINAL STATEMENT:
{statement_text}

ORIGINAL EXAMPLES:
{examples}

BASELINE KAPPA PROFILE ON ALL 60 RESPONSE CASES:
{baseline_profile_json}

COMPILER INPUT CASE IDS:
{case_ids_json}

This is the minimal-input ablation. You do NOT get judge rationales or quote fields. Propose repairs based only on the statement text, examples, kappa profile, and the fact that the listed cases were selected according to CASE_SPLIT_STRATEGY.

PREVIOUS ROUND FAILURE SUMMARY:
{previous_failure_summary}

ROUND-SPECIFIC GUIDANCE:
{round_guidance}

Produce {n_candidates} meaning-preserving rewrite candidates.
"""

ROUND_1_GUIDANCE = """This is the first repair round. Generate diverse, conservative clarifications that target phrase-level ambiguity, not case-specific outcomes."""

ANTI_OVERFIT_GUIDANCE = """Previous candidates failed mainly because compiler-input cases improved more than held-out cases. The gate rejects edits when the absolute gap between shown-case delta-kappa and held-out delta-kappa exceeds 0.10.

For this round:
- Do not optimize the listed cases directly.
- Extract the general ambiguity that explains the listed cases, then rewrite only that phrase.
- Prefer narrower wording changes that should apply uniformly across unseen scenarios.
- Avoid adding examples, domains, new exceptions, or operational tests that only fit the shown responses.
- Set predicted_delta_kappa for held-out generalization, not for the listed cases."""


def default_case_strategy(round_id: int) -> str:
    if round_id == 1:
        return "top_disagreement"
    if round_id == 2:
        return "stratified"
    if round_id == 3:
        return "rotated_top"
    return "stratified"


def call_compiler(log: RawAPILogger, client: OpenAI, key: dict[str, Any], user: str) -> dict[str, Any]:
    raw = log.call(
        role="repair_compiler",
        key=key,
        fn=lambda: client.chat.completions.create(
            model=GPT,
            messages=[{"role": "system", "content": COMPILER_SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            max_completion_tokens=6000,
            reasoning_effort="none",
            response_format={"type": "json_object"},
        ),
    )
    return parse_json(raw.choices[0].message.content or "")


def build_case_bundle(statement_id: str, compiler_keys: list[tuple[str, int, str]]) -> list[dict[str, Any]]:
    var_a = load_all_score_maps(VAR_A_PATHS)
    phase4 = load_all_score_maps(PHASE4_PATHS)
    cases_by_key = {case_key(row): row for row in response_cases(statement_id)}
    bundle = []
    for key in compiler_keys:
        row = cases_by_key[key]
        judgments: dict[str, Any] = {"var_A": {}, "phase_4": {}}
        for judge, mapping in var_a.items():
            if key in mapping:
                judgments["var_A"][judge] = summarize_judgment(mapping[key])
        for judge, mapping in phase4.items():
            if key in mapping:
                judgments["phase_4"][judge] = summarize_judgment(mapping[key])
        bundle.append(
            {
                "case_id": case_key_string(key),
                "scenario_idx": key[1],
                "generator": key[2],
                "user_query": row["user_query"],
                "response": row["response"],
                "judgments": judgments,
            }
        )
    return bundle


def normalize_candidates(data: dict[str, Any], n_candidates: int) -> list[dict[str, Any]]:
    raw = data.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("compiler output missing candidates list")
    candidates = []
    for idx, cand in enumerate(raw[:n_candidates]):
        if not isinstance(cand, dict):
            continue
        rewritten = cand.get("rewritten_text")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        candidates.append(
            {
                "label": str(cand.get("label") or f"candidate_{idx:02d}"),
                "rewritten_text": rewritten.strip(),
                "phrases_changed": cand.get("phrases_changed") if isinstance(cand.get("phrases_changed"), list) else [],
                "rationale": str(cand.get("rationale") or ""),
                "predicted_delta_kappa": cand.get("predicted_delta_kappa"),
                "risk_notes": str(cand.get("risk_notes") or ""),
                "meaning_preservation_notes": str(cand.get("meaning_preservation_notes") or ""),
            }
        )
    if len(candidates) != n_candidates:
        raise ValueError(f"compiler returned {len(candidates)} usable candidates, expected {n_candidates}")
    return candidates


def previous_failure_summary(round_id: int, statement_id: str, limit: int = 5) -> str:
    if round_id <= 1:
        return "None; this is the first round."
    prev_root = REPAIR_DIR / f"round_{round_id - 1}" / statement_id
    if not prev_root.exists():
        return "No previous-round verdicts found."
    rows = []
    for verdict_path in sorted(prev_root.glob("*/verdict.json")):
        verdict = json.loads(verdict_path.read_text())
        candidate = verdict.get("candidate") or {}
        var_a = verdict.get("var_A") or {}
        cross = verdict.get("cross") or {}
        gate = verdict.get("gate") or {}
        rows.append(
            {
                "candidate_id": verdict.get("candidate_id"),
                "mode": candidate.get("mode"),
                "rewritten_text": candidate.get("rewritten_text"),
                "gate_reasons": gate.get("reasons") or [],
                "delta_kappa_held_out_var_A": var_a.get("delta_kappa_held_out_var_A"),
                "delta_kappa_compiler_input_var_A": var_a.get("delta_kappa_compiler_input_var_A"),
                "delta_kappa_phase_4": cross.get("delta_kappa_phase_4"),
                "delta_kappa_full_spec": cross.get("delta_kappa_full_spec"),
            }
        )
    if not rows:
        return "No previous-round verdicts found."
    rows.sort(key=lambda row: (row["delta_kappa_held_out_var_A"] is None, -(row["delta_kappa_held_out_var_A"] or -999)))
    return json.dumps(rows[:limit], ensure_ascii=False, indent=2)


def write_candidates(
    *,
    round_id: int,
    statement: dict[str, Any],
    mode: str,
    candidates: list[dict[str, Any]],
    compiler_keys: list[tuple[str, int, str]],
    held_out_keys: list[tuple[str, int, str]],
    case_strategy: str,
    force: bool,
) -> None:
    for idx, cand in enumerate(candidates):
        candidate_id = f"{mode}_{idx:02d}"
        cand_dir = statement_candidate_dir(round_id, statement["id"], candidate_id)
        cand_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cand_dir / "candidate.json"
        stmt_path = cand_dir / "statement.jsonl"
        if meta_path.exists() and stmt_path.exists() and not force:
            print(f"    skip existing {statement['id']} {candidate_id}")
            continue
        metadata = {
            "round": round_id,
            "statement_id": statement["id"],
            "candidate_id": candidate_id,
            "mode": mode,
            "case_strategy": case_strategy,
            "compiler_input_case_ids": [case_key_string(k) for k in compiler_keys],
            "held_out_case_ids": [case_key_string(k) for k in held_out_keys],
            **cand,
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        write_candidate_statement(
            statement,
            cand["rewritten_text"],
            stmt_path,
            {
                "round": round_id,
                "candidate_id": candidate_id,
                "mode": mode,
                "label": cand["label"],
                "predicted_delta_kappa": cand.get("predicted_delta_kappa"),
            },
        )
        print(f"    wrote {meta_path}")


def compile_for_statement(
    *,
    round_id: int,
    statement_id: str,
    n_candidates_per_mode: int,
    modes: list[str],
    case_strategy: str,
    force: bool,
    log: RawAPILogger,
    client: OpenAI,
) -> None:
    spec = load_spec_by_id()
    statement = spec[statement_id]
    compiler_keys, held_out_keys = select_compiler_cases(statement_id, limit=10, strategy=case_strategy)
    all_profile = baseline_profile(statement_id)
    compiler_profile = baseline_profile(statement_id, keys=compiler_keys)
    held_out_profile = baseline_profile(statement_id, keys=held_out_keys)
    split_path = REPAIR_DIR / f"round_{round_id}" / statement_id / "case_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "statement_id": statement_id,
                "case_strategy": case_strategy,
                "compiler_input_case_ids": [case_key_string(k) for k in compiler_keys],
                "held_out_case_ids": [case_key_string(k) for k in held_out_keys],
                "baseline_all": all_profile,
                "baseline_compiler_input": compiler_profile,
                "baseline_held_out": held_out_profile,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"  {statement_id}: strategy={case_strategy} compiler cases={len(compiler_keys)}, held_out={len(held_out_keys)}"
    )
    examples = statement_examples_text(statement)
    case_bundle = build_case_bundle(statement_id, compiler_keys)
    failure_summary = previous_failure_summary(round_id, statement_id)
    round_guidance = ROUND_1_GUIDANCE if round_id == 1 else ANTI_OVERFIT_GUIDANCE
    for mode in modes:
        key = {"round": round_id, "statement_id": statement_id, "mode": mode}
        if mode == "rich":
            user = RICH_USER_TEMPLATE.format(
                statement_id=statement_id,
                round_id=round_id,
                case_strategy=case_strategy,
                statement_text=statement["text"],
                examples=examples,
                baseline_profile_json=json.dumps(
                    {
                        "all": all_profile,
                        "compiler_input": compiler_profile,
                        "held_out": held_out_profile,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                case_bundle_json=json.dumps(case_bundle, ensure_ascii=False, indent=2),
                previous_failure_summary=failure_summary,
                round_guidance=round_guidance,
                n_candidates=n_candidates_per_mode,
            )
        elif mode == "minimal":
            user = MINIMAL_USER_TEMPLATE.format(
                statement_id=statement_id,
                round_id=round_id,
                case_strategy=case_strategy,
                statement_text=statement["text"],
                examples=examples,
                baseline_profile_json=json.dumps(
                    {
                        "all": all_profile,
                        "compiler_input": compiler_profile,
                        "held_out": held_out_profile,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                case_ids_json=json.dumps([case_key_string(k) for k in compiler_keys], indent=2),
                previous_failure_summary=failure_summary,
                round_guidance=round_guidance,
                n_candidates=n_candidates_per_mode,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
        data = call_compiler(log, client, key, user)
        candidates = normalize_candidates(data, n_candidates_per_mode)
        write_candidates(
            round_id=round_id,
            statement=statement,
            mode=mode,
            candidates=candidates,
            compiler_keys=compiler_keys,
            held_out_keys=held_out_keys,
            case_strategy=case_strategy,
            force=force,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--statements", nargs="*", default=E1_TARGETS)
    parser.add_argument("--modes", nargs="*", choices=["rich", "minimal"], default=["rich", "minimal"])
    parser.add_argument("--candidates-per-mode", type=int, default=4)
    parser.add_argument("--case-strategy", choices=["top_disagreement", "stratified", "rotated_top"], default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    log = RawAPILogger(f"e9_compile_edit_round_{args.round}")
    case_strategy = args.case_strategy or default_case_strategy(args.round)
    print(f"raw run dir: {log.run_dir}")
    print(f"round={args.round} statements={args.statements} modes={args.modes} case_strategy={case_strategy}")
    for statement_id in args.statements:
        compile_for_statement(
            round_id=args.round,
            statement_id=statement_id,
            n_candidates_per_mode=args.candidates_per_mode,
            modes=args.modes,
            case_strategy=case_strategy,
            force=args.force,
            log=log,
            client=client,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
