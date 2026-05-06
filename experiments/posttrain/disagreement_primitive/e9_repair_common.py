# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for E9 spec-repair loop experiments.

This module is intentionally small and data-oriented. It does not make LM API
calls; scripts that call models must use RawAPILogger directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from e8_paired_indirection import DIR, SPEC_PATH, load_jsonl, render_examples, spearman, write_jsonl
from e8_phase3_analysis import fleiss_kappa

REPAIR_DIR = DIR / "repair_v0"

GENERATOR_COLUMNS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]

VAR_A_PATHS = {
    "gpt": DIR / "e8_va_judgments.jsonl",
    "gemini": DIR / "phase2_gemini/va_judgments.jsonl",
    "glm": DIR / "phase2_glm/va_judgments.jsonl",
}

VAR_B_PATHS = {
    "gpt": DIR / "e8_vb_judgments.jsonl",
    "gemini": DIR / "phase2_gemini/vb_judgments.jsonl",
    "glm": DIR / "phase2_glm/vb_judgments.jsonl",
}

PHASE3_PATHS = {
    "gpt": DIR / "phase3_gpt/judgments.jsonl",
    "gemini": DIR / "phase3_gemini/judgments.jsonl",
    "glm": DIR / "phase3_glm/judgments.jsonl",
}

PHASE4_PATHS = {
    "gpt": DIR / "phase4_gpt/judgments.jsonl",
    "gemini": DIR / "phase4_gemini/judgments.jsonl",
    "glm": DIR / "phase4_glm/judgments.jsonl",
}

TIER_A = ["avoid_abuse", "assume_objective_pov", "comply_with_laws", "refusal_style"]
TIER_B = ["formatting"]
TIER_C = ["do_not_encourage_self_harm", "be_clear"]
E1_TARGETS = TIER_A + TIER_B + TIER_C

QUALIFIER_DROP_STATEMENTS = [
    "ask_clarifying_questions",
    "assume_objective_pov",
    "avoid_abuse",
    "avoid_extremist_content",
    "avoid_regulated_advice",
    "avoid_targeted_political_manipulation",
    "be_clear",
    "do_not_facilitate_illicit_behavior",
    "do_not_lie",
    "follow_all_applicable_instructions",
    "highlight_misalignments",
    "ignore_untrusted_data",
    "letter_and_spirit",
    "protect_privileged_messages",
    "refusal_style",
    "transformation_exception",
]


def load_spec(path: Path = SPEC_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def load_spec_by_id(path: Path = SPEC_PATH) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_spec(path)}


def statement_examples_text(statement: dict[str, Any]) -> str:
    examples = (statement.get("metadata") or {}).get("examples") or []
    return render_examples(examples if isinstance(examples, list) else [])


def statement_candidate_dir(round_id: int, statement_id: str, candidate_id: str) -> Path:
    return REPAIR_DIR / f"round_{round_id}" / statement_id / candidate_id


def score_to_3way(score: int | None) -> str | None:
    if score not in {1, 2, 3, 4, 5}:
        return None
    if score <= 2:
        return "non-compliant"
    if score == 3:
        return "ambiguous"
    return "compliant"


def decision_to_3way(decision: str | None) -> str | None:
    if decision in {"compliant", "non-compliant", "ambiguous"}:
        return decision
    return None


def is_problematic(decision: str) -> bool:
    return decision in {"non-compliant", "ambiguous"}


def load_score_map(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    out = {}
    for row in load_jsonl(path):
        if "error" in row:
            continue
        if row.get("score") not in {1, 2, 3, 4, 5}:
            continue
        out[(row["statement_id"], int(row["scenario_idx"]), row["generator"])] = row
    return out


def load_decision_map(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    out = {}
    for row in load_jsonl(path):
        if "error" in row:
            continue
        if row.get("decision") not in {"compliant", "non-compliant", "ambiguous"}:
            continue
        out[(row["statement_id"], int(row["scenario_idx"]), row["generator"])] = row
    return out


def load_all_score_maps(paths: dict[str, Path]) -> dict[str, dict[tuple[str, int, str], dict[str, Any]]]:
    return {judge: load_score_map(path) for judge, path in paths.items()}


def load_all_decision_maps(paths: dict[str, Path]) -> dict[str, dict[tuple[str, int, str], dict[str, Any]]]:
    return {judge: load_decision_map(path) for judge, path in paths.items()}


def response_cases(statement_id: str) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(DIR / "e8_responses.jsonl"):
        if row["statement_id"] != statement_id:
            continue
        for generator, col in GENERATOR_COLUMNS:
            response = row.get(col)
            if not response:
                continue
            rows.append(
                {
                    "statement_id": statement_id,
                    "scenario_idx": int(row["scenario_idx"]),
                    "generator": generator,
                    "user_query": row["user_query"],
                    "response": response,
                }
            )
    rows.sort(key=lambda r: (r["scenario_idx"], r["generator"]))
    return rows


def case_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (row["statement_id"], int(row["scenario_idx"]), row["generator"])


def case_key_string(key: tuple[str, int, str]) -> str:
    return f"{key[0]}::{key[1]}::{key[2]}"


def parse_case_key(text: str) -> tuple[str, int, str]:
    sid, idx, gen = text.split("::", 2)
    return sid, int(idx), gen


def ranked_response_case_keys(statement_id: str) -> list[tuple[str, int, str]]:
    """Rank response cases by baseline disagreement signal."""
    var_a = load_all_score_maps(VAR_A_PATHS)
    phase4 = load_all_score_maps(PHASE4_PATHS)
    all_cases = [case_key(row) for row in response_cases(statement_id)]
    ranked = []
    for key in all_cases:
        scores = []
        tension_count = 0
        for maps in (var_a, phase4):
            for judge_map in maps.values():
                row = judge_map.get(key)
                if not row:
                    continue
                scores.append(row["score"])
                if row.get("rubric_spec_tension") is True:
                    tension_count += 1
        spread = max(scores) - min(scores) if len(scores) >= 2 else 0
        ranked.append((-(spread + 0.25 * tension_count), key))
    ranked.sort(key=lambda item: (item[0], item[1][1], item[1][2]))
    return [key for _, key in ranked]


def evenly_spaced_cases(keys: list[tuple[str, int, str]], limit: int) -> list[tuple[str, int, str]]:
    if limit >= len(keys):
        return list(keys)
    if limit <= 0:
        return []
    if limit == 1:
        return [keys[0]]
    step = (len(keys) - 1) / (limit - 1)
    selected = []
    used = set()
    for idx in range(limit):
        candidate = keys[round(idx * step)]
        if candidate in used:
            continue
        selected.append(candidate)
        used.add(candidate)
    for candidate in keys:
        if len(selected) == limit:
            break
        if candidate in used:
            continue
        selected.append(candidate)
        used.add(candidate)
    return selected


def select_compiler_cases(
    statement_id: str, limit: int = 10, strategy: str = "top_disagreement"
) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """Select compiler-input response cases and the corresponding held-out set.

    There are 60 response cases per statement: 20 scenario indices x 3
    generators. `top_disagreement` uses the original handoff plan: show the 10
    highest-disagreement cases. `stratified` keeps a few high-disagreement cases
    but adds medium/low-disagreement anchors to reduce compiler overfit.
    `rotated_top` uses the next highest-disagreement decile for failure probing.
    """
    ranked = ranked_response_case_keys(statement_id)
    if strategy == "top_disagreement":
        compiler = ranked[:limit]
    elif strategy == "stratified":
        high_count = min(4, limit)
        high = ranked[:high_count]
        anchors = evenly_spaced_cases(ranked[high_count:], limit - high_count)
        compiler = high + anchors
    elif strategy == "rotated_top":
        compiler = ranked[limit : 2 * limit]
        if len(compiler) < limit:
            compiler.extend(key for key in ranked if key not in compiler)
            compiler = compiler[:limit]
    else:
        raise ValueError(f"unknown compiler case strategy: {strategy}")
    compiler_set = set(compiler)
    held_out = [key for key in ranked if key not in compiler_set]
    return compiler, held_out


def fleiss_kappa_for_score_maps(
    maps_by_judge: dict[str, dict[tuple[str, int, str], dict[str, Any]]],
    keys: list[tuple[str, int, str]],
    judges: list[str],
) -> tuple[float | None, int]:
    rows = []
    for key in keys:
        flags = []
        for judge in judges:
            row = maps_by_judge.get(judge, {}).get(key)
            if not row:
                break
            d = score_to_3way(row.get("score"))
            if d is None:
                break
            flags.append(is_problematic(d))
        if len(flags) != len(judges):
            continue
        n_pos = sum(flags)
        rows.append([len(judges) - n_pos, n_pos])
    return fleiss_kappa(rows), len(rows)


def fleiss_kappa_for_decision_maps(
    maps_by_judge: dict[str, dict[tuple[str, int, str], dict[str, Any]]],
    keys: list[tuple[str, int, str]],
    judges: list[str],
) -> tuple[float | None, int]:
    rows = []
    for key in keys:
        flags = []
        for judge in judges:
            row = maps_by_judge.get(judge, {}).get(key)
            if not row:
                break
            d = decision_to_3way(row.get("decision"))
            if d is None:
                break
            flags.append(is_problematic(d))
        if len(flags) != len(judges):
            continue
        n_pos = sum(flags)
        rows.append([len(judges) - n_pos, n_pos])
    return fleiss_kappa(rows), len(rows)


def spearman_between_score_maps(
    left_by_judge: dict[str, dict[tuple[str, int, str], dict[str, Any]]],
    right_by_judge: dict[str, dict[tuple[str, int, str], dict[str, Any]]],
    keys: list[tuple[str, int, str]],
    judges: list[str],
) -> dict[str, float | None]:
    out = {}
    for judge in judges:
        pairs = []
        left = left_by_judge.get(judge, {})
        right = right_by_judge.get(judge, {})
        for key in keys:
            lrow = left.get(key)
            rrow = right.get(key)
            if not lrow or not rrow:
                continue
            pairs.append((lrow["score"], rrow["score"]))
        out[judge] = spearman(pairs)
    return out


def baseline_profile(
    statement_id: str, keys: list[tuple[str, int, str]] | None = None, judges: list[str] | None = None
) -> dict[str, Any]:
    judges = judges or ["gpt", "gemini", "glm"]
    keys = keys or [case_key(row) for row in response_cases(statement_id)]
    var_a = load_all_score_maps(VAR_A_PATHS)
    var_b = load_all_score_maps(VAR_B_PATHS)
    phase4 = load_all_score_maps(PHASE4_PATHS)
    full_spec = load_all_decision_maps(PHASE3_PATHS)
    k_single, n_single = fleiss_kappa_for_score_maps(var_a, keys, judges)
    k_rubric, n_rubric = fleiss_kappa_for_score_maps(var_b, keys, judges)
    k_phase4, n_phase4 = fleiss_kappa_for_score_maps(phase4, keys, judges)
    k_full, n_full = fleiss_kappa_for_decision_maps(full_spec, keys, judges)
    return {
        "statement_id": statement_id,
        "judges": judges,
        "n_keys": len(keys),
        "kappa_var_A": k_single,
        "n_var_A": n_single,
        "kappa_rubric": k_rubric,
        "n_rubric": n_rubric,
        "kappa_phase_4": k_phase4,
        "n_phase_4": n_phase4,
        "kappa_full_spec": k_full,
        "n_full_spec": n_full,
        "spearman_var_A_phase_4": spearman_between_score_maps(var_a, phase4, keys, judges),
    }


def write_candidate_statement(
    statement: dict[str, Any], edited_text: str, out_path: Path, candidate_meta: dict[str, Any]
) -> None:
    edited = json.loads(json.dumps(statement))
    edited["text"] = edited_text
    metadata = edited.setdefault("metadata", {})
    metadata["_repair_candidate"] = candidate_meta
    write_jsonl([edited], out_path)


def build_phantom_spec(candidate_statement: dict[str, Any], out_path: Path, base_spec_path: Path = SPEC_PATH) -> None:
    rows = []
    for row in load_spec(base_spec_path):
        if row["id"] == candidate_statement["id"]:
            rows.append(candidate_statement)
        else:
            rows.append(row)
    write_jsonl(rows, out_path)


def load_candidate_statement(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one statement in {path}, got {len(rows)}")
    return rows[0]


def summarize_judgment(row: dict[str, Any], *, max_reasoning: int = 700, max_response: int = 700) -> dict[str, Any]:
    return {
        "score": row.get("score"),
        "decision": row.get("decision"),
        "reasoning": (row.get("reasoning") or "")[:max_reasoning],
        "spec_quotes": row.get("spec_quotes") or [],
        "rubric_quotes": row.get("rubric_quotes") or [],
        "rubric_spec_tension": row.get("rubric_spec_tension"),
        "tension_description": row.get("tension_description") or "",
        "response": (row.get("response") or "")[:max_response],
        "user_query": (row.get("user_query") or "")[:max_response],
    }


def append_round_summary(path: Path, title: str, body: str) -> None:
    with path.open("a") as fh:
        fh.write(f"\n### {title}\n\n{body.rstrip()}\n")
