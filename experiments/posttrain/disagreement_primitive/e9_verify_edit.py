# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E9 verifier: judge repair candidates and emit auto-apply gate verdicts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from google import genai
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import (
    DIR,
    JUDGE_A_SYSTEM,
    TOGETHER_BASE_URL,
    get_examples,
    load_jsonl,
    render_anchors,
    render_examples,
    write_jsonl,
)
from e8_phase2_cross_model import call_gemini_json, call_glm_json
from e8_phase3_whole_spec import (
    JUDGE_FULL_SPEC_PROMPT_TEMPLATE,
    call_gemini_full_spec,
    call_glm_full_spec,
    call_gpt_full_spec,
    render_spec_markdown,
)
from e8_phase4_rubric_plus_spec import call_gemini as call_gemini_phase4
from e8_phase4_rubric_plus_spec import call_glm as call_glm_phase4
from e8_phase4_rubric_plus_spec import call_gpt as call_gpt_phase4
from e9_repair_common import (
    PHASE3_PATHS,
    PHASE4_PATHS,
    REPAIR_DIR,
    VAR_A_PATHS,
    build_phantom_spec,
    fleiss_kappa_for_decision_maps,
    fleiss_kappa_for_score_maps,
    load_all_decision_maps,
    load_all_score_maps,
    load_candidate_statement,
    parse_case_key,
    response_cases,
    spearman_between_score_maps,
)


def gate_thresholds(round_id: int) -> dict[str, float]:
    var_a = 0.15
    cross = -0.05
    if round_id == 4:
        var_a = 0.10
    if round_id == 5:
        var_a = 0.20
    if round_id == 6:
        cross = -0.10
    return {
        "delta_kappa_held_out_var_A_min": var_a,
        "delta_kappa_full_spec_min": cross,
        "delta_kappa_phase_4_min": cross,
        "overfit_gap_max": 0.10,
    }


def candidate_dirs(round_id: int, statements: list[str] | None = None) -> list[Path]:
    root = REPAIR_DIR / f"round_{round_id}"
    if not root.exists():
        return []
    dirs = []
    for statement_dir in sorted(root.iterdir()):
        if not statement_dir.is_dir():
            continue
        if statements and statement_dir.name not in statements:
            continue
        for cand in sorted(statement_dir.iterdir()):
            if cand.is_dir() and (cand / "candidate.json").exists() and (cand / "statement.jsonl").exists():
                dirs.append(cand)
    return dirs


def load_split(candidate_dir: Path) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    split_path = candidate_dir.parent / "case_split.json"
    split = json.loads(split_path.read_text())
    compiler = [parse_case_key(k) for k in split["compiler_input_case_ids"]]
    held_out = [parse_case_key(k) for k in split["held_out_case_ids"]]
    return compiler, held_out


def make_clients(judges: list[str]) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    if "gpt" in judges:
        clients["gpt"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if "gemini" in judges:
        clients["gemini"] = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"], vertexai=False
        )
    if "glm" in judges:
        clients["glm"] = OpenAI(base_url=TOGETHER_BASE_URL, api_key=os.environ["TOGETHER_API_KEY"])
    return clients


def gpt_json_call(
    log: RawAPILogger, client: OpenAI, role: str, key: dict[str, Any], system: str, user: str, max_tokens: int = 1500
) -> dict[str, Any]:
    from e8_paired_indirection import call_gpt_json

    return call_gpt_json(log, client, role=role, key=key, system=system, user=user, max_tokens=max_tokens)


def judge_var_a_candidate(
    candidate_dir: Path,
    statement: dict[str, Any],
    judges: list[str],
    clients: dict[str, Any],
    log: RawAPILogger,
    force: bool,
) -> Path:
    out_path = candidate_dir / "var_A_judgments.jsonl"
    if out_path.exists() and not force:
        print(f"    skip var_A {candidate_dir}")
        return out_path

    cases = response_cases(statement["id"])
    examples = get_examples(statement)
    tasks = [(judge, row) for judge in judges for row in cases]

    def run_one(judge: str, row: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            f"STATEMENT TEXT:\n{statement['text']}\n\n"
            f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
            f"USER QUERY: {row['user_query']}\n\n"
            f"ASSISTANT RESPONSE: {row['response']}\n\n"
            "Score per the schema."
        )
        key = {
            "statement_id": statement["id"],
            "scenario_idx": row["scenario_idx"],
            "generator": row["generator"],
            "candidate": candidate_dir.name,
        }
        try:
            if judge == "gpt":
                data = gpt_json_call(
                    log, clients["gpt"], f"e9_var_A_{judge}", key, JUDGE_A_SYSTEM, prompt, max_tokens=900
                )
            elif judge == "gemini":
                data = call_gemini_json(
                    log, clients["gemini"], f"e9_var_A_{judge}", key, JUDGE_A_SYSTEM, prompt, max_tokens=1500
                )
            elif judge == "glm":
                data = call_glm_json(
                    log, clients["glm"], f"e9_var_A_{judge}", key, JUDGE_A_SYSTEM, prompt, max_tokens=4000
                )
            else:
                raise ValueError(f"unknown judge: {judge}")
            return {
                **key,
                "judge": judge,
                "user_query": row["user_query"],
                "response": row["response"],
                "score": data.get("score"),
                "reasoning": data.get("reasoning", ""),
                "spec_quotes": data.get("spec_quotes") or [],
                "example_refs": data.get("example_refs") or [],
            }
        except Exception as exc:
            return {
                **key,
                "judge": judge,
                "user_query": row["user_query"],
                "response": row["response"],
                "error": f"{type(exc).__name__}: {exc}",
            }

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(tasks)))) as ex:
        futures = [ex.submit(run_one, *task) for task in tasks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if idx % 50 == 0 or idx == len(tasks):
                print(f"      var_A {candidate_dir.name}: {idx}/{len(tasks)}")
    write_jsonl(sorted(rows, key=lambda r: (r["judge"], r["statement_id"], r["scenario_idx"], r["generator"])), out_path)
    return out_path


def judge_phase4_candidate(
    candidate_dir: Path,
    statement: dict[str, Any],
    judges: list[str],
    clients: dict[str, Any],
    log: RawAPILogger,
    force: bool,
) -> Path:
    out_path = candidate_dir / "phase4_judgments.jsonl"
    if out_path.exists() and not force:
        print(f"    skip phase4 {candidate_dir}")
        return out_path

    rubrics_by_id = {r["statement_id"]: r["rubric"] for r in load_jsonl(DIR / "e8_rubrics.jsonl") if "error" not in r}
    rubric = rubrics_by_id[statement["id"]]
    examples = get_examples(statement)
    cases = response_cases(statement["id"])
    tasks = [(judge, row) for judge in judges for row in cases]

    def run_one(judge: str, row: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            f"SPEC STATEMENT:\n{statement['text']}\n\n"
            f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
            f"RUBRIC:\n{render_anchors(rubric)}\n\n"
            f"USER QUERY: {row['user_query']}\n\n"
            f"ASSISTANT RESPONSE: {row['response']}\n\n"
            "Score per the schema."
        )
        key = {
            "statement_id": statement["id"],
            "scenario_idx": row["scenario_idx"],
            "generator": row["generator"],
            "candidate": candidate_dir.name,
        }
        try:
            if judge == "gpt":
                data = call_gpt_phase4(log, clients["gpt"], f"e9_phase4_{judge}", key, prompt)
            elif judge == "gemini":
                data = call_gemini_phase4(log, clients["gemini"], f"e9_phase4_{judge}", key, prompt)
            elif judge == "glm":
                data = call_glm_phase4(log, clients["glm"], f"e9_phase4_{judge}", key, prompt)
            else:
                raise ValueError(f"unknown judge: {judge}")
            return {
                **key,
                "judge": judge,
                "user_query": row["user_query"],
                "response": row["response"],
                "score": data.get("score"),
                "reasoning": data.get("reasoning", ""),
                "spec_quotes": data.get("spec_quotes") or [],
                "example_refs": data.get("example_refs") or [],
                "rubric_quotes": data.get("rubric_quotes") or [],
                "rubric_spec_tension": data.get("rubric_spec_tension"),
                "tension_description": data.get("tension_description", ""),
            }
        except Exception as exc:
            return {
                **key,
                "judge": judge,
                "user_query": row["user_query"],
                "response": row["response"],
                "error": f"{type(exc).__name__}: {exc}",
            }

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(tasks)))) as ex:
        futures = [ex.submit(run_one, *task) for task in tasks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if idx % 50 == 0 or idx == len(tasks):
                print(f"      phase4 {candidate_dir.name}: {idx}/{len(tasks)}")
    write_jsonl(sorted(rows, key=lambda r: (r["judge"], r["statement_id"], r["scenario_idx"], r["generator"])), out_path)
    return out_path


def judge_full_spec_candidate(
    candidate_dir: Path,
    statement: dict[str, Any],
    judges: list[str],
    clients: dict[str, Any],
    log: RawAPILogger,
    force: bool,
) -> Path:
    out_path = candidate_dir / "full_spec_judgments.jsonl"
    if out_path.exists() and not force:
        print(f"    skip full_spec {candidate_dir}")
        return out_path

    phantom_path = candidate_dir / "phantom_full_spec.jsonl"
    build_phantom_spec(statement, phantom_path)
    spec_md = render_spec_markdown(phantom_path)
    cases = response_cases(statement["id"])
    tasks = [(judge, row) for judge in judges for row in cases]

    def run_one(judge: str, row: dict[str, Any]) -> dict[str, Any]:
        prompt = JUDGE_FULL_SPEC_PROMPT_TEMPLATE.format(
            spec_md=spec_md, user_query=row["user_query"], response=row["response"]
        )
        key = {
            "statement_id": statement["id"],
            "scenario_idx": row["scenario_idx"],
            "generator": row["generator"],
            "candidate": candidate_dir.name,
        }
        try:
            if judge == "gpt":
                data = call_gpt_full_spec(log, clients["gpt"], f"e9_full_spec_{judge}", key, prompt)
            elif judge == "gemini":
                data = call_gemini_full_spec(log, clients["gemini"], f"e9_full_spec_{judge}", key, prompt)
            elif judge == "glm":
                data = call_glm_full_spec(log, clients["glm"], f"e9_full_spec_{judge}", key, prompt)
            else:
                raise ValueError(f"unknown judge: {judge}")
            decision = data.get("decision", "")
            if isinstance(decision, str):
                decision = decision.lower().strip()
            return {
                **key,
                "judge": judge,
                "user_query": row["user_query"],
                "response": row["response"],
                "decision": decision,
                "reasoning": data.get("reasoning", ""),
            }
        except Exception as exc:
            return {
                **key,
                "judge": judge,
                "user_query": row["user_query"],
                "response": row["response"],
                "error": f"{type(exc).__name__}: {exc}",
            }

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(tasks)))) as ex:
        futures = [ex.submit(run_one, *task) for task in tasks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if idx % 50 == 0 or idx == len(tasks):
                print(f"      full_spec {candidate_dir.name}: {idx}/{len(tasks)}")
    write_jsonl(sorted(rows, key=lambda r: (r["judge"], r["statement_id"], r["scenario_idx"], r["generator"])), out_path)
    return out_path


def candidate_score_maps(path: Path) -> dict[str, dict[tuple[str, int, str], dict[str, Any]]]:
    by_judge: dict[str, dict[tuple[str, int, str], dict[str, Any]]] = {}
    for row in load_jsonl(path):
        if "error" in row or row.get("score") not in {1, 2, 3, 4, 5}:
            continue
        by_judge.setdefault(row["judge"], {})[(row["statement_id"], int(row["scenario_idx"]), row["generator"])] = row
    return by_judge


def candidate_decision_maps(path: Path) -> dict[str, dict[tuple[str, int, str], dict[str, Any]]]:
    by_judge: dict[str, dict[tuple[str, int, str], dict[str, Any]]] = {}
    for row in load_jsonl(path):
        if "error" in row or row.get("decision") not in {"compliant", "non-compliant", "ambiguous"}:
            continue
        by_judge.setdefault(row["judge"], {})[(row["statement_id"], int(row["scenario_idx"]), row["generator"])] = row
    return by_judge


def compute_var_a_metrics(
    candidate_dir: Path,
    judges: list[str],
    compiler_keys: list[tuple[str, int, str]],
    held_out_keys: list[tuple[str, int, str]],
) -> dict[str, Any]:
    baseline_var_a = load_all_score_maps(VAR_A_PATHS)
    cand_var_a = candidate_score_maps(candidate_dir / "var_A_judgments.jsonl")
    k_base_held, n_base_held = fleiss_kappa_for_score_maps(baseline_var_a, held_out_keys, judges)
    k_cand_held, n_cand_held = fleiss_kappa_for_score_maps(cand_var_a, held_out_keys, judges)
    k_base_input, n_base_input = fleiss_kappa_for_score_maps(baseline_var_a, compiler_keys, judges)
    k_cand_input, n_cand_input = fleiss_kappa_for_score_maps(cand_var_a, compiler_keys, judges)
    return {
        "baseline_kappa_held_out_var_A": k_base_held,
        "candidate_kappa_held_out_var_A": k_cand_held,
        "delta_kappa_held_out_var_A": None if k_base_held is None or k_cand_held is None else k_cand_held - k_base_held,
        "n_baseline_held_out_var_A": n_base_held,
        "n_candidate_held_out_var_A": n_cand_held,
        "baseline_kappa_compiler_input_var_A": k_base_input,
        "candidate_kappa_compiler_input_var_A": k_cand_input,
        "delta_kappa_compiler_input_var_A": (
            None if k_base_input is None or k_cand_input is None else k_cand_input - k_base_input
        ),
        "n_baseline_compiler_input_var_A": n_base_input,
        "n_candidate_compiler_input_var_A": n_cand_input,
    }


def compute_cross_metrics(
    candidate_dir: Path, judges: list[str], held_out_keys: list[tuple[str, int, str]]
) -> dict[str, Any]:
    baseline_phase4 = load_all_score_maps(PHASE4_PATHS)
    baseline_full = load_all_decision_maps(PHASE3_PATHS)
    baseline_var_a = load_all_score_maps(VAR_A_PATHS)
    cand_var_a = candidate_score_maps(candidate_dir / "var_A_judgments.jsonl")
    cand_phase4 = candidate_score_maps(candidate_dir / "phase4_judgments.jsonl")
    cand_full = candidate_decision_maps(candidate_dir / "full_spec_judgments.jsonl")

    k_base_phase4, n_base_phase4 = fleiss_kappa_for_score_maps(baseline_phase4, held_out_keys, judges)
    k_cand_phase4, n_cand_phase4 = fleiss_kappa_for_score_maps(cand_phase4, held_out_keys, judges)
    k_base_full, n_base_full = fleiss_kappa_for_decision_maps(baseline_full, held_out_keys, judges)
    k_cand_full, n_cand_full = fleiss_kappa_for_decision_maps(cand_full, held_out_keys, judges)
    base_spearman = spearman_between_score_maps(baseline_var_a, baseline_phase4, held_out_keys, judges)
    cand_spearman = spearman_between_score_maps(cand_var_a, cand_phase4, held_out_keys, judges)
    spearman_improved = {}
    for judge in judges:
        b = base_spearman.get(judge)
        c = cand_spearman.get(judge)
        spearman_improved[judge] = None if b is None or c is None else c > b
    return {
        "baseline_kappa_phase_4": k_base_phase4,
        "candidate_kappa_phase_4": k_cand_phase4,
        "delta_kappa_phase_4": None if k_base_phase4 is None or k_cand_phase4 is None else k_cand_phase4 - k_base_phase4,
        "n_baseline_phase_4": n_base_phase4,
        "n_candidate_phase_4": n_cand_phase4,
        "baseline_kappa_full_spec": k_base_full,
        "candidate_kappa_full_spec": k_cand_full,
        "delta_kappa_full_spec": None if k_base_full is None or k_cand_full is None else k_cand_full - k_base_full,
        "n_baseline_full_spec": n_base_full,
        "n_candidate_full_spec": n_cand_full,
        "baseline_spearman_var_A_phase_4": base_spearman,
        "candidate_spearman_var_A_phase_4": cand_spearman,
        "spearman_improved": spearman_improved,
    }


def pass_var_a_gate(metrics: dict[str, Any], thresholds: dict[str, float]) -> bool:
    delta = metrics.get("delta_kappa_held_out_var_A")
    if delta is None or delta < thresholds["delta_kappa_held_out_var_A_min"]:
        return False
    input_delta = metrics.get("delta_kappa_compiler_input_var_A")
    if input_delta is None:
        return False
    return abs(delta - input_delta) <= thresholds["overfit_gap_max"]


def full_gate_verdict(
    var_a_metrics: dict[str, Any], cross_metrics: dict[str, Any] | None, thresholds: dict[str, float], judges: list[str]
) -> dict[str, Any]:
    reasons = []
    var_a_delta = var_a_metrics.get("delta_kappa_held_out_var_A")
    input_delta = var_a_metrics.get("delta_kappa_compiler_input_var_A")
    if var_a_delta is None or var_a_delta < thresholds["delta_kappa_held_out_var_A_min"]:
        reasons.append("held_out_var_A_delta_below_threshold")
    if input_delta is None or var_a_delta is None or abs(var_a_delta - input_delta) > thresholds["overfit_gap_max"]:
        reasons.append("overfit_gap_failed")
    if cross_metrics is None:
        reasons.append("cross_condition_not_run")
        return {"passed": False, "reasons": reasons}
    full_delta = cross_metrics.get("delta_kappa_full_spec")
    phase4_delta = cross_metrics.get("delta_kappa_phase_4")
    if full_delta is None or full_delta < thresholds["delta_kappa_full_spec_min"]:
        reasons.append("full_spec_delta_below_threshold")
    if phase4_delta is None or phase4_delta < thresholds["delta_kappa_phase_4_min"]:
        reasons.append("phase_4_delta_below_threshold")
    improved = cross_metrics.get("spearman_improved") or {}
    n_improved = sum(1 for judge in judges if improved.get(judge) is True)
    required = min(2, len(judges))
    if n_improved < required:
        reasons.append("spearman_improved_count_below_threshold")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "spearman_improved_count": n_improved,
        "spearman_required": required,
    }


def verify_candidate(
    candidate_dir: Path,
    round_id: int,
    judges: list[str],
    clients: dict[str, Any],
    log: RawAPILogger,
    force: bool,
    skip_cross_if_var_a_fails: bool,
) -> dict[str, Any]:
    statement = load_candidate_statement(candidate_dir / "statement.jsonl")
    compiler_keys, held_out_keys = load_split(candidate_dir)
    thresholds = gate_thresholds(round_id)
    print(f"  verify {statement['id']} / {candidate_dir.name}")
    judge_var_a_candidate(candidate_dir, statement, judges, clients, log, force)
    var_a_metrics = compute_var_a_metrics(candidate_dir, judges, compiler_keys, held_out_keys)
    cross_metrics = None
    if pass_var_a_gate(var_a_metrics, thresholds) or not skip_cross_if_var_a_fails:
        print("    var_A survivor; running full_spec + phase4")
        judge_phase4_candidate(candidate_dir, statement, judges, clients, log, force)
        judge_full_spec_candidate(candidate_dir, statement, judges, clients, log, force)
        cross_metrics = compute_cross_metrics(candidate_dir, judges, held_out_keys)
    else:
        print("    var_A gate failed early; skipping cross-condition spend")
    verdict = full_gate_verdict(var_a_metrics, cross_metrics, thresholds, judges)
    candidate_meta = json.loads((candidate_dir / "candidate.json").read_text())
    out = {
        "round": round_id,
        "statement_id": statement["id"],
        "candidate_id": candidate_dir.name,
        "candidate": candidate_meta,
        "judges": judges,
        "thresholds": thresholds,
        "var_A": var_a_metrics,
        "cross": cross_metrics,
        "gate": verdict,
    }
    out_path = candidate_dir / "verdict.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"    gate passed={verdict['passed']} reasons={verdict['reasons']}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--statements", nargs="*", default=None)
    parser.add_argument("--judges", nargs="*", choices=["gpt", "gemini", "glm"], default=["gpt", "gemini"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-skip-cross", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()

    dirs = candidate_dirs(args.round, args.statements)
    if args.max_candidates is not None:
        dirs = dirs[: args.max_candidates]
    if not dirs:
        print("No candidates found.")
        return 1

    print(f"round={args.round} candidates={len(dirs)} judges={args.judges}")
    print("baseline gate profiles are computed on GPT+Gemini unless --judges includes glm")
    clients = make_clients(args.judges)
    log = RawAPILogger(f"e9_verify_edit_round_{args.round}")
    print(f"raw run dir: {log.run_dir}")

    verdicts = []
    for cand_dir in dirs:
        verdicts.append(
            verify_candidate(
                cand_dir,
                args.round,
                args.judges,
                clients,
                log,
                force=args.force,
                skip_cross_if_var_a_fails=not args.no_skip_cross,
            )
        )
    out_path = REPAIR_DIR / f"round_{args.round}" / "verdicts.jsonl"
    write_jsonl(verdicts, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
