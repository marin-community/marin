# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  -- long log strings + CLI help

"""Stage 2 (scenario generation) orchestrator for the multi-LM diversity-gen pipeline.

See `.agents/logbooks/dart.md` §11.6 for design context.

Inputs:
  - Stage 1 understanding records (from `run_stage1_understanding.py`) —
    pointed to via `--stage1-dir`. Loads `understandings.jsonl`.
  - The Model Spec (for the statement text + examples; identical to Stage 1).

Generates `N` scenarios per statement (default 20) via:
  - One LLM call per scenario (NOT one call per statement — caching strategy).
  - Each call assembles three prefix levels:
      L1 (universal, ~1500 tokens)       — STAGE2_SYSTEM_PROMPT + methodology
                                            framing + JSON output schema.
                                            Cached across ALL statements + scenarios.
      L2 (per-statement, ~2500-3500 tok) — statement text + understanding +
                                            axes + spec examples. Cached across
                                            the 20 calls for ONE statement.
      L3 (variable, ~100 tok)            — scenario index + primary-axis hint.
                                            Differs per call.
  - OpenAI's prompt cache automatically reuses the longest matching prefix,
    so 1 cache write per universal-prefix-change (~1) + 1 per statement (~46)
    + 920 cache reads across the run.

Variation strategies:
  - `axis_rotation` (default): scenario N's primary axis = axes[N % len(axes)];
    primary spectrum value = axis.spectrum[(N // len(axes)) % len(spectrum)].
    Deterministic; each axis primary for ~20/8 = 2-3 scenarios.
  - `free_form`: no axis targeting per call; relies on temperature + diversity instruction.

Two execution modes (same as Stage 1):
  - sync  (immediate; recommended for smoke / small N)
  - batch (default for production; 50% cost, ≤24h SLA)

Output layout under `<output-base-dir>/<run_id>/`:
    manifest.json                   run config + stage1 source + git commit
    scenarios.jsonl                 one record per successful scenario
    parse_failures.jsonl            scenarios that failed all retries
    stage_status.json               {(statement_id, scenario_n) -> status}
    attempts/<sid>/scenario_NN/attempt_M__raw.json   per-retry raw responses
    batch_input.jsonl / state.json  if --mode batch

Usage:
    set -a; source .env; set +a
    PYENV_VERSION=3.12.0 uv run python -m experiments.posttrain.disagreement_primitive.diversity_gen.run_stage2_scenarios \\
        --mode sync \\
        --stage1-dir experiments/posttrain/disagreement_primitive/diversity_gen/gpt_5_1/stage1_understanding/<run_id> \\
        --statements be_engaging,refusal_style \\
        --n-scenarios 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI  # still used by batch-mode path (OpenAI-only)

from experiments.posttrain.disagreement_primitive.diversity_gen.lm_client import call_lm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_api_logger import RawAPILogger

from experiments.posttrain.disagreement_primitive.diversity_gen.parse_scenario import (
    parse_scenario_response,
    parse_single_call_diverse_response,
)
from experiments.posttrain.disagreement_primitive.diversity_gen.prompts import (
    STAGE2_SYSTEM_PROMPT,
    make_stage2_single_call_diverse_suffix,
    make_stage2_statement_prefix,
    make_stage2_universal_prefix,
    make_stage2_variation_suffix,
)

WORKTREE = Path(__file__).resolve().parents[4]
DEFAULT_SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
DEFAULT_OUTPUT_ROOT = WORKTREE / "experiments/posttrain/disagreement_primitive/diversity_gen"


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(WORKTREE))
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_spec(path: Path) -> dict[str, dict[str, Any]]:
    spec = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            spec[r["id"]] = r
    return spec


def _load_understandings(stage1_dir: Path) -> dict[str, dict[str, Any]]:
    path = stage1_dir / "understandings.jsonl"
    if not path.exists():
        raise SystemExit(f"missing understandings file: {path}")
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["statement_id"]] = r
    return out


@dataclass(frozen=True)
class ScenarioAssignment:
    """One row in the per-statement scenario enumeration.

    is_default=True identifies the baseline scenario (scenario 0) where every
    axis is at its default. is_default=False identifies a single-axis
    variation: `varied_axis` is at `varied_value`; every other axis stays at
    its default.
    """

    scenario_n: int
    is_default: bool
    varied_axis: str
    varied_value: str


def _compute_axis_assignment(axes: list[dict[str, Any]]) -> list[ScenarioAssignment]:
    """One-axis-at-a-time enumeration from defaults (dart.md §11.6).

    Returns:
      [
        ScenarioAssignment(0, is_default=True, ...),
        ScenarioAssignment(1, is_default=False, varied_axis=A.name, varied_value=v),
        ...
      ]

    Total = 1 + sum_over_axes(spectrum_size_i - 1). For each axis A, generates
    one variation scenario per non-default spectrum value (preserving
    spectrum order). The first scenario is always the all-default baseline.
    """
    out: list[ScenarioAssignment] = [ScenarioAssignment(scenario_n=0, is_default=True, varied_axis="", varied_value="")]
    n = 1
    for ax in axes:
        axis_name = ax["axis"]
        spectrum = ax.get("spectrum") or []
        default_value = ax.get("default_spectrum_value") or ""
        if not default_value:
            raise ValueError(
                f"axis {axis_name!r} is missing default_spectrum_value — re-run Stage 1 "
                f"with the updated prompt schema"
            )
        for v in spectrum:
            if v == default_value:
                continue
            out.append(ScenarioAssignment(scenario_n=n, is_default=False, varied_axis=axis_name, varied_value=v))
            n += 1
    return out


def _build_user_prompt(
    universal_prefix: str,
    statement_prefix: str,
    variation_suffix: str,
) -> str:
    """Assemble L1 + L2 + L3 in strict prefix order so cache reuse works."""
    return universal_prefix + statement_prefix + variation_suffix


@dataclass
class ScenarioResult:
    statement_id: str
    scenario_n: int
    is_default_scenario: bool
    varied_axis: str
    varied_value: str
    success: bool
    attempts: int
    parsed: dict[str, Any] | None
    last_error: str | None
    last_raw: str | None


def _save_attempt(
    attempts_dir: Path,
    statement_id: str,
    scenario_n: int,
    attempt: int,
    content: str,
    error: str | None,
) -> None:
    sub = attempts_dir / statement_id / f"scenario_{scenario_n:02d}"
    sub.mkdir(parents=True, exist_ok=True)
    rec = {
        "statement_id": statement_id,
        "scenario_n": scenario_n,
        "attempt": attempt,
        "raw_response": content,
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (sub / f"attempt_{attempt:02d}__raw.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))


def process_scenario_sync(
    client: OpenAI,
    log: RawAPILogger,
    statement: dict[str, Any],
    understanding: dict[str, Any],
    assignment: ScenarioAssignment,
    n_total: int,
    model: str,
    temperature: float,
    max_retries: int,
    attempts_dir: Path,
    universal_prefix: str,
    statement_prefix: str,
) -> ScenarioResult:
    sid = statement["id"]
    scenario_n = assignment.scenario_n
    suffix = make_stage2_variation_suffix(
        scenario_n,
        n_total,
        is_default_scenario=assignment.is_default,
        varied_axis=assignment.varied_axis,
        varied_value=assignment.varied_value,
    )
    user_content = _build_user_prompt(universal_prefix, statement_prefix, suffix)

    last_err: str | None = None
    last_content: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            content = log.call(
                role="stage2_scenarios",
                key={"statement_id": sid, "scenario_n": scenario_n, "attempt": attempt, "model": model},
                fn=lambda: call_lm(
                    model=model,
                    system=STAGE2_SYSTEM_PROMPT,
                    user=user_content,
                    max_output_tokens=3000,
                    temperature=temperature,
                    response_schema=None,
                ),
            )
        except Exception as exc:
            last_err = f"api_error: {type(exc).__name__}: {exc}"
            _save_attempt(attempts_dir, sid, scenario_n, attempt, content="", error=last_err)
            time.sleep(1 + attempt)
            continue

        content = (content or "").strip()
        last_content = content
        try:
            parsed = parse_scenario_response(content)
        except ValueError as exc:
            last_err = f"parse_error: {exc}"
            _save_attempt(attempts_dir, sid, scenario_n, attempt, content=content, error=last_err)
            time.sleep(1 + attempt)
            continue

        _save_attempt(attempts_dir, sid, scenario_n, attempt, content=content, error=None)
        return ScenarioResult(
            statement_id=sid,
            scenario_n=scenario_n,
            is_default_scenario=assignment.is_default,
            varied_axis=assignment.varied_axis,
            varied_value=assignment.varied_value,
            success=True,
            attempts=attempt,
            parsed=parsed,
            last_error=None,
            last_raw=content,
        )

    return ScenarioResult(
        statement_id=sid,
        scenario_n=scenario_n,
        is_default_scenario=assignment.is_default,
        varied_axis=assignment.varied_axis,
        varied_value=assignment.varied_value,
        success=False,
        attempts=max_retries,
        parsed=None,
        last_error=last_err,
        last_raw=last_content,
    )


def _build_output_record(
    statement: dict[str, Any],
    result: ScenarioResult,
    model: str,
    temperature: float,
    mode: str,
    stage1_run_id: str,
) -> dict[str, Any]:
    if not result.success or result.parsed is None:
        raise ValueError(f"cannot build record for failed scenario {result.statement_id}/{result.scenario_n}")
    p = result.parsed
    return {
        "statement_id": statement["id"],
        "scenario_n": result.scenario_n,
        "scenario_id": f"{statement['id']}__s{result.scenario_n:03d}",
        "is_default_scenario": result.is_default_scenario,
        "varied_axis": result.varied_axis,
        "varied_value": result.varied_value,
        "strategy": "one_axis_at_a_time_from_default",
        "scenario_text": p["scenario_text"],
        "user_query": p["user_query"],
        "system_prompt": p["system_prompt"],
        "axis_values_embodied": p["axis_values_embodied"],
        "rubric": p["rubric"],
        "model": model,
        "temperature": temperature,
        "reasoning_effort": "none",
        "mode": mode,
        "stage1_run_id": stage1_run_id,
        "attempt_index": result.attempts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Strategy: single_call_diverse — one call per statement, N+1 scenarios in
# one JSON array, hard topic-diversity constraint. See dart.md §11.9 follow-up
# and the prompt builder `make_stage2_single_call_diverse_suffix`.
# ---------------------------------------------------------------------------


def run_single_call_diverse_sync(
    spec: dict[str, dict[str, Any]],
    understandings: dict[str, dict[str, Any]],
    statement_ids: list[str],
    out_dir: Path,
    model: str,
    temperature: float,
    max_retries: int,
    workers: int,
    stage1_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not in environment. `set -a; source .env; set +a` first.")
    client = OpenAI(api_key=api_key)
    log = RawAPILogger(f"diversity_gen_{_model_slug(model)}_stage2_scd")
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    universal_prefix = make_stage2_universal_prefix()

    print(
        f"sync mode + single_call_diverse: 1 call per statement, " f"{len(statement_ids)} statements, workers={workers}",
        flush=True,
    )

    def process(sid: str) -> dict[str, Any]:
        statement = spec[sid]
        understanding = understandings[sid]
        axes = understanding.get("behavior_specific_axes") or []
        axes_names = [ax["axis"] for ax in axes]
        n_axes = len(axes)
        n_total = n_axes + 1
        statement_prefix = make_stage2_statement_prefix(statement, understanding)
        suffix = make_stage2_single_call_diverse_suffix(n_axes)
        user_content = universal_prefix + statement_prefix + suffix

        last_err: str | None = None
        last_content: str | None = None
        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                resp = log.call(
                    role="stage2_single_call_diverse",
                    key={"statement_id": sid, "attempt": attempt, "model": model},
                    fn=lambda: client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        temperature=temperature,
                        max_completion_tokens=16000,
                        reasoning_effort="none",
                        response_format={"type": "json_object"},
                    ),
                )
            except Exception as exc:
                last_err = f"api_error: {type(exc).__name__}: {exc}"
                print(f"  [{sid}] attempt {attempt} api error: {exc}", flush=True)
                _save_attempt(attempts_dir, sid, 0, attempt, content="", error=last_err)
                time.sleep(1 + attempt)
                continue

            content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
            last_content = content
            try:
                scenarios = parse_single_call_diverse_response(content, n_total, axes_names)
            except ValueError as exc:
                last_err = f"parse_error: {exc}"
                print(f"  [{sid}] attempt {attempt} parse error: {exc}", flush=True)
                _save_attempt(attempts_dir, sid, 0, attempt, content=content, error=last_err)
                time.sleep(1 + attempt)
                continue

            _save_attempt(attempts_dir, sid, 0, attempt, content=content, error=None)
            print(
                f"  [{sid}] attempt {attempt} OK ({n_total} scenarios, {time.time() - t0:.1f}s)",
                flush=True,
            )
            return {"sid": sid, "success": True, "scenarios": scenarios, "attempts": attempt, "raw": content}

        return {"sid": sid, "success": False, "error": last_err, "attempts": max_retries, "raw": last_content}

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, sid): sid for sid in statement_ids}
        completed = 0
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                failures.append({"statement_id": sid, "error": f"worker_crash: {type(exc).__name__}: {exc}"})
                continue
            completed += 1
            print(f"  done {completed}/{len(statement_ids)}: {sid}", flush=True)
            if not res["success"]:
                failures.append(
                    {
                        "statement_id": sid,
                        "error": res["error"],
                        "attempts": res["attempts"],
                        "last_raw_truncated": (res.get("raw") or "")[:500],
                    }
                )
                continue
            # Convert each scenario to the canonical per-record output shape
            for i, s in enumerate(res["scenarios"]):
                rec = {
                    "statement_id": sid,
                    "scenario_n": i,
                    "scenario_id": f"{sid}__s{i:03d}",
                    "is_default_scenario": bool(s["is_default_scenario"]),
                    "varied_axis": s["varied_axis"],
                    "varied_value": s["varied_value"],
                    "strategy": "single_call_diverse",
                    "scenario_text": s["scenario_text"],
                    "user_query": s["user_query"],
                    "system_prompt": s["system_prompt"],
                    "axis_values_embodied": s["axis_values_embodied"],
                    "rubric": s["rubric"],
                    "context_summary": s["context_summary"],
                    "model": model,
                    "temperature": temperature,
                    "reasoning_effort": "none",
                    "mode": "sync",
                    "stage1_run_id": stage1_run_id,
                    "attempt_index": res["attempts"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                successes.append(rec)

    return successes, failures


def run_sync_mode(
    spec: dict[str, dict[str, Any]],
    understandings: dict[str, dict[str, Any]],
    statement_ids: list[str],
    out_dir: Path,
    model: str,
    temperature: float,
    max_retries: int,
    workers: int,
    stage1_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not in environment. `set -a; source .env; set +a` first.")
    client = OpenAI(api_key=api_key)
    log = RawAPILogger(f"diversity_gen_{_model_slug(model)}_stage2")
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    universal_prefix = make_stage2_universal_prefix()

    # Build the full job list per statement. Total per statement =
    # 1 (default baseline) + sum(spectrum_size_i - 1) for each axis i.
    jobs: list[dict[str, Any]] = []
    for sid in statement_ids:
        statement = spec[sid]
        understanding = understandings[sid]
        axes = understanding.get("behavior_specific_axes") or []
        assignments = _compute_axis_assignment(axes)
        statement_prefix = make_stage2_statement_prefix(statement, understanding)
        n_total_for_stmt = len(assignments)
        for assignment in assignments:
            jobs.append(
                {
                    "statement": statement,
                    "understanding": understanding,
                    "assignment": assignment,
                    "n_total": n_total_for_stmt,
                    "universal_prefix": universal_prefix,
                    "statement_prefix": statement_prefix,
                }
            )

    per_stmt_counts = [(sid, sum(1 for j in jobs if j["statement"]["id"] == sid)) for sid in statement_ids]
    print(
        f"sync mode: {len(jobs)} total calls across {len(statement_ids)} statements "
        f"(strategy=one_axis_at_a_time_from_default, workers={workers}). Per-statement counts:",
        flush=True,
    )
    for sid, c in per_stmt_counts:
        print(f"  {sid}: {c} scenarios (1 default + variations)", flush=True)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def worker(j: dict[str, Any]) -> ScenarioResult:
        return process_scenario_sync(
            client,
            log,
            j["statement"],
            j["understanding"],
            j["assignment"],
            j["n_total"],
            model,
            temperature,
            max_retries,
            attempts_dir,
            j["universal_prefix"],
            j["statement_prefix"],
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                failures.append(
                    {
                        "statement_id": j["statement"]["id"],
                        "scenario_n": j["assignment"].scenario_n,
                        "error": f"worker_crash: {type(exc).__name__}: {exc}",
                    }
                )
                continue
            completed += 1
            if result.success:
                successes.append(
                    _build_output_record(
                        j["statement"],
                        result,
                        model,
                        temperature,
                        "sync",
                        stage1_run_id,
                    )
                )
                if completed % 10 == 0 or completed == len(jobs):
                    label = "default" if result.is_default_scenario else f"vary[{result.varied_axis}]"
                    print(
                        f"  progress: {completed}/{len(jobs)} (last: {result.statement_id}/s{result.scenario_n:03d} {label})",
                        flush=True,
                    )
            else:
                failures.append(
                    {
                        "statement_id": result.statement_id,
                        "scenario_n": result.scenario_n,
                        "is_default_scenario": result.is_default_scenario,
                        "varied_axis": result.varied_axis,
                        "varied_value": result.varied_value,
                        "attempts": result.attempts,
                        "error": result.last_error,
                        "last_raw_truncated": (result.last_raw or "")[:500],
                    }
                )
    return successes, failures


def run_batch_anthropic_mode(
    spec: dict[str, dict[str, Any]],
    understandings: dict[str, dict[str, Any]],
    statement_ids: list[str],
    out_dir: Path,
    model: str,
    temperature: float,
    stage1_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Anthropic Message Batches API for Stage 2a oaat.

    Caching: the cacheable prefix per call is `universal_prefix + statement_prefix`
    (~2.5-3k tokens). Across the ~22 calls per statement, all share the same
    per-statement prefix. Anthropic ephemeral caching engages because the prefix
    exceeds the 1024-token threshold.
    """
    import batch_anthropic as ba  # type: ignore

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set — `unset ANTHROPIC_API_KEY; set -a; source .env; source .env2; set +a`")
    log = RawAPILogger(f"diversity_gen_{_model_slug(model)}_stage2")
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    universal_prefix = make_stage2_universal_prefix()

    requests = []
    cmap: dict[str, dict[str, Any]] = {}
    for sid in statement_ids:
        statement = spec[sid]
        understanding = understandings[sid]
        axes = understanding.get("behavior_specific_axes") or []
        assignments = _compute_axis_assignment(axes)
        statement_prefix = make_stage2_statement_prefix(statement, understanding)
        cacheable_prefix = universal_prefix + statement_prefix  # ~2-3k tokens, >1024 threshold
        n_total_for_stmt = len(assignments)
        for assignment in assignments:
            n = assignment.scenario_n
            custom_id = f"stage2__{sid}__s{n:03d}"
            cmap[custom_id] = {
                "statement_id": sid,
                "scenario_n": n,
                "is_default_scenario": assignment.is_default,
                "varied_axis": assignment.varied_axis,
                "varied_value": assignment.varied_value,
                "n_total_for_stmt": n_total_for_stmt,
            }
            suffix = make_stage2_variation_suffix(
                n,
                n_total_for_stmt,
                is_default_scenario=assignment.is_default,
                varied_axis=assignment.varied_axis,
                varied_value=assignment.varied_value,
            )
            user_content = _build_user_prompt(universal_prefix, statement_prefix, suffix)
            req = ba.build_request(
                custom_id=custom_id,
                model=model,
                system=STAGE2_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=3000,
                temperature=temperature,
                cache=True,
                cache_user_prefix=cacheable_prefix,
            )
            requests.append(req)

    (out_dir / "batch_custom_id_map.json").write_text(json.dumps(cmap, indent=2))
    print(f"submitting anthropic batch with {len(requests)} rows", flush=True)
    state = ba.submit(api_key, requests, out_dir, name="batch")
    print(f"  batch_id: {state['batch_id']}", flush=True)
    ba.poll(api_key, out_dir, name="batch", interval=30.0, timeout=86400.0)
    entries = ba.collect(api_key, out_dir, name="batch")

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entry in entries:
        custom_id = entry.get("custom_id", "?")
        cinfo = cmap.get(custom_id) or {}
        sid = cinfo.get("statement_id", custom_id)
        n = cinfo.get("scenario_n", 0)
        log.call(
            role="stage2_scenarios_batch_anthropic",
            key={"statement_id": sid, "scenario_n": n, "custom_id": custom_id, "model": model},
            fn=lambda e=entry: e,
        )
        result = entry.get("result") or {}
        if result.get("type") != "succeeded":
            err = json.dumps(result)[:300]
            failures.append({"statement_id": sid, "scenario_n": n, "error": f"batch_error: {err}"})
            _save_attempt(attempts_dir, sid, n, 1, content="", error=err)
            continue
        msg = result.get("message") or {}
        blocks = msg.get("content") or []
        content = ""
        for b in blocks:
            if b.get("type") == "text":
                content += b.get("text", "")
        content = content.strip()
        try:
            parsed = parse_scenario_response(content)
        except ValueError as exc:
            failures.append({"statement_id": sid, "scenario_n": n, "error": f"parse_error: {exc}", "last_raw_truncated": content[:500]})
            _save_attempt(attempts_dir, sid, n, 1, content=content, error=f"parse_error: {exc}")
            continue
        _save_attempt(attempts_dir, sid, n, 1, content=content, error=None)
        sr = ScenarioResult(
            statement_id=sid,
            scenario_n=n,
            is_default_scenario=cinfo.get("is_default_scenario", False),
            varied_axis=cinfo.get("varied_axis", ""),
            varied_value=cinfo.get("varied_value", ""),
            success=True,
            attempts=1,
            parsed=parsed,
            last_error=None,
            last_raw=content,
        )
        rec = _build_output_record(spec[sid], sr, model, temperature, "batch_anthropic", stage1_run_id)
        successes.append(rec)
    return successes, failures


def run_batch_mode(
    spec: dict[str, dict[str, Any]],
    understandings: dict[str, dict[str, Any]],
    statement_ids: list[str],
    out_dir: Path,
    model: str,
    temperature: float,
    stage1_run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One-shot OpenAI batch. Each row is a single scenario call."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not in environment. `set -a; source .env; set +a` first.")
    client = OpenAI(api_key=api_key)
    log = RawAPILogger(f"diversity_gen_{_model_slug(model)}_stage2")
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    universal_prefix = make_stage2_universal_prefix()

    # Build batch JSONL
    input_path = out_dir / "batch_input.jsonl"
    cmap: dict[str, dict[str, Any]] = {}
    with input_path.open("w") as fh:
        for sid in statement_ids:
            statement = spec[sid]
            understanding = understandings[sid]
            axes = understanding.get("behavior_specific_axes") or []
            assignments = _compute_axis_assignment(axes)
            statement_prefix = make_stage2_statement_prefix(statement, understanding)
            n_total_for_stmt = len(assignments)
            for assignment in assignments:
                n = assignment.scenario_n
                custom_id = f"stage2__{sid}__s{n:03d}"
                cmap[custom_id] = {
                    "statement_id": sid,
                    "scenario_n": n,
                    "is_default_scenario": assignment.is_default,
                    "varied_axis": assignment.varied_axis,
                    "varied_value": assignment.varied_value,
                    "n_total_for_stmt": n_total_for_stmt,
                }
                suffix = make_stage2_variation_suffix(
                    n,
                    n_total_for_stmt,
                    is_default_scenario=assignment.is_default,
                    varied_axis=assignment.varied_axis,
                    varied_value=assignment.varied_value,
                )
                user_content = _build_user_prompt(universal_prefix, statement_prefix, suffix)
                body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": temperature,
                    "max_completion_tokens": 3000,
                    "reasoning_effort": "none",
                    "response_format": {"type": "json_object"},
                }
                fh.write(
                    json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body})
                    + "\n"
                )

    (out_dir / "batch_custom_id_map.json").write_text(json.dumps(cmap, indent=2))
    print(f"batch input written: {input_path} ({len(cmap)} rows)", flush=True)

    upload = client.files.create(file=input_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"diversity_gen stage2 scenarios ({model}, one-axis-at-a-time)"},
    )
    (out_dir / "batch_state.json").write_text(json.dumps({"batch_id": batch.id, "input_file_id": upload.id}, indent=2))
    print(f"submitted batch {batch.id}", flush=True)

    terminal = {"completed", "failed", "expired", "cancelled"}
    while batch.status not in terminal:
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)
        rc = batch.request_counts
        print(
            f"  poll: status={batch.status} counts=total={rc.total if rc else '?'} "
            f"completed={rc.completed if rc else '?'} failed={rc.failed if rc else '?'}",
            flush=True,
        )

    if batch.status != "completed":
        raise SystemExit(f"batch ended in non-completed status: {batch.status}")

    output_blob = client.files.content(batch.output_file_id).text
    (out_dir / "batch_output.jsonl").write_text(output_blob)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for line in output_blob.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        custom_id = entry.get("custom_id", "?")
        meta = cmap.get(custom_id, {})
        sid = meta.get("statement_id", custom_id)
        scenario_n = meta.get("scenario_n", -1)
        statement = spec.get(sid)
        if statement is None:
            failures.append({"statement_id": sid, "scenario_n": scenario_n, "error": "no matching spec record"})
            continue

        log.call(
            role="stage2_scenarios_batch",
            key={"statement_id": sid, "scenario_n": scenario_n, "custom_id": custom_id, "model": model},
            fn=lambda e=entry: e,
        )

        response = entry.get("response") or {}
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            failures.append({"statement_id": sid, "scenario_n": scenario_n, "error": "no choices"})
            _save_attempt(attempts_dir, sid, scenario_n, 1, "", "no choices")
            continue
        content = (choices[0].get("message") or {}).get("content") or ""
        try:
            parsed = parse_scenario_response(content)
        except ValueError as exc:
            failures.append(
                {
                    "statement_id": sid,
                    "scenario_n": scenario_n,
                    "error": f"parse_error: {exc}",
                    "last_raw_truncated": content[:500],
                }
            )
            _save_attempt(attempts_dir, sid, scenario_n, 1, content, f"parse_error: {exc}")
            continue

        _save_attempt(attempts_dir, sid, scenario_n, 1, content, None)
        result = ScenarioResult(
            statement_id=sid,
            scenario_n=scenario_n,
            is_default_scenario=meta.get("is_default_scenario", False),
            varied_axis=meta.get("varied_axis", ""),
            varied_value=meta.get("varied_value", ""),
            success=True,
            attempts=1,
            parsed=parsed,
            last_error=None,
            last_raw=content,
        )
        successes.append(
            _build_output_record(
                statement,
                result,
                model,
                temperature,
                "batch",
                stage1_run_id,
            )
        )

    return successes, failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 2 (scenario generation) for diversity-gen pipeline.")
    p.add_argument("--mode", choices=["sync", "batch"], default="batch", help="execution mode (default: batch)")
    p.add_argument("--model", default="gpt-5.1", help="LM to use (default: gpt-5.1)")
    p.add_argument("--temperature", type=float, default=1.0, help="generation temperature (default: 1.0)")
    p.add_argument(
        "--strategy",
        choices=["one_axis_at_a_time_from_default", "single_call_diverse"],
        default="one_axis_at_a_time_from_default",
        help=(
            "scenario-generation strategy. one_axis_at_a_time_from_default: one LM call per scenario, "
            "single-axis variation from default (dart.md §11.6). single_call_diverse: one call per "
            "statement returning N+1 scenarios (1 default + N axis variations) with hard topic-diversity "
            "constraint (dart.md §11.9 follow-up)."
        ),
    )
    p.add_argument("--stage1-dir", type=Path, required=True, help="path to a stage1_understanding/<run_id>/ directory")
    p.add_argument("--statements", default=None, help="comma-separated statement ids (default: all in stage1 output)")
    p.add_argument("--spec-path", type=Path, default=DEFAULT_SPEC_PATH)
    p.add_argument("--max-retries", type=int, default=5, help="sync-mode max retries per scenario")
    p.add_argument("--workers", type=int, default=16, help="sync-mode concurrency (default: 16)")
    p.add_argument(
        "--output-base-dir",
        type=Path,
        default=None,
        help="output root; defaults to diversity_gen/<model_slug>/stage2_scenarios/",
    )
    args = p.parse_args(argv)

    spec = _load_spec(args.spec_path)
    understandings = _load_understandings(args.stage1_dir)

    stage1_manifest_path = args.stage1_dir / "manifest.json"
    stage1_run_id = "unknown"
    if stage1_manifest_path.exists():
        try:
            stage1_run_id = json.loads(stage1_manifest_path.read_text()).get("run_id", "unknown")
        except Exception:
            pass

    if args.statements:
        wanted = [s.strip() for s in args.statements.split(",") if s.strip()]
    else:
        wanted = sorted(understandings.keys())
    missing = [sid for sid in wanted if sid not in understandings]
    if missing:
        raise SystemExit(f"--statements references ids missing from stage1 output: {missing}")
    missing_spec = [sid for sid in wanted if sid not in spec]
    if missing_spec:
        raise SystemExit(f"statements missing from spec file: {missing_spec}")

    # Strategy-specific output subdir + scenario count computation
    strategy_slug = "scd" if args.strategy == "single_call_diverse" else "oaat"
    out_base = args.output_base_dir or (
        DEFAULT_OUTPUT_ROOT / _model_slug(args.model) / "stage2_scenarios" / strategy_slug
    )
    run_id = _now_stamp()
    out_dir = out_base / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)

    per_stmt_counts: list[tuple[str, int]] = []
    if args.strategy == "single_call_diverse":
        for sid in wanted:
            axes = understandings[sid].get("behavior_specific_axes") or []
            per_stmt_counts.append((sid, len(axes) + 1))  # 1 default + N variations
    else:
        for sid in wanted:
            axes = understandings[sid].get("behavior_specific_axes") or []
            per_stmt_counts.append((sid, len(_compute_axis_assignment(axes))))
    total_scenarios = sum(c for _, c in per_stmt_counts)
    print(
        f"will generate {total_scenarios} scenarios across {len(wanted)} statements (strategy={args.strategy}):",
        flush=True,
    )
    for sid, c in per_stmt_counts:
        print(f"  {sid}: {c} scenarios", flush=True)

    manifest = {
        "run_id": run_id,
        "model": args.model,
        "temperature": args.temperature,
        "reasoning_effort": "none",
        "mode": args.mode,
        "strategy": args.strategy,
        "stage1_dir": str(args.stage1_dir),
        "stage1_run_id": stage1_run_id,
        "statements": wanted,
        "per_statement_scenario_counts": dict(per_stmt_counts),
        "total_scenarios": total_scenarios,
        "max_retries": args.max_retries,
        "workers": args.workers,
        "git_commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.strategy == "single_call_diverse":
        if args.mode == "batch":
            raise SystemExit("single_call_diverse + batch mode not implemented yet; use --mode sync")
        successes, failures = run_single_call_diverse_sync(
            spec,
            understandings,
            wanted,
            out_dir,
            args.model,
            args.temperature,
            args.max_retries,
            args.workers,
            stage1_run_id,
        )
    elif args.mode == "sync":
        successes, failures = run_sync_mode(
            spec,
            understandings,
            wanted,
            out_dir,
            args.model,
            args.temperature,
            args.max_retries,
            args.workers,
            stage1_run_id,
        )
    elif args.model.lower().startswith("claude-"):
        successes, failures = run_batch_anthropic_mode(
            spec, understandings, wanted, out_dir, args.model, args.temperature, stage1_run_id,
        )
    else:
        successes, failures = run_batch_mode(
            spec,
            understandings,
            wanted,
            out_dir,
            args.model,
            args.temperature,
            stage1_run_id,
        )

    with (out_dir / "scenarios.jsonl").open("w") as f:
        for rec in sorted(successes, key=lambda r: (r["statement_id"], r["scenario_n"])):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "parse_failures.jsonl").open("w") as f:
        for rec in sorted(failures, key=lambda r: (r.get("statement_id", ""), r.get("scenario_n", -1))):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    status = {
        **{f"{r['statement_id']}__s{r['scenario_n']:03d}": "succeeded" for r in successes},
        **{f"{r.get('statement_id', '?')}__s{r.get('scenario_n', -1):03d}": "failed" for r in failures},
    }
    (out_dir / "stage_status.json").write_text(json.dumps(status, indent=2))

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_succeeded"] = len(successes)
    manifest["n_failed"] = len(failures)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDONE. {len(successes)}/{total_scenarios} succeeded, {len(failures)} failed.", flush=True)
    print(f"output dir: {out_dir}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
