# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  -- long log strings + CLI help

"""Stage 2 (repair) orchestrator — post-hoc diversity rewrite over an existing
Set B (`rubric-default-style`) corpus.

See `.agents/logbooks/dart.md` §11.11 for the design rationale and the
sign-off-ready pilot proposal.

Pilot scope (single-pass, no auditor):
  Input  : `<source_set_b_dir>/scenarios.jsonl` + `<stage1_dir>/understandings.jsonl`
  Output : `<output_base_dir>/repaired/<run_id>/scenarios.jsonl`
  Per statement: ONE GPT-5.1 call that takes all of that statement's Set B
  scenarios and returns rewritten surface fields, preserving the immutable
  axis-assignment fields verbatim. No auditor pass.

Usage:
    set -a; source .env; set +a
    PYENV_VERSION=3.12.0 uv run python -m experiments.posttrain.disagreement_primitive.diversity_gen.run_stage2_repair \\
        --source-set-b-dir experiments/posttrain/disagreement_primitive/diversity_gen/gpt_5_1/stage2_scenarios/20260516T174023Z \\
        --stage1-dir       experiments/posttrain/disagreement_primitive/diversity_gen/gpt_5_1/stage1_understanding/20260516T172804Z \\
        --statements       no_topic_off_limits,avoid_hateful_content
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

from openai import OpenAI  # kept for type hints / future batch-mode use

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_api_logger import RawAPILogger

from experiments.posttrain.disagreement_primitive.diversity_gen.lm_client import call_lm
from experiments.posttrain.disagreement_primitive.diversity_gen.parse_scenario import (
    parse_repair_response,
)
from experiments.posttrain.disagreement_primitive.diversity_gen.prompts import (
    REPAIR_OUTPUT_JSON_SCHEMA,
    STAGE2_REPAIR_SYSTEM_PROMPT,
    make_stage2_repair_iterative_prompt,
    make_stage2_repair_singlepass_prompt,
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
    spec: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            spec[r["id"]] = r
    return spec


def _load_understandings(stage1_dir: Path) -> dict[str, dict[str, Any]]:
    p = stage1_dir / "understandings.jsonl"
    if not p.exists():
        raise SystemExit(f"missing understandings file: {p}")
    out: dict[str, dict[str, Any]] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["statement_id"]] = r
    return out


def _load_set_b_scenarios(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    p = source_dir / "scenarios.jsonl"
    if not p.exists():
        raise SystemExit(f"missing source scenarios file: {p}")
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_sid.setdefault(r["statement_id"], []).append(r)
    for sid, lst in by_sid.items():
        lst.sort(key=lambda x: x["scenario_n"])
    return by_sid


@dataclass
class RepairResult:
    statement_id: str
    n_source: int
    success: bool
    attempts: int
    rewritten: list[dict[str, Any]] | None
    last_error: str | None
    last_raw: str | None


def _save_attempt(
    attempts_dir: Path,
    statement_id: str,
    attempt: int,
    content: str,
    error: str | None,
) -> None:
    sub = attempts_dir / statement_id
    sub.mkdir(parents=True, exist_ok=True)
    rec = {
        "statement_id": statement_id,
        "attempt": attempt,
        "raw_response": content,
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (sub / f"attempt_{attempt:02d}__raw.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False)
    )


def process_statement_iterative(
    client: OpenAI,
    log: RawAPILogger,
    statement: dict[str, Any],
    understanding: dict[str, Any],
    source_scenarios: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_retries: int,
    max_completion_tokens: int,
    attempts_dir: Path,
) -> RepairResult:
    """Sequential per-scenario repair: each rewrite sees all prior rewrites.

    Within a statement, scenarios are processed in scenario_n order. Each call
    is one LM call producing one rewritten scenario. The running list of prior
    rewrites is included in every subsequent prompt so the LM can avoid
    repeating their target referents / personas / countries / domains.

    A single scenario failure (after max_retries) ends the statement run and
    returns a partial RepairResult marked failed — we do not silently produce
    a partial-set output, so an out-of-band fix can be made.
    """
    sid = statement["id"]
    n_total = len(source_scenarios)
    prior_rewrites: list[dict[str, Any]] = []

    for source in source_scenarios:
        scenario_n = source["scenario_n"]
        user_content = make_stage2_repair_iterative_prompt(
            statement_record=statement,
            understanding_record=understanding,
            source_scenario=source,
            prior_rewrites=prior_rewrites,
            total_in_statement=n_total,
        )

        last_err: str | None = None
        last_content: str | None = None
        rewritten_one: dict[str, Any] | None = None
        for attempt in range(1, max_retries + 1):
            try:
                content = log.call(
                    role="stage2_repair_iterative",
                    key={"statement_id": sid, "scenario_n": scenario_n, "attempt": attempt, "model": model},
                    fn=lambda: call_lm(
                        model=model,
                        system=STAGE2_REPAIR_SYSTEM_PROMPT,
                        user=user_content,
                        max_output_tokens=max_completion_tokens,
                        temperature=temperature,
                        response_schema=None,
                    ),
                )
            except Exception as exc:
                last_err = f"api_error: {type(exc).__name__}: {exc}"
                _save_attempt(attempts_dir, f"{sid}__s{scenario_n:03d}", attempt, content="", error=last_err)
                time.sleep(1 + attempt)
                continue

            content = (content or "").strip()
            last_content = content
            try:
                parsed = parse_repair_response(content, [source])
            except ValueError as exc:
                last_err = f"parse_error: {exc}"
                _save_attempt(attempts_dir, f"{sid}__s{scenario_n:03d}", attempt, content=content, error=last_err)
                time.sleep(1 + attempt)
                continue

            _save_attempt(attempts_dir, f"{sid}__s{scenario_n:03d}", attempt, content=content, error=None)
            rewritten_one = parsed[0]
            break

        if rewritten_one is None:
            return RepairResult(
                statement_id=sid,
                n_source=n_total,
                success=False,
                attempts=max_retries,
                rewritten=prior_rewrites,
                last_error=f"failed on scenario_n={scenario_n}: {last_err}",
                last_raw=last_content,
            )

        prior_rewrites.append(rewritten_one)

    return RepairResult(
        statement_id=sid,
        n_source=n_total,
        success=True,
        attempts=1,
        rewritten=prior_rewrites,
        last_error=None,
        last_raw=None,
    )


def process_statement_sync(
    client: OpenAI,
    log: RawAPILogger,
    statement: dict[str, Any],
    understanding: dict[str, Any],
    source_scenarios: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_retries: int,
    max_completion_tokens: int,
    attempts_dir: Path,
    max_per_surface_value: int | None = None,
    strict_schema: bool = False,
) -> RepairResult:
    sid = statement["id"]
    user_content = make_stage2_repair_singlepass_prompt(
        statement_record=statement,
        understanding_record=understanding,
        scenarios=source_scenarios,
        max_per_surface_value=max_per_surface_value,
    )
    schema = REPAIR_OUTPUT_JSON_SCHEMA if strict_schema else None

    last_err: str | None = None
    last_content: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            content = log.call(
                role="stage2_repair_singlepass",
                key={"statement_id": sid, "attempt": attempt, "model": model},
                fn=lambda: call_lm(
                    model=model,
                    system=STAGE2_REPAIR_SYSTEM_PROMPT,
                    user=user_content,
                    max_output_tokens=max_completion_tokens,
                    temperature=temperature,
                    response_schema=schema,
                ),
            )
        except Exception as exc:
            last_err = f"api_error: {type(exc).__name__}: {exc}"
            _save_attempt(attempts_dir, sid, attempt, content="", error=last_err)
            time.sleep(1 + attempt)
            continue

        content = (content or "").strip()
        last_content = content
        try:
            rewritten = parse_repair_response(content, source_scenarios)
        except ValueError as exc:
            last_err = f"parse_error: {exc}"
            _save_attempt(attempts_dir, sid, attempt, content=content, error=last_err)
            time.sleep(1 + attempt)
            continue

        _save_attempt(attempts_dir, sid, attempt, content=content, error=None)
        return RepairResult(
            statement_id=sid,
            n_source=len(source_scenarios),
            success=True,
            attempts=attempt,
            rewritten=rewritten,
            last_error=None,
            last_raw=content,
        )

    return RepairResult(
        statement_id=sid,
        n_source=len(source_scenarios),
        success=False,
        attempts=max_retries,
        rewritten=None,
        last_error=last_err,
        last_raw=last_content,
    )


def run_anthropic_batch_singlepass(
    spec: dict[str, dict[str, Any]],
    understandings: dict[str, dict[str, Any]],
    by_sid: dict[str, list[dict[str, Any]]],
    statement_ids: list[str],
    out_root: Path,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    max_per_surface_value: int | None,
    strict_schema: bool,
    attempts_dir: Path,
) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    """Anthropic Message Batches API for Stage 2b single_pass repair.

    One batch row per statement. No `cache_user_prefix` — the V2.5a prompt
    layout puts statement-specific content BEFORE the methodology, so there's
    no shared byte-prefix across the 46 calls to amortize.

    Returns (n_admitted, n_failed_statements, admitted_records, failure_records).
    """
    import batch_anthropic as ba  # type: ignore

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set — `unset ANTHROPIC_API_KEY; set -a; source .env; source .env2; set +a`")
    log = RawAPILogger(f"diversity_gen_{_model_slug(model)}_stage2_repair")

    schema = REPAIR_OUTPUT_JSON_SCHEMA if strict_schema else None

    requests = []
    cmap: dict[str, str] = {}  # custom_id -> statement_id
    for sid in statement_ids:
        statement = spec[sid]
        understanding = understandings[sid]
        source_scenarios = by_sid[sid]
        user_content = make_stage2_repair_singlepass_prompt(
            statement_record=statement,
            understanding_record=understanding,
            scenarios=source_scenarios,
            max_per_surface_value=max_per_surface_value,
        )
        custom_id = f"stage2b__{sid}"
        cmap[custom_id] = sid
        req = ba.build_request(
            custom_id=custom_id,
            model=model,
            system=STAGE2_REPAIR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=max_completion_tokens,
            temperature=temperature,
            cache=False,
            # Strict JSON schema enforced server-side. Anthropic takes the raw
            # schema dict (no `{name, strict, schema}` wrapper). Eliminates the
            # planning-preamble parse failure mode we saw without it.
            output_schema=schema["schema"] if schema is not None else None,
        )
        requests.append(req)

    (out_root / "batch_custom_id_map.json").write_text(json.dumps(cmap, indent=2))
    print(f"submitting anthropic batch with {len(requests)} statements", flush=True)
    state = ba.submit(api_key, requests, out_root, name="batch")
    print(f"  batch_id: {state['batch_id']}", flush=True)
    ba.poll(api_key, out_root, name="batch", interval=30.0, timeout=86400.0)
    entries = ba.collect(api_key, out_root, name="batch")

    admitted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_admitted = 0
    n_failed = 0
    for entry in entries:
        custom_id = entry.get("custom_id", "?")
        sid = cmap.get(custom_id, custom_id)
        log.call(
            role="stage2_repair_singlepass_batch_anthropic",
            key={"statement_id": sid, "custom_id": custom_id, "model": model},
            fn=lambda e=entry: e,
        )
        result = entry.get("result") or {}
        if result.get("type") != "succeeded":
            n_failed += 1
            err = json.dumps(result)[:400]
            failures.append({"statement_id": sid, "n_source": len(by_sid[sid]), "error": f"batch_error: {err}"})
            print(f"   [FAIL] {sid}: {err[:200]}")
            continue
        msg = result.get("message") or {}
        blocks = msg.get("content") or []
        content = ""
        for b in blocks:
            if b.get("type") == "text":
                content += b.get("text", "")
        content = content.strip()
        try:
            rewritten = parse_repair_response(content, by_sid[sid])
        except ValueError as exc:
            n_failed += 1
            failures.append({"statement_id": sid, "n_source": len(by_sid[sid]), "error": f"parse_error: {exc}", "last_raw_truncated": content[:500]})
            print(f"   [FAIL] {sid}: parse_error: {str(exc)[:200]}")
            continue
        for rew, src in zip(rewritten, by_sid[sid], strict=True):
            rec = {
                "statement_id": sid,
                "scenario_n": rew["scenario_n"],
                "scenario_id": rew["scenario_id"],
                "is_default_scenario": rew["is_default_scenario"],
                "varied_axis": rew["varied_axis"],
                "varied_value": rew["varied_value"],
                "strategy": "repair_singlepass",
                "source_strategy": src.get("strategy", "one_axis_at_a_time_from_default"),
                "scenario_text": rew["scenario_text"],
                "user_query": rew["user_query"],
                "system_prompt": rew["system_prompt"],
                "rubric": rew["rubric"],
            }
            if "contains_placeholder" in rew:
                rec["contains_placeholder"] = rew["contains_placeholder"]
                rec["placeholder_notes"] = rew.get("placeholder_notes", "")
            admitted.append(rec)
            n_admitted += 1
        print(f"   [OK]   {sid}: {len(rewritten)} rewritten")

    return n_admitted, n_failed, admitted, failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-set-b-dir", type=Path, required=True,
                    help="dir with the source rubric-default-style run; reads scenarios.jsonl")
    ap.add_argument("--stage1-dir", type=Path, required=True,
                    help="dir with the matching Stage 1 understandings.jsonl")
    ap.add_argument("--spec-path", type=Path, default=DEFAULT_SPEC_PATH)
    ap.add_argument("--statements", type=str, required=True,
                    help='comma-separated statement IDs or "all" (where "all" means every '
                         "statement_id present in the source Set B scenarios.jsonl)")
    ap.add_argument("--mode", type=str, default="single_pass",
                    choices=["single_pass", "iterative"],
                    help="single_pass: one LM call per statement, rewrites all N scenarios at once. "
                         "iterative: N LM calls per statement, each with running list of prior rewrites visible.")
    ap.add_argument("--model", type=str, default="gpt-5.1")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-completion-tokens", type=int, default=20000,
                    help="per-call output cap. single_pass: ~10-15k tokens for 25 scenarios. "
                         "iterative: 1 scenario per call, ~2000 tokens suffices.")
    ap.add_argument("--output-base-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--max-workers", type=int, default=8,
                    help="across-statement parallelism. Within a statement, iterative mode is sequential.")
    ap.add_argument("--max-per-surface-value", type=int, default=None,
                    help="single_pass only: hard cap on how many scenarios may share the same "
                         "surface-dimension value (target referent / persona / country / domain). "
                         "Default None = use ⌈N/5⌉ (V2 pilot). Set 2 for V2.5a; set 1 for V2.5b.")
    ap.add_argument("--strict-schema", action="store_true",
                    help="Use OpenAI strict json_schema response_format (server-side schema enforcement). "
                         "Adds two diagnostic fields per scenario: contains_placeholder + placeholder_notes "
                         "(LM self-reports if it hedged or used a placeholder). Default off (json_object mode).")
    ap.add_argument("--run-tag", type=str, default=None,
                    help="optional short tag appended to the output run_id (e.g., 'v25a_cap2'). "
                         "Helps name parallel pilot runs without collisions.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan + first ~500 chars of one prompt, then exit (no API calls)")
    args = ap.parse_args()

    # Backend keys validated lazily inside lm_client.call_lm() on first call
    # (OPENAI_API_KEY for gpt-*, GEMINI_API_KEY for gemini-*, etc.).

    spec = _load_spec(args.spec_path)
    understandings = _load_understandings(args.stage1_dir)
    by_sid = _load_set_b_scenarios(args.source_set_b_dir)

    if args.statements.strip() == "all":
        statement_ids = sorted(by_sid.keys())
    else:
        statement_ids = [s.strip() for s in args.statements.split(",") if s.strip()]

    missing_spec = [s for s in statement_ids if s not in spec]
    missing_u = [s for s in statement_ids if s not in understandings]
    missing_b = [s for s in statement_ids if s not in by_sid]
    if missing_spec or missing_u or missing_b:
        raise SystemExit(
            f"missing inputs:\n  spec={missing_spec}\n  understandings={missing_u}\n  set_b={missing_b}"
        )

    run_id = _now_stamp()
    if args.run_tag:
        run_id = f"{run_id}_{re.sub(r'[^A-Za-z0-9_]+', '_', args.run_tag).strip('_')}"
    model_slug = _model_slug(args.model)
    mode_slug = "repaired" if args.mode == "single_pass" else "repaired_iterative"
    out_root = args.output_base_dir / model_slug / "stage2_scenarios" / mode_slug / run_id
    attempts_dir = out_root / "rewrite_attempts"
    out_root.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "mode": args.mode,
        "model": args.model,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "max_completion_tokens": args.max_completion_tokens,
        "max_per_surface_value": args.max_per_surface_value,
        "strict_schema": args.strict_schema,
        "source_set_b_dir": str(args.source_set_b_dir.resolve()),
        "stage1_dir": str(args.stage1_dir.resolve()),
        "spec_path": str(args.spec_path.resolve()),
        "statements": statement_ids,
        "scenario_counts": {sid: len(by_sid[sid]) for sid in statement_ids},
        "git_commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"=> output dir: {out_root}")
    for sid in statement_ids:
        print(f"   {sid}: {len(by_sid[sid])} source scenarios")

    if args.dry_run:
        sample_sid = statement_ids[0]
        prompt = make_stage2_repair_singlepass_prompt(
            statement_record=spec[sample_sid],
            understanding_record=understandings[sample_sid],
            scenarios=by_sid[sample_sid],
        )
        print(f"\n=== DRY RUN: first 1200 chars of prompt for {sample_sid} ===\n")
        print(prompt[:1200])
        print(f"\n=== full prompt length: {len(prompt)} chars ===")
        return

    client = None  # unused — lm_client.call_lm() lazy-inits per-backend
    log = RawAPILogger(experiment_name="stage2_repair_singlepass", base_dir=out_root / "raw_api_log")

    out_scenarios_path = out_root / "scenarios.jsonl"
    out_failures_path = out_root / "parse_failures.jsonl"

    # Anthropic batch path: skip the ThreadPoolExecutor and submit all statements as
    # a single batch. Only supported for single_pass (iterative is sequential by design).
    if args.model.lower().startswith("claude-") and args.mode == "single_pass":
        n_admitted, n_failed, admitted, failures = run_anthropic_batch_singlepass(
            spec=spec,
            understandings=understandings,
            by_sid=by_sid,
            statement_ids=statement_ids,
            out_root=out_root,
            model=args.model,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            max_per_surface_value=args.max_per_surface_value,
            strict_schema=args.strict_schema,
            attempts_dir=attempts_dir,
        )
        with out_scenarios_path.open("w") as fout:
            for rec in admitted:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with out_failures_path.open("w") as ferr:
            for rec in failures:
                ferr.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["n_admitted"] = n_admitted
        manifest["n_failed_statements"] = n_failed
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\n== summary ==\nadmitted scenarios: {n_admitted}\nfailed statements:  {n_failed}\nout: {out_root}")
        return

    n_admitted = 0
    n_failed = 0
    with (
        out_scenarios_path.open("w") as fout,
        out_failures_path.open("w") as ferr,
        ThreadPoolExecutor(max_workers=args.max_workers) as ex,
    ):
        def _submit(sid: str):
            common = dict(
                client=client,
                log=log,
                statement=spec[sid],
                understanding=understandings[sid],
                source_scenarios=by_sid[sid],
                model=args.model,
                temperature=args.temperature,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
                attempts_dir=attempts_dir,
            )
            if args.mode == "single_pass":
                return ex.submit(
                    process_statement_sync,
                    max_per_surface_value=args.max_per_surface_value,
                    strict_schema=args.strict_schema,
                    **common,
                )
            return ex.submit(process_statement_iterative, **common)

        futs = {_submit(sid): sid for sid in statement_ids}
        for fut in as_completed(futs):
            sid = futs[fut]
            res = fut.result()
            if not res.success:
                n_failed += 1
                ferr.write(
                    json.dumps(
                        {
                            "statement_id": res.statement_id,
                            "n_source": res.n_source,
                            "attempts": res.attempts,
                            "error": res.last_error,
                            "last_raw_truncated": (res.last_raw or "")[:500],
                        }
                    )
                    + "\n"
                )
                print(f"   [FAIL] {sid}: {res.last_error}")
                continue

            assert res.rewritten is not None
            for rew, src in zip(res.rewritten, by_sid[sid], strict=True):
                rec = {
                    "statement_id": sid,
                    "scenario_n": rew["scenario_n"],
                    "scenario_id": rew["scenario_id"],
                    "is_default_scenario": rew["is_default_scenario"],
                    "varied_axis": rew["varied_axis"],
                    "varied_value": rew["varied_value"],
                    "strategy": "repair_singlepass",
                    "source_strategy": src.get("strategy", "one_axis_at_a_time_from_default"),
                    "scenario_text": rew["scenario_text"],
                    "user_query": rew["user_query"],
                    "system_prompt": rew["system_prompt"],
                    "rubric": rew["rubric"],
                }
                # Pass through LM self-report fields when present (strict-schema runs).
                if "contains_placeholder" in rew:
                    rec["contains_placeholder"] = rew["contains_placeholder"]
                    rec["placeholder_notes"] = rew.get("placeholder_notes", "")
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_admitted += 1
            print(f"   [OK]   {sid}: {len(res.rewritten)} rewritten in {res.attempts} attempt(s)")

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_admitted"] = n_admitted
    manifest["n_failed_statements"] = n_failed
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n== summary ==\nadmitted scenarios: {n_admitted}\nfailed statements:  {n_failed}\nout: {out_root}")


if __name__ == "__main__":
    main()
