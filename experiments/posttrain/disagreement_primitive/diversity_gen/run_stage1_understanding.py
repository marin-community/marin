# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  -- long log strings + CLI help text

"""Stage 1 (understanding) orchestrator for the multi-LM diversity-gen pipeline.

See `.agents/logbooks/dart.md` §11.6 for design context.

Two execution modes:
  - sync  (recommended for smoke / small batches; immediate)
  - batch (recommended default for production; ~50% cost; ≤24h SLA)

Both modes:
  - Use the marin-derived JSON prompt in `prompts.py` (statement + section +
    text + positive/negative examples).
  - Save EVERY raw API response through `raw_api_logger` to
    `results/raw/diversity_gen_<model_slug>_stage1/<ts>/<role>/...`
  - Save parsed records to `<output-base-dir>/<run_id>/understandings.jsonl`.
  - Save per-attempt pretty-printed raw outputs to `<run_id>/attempts/<sid>/`.
  - Save parse failures (after retries) to `<run_id>/parse_failures.jsonl`.

Layout under `<output-base-dir>/<run_id>/`:
    manifest.json                    run metadata (model, temperature, statements, git_commit)
    understandings.jsonl             one parsed record per successful statement
    parse_failures.jsonl             one record per statement that failed all retries
    stage_status.json                {statement_id -> "succeeded"|"failed"|"pending"}
    attempts/<statement_id>/attempt_<N>__raw.json     per-retry raw responses

Usage:
    # Smoke on 2 statements, sync mode:
    set -a; source .env; set +a
    PYENV_VERSION=3.12.0 uv run python -m experiments.posttrain.disagreement_primitive.diversity_gen.run_stage1_understanding \\
        --mode sync \\
        --statements be_engaging,refusal_style

    # Full 46, default (batch) mode:
    set -a; source .env; set +a
    PYENV_VERSION=3.12.0 uv run python -m experiments.posttrain.disagreement_primitive.diversity_gen.run_stage1_understanding

NEVER pass `.env` as a positional argument or include its contents in logs.
The script reads `OPENAI_API_KEY` from `os.environ` only.
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

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_api_logger import RawAPILogger

from experiments.posttrain.disagreement_primitive.diversity_gen.parse_understanding import (
    parse_understanding_response,
)
from experiments.posttrain.disagreement_primitive.diversity_gen.prompts import (
    STAGE1_SYSTEM_PROMPT,
    make_stage1_user_prompt,
)

WORKTREE = Path(__file__).resolve().parents[4]
DEFAULT_SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
DEFAULT_OUTPUT_ROOT = WORKTREE / "experiments/posttrain/disagreement_primitive/diversity_gen"


def _model_slug(model: str) -> str:
    """`gpt-5.1` -> `gpt5_1`, `claude-sonnet-4-6` -> `claude_sonnet_4_6`."""
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def _now_stamp() -> str:
    """`YYYYMMDDTHHMMSSZ` (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(WORKTREE))
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_spec(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _filter_statements(spec: list[dict[str, Any]], statements_filter: list[str] | None) -> list[dict[str, Any]]:
    if not statements_filter:
        return spec
    wanted = set(statements_filter)
    have = {s["id"] for s in spec}
    missing = wanted - have
    if missing:
        raise SystemExit(f"--statements references unknown ids: {sorted(missing)}")
    return [s for s in spec if s["id"] in wanted]


@dataclass
class StatementResult:
    statement_id: str
    success: bool
    attempts: int
    parsed: dict[str, Any] | None
    last_error: str | None
    last_raw: str | None
    raw_seq: list[str]  # raw_api_logger sequence ids per attempt


def _save_attempt(attempts_dir: Path, statement_id: str, attempt: int, content: str, error: str | None) -> None:
    sub = attempts_dir / statement_id
    sub.mkdir(parents=True, exist_ok=True)
    rec = {
        "statement_id": statement_id,
        "attempt": attempt,
        "raw_response": content,
        "error": error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (sub / f"attempt_{attempt:02d}__raw.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))


def process_statement_sync(
    client: OpenAI,
    log: RawAPILogger,
    statement: dict[str, Any],
    model: str,
    temperature: float,
    max_retries: int,
    attempts_dir: Path,
) -> StatementResult:
    sid = statement["id"]
    user_prompt = make_stage1_user_prompt(statement)
    last_err: str | None = None
    last_content: str | None = None
    raw_seq: list[str] = []

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = log.call(
                role="stage1_understanding",
                key={"statement_id": sid, "attempt": attempt, "model": model},
                fn=lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_completion_tokens=4000,
                    reasoning_effort="none",
                    response_format={"type": "json_object"},
                ),
            )
        except Exception as exc:
            last_err = f"api_error: {type(exc).__name__}: {exc}"
            print(f"  [{sid}] attempt {attempt} api error: {exc}", flush=True)
            _save_attempt(attempts_dir, sid, attempt, content="", error=last_err)
            time.sleep(1 + attempt)
            continue

        content = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        last_content = content
        # raw_api_logger files are timestamped with seq; we record the elapsed
        # time as a proxy. Full provenance is in results/raw/...
        raw_seq.append(f"attempt_{attempt}_t{time.time() - t0:.1f}s")

        try:
            parsed = parse_understanding_response(content)
        except ValueError as exc:
            last_err = f"parse_error: {exc}"
            print(f"  [{sid}] attempt {attempt} parse error: {exc}", flush=True)
            _save_attempt(attempts_dir, sid, attempt, content=content, error=last_err)
            time.sleep(1 + attempt)
            continue

        # save the successful attempt too so the audit trail is uniform
        _save_attempt(attempts_dir, sid, attempt, content=content, error=None)
        print(
            f"  [{sid}] attempt {attempt} OK ({len(parsed.get('variation_axes', []))} axes, "
            f"{time.time() - t0:.1f}s)",
            flush=True,
        )
        return StatementResult(
            statement_id=sid,
            success=True,
            attempts=attempt,
            parsed=parsed,
            last_error=None,
            last_raw=content,
            raw_seq=raw_seq,
        )

    return StatementResult(
        statement_id=sid,
        success=False,
        attempts=max_retries,
        parsed=None,
        last_error=last_err,
        last_raw=last_content,
        raw_seq=raw_seq,
    )


def _build_record(
    statement: dict[str, Any], result: StatementResult, model: str, temperature: float, mode: str
) -> dict[str, Any]:
    """Assemble the canonical output record for a successful statement."""
    if not result.success or result.parsed is None:
        raise ValueError(f"cannot build record for failed statement {result.statement_id}")
    parsed = result.parsed
    axes = parsed["variation_axes"]
    examples = (statement.get("metadata") or {}).get("examples", []) or []
    return {
        "statement_id": statement["id"],
        "statement_title": statement["id"],
        "section": statement.get("section", ""),
        "subsection": statement.get("subsection", ""),
        "statement_text": statement.get("text", ""),
        "examples_used": [
            {
                "description": ex.get("description", ""),
                "user_query": ex.get("user_query", ""),
                "good_response": ex.get("good_response", ""),
                "bad_response": ex.get("bad_response", ""),
            }
            for ex in examples
        ],
        "behavior_understanding": parsed["behavior_understanding"],
        "scientific_motivation": parsed["scientific_motivation"],
        # `behavior_specific_axes` is the canonical axes list — each element has
        # axis, description, spectrum, why_it_matters, default_spectrum_value.
        # NO demographic axes are merged in (see dart.md §11.6 scope decision).
        "behavior_specific_axes": axes,
        "model": model,
        "temperature": temperature,
        "reasoning_effort": "none",
        "mode": mode,
        "attempt_index": result.attempts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_sync_mode(
    spec: list[dict[str, Any]],
    out_dir: Path,
    model: str,
    temperature: float,
    max_retries: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all statements via sync chat-completions. Returns (successes, failures)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not in environment. Did you `set -a; source .env; set +a` first?")
    client = OpenAI(api_key=api_key)
    log = RawAPILogger(f"diversity_gen_{_model_slug(model)}_stage1")
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"sync mode: {len(spec)} statements x max {max_retries} retries x {workers} workers",
        flush=True,
    )
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def worker(statement: dict[str, Any]) -> StatementResult:
        return process_statement_sync(client, log, statement, model, temperature, max_retries, attempts_dir)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(worker, s): s for s in spec}
        for fut in as_completed(futures):
            statement = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                failures.append(
                    {
                        "statement_id": statement["id"],
                        "error": f"worker_crash: {type(exc).__name__}: {exc}",
                    }
                )
                print(f"  [{statement['id']}] WORKER CRASH: {exc}", flush=True)
                continue
            if result.success:
                successes.append(_build_record(statement, result, model, temperature, "sync"))
            else:
                failures.append(
                    {
                        "statement_id": result.statement_id,
                        "attempts": result.attempts,
                        "error": result.last_error,
                        "last_raw_truncated": (result.last_raw or "")[:500],
                    }
                )
    return successes, failures


def run_batch_mode(
    spec: list[dict[str, Any]],
    out_dir: Path,
    model: str,
    temperature: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all statements via OpenAI Batch API (single shot, no in-batch retry).

    For per-statement retry on parse failures, re-run with `--mode sync
    --statements <failed_ids>` afterward.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not in environment. Did you `set -a; source .env; set +a` first?")
    client = OpenAI(api_key=api_key)
    log = RawAPILogger(f"diversity_gen_{_model_slug(model)}_stage1")
    attempts_dir = out_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    # Assemble batch JSONL
    input_path = out_dir / "batch_input.jsonl"
    cmap: dict[str, dict[str, Any]] = {}
    with input_path.open("w") as fh:
        for s in spec:
            sid = s["id"]
            custom_id = f"stage1__{sid}"
            cmap[custom_id] = {"statement_id": sid}
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
                    {"role": "user", "content": make_stage1_user_prompt(s)},
                ],
                "temperature": temperature,
                "max_completion_tokens": 4000,
                "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
            }
            fh.write(
                json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body})
                + "\n"
            )

    (out_dir / "batch_custom_id_map.json").write_text(json.dumps(cmap, indent=2))
    print(f"batch input written to {input_path}", flush=True)

    # Submit
    upload = client.files.create(file=input_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"diversity_gen stage1 understanding ({model})"},
    )
    (out_dir / "batch_state.json").write_text(json.dumps({"batch_id": batch.id, "input_file_id": upload.id}, indent=2))
    print(f"submitted batch {batch.id}", flush=True)

    # Poll
    terminal = {"completed", "failed", "expired", "cancelled"}
    while batch.status not in terminal:
        time.sleep(30)
        batch = client.batches.retrieve(batch.id)
        print(
            f"  poll: status={batch.status} counts=total={batch.request_counts.total if batch.request_counts else '?'} "
            f"completed={batch.request_counts.completed if batch.request_counts else '?'} "
            f"failed={batch.request_counts.failed if batch.request_counts else '?'}",
            flush=True,
        )

    if batch.status != "completed":
        raise SystemExit(f"batch ended in non-completed status: {batch.status}")

    output_file_id = batch.output_file_id
    if not output_file_id:
        raise SystemExit("batch completed but no output_file_id")

    output_blob = client.files.content(output_file_id).text
    (out_dir / "batch_output.jsonl").write_text(output_blob)

    # Parse each line, route through raw_api_logger for archival
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for line in output_blob.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        custom_id = entry.get("custom_id", "?")
        sid = cmap.get(custom_id, {}).get("statement_id", custom_id)
        statement = next((s for s in spec if s["id"] == sid), None)
        if statement is None:
            failures.append({"statement_id": sid, "error": "no matching spec record"})
            continue

        # Re-route the batch response through raw_api_logger for uniform archival
        log.call(
            role="stage1_understanding_batch",
            key={"statement_id": sid, "custom_id": custom_id, "model": model},
            fn=lambda e=entry: e,  # logger just persists the dict
        )

        response = entry.get("response") or {}
        body = response.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            failures.append({"statement_id": sid, "error": "no choices in batch response", "raw_line": line[:500]})
            _save_attempt(attempts_dir, sid, 1, content="", error="no choices")
            continue
        content = (choices[0].get("message") or {}).get("content") or ""
        try:
            parsed = parse_understanding_response(content)
        except ValueError as exc:
            failures.append({"statement_id": sid, "error": f"parse_error: {exc}", "last_raw_truncated": content[:500]})
            _save_attempt(attempts_dir, sid, 1, content=content, error=f"parse_error: {exc}")
            continue

        _save_attempt(attempts_dir, sid, 1, content=content, error=None)
        result = StatementResult(
            statement_id=sid,
            success=True,
            attempts=1,
            parsed=parsed,
            last_error=None,
            last_raw=content,
            raw_seq=["batch_attempt_1"],
        )
        successes.append(_build_record(statement, result, model, temperature, "batch"))

    return successes, failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 1 (understanding) for diversity-gen pipeline.")
    p.add_argument(
        "--mode", choices=["sync", "batch"], default="batch", help="execution mode (default: batch — 50%% off, ≤24h SLA)"
    )
    p.add_argument("--model", default="gpt-5.1", help="LM to use (default: gpt-5.1)")
    p.add_argument("--temperature", type=float, default=1.0, help="generation temperature (default: 1.0 per marin)")
    p.add_argument("--max-retries", type=int, default=5, help="sync-mode max retries per statement on parse error")
    p.add_argument("--workers", type=int, default=8, help="sync-mode concurrency")
    p.add_argument("--statements", default=None, help="comma-separated statement ids (default: all)")
    p.add_argument(
        "--spec-path", type=Path, default=DEFAULT_SPEC_PATH, help=f"spec JSONL path (default: {DEFAULT_SPEC_PATH})"
    )
    p.add_argument(
        "--output-base-dir",
        type=Path,
        default=None,
        help="output root; defaults to diversity_gen/<model_slug>/stage1_understanding/",
    )
    args = p.parse_args(argv)

    spec = _load_spec(args.spec_path)
    statements_filter = [s.strip() for s in args.statements.split(",") if s.strip()] if args.statements else None
    spec = _filter_statements(spec, statements_filter)
    print(f"loaded {len(spec)} statements from {args.spec_path}", flush=True)
    if not spec:
        raise SystemExit("no statements selected")

    out_base = args.output_base_dir or (DEFAULT_OUTPUT_ROOT / _model_slug(args.model) / "stage1_understanding")
    run_id = _now_stamp()
    out_dir = out_base / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)

    manifest = {
        "run_id": run_id,
        "model": args.model,
        "temperature": args.temperature,
        "reasoning_effort": "none",
        "mode": args.mode,
        "statements_filter": statements_filter,
        "n_statements": len(spec),
        "max_retries": args.max_retries,
        "workers": args.workers,
        "spec_path": str(args.spec_path),
        "git_commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.mode == "sync":
        successes, failures = run_sync_mode(spec, out_dir, args.model, args.temperature, args.max_retries, args.workers)
    else:
        successes, failures = run_batch_mode(spec, out_dir, args.model, args.temperature)

    # Write outputs
    with (out_dir / "understandings.jsonl").open("w") as f:
        for rec in sorted(successes, key=lambda r: r["statement_id"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "parse_failures.jsonl").open("w") as f:
        for rec in sorted(failures, key=lambda r: r["statement_id"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    status = {
        **{rec["statement_id"]: "succeeded" for rec in successes},
        **{rec["statement_id"]: "failed" for rec in failures},
    }
    (out_dir / "stage_status.json").write_text(json.dumps(status, indent=2))

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_succeeded"] = len(successes)
    manifest["n_failed"] = len(failures)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"\nDONE. {len(successes)}/{len(spec)} succeeded, {len(failures)} failed.",
        flush=True,
    )
    print(f"output dir: {out_dir}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
