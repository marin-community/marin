#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit Bloom-compatible GPT-4.1 LM-as-judge work via OpenAI Batch API.

This is the local Batch-API equivalent of `run_bloom_judge.py` for eval
inference JSONL files. It intentionally uses the same prompt constructors:

- `build_judge_system_prompt()`
- `build_compliance_judge_prompt(...)`

and the same GPT-4.1 judge settings used by the prior Bloom-compatible runs:

- model: `gpt-4.1-2025-04-14`
- temperature: `0.0`
- max_tokens: `4000`
- no JSON response_format override

The script runs locally only. It never submits Iris jobs.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments" / "posttrain"))
sys.path.insert(0, str(_REPO_ROOT / "lib" / "marin" / "src"))

import batch_lib as bl  # noqa: E402
from marin.alignment.generate_prompts import load_spec  # noqa: E402
from marin.alignment.judge import build_judge_messages  # noqa: E402
from marin.alignment.prompts.judge import (  # noqa: E402
    build_compliance_judge_prompt,
    build_judge_system_prompt,
)
from marin.alignment.types import Statement  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-4.1-2025-04-14"
MAX_TOKENS = 4000
TEMPERATURE = 0.0
SPEC_PATH = _REPO_ROOT / "experiments" / "posttrain" / "specs" / "openai_model_spec.jsonl"
DEFAULT_JOB_ROOT = _REPO_ROOT / "scratch" / "validate_bloom_azero_judge_gpt41_batch_20260425_2218Z"
DEFAULT_INPUT_ROOT = _REPO_ROOT / "scratch" / "validate_bloom_azero_inference_20260425_2213Z"

TARGETS: dict[str, Path] = {
    "azero_lr1e6_seed0_step1699": DEFAULT_INPUT_ROOT / "azero_lr1e6_seed0_step1699" / "inference.jsonl",
    "azero_lr1e5_seed0_step1699": DEFAULT_INPUT_ROOT / "azero_lr1e5_seed0_step1699" / "inference.jsonl",
    "azero_lr8p75e6_seed0_step1699": DEFAULT_INPUT_ROOT / "azero_lr8p75e6_seed0_step1699" / "inference.jsonl",
    "azero_lr5e6_seed0_step1699": DEFAULT_INPUT_ROOT / "azero_lr5e6_seed0_step1699" / "inference.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def target_job_dir(job_root: Path, target: str) -> Path:
    return job_root / target


def _statement_record(statement: Statement) -> dict[str, Any]:
    return {
        "id": statement.id,
        "text": statement.text,
        "type": statement.type.value,
        "authority_level": statement.authority_level.value,
        "section": statement.section,
        "subsection": statement.subsection,
    }


def build_requests(
    *,
    target: str,
    inference_path: Path,
    statements: dict[str, Statement],
    require_rubric: bool,
    max_items: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build Batch API requests with Bloom-compatible GPT-4.1 judge prompts."""
    records = read_jsonl(inference_path)
    logger.info("target=%s loaded %d inference records from %s", target, len(records), inference_path)

    by_statement: dict[str, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for source_idx, record in enumerate(records):
        by_statement[record.get("behavior_id", "")].append((source_idx, record))

    system_prompt = build_judge_system_prompt()
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    skipped_no_statement = 0
    skipped_no_response = 0
    skipped_no_rubric = 0
    prompt_mismatch = 0

    for statement_id in sorted(by_statement):
        statement = statements.get(statement_id)
        rows = by_statement[statement_id]
        if statement is None:
            skipped_no_statement += len(rows)
            continue

        for source_idx, record in rows:
            if max_items is not None and len(requests) >= max_items:
                break

            response_text = (record.get("response_text") or "").strip()
            if not response_text:
                skipped_no_response += 1
                continue

            rubric = (record.get("rubric") or "").strip() or None
            if require_rubric and not rubric:
                skipped_no_rubric += 1
                continue

            user_message = record.get("user_message", "")
            user_prompt = build_compliance_judge_prompt(
                statement=statement,
                user_input=user_message,
                model_response=response_text,
                question_rubric=rubric,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            reference_messages = build_judge_messages(
                statement=statement,
                user_message=user_message,
                model_response=response_text,
                rubric=rubric,
            )
            if messages != reference_messages:
                prompt_mismatch += 1

            custom_id = f"{target}::{len(requests):07d}"
            requests.append(
                bl.build_request(
                    custom_id=custom_id,
                    model=JUDGE_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
            )
            manifest.append(
                {
                    "custom_id": custom_id,
                    "source_idx": source_idx,
                    "prompt_id": record.get("prompt_id", ""),
                    "behavior_id": statement_id,
                    "sample_idx": record.get("sample_idx", 0),
                    "model": record.get("model", ""),
                    "user_message": user_message,
                    "response_text": response_text,
                    "rubric": rubric,
                    "statement": _statement_record(statement),
                }
            )
        if max_items is not None and len(requests) >= max_items:
            break

    audit = {
        "target": target,
        "inference_path": str(inference_path),
        "judge_model": JUDGE_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "require_rubric": require_rubric,
        "source_records": len(records),
        "request_count": len(requests),
        "skipped_no_statement": skipped_no_statement,
        "skipped_no_response": skipped_no_response,
        "skipped_no_rubric": skipped_no_rubric,
        "prompt_mismatch_against_marin_judge_build_messages": prompt_mismatch,
        "built_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    return requests, manifest, audit


def build_job_files(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    if args.target not in TARGETS:
        raise SystemExit(f"unknown target={args.target!r}; choices={sorted(TARGETS)}")

    inference_path = Path(args.inference_path) if args.inference_path else TARGETS[args.target]
    if not inference_path.exists():
        raise SystemExit(f"inference path does not exist: {inference_path}")

    statements = load_spec(str(args.spec_path))
    logger.info("loaded %d statements from %s", len(statements), args.spec_path)

    requests, manifest, audit = build_requests(
        target=args.target,
        inference_path=inference_path,
        statements=statements,
        require_rubric=args.require_rubric,
        max_items=args.max_items,
    )
    if not requests:
        raise SystemExit("built zero requests; refusing to submit")
    if audit["prompt_mismatch_against_marin_judge_build_messages"] != 0:
        raise SystemExit(f"prompt mismatch audit failed: {audit}")

    job_dir = target_job_dir(args.job_root, args.target)
    job_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(job_dir / "manifest.jsonl", manifest)
    write_json(job_dir / "audit.json", audit)
    logger.info("target=%s built %d requests; audit=%s", args.target, len(requests), audit)
    return job_dir, requests, audit


def cmd_build(args: argparse.Namespace) -> None:
    job_dir, requests, audit = build_job_files(args)
    input_path = job_dir / "requests.jsonl"
    write_jsonl(input_path, requests)
    size_mb = input_path.stat().st_size / (1024 * 1024)
    logger.info("wrote %s (%d requests, %.1f MB)", input_path, len(requests), size_mb)
    if size_mb > bl.MAX_FILE_MB:
        raise SystemExit(f"request file too large for Batch API: {size_mb:.1f} MB")
    if len(requests) > bl.MAX_REQUESTS_PER_BATCH:
        raise SystemExit(f"too many requests for Batch API: {len(requests)}")
    print(json.dumps({"job_dir": str(job_dir), "requests_path": str(input_path), "size_mb": size_mb, **audit}, indent=2))


def cmd_submit(args: argparse.Namespace) -> None:
    job_dir, requests, audit = build_job_files(args)
    state_path = job_dir / "batch_state.json"
    if state_path.exists() and not args.force:
        raise SystemExit(f"{state_path} already exists; refusing to resubmit without --force")

    client = OpenAI()
    state = bl.submit(
        client,
        requests,
        job_dir,
        metadata={
            "project": "validate_bloom_azero",
            "target": args.target,
            "judge_model": JUDGE_MODEL,
            "num_items": str(len(requests)),
            "prompt_path": "marin.alignment.prompts.judge",
        },
    )
    state["audit"] = audit
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    logger.info("submitted target=%s batch_id=%s status=%s", args.target, state["batch_id"], state["status"])
    print(json.dumps(state, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    client = OpenAI()
    state = bl.refresh_status(client, target_job_dir(args.job_root, args.target))
    print(json.dumps(state, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--target", required=True, choices=sorted(TARGETS))
        subparser.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
        subparser.add_argument("--inference-path", type=Path, default=None)
        subparser.add_argument("--spec-path", type=Path, default=SPEC_PATH)
        subparser.add_argument("--max-items", type=int, default=None)
        subparser.add_argument("--require-rubric", action="store_true", default=True)
        subparser.add_argument("--no-require-rubric", dest="require_rubric", action="store_false")

    build_parser = subparsers.add_parser("build", help="Build local requests/manifest/audit without submitting.")
    add_common(build_parser)
    build_parser.set_defaults(func=cmd_build)

    submit_parser = subparsers.add_parser("submit", help="Build and submit one target to the OpenAI Batch API.")
    add_common(submit_parser)
    submit_parser.add_argument("--force", action="store_true")
    submit_parser.set_defaults(func=cmd_submit)

    status_parser = subparsers.add_parser("status", help="Refresh status for one submitted target.")
    status_parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    status_parser.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
