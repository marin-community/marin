# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overnight IFBench rollout utilities.

This module keeps the 20k capped rollout run reproducible:

- Gemini 3 Flash thinking sweep on the 100-prompt smoke slice.
- Provider batch submission/poll/download for the 20k slice.
- Strict/loose verifier summaries over Rollout JSONL files.

It intentionally does not launch Iris jobs; those depend on cluster-specific
Iris CLI flags and are recorded separately in the logbook.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import pathlib
from collections.abc import Iterable
from typing import Any

from experiments.ifbench.data.prepare import PreparedRow
from experiments.ifbench.rollout.backend import BatchHandle, BatchStatus, Rollout, RolloutRequest, SamplingConfig
from experiments.ifbench.rollout.extract import extract_pairs_and_sft, write_pairs_jsonl, write_sft_jsonl
from experiments.ifbench.rollout.gemini_backend import GeminiBackend
from experiments.ifbench.rollout.sync_runner import run_sync_batch, write_rollouts_jsonl
from experiments.ifbench.rollout.together_backend import TogetherBackend
from experiments.ifbench.verifiers.parse import parse_ground_truth
from experiments.ifbench.verifiers.scoring import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

logger = logging.getLogger(__name__)

DEFAULT_PREPARED = pathlib.Path("/tmp/ifbench_port/prepared_v2/train.jsonl")
DEFAULT_WORK_DIR = pathlib.Path("/tmp/ifbench_port/overnight_20k")
FLASH_MODEL = "gemini-3-flash-preview"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
LLAMA_8B = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
BASE_SAMPLING = SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=1024)
THINKING_LEVELS = ("minimal", "low", "medium", "high")
POLL_ERROR_STATUS = "poll_error"
TERMINAL_BAD_STATUSES = {BatchStatus.FAILED.value, BatchStatus.CANCELLED.value, BatchStatus.EXPIRED.value}


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()


def load_prompts(path: pathlib.Path, n: int) -> list[PreparedRow]:
    rows: list[PreparedRow] = []
    with path.open() as f:
        for line in f:
            if len(rows) >= n:
                break
            rows.append(PreparedRow(**json.loads(line)))
    if len(rows) < n:
        raise RuntimeError(f"asked for {n} rows but only found {len(rows)} in {path}")
    return rows


def build_requests(rows: Iterable[PreparedRow], model_id: str, sampling: SamplingConfig) -> list[RolloutRequest]:
    return [
        RolloutRequest(
            prompt_id=row.prompt_id,
            model_id=model_id,
            messages=row.messages,
            sampling=sampling,
            seed=0,
        )
        for row in rows
    ]


def read_rollouts_jsonl(path: pathlib.Path) -> list[Rollout]:
    rollouts: list[Rollout] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rollouts.append(Rollout(**json.loads(line)))
    return rollouts


def _input_example(row: PreparedRow) -> InputExample:
    parsed = parse_ground_truth(row.ground_truth)
    return InputExample(
        key=0,
        instruction_id_list=parsed.instruction_id_list,
        prompt=row.messages[0].get("content", ""),
        kwargs=parsed.kwargs,
    )


def summarize_rollouts(rows: list[PreparedRow], rollout_paths: list[pathlib.Path]) -> dict[str, Any]:
    rows_by_prompt = {row.prompt_id: row for row in rows}
    per_model: dict[str, dict[str, Any]] = {}
    all_rollouts: list[Rollout] = []
    for path in rollout_paths:
        rollouts = read_rollouts_jsonl(path)
        all_rollouts.extend(rollouts)
        for rollout in rollouts:
            stats = per_model.setdefault(
                rollout.model_id,
                {
                    "n": 0,
                    "empty": 0,
                    "strict_pass_all": 0,
                    "loose_pass_all": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "paths": set(),
                },
            )
            stats["n"] += 1
            stats["paths"].add(str(path))
            stats["input_tokens"] += rollout.input_tokens or 0
            stats["output_tokens"] += rollout.output_tokens or 0
            stats["thinking_tokens"] += rollout.thinking_tokens or 0
            if not rollout.response_text or not rollout.response_text.strip():
                stats["empty"] += 1
                continue
            row = rows_by_prompt.get(rollout.prompt_id)
            if row is None:
                continue
            inp = _input_example(row)
            strict = test_instruction_following_strict(inp, rollout.response_text)
            loose = test_instruction_following_loose(inp, rollout.response_text)
            stats["strict_pass_all"] += int(strict.follow_all_instructions)
            stats["loose_pass_all"] += int(loose.follow_all_instructions)

    for stats in per_model.values():
        n = stats["n"]
        stats["strict_pass_rate"] = stats["strict_pass_all"] / n if n else 0.0
        stats["loose_pass_rate"] = stats["loose_pass_all"] / n if n else 0.0
        stats["paths"] = sorted(stats["paths"])

    rollouts_by_prompt: dict[str, list[Rollout]] = {}
    for rollout in all_rollouts:
        rollouts_by_prompt.setdefault(rollout.prompt_id, []).append(rollout)
    pairs, sft, extraction = extract_pairs_and_sft(rows, rollouts_by_prompt)
    return {
        "created_at_utc": utc_now(),
        "n_prompts": len(rows),
        "rollout_paths": [str(p) for p in rollout_paths],
        "per_model": per_model,
        "n_pairs": len(pairs),
        "n_sft": len(sft),
        "pair_yield": extraction.pair_yield,
        "n_no_passers": extraction.n_prompts_no_passers,
        "n_no_failers": extraction.n_prompts_no_failers,
        "yield_by_num_constraints": extraction.yield_by_num_constraints,
        "skip_by_num_constraints": extraction.skip_by_num_constraints,
    }


def write_summary(summary: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_sweep_gemini(args: argparse.Namespace) -> None:
    rows = load_prompts(args.prepared, args.n)
    sweep_dir = args.work_dir / "gemini_flash_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for level in THINKING_LEVELS:
        out_path = sweep_dir / f"rollouts_flash_{level}.jsonl"
        if out_path.exists() and not args.force:
            logger.info("Skipping %s because %s already exists", level, out_path)
        else:
            sampling = dataclasses.replace(BASE_SAMPLING, thinking_level=level)
            requests = build_requests(rows, FLASH_MODEL, sampling)
            logger.info("Running Gemini Flash sync sweep level=%s n=%d", level, len(requests))
            rollouts = asyncio.run(run_sync_batch(requests, "gemini", concurrency=args.concurrency))
            write_rollouts_jsonl(rollouts, out_path)
        summary = summarize_rollouts(rows, [out_path])
        summary_path = sweep_dir / f"summary_flash_{level}.json"
        write_summary(summary, summary_path)
        summaries[level] = summary
        model_stats = summary["per_model"].get(FLASH_MODEL, {})
        logger.info(
            "level=%s strict=%s/%s loose=%s/%s input=%s output=%s thinking=%s",
            level,
            model_stats.get("strict_pass_all"),
            model_stats.get("n"),
            model_stats.get("loose_pass_all"),
            model_stats.get("n"),
            model_stats.get("input_tokens"),
            model_stats.get("output_tokens"),
            model_stats.get("thinking_tokens"),
        )

    high_strict = summaries["high"]["per_model"].get(FLASH_MODEL, {}).get("strict_pass_all", -1)
    selected = "high"
    for level in THINKING_LEVELS:
        strict = summaries[level]["per_model"].get(FLASH_MODEL, {}).get("strict_pass_all", -1)
        if strict >= high_strict:
            selected = level
            break
    decision = {
        "created_at_utc": utc_now(),
        "selection_rule": "lowest thinking level with strict_pass_all >= high strict_pass_all",
        "selected_level": selected,
        "levels": {level: summaries[level]["per_model"].get(FLASH_MODEL, {}) for level in THINKING_LEVELS},
    }
    write_summary(decision, sweep_dir / "decision.json")
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def _handle_to_json(handle: BatchHandle, mode: str) -> dict[str, Any]:
    out = dataclasses.asdict(handle)
    out["mode"] = mode
    return out


def _backend_for_handle(handle_data: dict[str, Any]) -> TogetherBackend | GeminiBackend:
    if handle_data["backend"] == "together":
        return TogetherBackend()
    if handle_data["backend"] == "gemini":
        return GeminiBackend(use_file_batch=handle_data.get("mode") == "file")
    raise ValueError(f"unknown backend {handle_data['backend']}")


def cmd_submit_provider_20k(args: argparse.Namespace) -> None:
    rows = load_prompts(args.prepared, args.n)
    handles: list[dict[str, Any]] = []
    batch_dir = args.work_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    args.handles.parent.mkdir(parents=True, exist_ok=True)

    def save_handles() -> None:
        args.handles.write_text(json.dumps(handles, indent=2, ensure_ascii=False))

    for model_id in (LLAMA_70B, LLAMA_8B):
        requests = build_requests(rows, model_id, BASE_SAMPLING)
        logger.info("Submitting Together batch model=%s n=%d", model_id, len(requests))
        handle = TogetherBackend().submit_batch(model_id, requests, batch_dir)
        handles.append(_handle_to_json(handle, "batch"))
        save_handles()

    flash_sampling = dataclasses.replace(BASE_SAMPLING, thinking_level=args.thinking_level)
    flash_requests = build_requests(rows, FLASH_MODEL, flash_sampling)
    if args.thinking_level != "high":
        logger.warning(
            "Gemini file-batch currently rejects thinkingConfig; requested thinking=%s will be omitted "
            "and Gemini will use its default dynamic/high thinking.",
            args.thinking_level,
        )
    logger.info("Submitting Gemini file batch model=%s thinking=%s n=%d", FLASH_MODEL, args.thinking_level, len(rows))
    flash_handle = GeminiBackend(use_file_batch=True).submit_batch(FLASH_MODEL, flash_requests, batch_dir)
    handles.append(_handle_to_json(flash_handle, "file"))
    save_handles()
    print(json.dumps(handles, indent=2, ensure_ascii=False))


def cmd_poll(args: argparse.Namespace) -> None:
    handles = json.loads(args.handles.read_text())
    statuses = []
    for handle_data in handles:
        backend = _backend_for_handle(handle_data)
        handle = BatchHandle(
            backend=handle_data["backend"],
            batch_id=handle_data["batch_id"],
            model_id=handle_data["model_id"],
            submitted_at_iso=handle_data["submitted_at_iso"],
            expected_request_count=handle_data["expected_request_count"],
        )
        try:
            status = backend.poll(handle).value
            status_data = {**handle_data, "status": status, "polled_at_utc": utc_now()}
        except Exception as exc:
            logger.warning("Poll failed for %s %s: %s", handle.backend, handle.batch_id, exc, exc_info=True)
            status_data = {
                **handle_data,
                "status": POLL_ERROR_STATUS,
                "error": f"{type(exc).__name__}: {exc}",
                "polled_at_utc": utc_now(),
            }
        statuses.append(status_data)
    print(json.dumps(statuses, indent=2, ensure_ascii=False))
    if any(s["status"] in {BatchStatus.PENDING.value, POLL_ERROR_STATUS} for s in statuses):
        raise SystemExit(2)
    if any(s["status"] in TERMINAL_BAD_STATUSES for s in statuses):
        raise SystemExit(1)


def cmd_download(args: argparse.Namespace) -> None:
    handles = json.loads(args.handles.read_text())
    out_dir = args.work_dir / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[pathlib.Path] = []
    for handle_data in handles:
        backend = _backend_for_handle(handle_data)
        handle = BatchHandle(
            backend=handle_data["backend"],
            batch_id=handle_data["batch_id"],
            model_id=handle_data["model_id"],
            submitted_at_iso=handle_data["submitted_at_iso"],
            expected_request_count=handle_data["expected_request_count"],
        )
        try:
            status = backend.poll(handle)
        except Exception as exc:
            logger.warning("Skipping download for %s after poll error: %s", handle.batch_id, exc, exc_info=True)
            continue
        if status is not BatchStatus.COMPLETED:
            logger.info("Skipping download for %s status=%s", handle.batch_id, status.value)
            continue
        safe_model = handle.model_id.replace("/", "_")
        out_path = out_dir / f"{handle.backend}_{safe_model}.jsonl"
        write_rollouts_jsonl(backend.download(handle), out_path)
        downloaded.append(out_path)
        logger.info("Downloaded %s → %s", handle.batch_id, out_path)
    print(json.dumps({"downloaded": [str(p) for p in downloaded]}, indent=2))


def cmd_score(args: argparse.Namespace) -> None:
    rows = load_prompts(args.prepared, args.n)
    summary = summarize_rollouts(rows, args.rollouts)
    write_summary(summary, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_extract(args: argparse.Namespace) -> None:
    rows = load_prompts(args.prepared, args.n)
    rollouts_by_prompt: dict[str, list[Rollout]] = {}
    for path in args.rollouts:
        for rollout in read_rollouts_jsonl(path):
            rollouts_by_prompt.setdefault(rollout.prompt_id, []).append(rollout)
    pairs, sft, stats = extract_pairs_and_sft(rows, rollouts_by_prompt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs_jsonl(pairs, args.output_dir / "dpo_pairs.jsonl")
    write_sft_jsonl(sft, args.output_dir / "sft_examples.jsonl")
    summary = {
        "created_at_utc": utc_now(),
        "n_prompts": len(rows),
        "n_pairs": len(pairs),
        "n_sft": len(sft),
        "pair_yield": stats.pair_yield,
        "n_no_passers": stats.n_prompts_no_passers,
        "n_no_failers": stats.n_prompts_no_failers,
        "yield_by_num_constraints": stats.yield_by_num_constraints,
        "skip_by_num_constraints": stats.skip_by_num_constraints,
        "rollout_paths": [str(p) for p in args.rollouts],
    }
    write_summary(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=pathlib.Path, default=DEFAULT_PREPARED)
    parser.add_argument("--work-dir", type=pathlib.Path, default=DEFAULT_WORK_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sweep = sub.add_parser("sweep-gemini")
    sweep.add_argument("-n", type=int, default=100)
    sweep.add_argument("--concurrency", type=int, default=10)
    sweep.add_argument("--force", action="store_true")
    sweep.set_defaults(func=cmd_sweep_gemini)

    submit = sub.add_parser("submit-provider-20k")
    submit.add_argument("-n", type=int, default=20_000)
    submit.add_argument("--thinking-level", choices=THINKING_LEVELS, required=True)
    submit.add_argument("--handles", type=pathlib.Path, default=DEFAULT_WORK_DIR / "handles.json")
    submit.set_defaults(func=cmd_submit_provider_20k)

    poll = sub.add_parser("poll")
    poll.add_argument("--handles", type=pathlib.Path, default=DEFAULT_WORK_DIR / "handles.json")
    poll.set_defaults(func=cmd_poll)

    download = sub.add_parser("download")
    download.add_argument("--handles", type=pathlib.Path, default=DEFAULT_WORK_DIR / "handles.json")
    download.set_defaults(func=cmd_download)

    score = sub.add_parser("score")
    score.add_argument("-n", type=int, default=20_000)
    score.add_argument("--rollouts", type=pathlib.Path, nargs="+", required=True)
    score.add_argument("--output", type=pathlib.Path, required=True)
    score.set_defaults(func=cmd_score)

    extract = sub.add_parser("extract")
    extract.add_argument("-n", type=int, default=20_000)
    extract.add_argument("--rollouts", type=pathlib.Path, nargs="+", required=True)
    extract.add_argument("--output-dir", type=pathlib.Path, required=True)
    extract.set_defaults(func=cmd_extract)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
