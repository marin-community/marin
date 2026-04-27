# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IFBench smoke driver: 100 prompts x 3-model pool via batch APIs.

Does the full Stage 2 → Stage 3 → Stage 4 flow end-to-end. Saves batch
handles to disk so a connection drop doesn't lose the submission.

Usage:
    source .env  # loads TOGETHER_API_KEY + GEMINI_API_KEY
    python -m experiments.ifbench.rollout.smoke submit  # submit all 3 batches
    python -m experiments.ifbench.rollout.smoke poll    # check status
    python -m experiments.ifbench.rollout.smoke run     # download + extract + report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import time
from typing import Any

from experiments.ifbench.data.prepare import PreparedRow
from experiments.ifbench.rollout.backend import (
    BatchHandle,
    BatchStatus,
    Rollout,
    RolloutBackend,
    RolloutRequest,
    SamplingConfig,
)
from experiments.ifbench.rollout.extract import (
    extract_pairs_and_sft,
    write_pairs_jsonl,
    write_sft_jsonl,
)
from experiments.ifbench.rollout.gemini_backend import GeminiBackend
from experiments.ifbench.rollout.together_backend import TogetherBackend

logger = logging.getLogger(__name__)

# v1 pool (D-007). Each entry: (model_id, backend instance, sampling overrides).
_BASE_SAMPLING = SamplingConfig(temperature=0.7, top_p=0.95, max_new_tokens=1024)
_GEMINI_SAMPLING = dataclasses.replace(_BASE_SAMPLING, thinking_level="high")


def _v1_pool() -> list[tuple[str, RolloutBackend, SamplingConfig]]:
    return [
        ("gemini-3-flash-preview", GeminiBackend(), _GEMINI_SAMPLING),
        ("meta-llama/Llama-3.3-70B-Instruct-Turbo", TogetherBackend(), _BASE_SAMPLING),
        ("meta-llama/Meta-Llama-3-8B-Instruct-Lite", TogetherBackend(), _BASE_SAMPLING),
    ]


def _load_prompts(prepared_jsonl: pathlib.Path, n: int) -> list[PreparedRow]:
    out: list[PreparedRow] = []
    with prepared_jsonl.open() as f:
        for line in f:
            if len(out) >= n:
                break
            row = json.loads(line)
            out.append(PreparedRow(**row))
    if len(out) < n:
        raise RuntimeError(f"asked for {n} rows but only {len(out)} available in {prepared_jsonl}")
    return out


def _build_requests(prompts: list[PreparedRow], model_id: str, sampling: SamplingConfig) -> list[RolloutRequest]:
    return [
        RolloutRequest(
            prompt_id=p.prompt_id,
            model_id=model_id,
            messages=p.messages,
            sampling=sampling,
            seed=0,
        )
        for p in prompts
    ]


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit all 3 batches and persist handles to args.state."""
    prepared = _load_prompts(args.prepared, args.n)
    logger.info("Loaded %d prompts from %s", len(prepared), args.prepared)

    handles: list[dict[str, Any]] = []
    for model_id, backend, sampling in _v1_pool():
        if args.skip_model and model_id in args.skip_model:
            logger.info("Skipping model %s by --skip-model", model_id)
            continue
        requests = _build_requests(prepared, model_id, sampling)
        logger.info("Submitting %d requests to %s via %s …", len(requests), model_id, backend.name)
        handle = backend.submit_batch(model_id, requests, args.work_dir / "batches")
        handles.append(dataclasses.asdict(handle))
        logger.info("  → batch_id=%s", handle.batch_id)

    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(handles, indent=2))
    logger.info("Wrote %d handles to %s", len(handles), args.state)


def cmd_poll(args: argparse.Namespace) -> None:
    """Print current status of each batch, return non-zero if any are still PENDING."""
    handles_data = json.loads(args.state.read_text())
    pool_by_name = {b.name: b for _, b, _ in _v1_pool()}
    any_pending = False
    for h in handles_data:
        backend = pool_by_name[h["backend"]]
        handle = BatchHandle(**h)
        status = backend.poll(handle)
        print(f"{h['backend']:10s}  {h['model_id']:50s}  {status.value}")
        if status is BatchStatus.PENDING:
            any_pending = True
    if any_pending:
        sys.exit(2)


def _wait_for_all(handles_data: list[dict[str, Any]], poll_interval: float, timeout: float) -> None:
    pool_by_name = {b.name: b for _, b, _ in _v1_pool()}
    started = time.time()
    while True:
        statuses: list[BatchStatus] = []
        for h in handles_data:
            backend = pool_by_name[h["backend"]]
            handle = BatchHandle(**h)
            statuses.append(backend.poll(handle))
        if all(
            s in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED)
            for s in statuses
        ):
            return
        if time.time() - started > timeout:
            raise RuntimeError(f"timed out after {timeout}s with statuses={statuses}")
        time.sleep(poll_interval)


def cmd_run(args: argparse.Namespace) -> None:
    """Wait for all batches, download, score, extract pairs + SFT, write outputs."""
    handles_data = json.loads(args.state.read_text())

    if args.wait:
        logger.info(
            "Waiting for all batches to complete (poll every %ds, timeout %ds) …", args.poll_interval, args.timeout
        )
        _wait_for_all(handles_data, args.poll_interval, args.timeout)

    pool_by_name = {b.name: b for _, b, _ in _v1_pool()}
    rollouts_by_prompt: dict[str, list[Rollout]] = {}
    per_model_summary: dict[str, dict[str, Any]] = {}

    for h in handles_data:
        backend = pool_by_name[h["backend"]]
        handle = BatchHandle(**h)
        status = backend.poll(handle)
        if status is not BatchStatus.COMPLETED:
            logger.warning("Skipping batch %s (status=%s)", handle.batch_id, status.value)
            per_model_summary[h["model_id"]] = {"status": status.value, "rollouts": 0}
            continue

        n_rollouts = 0
        thinking_total = 0
        output_total = 0
        input_total = 0
        empty_or_err = 0
        for rollout in backend.download(handle):
            rollouts_by_prompt.setdefault(rollout.prompt_id, []).append(rollout)
            n_rollouts += 1
            input_total += rollout.input_tokens or 0
            output_total += rollout.output_tokens or 0
            thinking_total += rollout.thinking_tokens or 0
            if not rollout.response_text or not rollout.response_text.strip():
                empty_or_err += 1
        per_model_summary[h["model_id"]] = {
            "status": "completed",
            "rollouts": n_rollouts,
            "input_tokens_sum": input_total,
            "output_tokens_sum": output_total,
            "thinking_tokens_sum": thinking_total,
            "empty_or_error": empty_or_err,
            "backend": h["backend"],
        }
        logger.info(
            "%s: %d rollouts, %d input + %d output (+%d thinking), %d empty",
            h["model_id"],
            n_rollouts,
            input_total,
            output_total,
            thinking_total,
            empty_or_err,
        )

    # Reload prompts so we can run extraction
    prepared = _load_prompts(args.prepared, args.n)
    pairs, sft, stats = extract_pairs_and_sft(prepared, rollouts_by_prompt)

    out_dir = args.work_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pairs_jsonl(pairs, out_dir / "dpo_pairs.jsonl")
    write_sft_jsonl(sft, out_dir / "sft_examples.jsonl")

    # Cost calc (batch pricing, current).
    pricing = {
        "gemini-3-flash-preview": {"input": 0.25, "output": 1.50},  # batch
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.44, "output": 0.44},  # batch ~50% of $0.88
        "meta-llama/Meta-Llama-3-8B-Instruct-Lite": {"input": 0.05, "output": 0.05},  # batch ~50% of $0.10
    }
    total_cost = 0.0
    cost_breakdown: dict[str, float] = {}
    for model_id, stats_d in per_model_summary.items():
        if model_id not in pricing or stats_d.get("status") != "completed":
            continue
        # Gemini bills thinking as output.
        out_for_billing = stats_d["output_tokens_sum"] + stats_d["thinking_tokens_sum"]
        cost = (
            stats_d["input_tokens_sum"] / 1_000_000 * pricing[model_id]["input"]
            + out_for_billing / 1_000_000 * pricing[model_id]["output"]
        )
        cost_breakdown[model_id] = round(cost, 4)
        total_cost += cost

    summary = {
        "n_prompts": len(prepared),
        "n_pairs": len(pairs),
        "n_sft": len(sft),
        "pair_yield": round(stats.pair_yield, 4),
        "n_no_passers": stats.n_prompts_no_passers,
        "n_no_failers": stats.n_prompts_no_failers,
        "yield_by_num_constraints": stats.yield_by_num_constraints,
        "skip_by_num_constraints": stats.skip_by_num_constraints,
        "per_model": per_model_summary,
        "cost_usd_estimated": round(total_cost, 4),
        "cost_breakdown_usd": cost_breakdown,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 70)
    print(f"SMOKE SUMMARY - {len(prepared)} prompts x {len(per_model_summary)} models")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["submit", "poll", "run"])
    p.add_argument("--prepared", type=pathlib.Path, default=pathlib.Path("/tmp/ifbench_port/prepared_v2/train.jsonl"))
    p.add_argument("--work-dir", type=pathlib.Path, default=pathlib.Path("/tmp/ifbench_port/smoke"))
    p.add_argument("--state", type=pathlib.Path, default=pathlib.Path("/tmp/ifbench_port/smoke/handles.json"))
    p.add_argument("-n", type=int, default=100)
    p.add_argument(
        "--skip-model", action="append", default=None, help="Skip a specific model_id (can pass multiple times)"
    )
    p.add_argument("--wait", action="store_true", default=True, help="Poll until all batches finish (run only)")
    p.add_argument("--no-wait", dest="wait", action="store_false")
    p.add_argument("--poll-interval", type=float, default=15.0)
    p.add_argument("--timeout", type=float, default=3600.0)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "poll":
        cmd_poll(args)
    elif args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
