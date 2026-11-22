# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a finite response-cap cohort, retaining completions and censored attempts."""

import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict

import httpx
import numpy as np
from marin.inference.load_test import RETAINED_METRICS, LoadRequest, measure_request, request_digest, summarize_metrics
from pydantic import TypeAdapter
from rigging.filesystem.storage_path import StoragePath
from rigging.telemetry.prometheus import PrometheusScraper
from rolloutengine.repetition import detect_tail_loop
from taskcompendium.chat import assistant_message
from taskcompendium.models import AnswerFormat, AnswerType, ConversationTrace, GradingAttempt, TextMessage
from taskcompendium.submission import extract_submission

logger = logging.getLogger(__name__)
ANSWER_FORMAT = TypeAdapter(AnswerFormat)


class ResponseCapture:
    """Collect OpenAI deltas, including exact token IDs, without retaining SSE overhead."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.token_ids: list[int] = []
        self.tool_calls: dict[int, dict] = {}

    def add(self, event: dict) -> None:
        for choice in event.get("choices", []):
            self.token_ids.extend(choice.get("token_ids") or [])
            delta = choice.get("delta", {})
            if delta.get("content"):
                self.content.append(delta["content"])
            if delta.get("reasoning_content") or delta.get("reasoning"):
                self.reasoning.append(delta.get("reasoning_content") or delta["reasoning"])
            for tool in delta.get("tool_calls") or []:
                target = self.tool_calls.setdefault(
                    tool["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                if tool.get("id"):
                    target["id"] = tool["id"]
                for key in ("name", "arguments"):
                    target["function"][key] += tool.get("function", {}).get(key) or ""

    def message(self, prefix: str) -> dict:
        message = {"role": "assistant", "content": prefix + "".join(self.content)}
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[index] for index in sorted(self.tool_calls)]
        return message


def response_signals(capture: ResponseCapture, metadata: dict, output_tokens: int | None) -> dict:
    """Measure syntax and periodic tails; these signals do not establish answer correctness."""
    message = capture.message(metadata["assistant_prefix"])
    try:
        event = assistant_message(message)
        attempt = GradingAttempt(
            ConversationTrace(
                events=(TextMessage(role="user", content="Source context retained in request inventory"), event)
            )
        )
        extract_submission(
            ANSWER_FORMAT.validate_python(metadata["answer_format"]),
            attempt,
            answer_type=AnswerType(metadata["answer_type"]),
            context_length=1,
        )
        parseable = True
        parse_error = None
    except ValueError as exc:
        parseable, parse_error = False, str(exc)
    ids = capture.token_ids
    exact_ids = output_tokens is not None and len(ids) == output_tokens
    loop = (
        detect_tail_loop(ids, [1] * len(ids), max_period_tokens=64, tail_tokens=256, minimum_occurrences=4)
        if exact_ids
        else None
    )
    content = message["content"]
    return {
        "source_submission_parseable": parseable,
        "parse_error": parse_error,
        "explicit_final_marker": bool(message.get("tool_calls")) or "\\boxed{" in content or "<|end_think|>" in content,
        "token_id_coverage_complete": exact_ids,
        "hard_tail_loop": None if not exact_ids else loop is not None,
        "hard_tail_loop_span": None if loop is None else asdict(loop),
    }


async def complete_cohort(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    requests: Sequence[LoadRequest],
    metadata: dict,
    *,
    concurrency: int,
    repeats: int,
    max_tokens: int,
    seed: int,
    on_result: Callable[[dict, dict], None],
) -> list[dict]:
    """Issue each prompt/seed pair exactly once; write every result before releasing its slot."""
    prompts = list(requests)
    random.Random(seed).shuffle(prompts)
    order = [(request, repeat) for request in prompts for repeat in range(repeats)]
    queue = iter(order)
    rows = []

    async def drive() -> None:
        for request, repeat in queue:
            identity = f"{request.identity}-r{repeat}"
            capture = ResponseCapture()
            body = {
                **request.body,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "min_tokens": 0,
                "stop_token_ids": [128001, 128009],
                "skip_special_tokens": False,
                "ignore_eos": False,
                "logprobs": True,
                "top_logprobs": 0,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            }
            measurement = await measure_request(
                client,
                base_url,
                model,
                LoadRequest(identity, request.path, body),
                seed=seed + repeat,
                on_event=capture.add,
            )
            row = {
                **asdict(measurement),
                "prompt_identity": request.identity,
                "repeat": repeat,
                "split": metadata[request.identity]["split"],
                "ttft_seconds": measurement.ttft,
                "decode_tokens_per_second": None if not measurement.tpot else 1 / measurement.tpot,
                **response_signals(capture, metadata[request.identity], measurement.output_tokens),
            }
            artifact = {
                **row,
                "message": capture.message(metadata[request.identity]["assistant_prefix"]),
                "reasoning": "".join(capture.reasoning),
                "token_ids": capture.token_ids,
            }
            await asyncio.to_thread(on_result, row, artifact)
            rows.append(row)

    await asyncio.gather(*(drive() for _ in range(min(concurrency, len(order)))))
    return rows


def summarize_cohort(rows: list[dict], expected: int) -> dict:
    successful = [row for row in rows if row["error"] is None]
    lengths = [row["output_tokens"] for row in successful]
    loop_rows = [row for row in successful if row["hard_tail_loop"] is not None]
    length_stops = sum(row["finish_reason"] == "length" for row in successful)
    decode_rates = [row["decode_tokens_per_second"] for row in successful if row["decode_tokens_per_second"] is not None]
    return {
        "expected_attempts": expected,
        "retained_attempts": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "censored_or_unstarted": expected - len(rows),
        "cohort_complete": len(rows) == expected and len(successful) == expected,
        "length_stop_fraction_bounds_all_attempts": [
            length_stops / expected,
            (length_stops + expected - len(successful)) / expected,
        ],
        "length_stop_fraction_of_successful": None if not successful else length_stops / len(successful),
        "response_length": None if not lengths else {f"p{p}": float(np.percentile(lengths, p)) for p in (50, 90, 99)},
        "source_submission_parseable_fraction": (
            None if not successful else sum(row["source_submission_parseable"] for row in successful) / len(successful)
        ),
        "explicit_final_marker_fraction": (
            None if not successful else sum(row["explicit_final_marker"] for row in successful) / len(successful)
        ),
        "loop_observed_attempts": len(loop_rows),
        "hard_tail_loop_fraction": (
            None if not loop_rows else sum(row["hard_tail_loop"] for row in loop_rows) / len(loop_rows)
        ),
        "mean_request_decode_tokens_per_second": None if not decode_rates else float(np.mean(decode_rates)),
    }


async def run_cohort(base_url: str, metrics_url: str, model: str, requests: list[LoadRequest], spec: dict) -> dict:
    root = StoragePath(spec["output_uri"])
    cohort = spec["cohort"]
    metadata = json.loads(StoragePath(cohort["metadata_uri"]).read_text())
    if request_digest(metadata) != cohort["metadata_sha256"]:
        raise ValueError("Cohort metadata changed")
    # Test exact-ID streaming before spending a long window on unobservable loop metrics.
    capture = ResponseCapture()
    probe = LoadRequest(
        "cohort-token-id-qualification",
        requests[0].path,
        {
            **requests[0].body,
            "max_tokens": 8,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "logprobs": True,
            "top_logprobs": 0,
        },
    )
    async with httpx.AsyncClient(timeout=300) as client:
        qualification = await measure_request(client, base_url, model, probe, seed=17, on_event=capture.add)
    (root / "cohort-token-id-qualification.json").write_text(
        json.dumps(
            {
                "measurement": asdict(qualification),
                "token_ids": capture.token_ids,
            }
        )
    )
    if qualification.error is not None or not capture.token_ids or len(capture.token_ids) != qualification.output_tokens:
        raise ValueError("Exact streaming token IDs did not match usage; cannot measure loop share")
    expected = len(requests) * cohort["repeats"]
    rows: list[dict] = []
    samples = []
    first = last = None
    counter_valid = True
    gauge_integrals: dict[str, float] = defaultdict(float)
    gauge_maxima: dict[str, float] = defaultdict(float)
    gauge_series = {}
    stopped = asyncio.Event()
    ready = asyncio.Event()
    scraper = PrometheusScraper(metrics_url, timeout=cohort["metrics_timeout_seconds"])
    started = time.perf_counter()
    logger.info(
        "Starting finite cohort: cap=%s prompts=%s attempts=%s concurrency=%s unix_time=%s",
        cohort["max_tokens"],
        len(requests),
        expected,
        cohort["concurrency"],
        time.time(),
    )

    def retain(row: dict, artifact: dict) -> None:
        (root / "responses" / f"{row['identity']}.json").write_text(json.dumps(artifact))
        rows.append(row)

    def snapshot() -> dict:
        return {
            "time": time.perf_counter(),
            "unix_time": time.time(),
            "series": [
                {"name": sample.name, "labels": sample.labels, "value": float(sample.value)}
                for family in scraper.scrape()
                for sample in family.samples
                if sample.name in RETAINED_METRICS
            ],
        }

    async def observe() -> None:
        nonlocal first, last, counter_valid
        chunk = 0
        while True:
            current = await asyncio.to_thread(snapshot)
            if first is None:
                first = current
                ready.set()
            if last is not None:
                elapsed = current["time"] - last["time"]
                counter_valid &= summarize_metrics([last, current], 16)["output_tokens_per_second"] is not None
                for name in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"):
                    values = [row["value"] for row in last["series"] if row["name"] == name]
                    if values:
                        gauge_integrals[name] += sum(values) / len(values) * elapsed
                        gauge_maxima[name] = max(gauge_maxima[name], max(values))
                        gauge_series[name] = len(values)
            last = current
            samples.append(current)
            if len(samples) >= 6 or stopped.is_set():
                await asyncio.to_thread((root / "metrics" / f"chunk-{chunk:05}.json").write_text, json.dumps(samples))
                samples.clear()
                chunk += 1
                progress = {
                    "elapsed_seconds": time.perf_counter() - started,
                    "summary": summarize_cohort(list(rows), expected),
                    "server": summarize_metrics([first, last], 16),
                    "counter_identity_and_reset_check_passed": counter_valid,
                    "timeweighted_gauges_so_far": {
                        name: value / (last["time"] - first["time"]) for name, value in gauge_integrals.items()
                    },
                }
                await asyncio.to_thread((root / "progress.json").write_text, json.dumps(progress))
            if stopped.is_set():
                return
            try:
                await asyncio.wait_for(stopped.wait(), timeout=10)
            except TimeoutError:
                pass

    # Baseline precedes admission, final scrape follows all completions.
    timed_out = False
    async with asyncio.TaskGroup() as group:
        group.create_task(observe())
        await ready.wait()
        try:
            async with httpx.AsyncClient(
                timeout=cohort["http_idle_timeout"],
                limits=httpx.Limits(
                    max_connections=cohort["concurrency"], max_keepalive_connections=cohort["concurrency"]
                ),
            ) as client:
                try:
                    async with asyncio.timeout(cohort["wall_budget_seconds"]):
                        await complete_cohort(
                            client,
                            base_url,
                            model,
                            requests,
                            metadata,
                            concurrency=cohort["concurrency"],
                            repeats=cohort["repeats"],
                            max_tokens=cohort["max_tokens"],
                            seed=17,
                            on_result=retain,
                        )
                except TimeoutError:
                    timed_out = True
        finally:
            stopped.set()
    assert first is not None and last is not None
    elapsed = last["time"] - first["time"]
    server = summarize_metrics([first, last], 16)
    if not counter_valid:
        server["output_tokens_per_second"] = server["output_tokens_per_second_per_gpu"] = None
    server["generation_counter_series"] = sum(row["name"] == "vllm:generation_tokens_total" for row in last["series"])
    server["counter_identity_and_reset_check_passed"] = counter_valid
    server["timeweighted_gauges"] = {
        name: {
            "mean_per_engine": value / elapsed,
            "maximum_per_engine": gauge_maxima[name],
            "observed_series": gauge_series[name],
        }
        for name, value in gauge_integrals.items()
    }
    result = {
        "source_revision": spec["source_revision"],
        "cohort": cohort,
        "summary": summarize_cohort(rows, expected),
        "splits": {
            split: summarize_cohort(
                [row for row in rows if row["split"] == split],
                sum(meta["split"] == split for meta in metadata.values()) * cohort["repeats"],
            )
            for split in ("eval", "train")
        },
        "server": server,
        "wall_budget_reached": timed_out,
        "requests": rows,
        "notes": [
            "Fixed prompt/seed cohort, each admitted once; unfinished attempts are censored, not length stops.",
            "Source parseability uses normal extraction (Boxed allows raw text); "
            "explicit_final_marker is separate, neither implies correctness.",
            "Hard loop = periodic suffix, period <=64 tokens repeated >=4 times in final256; "
            "circular reasoning requires a retained-output audit.",
            "Whole-cohort rates include startup prefill and declining concurrency at the tail; "
            "use retained metric chunks for saturated-window rates.",
        ],
    }
    (root / "cohort.json").write_text(json.dumps(result, indent=2))
    return result
