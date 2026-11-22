# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure one-token HTTP completions on fresh and partially cached prompts."""

import json
import logging
import random
import statistics
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
PROMPT_LENGTHS = (2048, 8192, 32768, 51840, 64512)
SUFFIX_LENGTH = 256
REPEATS = 5


def _completion(session: requests.Session, endpoint: str, model: str, prompt: list[int]) -> tuple[float, int | None]:
    started = time.perf_counter()
    response = session.post(
        endpoint,
        json={"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0},
        timeout=600,
    )
    seconds = time.perf_counter() - started
    response.raise_for_status()
    result = response.json()
    usage = result["usage"]
    if (usage["prompt_tokens"], usage["completion_tokens"]) != (len(prompt), 1):
        raise RuntimeError(f"Expected {len(prompt)} prompt tokens and one completion token, got {usage}")
    if result["choices"][0]["finish_reason"] not in ("length", "stop"):
        raise RuntimeError(f"Completion did not finish successfully: {result['choices'][0]['finish_reason']}")
    details = usage.get("prompt_tokens_details") or {}
    return seconds, details.get("cached_tokens")


def measure_prefill(base_url: str, model: str, output: Path) -> None:
    """Write sample-level measurements, including the untimed warmup at each shape."""
    generator = random.Random(42)
    records = []
    endpoint = f"{base_url}/completions"
    with requests.Session() as session:
        for length in PROMPT_LENGTHS:
            for prefix_length in (0, length - SUFFIX_LENGTH):
                samples = []
                for repeat in range(REPEATS + 1):
                    # The first block is distinct for each trial, including warmup and priming.
                    prompt = [generator.randrange(256, 10_000) for _ in range(length)]
                    if prefix_length:
                        _completion(session, endpoint, model, prompt[:prefix_length])
                    seconds, cached_tokens = _completion(session, endpoint, model, prompt)
                    record = {
                        "prompt_tokens": length,
                        "primed_prefix_tokens": prefix_length,
                        "reported_cached_tokens": cached_tokens,
                        "seconds": seconds,
                        "warmup": repeat == 0,
                    }
                    records.append(record)
                    output.write_text(json.dumps(records, indent=2) + "\n")
                    if repeat:
                        samples.append(seconds)
                logger.info(
                    "Prefill probe: prompt=%d primed=%d median=%.3fs",
                    length,
                    prefix_length,
                    statistics.median(samples),
                )
