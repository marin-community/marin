# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure conversion throughput on a fixed set of sampled source chunks."""

import argparse
import asyncio
import json
import logging
import os
import time

from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import filesystem_for

from experiments.datakit.science_sft_conversion.batch_transport import GLMBatchChatClient
from experiments.datakit.science_sft_conversion.conversion import (
    QUESTION_SOLUTION_SOURCES,
    ConversionMode,
    _document,
    _row_request,
    format_for,
    sources,
)
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, resolve_glm_base_url

logger = logging.getLogger(__name__)


async def benchmark(input_url: str, output_url: str, relay_job: str, concurrency: int) -> None:
    """Save request timings, token counts, and first-attempt validation results."""
    endpoint = resolve_glm_base_url(relay_job).removesuffix("/v1")
    fs, path = filesystem_for(input_url)
    with fs.open(path, "rb") as stream:
        samples = json.load(stream)
    source_by_name = {source.name: source for source in sources()}
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()

    async with GLMBatchChatClient(endpoint, os.environ[GLM_BULK_TOKEN_ENV], batch_size=64, workers=2) as client:

        async def request(sample: dict) -> dict:
            source = source_by_name[sample["source"]]
            selected = format_for(source.name, sample["source_id"], sample["chunk_index"])
            mode = ConversionMode.STANDALONE if source.name in QUESTION_SOLUTION_SOURCES else ConversionMode.GROUNDED
            payload = _row_request(
                source, sample["source_chunk"], sample["chunk_index"], sample["chunk_count"], selected, mode
            )
            async with semaphore:
                request_started = time.monotonic()
                response = await client.post(f"{endpoint}/v1/chat/completions", json=payload)
                response.raise_for_status()
                result = response.json()
                finished = time.monotonic()
            validation_error = None
            try:
                _document(
                    source,
                    sample["source_id"],
                    sample["source_chunk"],
                    sample["chunk_index"],
                    json.loads(result["choices"][0]["message"]["content"]),
                    selected,
                    mode,
                )
            except ValueError as error:
                # First-attempt rejection is measured, rather than repaired, in this load test.
                validation_error = str(error)
            usage = result["usage"]
            logger.info("Completed %s/%s: %s", source.name, sample["source_id"], usage)
            return {
                "source": source.name,
                "source_id": sample["source_id"],
                "started": request_started - started,
                "finished": finished - started,
                "duration": finished - request_started,
                "usage": usage,
                "finish_reason": result["choices"][0]["finish_reason"],
                "validation_error": validation_error,
            }

        results = await asyncio.gather(*(request(sample) for sample in samples))
    elapsed = time.monotonic() - started
    completion_tokens = sum(result["usage"]["completion_tokens"] for result in results)
    report = {
        "relay_job": relay_job,
        "input": input_url,
        "concurrency": concurrency,
        "elapsed": elapsed,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / elapsed,
        "validation_failures": sum(result["validation_error"] is not None for result in results),
        "requests": results,
    }
    fs, path = filesystem_for(output_url)
    with atomic_rename(path, filesystem=fs) as temporary_path:
        with fs.open(temporary_path, "wb") as output:
            output.write(json.dumps(report, indent=2).encode())
    logger.info("Benchmark: %.1f completion tokens/second across %d requests", completion_tokens / elapsed, len(results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--relay-job", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(benchmark(args.input_path, args.output_path, args.relay_job, args.concurrency))


if __name__ == "__main__":
    main()
