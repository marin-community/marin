# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample science-forward source chunks for conversion quality review."""

import argparse
import asyncio
import json
import logging
import random

import httpx
import pyarrow.parquet as pq
from iris.client.client import iris_ctx
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import filesystem_for

from experiments.datakit.science_sft_conversion.conversion import (
    ENDPOINT,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    Source,
    _convert_chunk,
    _source_files,
    sources,
    split_source,
)

logger = logging.getLogger(__name__)


def _source_samples(source: Source, count: int, seed: int) -> list[tuple[Source, str, str, int, int]]:
    files = _source_files(source)
    random_state = random.Random(f"{source.name}:{seed}")
    samples = []
    seen_ids = set()
    for sample_index in range(count):
        for _ in range(20):
            input_url = files[0] if sample_index == 0 else random_state.choice(files)
            fs, path = filesystem_for(input_url)
            with fs.open(path, "rb") as stream:
                parquet = pq.ParquetFile(stream)
                row_group = 0 if sample_index == 0 else random_state.randrange(parquet.metadata.num_row_groups)
                batch = next(parquet.iter_batches(batch_size=128, row_groups=[row_group], columns=["id", "text"]))
                row_index = 0 if sample_index == 0 else random_state.randrange(batch.num_rows)
                row = batch.slice(row_index, 1).to_pylist()[0]
            source_id = str(row["id"])
            if source_id in seen_ids:
                continue
            chunks = split_source(row["text"])
            if not chunks:
                raise ValueError(f"Empty source row for {source.name}/{source_id}")
            chunk_index = 0 if sample_index == 0 else random_state.randrange(len(chunks))
            samples.append((source, source_id, chunks[chunk_index], chunk_index, len(chunks)))
            seen_ids.add(source_id)
            break
        else:
            raise ValueError(f"Could not sample {count} distinct source rows for {source.name}")
    return samples


async def probe(output_url: str, samples_per_source: int, seed: int) -> None:
    """Write sampled source chunks and validated conversions to JSON."""
    controller = iris_ctx().client
    if controller is None:
        raise RuntimeError("Run the probe as an Iris task")
    endpoint = controller.resolve_endpoint(ENDPOINT).rstrip("/")
    samples = [sample for source in sources() for sample in _source_samples(source, samples_per_source, seed)]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
        records = await asyncio.gather(
            *(
                _convert_chunk(client, semaphore, endpoint, source, source_id, chunk, chunk_index, chunk_count)
                for source, source_id, chunk, chunk_index, chunk_count in samples
            )
        )

    results = []
    for (source, source_id, chunk, chunk_index, chunk_count), result in zip(samples, records, strict=True):
        results.append(
            {
                "source": source.name,
                "source_id": source_id,
                "format": result.answer_format.name,
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "source_chunk": chunk,
                "record": result.record,
            }
        )
        logger.info("Validated %s in %s format", source.name, results[-1]["format"])

    fs, path = filesystem_for(output_url)
    with atomic_rename(path, filesystem=fs) as temporary_path:
        with fs.open(temporary_path, "wb") as output:
            output.write(json.dumps(results, ensure_ascii=False, indent=2).encode())
    logger.info("Wrote %d source samples to %s", len(results), output_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--samples-per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples_per_source < 1:
        parser.error("--samples-per-source must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(probe(args.output_path, args.samples_per_source, args.seed))


if __name__ == "__main__":
    main()
