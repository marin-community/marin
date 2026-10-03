# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample science-forward source chunks for conversion quality review."""

import argparse
import asyncio
import json
import logging
import os
import random
from itertools import islice

import pyarrow.parquet as pq
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import filesystem_for

from experiments.datakit.science_sft_conversion.batch_transport import GLMBatchChatClient
from experiments.datakit.science_sft_conversion.conversion import (
    Source,
    _convert_chunk,
    _source_files,
    sources,
    split_source,
)
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, resolve_glm_base_url

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
                row_index = (
                    0 if sample_index == 0 else random_state.randrange(parquet.metadata.row_group(row_group).num_rows)
                )
                batches = parquet.iter_batches(batch_size=128, row_groups=[row_group], columns=["id", "text"])
                batch = next(islice(batches, row_index // 128, row_index // 128 + 1))
                row = batch.slice(row_index % 128, 1).to_pylist()[0]
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


async def probe(
    output_url: str,
    samples_per_source: int,
    seed: int,
    relay_job: str,
    concurrency: int,
    source_names: tuple[str, ...],
) -> None:
    """Write sampled source chunks and validated conversions to JSON."""
    endpoint = resolve_glm_base_url(relay_job).removesuffix("/v1")
    selected = [source for source in sources() if not source_names or source.name in source_names]
    if source_names and {source.name for source in selected} != set(source_names):
        raise ValueError("Unknown source in probe selection")
    samples = [sample for source in selected for sample in _source_samples(source, samples_per_source, seed)]

    semaphore = asyncio.Semaphore(concurrency)
    async with GLMBatchChatClient(endpoint, os.environ[GLM_BULK_TOKEN_ENV], batch_size=32, workers=2) as client:
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
    parser.add_argument("--relay-job", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument(
        "--source", action="append", default=[], help="Source name; omit for one sample from every source"
    )
    parser.add_argument("--samples-per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples_per_source < 1:
        parser.error("--samples-per-source must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        probe(args.output_path, args.samples_per_source, args.seed, args.relay_job, args.concurrency, tuple(args.source))
    )


if __name__ == "__main__":
    main()
