# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert one chunk from each science-forward source for a quality review."""

import argparse
import asyncio
import json
import logging

import httpx
import pyarrow.parquet as pq
from iris.client.client import iris_ctx
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.buckets import filesystem_for

from experiments.datakit.science_sft_conversion.conversion import (
    ENDPOINT,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    _convert_chunk,
    _source_files,
    format_for,
    sources,
    split_source,
)

logger = logging.getLogger(__name__)


async def probe(output_url: str) -> None:
    """Write representative source chunks and validated conversions to JSON."""
    controller = iris_ctx().client
    if controller is None:
        raise RuntimeError("Run the probe as an Iris task")
    endpoint = controller.resolve_endpoint(ENDPOINT).rstrip("/")
    samples = []
    for source in sources():
        input_url = _source_files(source)[0]
        fs, path = filesystem_for(input_url)
        with fs.open(path, "rb") as stream:
            row = pq.ParquetFile(stream).read_row_group(0, columns=["id", "text"]).slice(0, 1).to_pylist()[0]
        chunks = split_source(row["text"])
        if not chunks:
            raise ValueError(f"Empty first source row for {source.name}")
        samples.append((source, str(row["id"]), chunks[0], len(chunks)))

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
        records = await asyncio.gather(
            *(
                _convert_chunk(client, semaphore, endpoint, source, source_id, chunk, 0, chunk_count)
                for source, source_id, chunk, chunk_count in samples
            )
        )

    results = []
    for (source, source_id, chunk, _), record in zip(samples, records, strict=True):
        results.append(
            {
                "source": source.name,
                "source_id": source_id,
                "format": format_for(source.name, source_id, 0).name,
                "source_chunk": chunk,
                "record": record,
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(probe(args.output_path))


if __name__ == "__main__":
    main()
