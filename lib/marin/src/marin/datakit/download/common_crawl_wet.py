# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingest Common Crawl WET (extracted plain-text) files into dolma JSONL.

WET files are the crawl's own text extraction (one ``conversion`` WARC record
per fetched page), so unlike the WARC path there is no HTML extraction step:
the record body *is* the text. This mirrors the Nemotron-CC ingress
(:mod:`marin.datakit.download.nemotron_v1`) but parses the WARC WET format
instead of a zstd-JSONL stream.

Two crawls are supported through different manifest sources:

- A published main crawl (e.g. ``CC-MAIN-2024-18``) exposes ``wet.paths.gz``.
- A project/focus crawl (e.g. ``CC-SUPPLEMENTAL-2026-22``) has no published
  manifest and its bucket prefix is not anonymously listable, so the WET file
  universe is derived from the crawl's columnar index parquet (distinct
  ``warc_filename`` mapped ``warc/ -> wet/``).

Both are read credential-free over the ``data.commoncrawl.org`` HTTPS gateway.
"""

import gzip
import json
import logging
import os
import random
from collections.abc import Iterator

import fsspec
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from rigging.filesystem import atomic_rename, open_url
from zephyr import Dataset, ZephyrContext

from marin.datakit.download.http_session import build_retrying_session
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_exists

logger = logging.getLogger(__name__)

CC_GATEWAY_URL = "https://data.commoncrawl.org"
_USER_AGENT = "marin-cc-wet-ingress/1.0"
_WET_RETRY_STATUS = (429, 500, 502, 503, 504)


def warc_to_wet_path(warc_path: str) -> str:
    """Map a WARC file key to its sibling WET key (``warc/`` -> ``wet/``)."""
    return warc_path.replace("/warc/", "/wet/").replace(".warc.gz", ".warc.wet.gz")


def iter_wet_text_records(raw_stream) -> Iterator[tuple[str, str, str]]:
    """Yield ``(record_id, target_uri, text)`` for each WET ``conversion`` record.

    WET files are multi-member-gzip WARC streams; ``gzip.GzipFile`` transparently
    spans the members. Records are framed by ``Content-Length`` per the WARC
    spec, so we read exactly that many body bytes rather than splitting on a
    delimiter that can occur inside the text.
    """
    with gzip.GzipFile(fileobj=raw_stream) as stream:
        while True:
            version = stream.readline()
            if not version:
                break
            if not version.startswith(b"WARC/"):
                continue
            headers: dict[bytes, bytes] = {}
            while True:
                line = stream.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                key, _, value = line.partition(b":")
                headers[key.strip().lower()] = value.strip()
            length = int(headers.get(b"content-length", b"0"))
            body = stream.read(length)
            if headers.get(b"warc-type") != b"conversion":
                continue
            yield (
                headers.get(b"warc-record-id", b"").decode(),
                headers.get(b"warc-target-uri", b"").decode(),
                body.decode("utf-8", "replace"),
            )


def download_single_wet_file(input_file_path: str, output_file_path: str, base_url: str = CC_GATEWAY_URL) -> dict:
    """Stream one WET file from the gateway, writing dolma records to zstd JSONL."""
    cc_url = f"{base_url}/{input_file_path}"
    logger.info(f"Downloading WET file {cc_url} to {output_file_path}")

    session = build_retrying_session(status_forcelist=_WET_RETRY_STATUS)
    response = session.get(cc_url, headers={"user-agent": _USER_AGENT}, stream=True)
    response.raise_for_status()

    num_records = 0
    with atomic_rename(output_file_path) as temp_path:
        with open_url(temp_path, "w", compression="zstd") as out:
            for record_id, target_uri, text in iter_wet_text_records(response.raw):
                if not text.strip():
                    continue
                dolma_record = {
                    "id": record_id,
                    "text": text,
                    "source": "common_crawl_wet",
                    "format": "text",
                    "metadata": {"warc_target_uri": target_uri, "warc_filename": input_file_path},
                }
                print(json.dumps(dolma_record), file=out)
                num_records += 1

    return {"input_file": input_file_path, "output_file": output_file_path, "num_records": num_records}


def main_crawl_wet_paths(crawl: str, base_url: str = CC_GATEWAY_URL) -> list[str]:
    """WET file keys for a published main crawl, read from its ``wet.paths.gz``."""
    paths_url = f"{base_url}/crawl-data/{crawl}/wet.paths.gz"
    with open_url(paths_url, "r", compression="gzip") as f:
        return [line.strip() for line in f if line.strip()]


def focus_crawl_wet_paths(index_parquet_url: str) -> list[str]:
    """WET file keys for a focus/project crawl, derived from its index parquet.

    The crawl's columnar index lists every fetched record's ``warc_filename``;
    distinct values mapped ``warc/ -> wet/`` give the WET file universe. A single
    index part suffices because the index is row-partitioned (every part covers
    the same file set).
    """
    pf = pq.ParquetFile(fsspec.open(index_parquet_url).open())
    warc_files: set[str] = set()
    for row_group in range(pf.metadata.num_row_groups):
        column = pf.read_row_group(row_group, columns=["warc_filename"]).column("warc_filename").to_pylist()
        warc_files.update(column)
    return sorted(warc_to_wet_path(f) for f in warc_files)


def _output_file_for(output_path: str, wet_path: str) -> str:
    """Mirror a source WET key under ``output_path`` as a ``.jsonl.zst`` file."""
    return os.path.join(output_path, wet_path.replace(".warc.wet.gz", ".jsonl.zst"))


def download_cc_wet(
    output_path: str,
    wet_paths: list[str],
    *,
    seed: int,
    num_files: int,
    base_url: str = CC_GATEWAY_URL,
    worker_ram: str = "4g",
) -> None:
    """Seeded-random-sample ``num_files`` WET files and download them in parallel.

    Sampling is over the sorted file universe so the selection is reproducible
    from ``seed`` regardless of the manifest's natural order. Each WET file is an
    independent Zephyr work unit; already-downloaded outputs are skipped so the
    step is resumable.
    """
    universe = sorted(wet_paths)
    rng = random.Random(seed)
    selected = universe if num_files >= len(universe) else rng.sample(universe, num_files)
    logger.info(f"Sampled {len(selected)} of {len(universe)} WET files (seed={seed}) into {output_path}")

    work = [(path, _output_file_for(output_path, path)) for path in selected]

    pipeline = (
        Dataset.from_list(work)
        .filter(lambda file_info: not fsspec_exists(file_info[1]))
        .map(lambda file_info: download_single_wet_file(file_info[0], file_info[1], base_url=base_url))
        .write_jsonl(os.path.join(output_path, ".metrics/download-{shard:05d}.jsonl"), skip_existing=True)
    )

    # Each WET file is ~20-95 MB gzipped and decompresses to a few hundred MB;
    # 4 GB per worker matches the Nemotron-CC ingress headroom.
    ctx = ZephyrContext(name="download-cc-wet", resources=ResourceConfig(cpu=1, ram=worker_ram))
    ctx.execute(pipeline)

    logger.info(f"Downloaded {len(selected)} WET files to {output_path}")


def download_cc_wet_step(
    *,
    name: str,
    crawl: str,
    wet_paths_fn,
    seed: int,
    num_files: int,
    base_url: str = CC_GATEWAY_URL,
    worker_ram: str = "4g",
) -> StepSpec:
    """A ``StepSpec`` that ingests a seeded WET sample of one crawl to dolma JSONL.

    ``wet_paths_fn`` is a zero-arg callable that returns the crawl's full WET file
    universe; it is invoked on the worker so the manifest read does not happen at
    graph-build time. ``crawl`` is the cache-identity key, so changing the
    manifest source without changing ``crawl`` will not invalidate the cache.
    """
    return StepSpec(
        name=name,
        fn=lambda output_path: download_cc_wet(
            output_path,
            wet_paths_fn(),
            seed=seed,
            num_files=num_files,
            base_url=base_url,
            worker_ram=worker_ram,
        ),
        hash_attrs={"crawl": crawl, "seed": seed, "num_files": num_files, "base_url": base_url},
    )
