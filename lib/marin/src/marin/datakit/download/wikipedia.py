# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Download script for the Wikipedia raw HTML data, provided by Wikimedia.

Home Page: https://dumps.wikimedia.org/other/enterprise_html/runs/

Note: The enwiki-NS0 file (English Wikipedia, namespace 0 = articles) is approximately 130 GB compressed.
      The simplewiki-NS0 file (Simple English Wikipedia) is much smaller at ~2 GB compressed.
"""

import logging
import os
import tarfile
from collections.abc import Iterable

import requests
from rigging.filesystem import open_url
from tqdm_loggable.auto import tqdm
from zephyr import Dataset, ZephyrContext, atomic_rename, load_jsonl

from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_size

logger = logging.getLogger(__name__)


def download_tar(url: str, output_prefix: str) -> str:
    shard_filename = url.split("/")[-1]
    output_filename = os.path.join(output_prefix, shard_filename)
    logger.info(f"Downloading URL: {url} to {output_filename}")

    try:
        total_size = fsspec_size(url)
        pbar = tqdm(total=total_size, desc="Downloading File", unit="B", unit_scale=True)

        with atomic_rename(output_filename) as tmp_filename, open_url(tmp_filename, "wb") as f:
            r = requests.get(url, stream=True)

            for chunk in r.raw.stream(20 * 1024 * 1024, decode_content=False):
                if chunk:
                    f.write(chunk)
                    f.flush()

                    pbar.update(len(chunk))

        return output_filename
    except Exception as e:
        logger.error(f"Error downloading URL: {url}")
        raise e


def process_file(input_file: str, output_path: str) -> Iterable[str]:
    logger.info(f"Processing file: {input_file}")
    logger.info(f"Output path: {output_path}")

    try:
        with open_url(input_file) as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tr:
                for info in tr:
                    extracted = tr.extractfile(info)
                    if extracted is None:
                        # Skip non-regular entries (directories, symlinks, etc.)
                        continue
                    with extracted as file:
                        file_content = file.read()
                        file_path = os.path.join(output_path, info.name + ".gz")

                    # Each file is a .ndjson file, which contains about 18k-21k articles
                    # per file with size ranging from 200MB to 300MB
                    with (
                        atomic_rename(file_path) as tmpfile_path,
                        open_url(tmpfile_path, "wb", compression="gzip") as output_f,
                    ):
                        output_f.write(file_content)
                        yield file_path

    except Exception as e:
        logger.error(f"Error processing file: {input_file}")
        raise e


def download_wikipedia(input_urls: list[str], revision: str, output_path: str) -> None:
    """Download and process Wikipedia data."""
    logger.info("Starting transfer of Wikipedia dump...")
    output_base = os.path.join(output_path, revision)

    ctx = ZephyrContext(name="download-wikipedia")
    download_metrics = ctx.execute(
        Dataset.from_list(input_urls)
        .map(lambda url: download_tar(url, output_base))
        .write_jsonl(f"{output_base}/.metrics/download-{{shard:05d}}.jsonl", skip_existing=True),
    ).results

    # load all of the output filenames to process
    downloads = ctx.execute(Dataset.from_list(download_metrics).flat_map(load_jsonl)).results

    extracted = ctx.execute(
        Dataset.from_list(downloads)
        .flat_map(lambda file: process_file(file, output_base))
        .write_jsonl(f"{output_base}/.metrics/process-{{shard:05d}}.jsonl", skip_existing=True),
    ).results

    logger.info("Wikipedia dump transfer complete, wrote: %s", extracted)


def download_wikipedia_step(
    *,
    input_urls: list[str] | None = None,
    revision: str | None = None,
) -> StepSpec:
    """Download Wikipedia HTML dumps."""

    def _run(output_path: str) -> None:
        assert input_urls is not None, "input_urls must be provided to download Wikipedia data"
        assert revision is not None, "revision must be provided to download Wikipedia data"
        download_wikipedia(input_urls, revision, output_path)

    return StepSpec(
        name="raw/wikipedia",
        fn=_run,
        hash_attrs={"input_urls": input_urls, "revision": revision},
        # NOTE: if no inputs are provided, use the previously downloaded 2024-12-01 data
        override_output_path="raw/wikipedia-9273e1" if input_urls is None else None,
    )
