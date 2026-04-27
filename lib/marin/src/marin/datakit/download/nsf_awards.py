# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download NSF award data and write one Parquet file per year.

The pipeline fetches presigned S3 URLs from the NSF API, downloads each year's
zip of per-award JSON files, filters to awards with abstracts, and writes a
single Parquet file per year.
"""

import json
import logging
import os
import zipfile
from io import BytesIO

import requests
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from zephyr import Dataset, ZephyrContext, counters
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

NSF_LIST_FILES_URL = "https://api.nsf.gov/services/v2/s3/list-files"
NSF_LIST_FILES_HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.nsf.gov",
    "Referer": "https://www.nsf.gov/",
}

MIN_YEAR = 1960
MAX_YEAR = 2025


def _fetch_download_urls() -> dict[int, str]:
    """Fetch presigned S3 download URLs for each year from the NSF API."""
    response = requests.get(NSF_LIST_FILES_URL, headers=NSF_LIST_FILES_HEADERS, timeout=60)
    response.raise_for_status()
    result = {}
    for entry in response.json()["files"]:
        name = entry["fileName"]
        if name.endswith(".zip"):
            try:
                year = int(name.removesuffix(".zip"))
                result[year] = entry["downloadUrl"]
            except ValueError:
                continue
    return result


KEEP_FIELDS = [
    "awd_id",
    "awd_titl_txt",
    "awd_abstract_narration",
    "awd_eff_date",
    "awd_exp_date",
    "awd_amount",
    "dir_abbr",
    "org_dir_long_name",
    "div_abbr",
    "org_div_long_name",
]


def _award_to_record(award: dict) -> dict | None:
    """Convert a raw NSF award JSON to a pretraining record, or None if no abstract."""
    abstract = award.get("awd_abstract_narration")
    if not abstract:
        return None
    title = award.get("awd_titl_txt", "")
    record = {k: award.get(k) for k in KEEP_FIELDS}
    record["text"] = f"{title}\n\n{abstract}" if title else abstract
    return record


def download_and_convert_year(year: int, download_url: str, output_path: str) -> dict:
    """Download NSF awards for a single year, filter to those with abstracts, and write a Parquet file."""
    logger.info(f"Downloading NSF awards for {year}")

    response = requests.get(download_url, timeout=300)
    response.raise_for_status()

    records = []
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            with z.open(name) as f:
                award = json.load(f)
            counters.increment("awards_total")
            record = _award_to_record(award)
            if record is None:
                counters.increment("awards_missing_abstract")
                continue
            for k in KEEP_FIELDS:
                if not award.get(k):
                    counters.increment(f"awards_missing_field.{k}")
            records.append(record)
    counters.increment("awards_kept", len(records))

    if not records:
        logger.warning(f"No awards with abstracts for {year}, skipping.")
        return {"year": year, "num_awards": 0}

    output_file = os.path.join(output_path, f"{year}.parquet")
    result = write_parquet_file(records, output_file)

    logger.info(f"Wrote {len(records)} awards for {year} to {output_file}")
    return {"year": year, "num_awards": len(records), **result}


def download_nsf_awards(min_year: int, max_year: int, output_path: str) -> None:
    """Download NSF awards and write one Parquet file per year."""
    logger.info(f"Downloading NSF awards for years {min_year}-{max_year}")

    url_map = _fetch_download_urls()
    tasks = [
        {"year": year, "url": url_map[year], "output_path": output_path}
        for year in range(min_year, max_year + 1)
        if year in url_map
    ]

    if not tasks:
        raise ValueError(f"No download URLs found for years {min_year}-{max_year}")

    logger.info(f"Found {len(tasks)} years to download")

    pipeline = (
        Dataset.from_list(tasks)
        .map(lambda task: download_and_convert_year(task["year"], task["url"], task["output_path"]))
        .write_jsonl(f"{output_path}/.metrics/download-{{shard:05d}}.jsonl", skip_existing=True)
    )
    ctx = ZephyrContext(name="download-nsf-awards")
    ctx.execute(pipeline)

    logger.info("NSF awards download complete.")


def download_nsf_awards_step(
    *,
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR,
) -> StepSpec:
    """Create a StepSpec that downloads NSF award data."""

    def _run(output_path: str) -> None:
        download_nsf_awards(min_year, max_year, output_path)

    return StepSpec(
        name="raw/nsf-awards",
        fn=_run,
        hash_attrs={"min_year": min_year, "max_year": max_year},
    )


def normalize_nsf_awards_step(download: StepSpec) -> StepSpec:
    """Normalize NSF awards: generate content-hash IDs, preserve awd_id as source_id."""
    return normalize_step(
        name="normalized/nsf-awards",
        download=download,
        text_field="text",
        id_field="awd_id",
        file_extensions=(".parquet",),
    )
