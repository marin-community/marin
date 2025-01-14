#!/usr/bin/env python3
"""
Given a parquet file with outlinks and the crawl results, compute crawl yield statistics.

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --pip_deps '--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html,w3lib,trafilatura,jax[tpu],flax,transformers,requests,warcio[all],resiliparse' \
    --no_wait -- \
    python scripts/crawl/get_fineweb_edu_crawl_yield.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-1M/ \
    --crawl_input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-1M/ \
    --urls_and_scores_output_directory gs://marin-us-central2/scratch/nfliu/urls_and_scores/fineweb-edu-1M/ \
    --statistics_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/fineweb-edu-1M/yield_statistics.json.gz
```
"""
import json
import logging
import math
import os
import pathlib
import random
from dataclasses import dataclass

import draccus
import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import trafilatura
import w3lib.url
from resiliparse.parse.encoding import bytes_to_str, detect_encoding
from tqdm_loggable.auto import tqdm
from trafilatura import extract
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
from warcio import ArchiveIterator

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists, fsspec_glob, fsspec_rm, remove_tpu_lockfile_on_exit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GetCrawlYieldConfig:
    urls_input_directory: str
    crawl_input_directory: str
    statistics_output_path: str
    urls_and_scores_output_directory: str


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def decode_html(html: bytes) -> str | None:
    """
    Given HTML (bytes), decode it into a string if possible. First try with
    utf-8. If that doesn't work, try to detect the encoding.
    """
    try:
        html = bytes_to_str(html, "utf-8")
    except Exception:
        encoding = detect_encoding(html)
        if encoding is None or encoding == "utf-8":
            return
        try:
            html = bytes_to_str(html, encoding)
        except Exception:
            return
    return html


@ray.remote(
    memory=32 * 1024 * 1024 * 1024,
)
def consolidate_parquet(parquet_paths: str, consolidated_parquet_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    all_examples = []
    for parquet_path in tqdm(parquet_paths, desc="Consolidating parquets"):
        with fsspec.open(parquet_path) as f:
            all_examples.extend(pd.read_parquet(f).to_dict("records"))
    write_examples_to_parquet(all_examples, consolidated_parquet_path)
    for parquet_path in tqdm(parquet_paths, desc="Deleting consolidated parquets"):
        fsspec_rm(parquet_path)


@ray.remote(
    memory=64 * 1024 * 1024 * 1024,
    num_cpus=4,
)
def extract_text_from_warc(
    urls_path: str,
    warc_path: str,
    robots_path: str,
    errors_path: str,
    extracted_text_output_path: str,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(f"Using trafilatura version {trafilatura.__version__}")
    success_path = extracted_text_output_path + ".SUCCESS"
    if fsspec_exists(success_path):
        logger.info(f"Success path {success_path} already exists, skipping...")

    with fsspec.open(urls_path) as f:
        df = pd.read_parquet(f)
    logger.info(f"Found {len(df)} examples in input file {urls_path}")
    # Extract the URLs from the "link_target" column
    urls = df["link_target"].tolist()
    # Deduplicate the URLs
    urls = set(urls)
    logger.info(f"Found {len(urls)} deduplicated URLs in input file {urls_path}")

    # Iterate through the WARC, decompressing as we go.
    fetched_urls = set()
    num_records_skipped = 0
    num_records_saved = 0
    extracted_text_records = []
    with fsspec.open(warc_path, "rb", compression="gzip") as file_stream:
        for record in tqdm(ArchiveIterator(file_stream)):
            if record.rec_type == "response":
                record_url = record.rec_headers.get_header("WARC-Target-URI")
                fetched_urls.add(record_url)

                content = record.content_stream().read()
                html_decoded: str | None = decode_html(content)
                if not html_decoded or not record_url:
                    num_records_skipped += 1
                    continue

                canonicalized_url = w3lib.url.canonicalize_url(record_url)
                extracted_text = extract(
                    html_decoded,
                    favor_precision=True,
                    include_comments=False,
                    deduplicate=True,
                )
                if not extracted_text:
                    num_records_skipped += 1
                    continue

                extracted_text_records.append(
                    {"url": record_url, "canonicalized_url": canonicalized_url, "extracted_text": extracted_text}
                )
                num_records_saved += 1
    # Count the number of URLs that weren't fetched
    unfetched_urls = urls - fetched_urls
    logger.info(f"Out of {len(urls)} URLs to fetch, {len(unfetched_urls)} were not successfully fetched")
    # As a sanity check, count the number of fetched_urls that aren't in the original set. This should hopefully be 0.
    logger.info(f"Out of {len(fetched_urls)} fetched_urls, {len(fetched_urls - urls)} were not in the input set of URLs")
    # Write examples from this shard to parquet
    write_examples_to_parquet(extracted_text_records, extracted_text_output_path)
    logger.info(f"Saved {num_records_saved} records from WARC, skipped {num_records_skipped} records")
    with fsspec.open(success_path, "w") as fout:
        json.dump(
            {
                "total_urls": len(urls),
                "total_urls_fetched": len(fetched_urls),
            },
            fout,
        )

    return (
        len(urls),
        len(fetched_urls),
    )


@ray.remote(
    memory=32 * 1024 * 1024 * 1024,
    num_cpus=8,
    resources={"TPU": 4, "TPU-v4-8-head": 1},
)
@remove_tpu_lockfile_on_exit
@cached_or_construct_output(success_suffix="SUCCESS", verbose=False)
def process_one_batch(input_path: str, output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Loading quality classifier...")
    # Load the quality classifier
    model = FlaxAutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
    logger.info("Loaded quality classifier...")

    logger.info("Reading input path with extracted text")
    examples_to_classify = []
    with fsspec.open(input_path, "rt", compression="gzip") as fin:
        for line in fin:
            examples_to_classify.append(json.loads(line))
    logger.info("Finished reading input path with extracted text")

    # Classify all of the examples in the shard
    examples_scores = []
    for examples_batch in tqdm(
        batched(examples_to_classify, 512), desc="Classifying text", total=math.ceil(len(examples_to_classify) / 512)
    ):
        documents = [ex["extracted_text"] for ex in examples_batch]
        inputs = tokenizer(documents, return_tensors="jax", padding="longest", truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        logits_list = logits.tolist()
        examples_scores.extend(logits_list)

    assert len(examples_scores) == len(examples_to_classify)
    logger.info(f"Ran quality classifier on {len(examples_to_classify)} examples")

    with fsspec.open(output_path, "w", compression="gzip") as fout:
        for (
            example,
            example_score,
        ) in zip(examples_to_classify, examples_scores):
            fout.write(
                json.dumps(
                    {
                        "url": example["url"],
                        "canonicalized_url": example["canonicalized_url"],
                        "score": example_score,
                    }
                )
                + "\n"
            )


def write_examples_to_parquet(examples: list[dict], output_path: str):
    table = pa.Table.from_pylist(examples)
    output_fs, output_path_in_fs = fsspec.core.url_to_fs(output_path)
    pq.write_table(table, output_path_in_fs, filesystem=output_fs, compression="snappy")


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def get_shard_indices_to_process(urls_input_directory: str) -> list[int]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".parquet").removeprefix(f"links."))
        for path in fsspec_glob(os.path.join(urls_input_directory, "links.*.parquet"))
    ]
    shard_indices = sorted(shard_indices)
    logger.info(f"Found {len(shard_indices)} shards to process")
    return shard_indices


@draccus.wrap()
def main(cfg: GetCrawlYieldConfig):
    shard_indices_to_process = ray.get(get_shard_indices_to_process.remote(cfg.urls_input_directory))
    random.shuffle(shard_indices_to_process)

    # Extract the text from the WARC for each shard
    unfinished = []
    for shard_index in shard_indices_to_process:
        urls_path = os.path.join(cfg.urls_input_directory, f"links.{shard_index}.parquet")
        warc_path = os.path.join(cfg.crawl_input_directory, f"links.{shard_index}.warc.gz")
        robots_path = os.path.join(cfg.crawl_input_directory, f"links.{shard_index}_robots.json.gz")
        errors_path = os.path.join(cfg.crawl_input_directory, f"links.{shard_index}_errors.json.gz")
        extracted_text_output_path = os.path.join(
            cfg.urls_and_scores_output_directory, f"links.{shard_index}_extracted_text.parquet"
        )

        unfinished.append(
            extract_text_from_warc.remote(urls_path, warc_path, robots_path, errors_path, extracted_text_output_path)
        )
    # Wait for text extraction jobs to finish
    total_urls = 0
    total_urls_fetched = 0
    while unfinished:
        finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
        try:
            results = ray.get(finished)
            for shard_urls, shard_urls_fetched in results:
                total_urls += shard_urls
                total_urls_fetched += shard_urls_fetched
        except Exception as e:
            logger.exception(f"Error processing shard: {e}")
            raise
    logger.info(f"Total URLs: {total_urls}\n" f"Total URLs fetched: {total_urls_fetched}\n")

    # Run the quality classifier on the extracted text
    unfinished = []
    for shard_index in shard_indices_to_process:
        extracted_text_path = os.path.join(
            cfg.urls_and_scores_output_directory, f"links.{shard_index}_extracted_text.parquet"
        )
        urls_and_scores_output_path = os.path.join(
            cfg.urls_and_scores_output_directory, f"links.{shard_index}_urls_and_scores.parquet"
        )
        unfinished.append(process_one_batch.remote(extracted_text_path, urls_and_scores_output_path))
    # Wait for text extraction jobs to finish
    total_urls_passing = 0
    while unfinished:
        finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
        try:
            results = ray.get(finished)
            for shard_urls_passing in results:
                total_urls_passing += shard_urls_passing
        except Exception as e:
            logger.exception(f"Error processing shard: {e}")
            raise
    logger.info(f"Total URLs passing: {total_urls_passing}\n")

    with fsspec.open(cfg.statistics_output_path, "w", compression="gzip") as fout:
        json.dump(
            {
                "total_urls": total_urls,
                "total_urls_fetched": total_urls_fetched,
                "total_urls_passing": total_urls_passing,
            },
            fout,
        )

    # Consolidate urls and scores into a single parqet
    parquet_paths = [
        os.path.join(cfg.urls_and_scores_output_directory, f"links.{shard_index}_urls_and_scores.parquet")
        for shard_index in shard_indices_to_process
    ]
    consolidated_parquet_path = os.path.join(cfg.urls_and_scores_output_directory, f"urls_and_scores.parquet")
    _ = ray.get(consolidate_parquet.remote(parquet_paths, consolidated_parquet_path))


if __name__ == "__main__":
    main()
