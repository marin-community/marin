#!/usr/bin/env python3
"""
Given a parquet file with outlinks and the crawl results, compute crawl yield statistics.

In particular, we compute:

1. The total number of URLs in the outlinks (N_total)
2. The total number of URLs that were successfully fetched (N_f)
3. The total number of successfully-fetched URLs that pass the FineWeb-Edu quality filtering pipeline (N_hq)

The quantities of interest are:

- N_f / N_total: How many of the URLs are actually crawlable / we get responses from?
- N_hq / N_total: What is the overall yield rate of the crawl frontier?
- N_hq / N_f: What is the yield rate of the successfully-fetched pages?

Running on OpenWebMath:

```
python marin/run/ray_run.py \
    --pip_deps 'resiliparse,fasttext,lxml,py-asciimath,tabulate,warcio[all],w3lib,cchardet' \
    --no_wait -- \
    python marin/crawl/get_open_web_math_crawl_yield.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/ \
    --crawl_input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/ \
    --data_source open-web-math-fde8ef8-10M \
    --text_output_directory gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/ \
    --urls_and_scores_output_directory gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-fde8ef8-10M/ \
    --statistics_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/yield_statistics.json.gz
```
"""
import json
import logging
import os
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Any

import draccus
import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import w3lib.url
from resiliparse.parse.encoding import bytes_to_str, detect_encoding
from tqdm_loggable.auto import tqdm
from warcio import ArchiveIterator

from marin.processing.classification.classifier import FasttextClassifier
from marin.processing.open_web_math.extract import extract_text
from marin.processing.open_web_math.text_normalizer import normalize
from marin.processing.open_web_math.utils import Config as OpenWebMathConfig
from marin.utils import fsspec_exists, fsspec_glob, fsspec_rm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DolmaFormattedOpenWebMathRecord:
    id: str
    source: str
    format: str
    text: str
    metadata: dict[str, Any]


@dataclass
class GetCrawlYieldConfig:
    urls_input_directory: str
    crawl_input_directory: str
    data_source: str
    text_output_directory: str
    statistics_output_path: str
    urls_and_scores_output_directory: str


def score_text(text, score_model):
    normalized_text = normalize(text).replace("\n", " ")
    # Remove any [EQUATION] tokens
    normalized_text = normalized_text.replace("[EQUATION]", "")
    pred = score_model.predict(normalized_text)
    if pred[0][0] == "__label__positive":
        prob = pred[1][0]
    else:
        prob = pred[1][1]

    return prob


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
    memory=32 * 1024 * 1024 * 1024,
    num_cpus=8,
)
def get_shard_yield(
    urls_path: str,
    warc_path: str,
    robots_path: str,
    errors_path: str,
    data_source: str,
    text_with_scores_output_path: str,
    urls_and_scores_output_path: str,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    success_path = urls_and_scores_output_path + ".SUCCESS"
    if fsspec_exists(success_path):
        logger.info(f"Success path {success_path} already exists, skipping...")
        with fsspec.open(success_path) as f:
            shard_stats = json.load(f)
        return (
            shard_stats["total_urls"],
            shard_stats["total_urls_fetched"],
            shard_stats["total_urls_passing"],
        )

    with fsspec.open(urls_path) as f:
        df = pd.read_parquet(f)
    logger.info(f"Found {len(df)} examples in input file {urls_path}")
    # Extract the URLs from the "link_target" column
    urls = df["link_target"].tolist()
    # Deduplicate the URLs
    urls = set(urls)
    logger.info(f"Found {len(urls)} deduplicated URLs in input file {urls_path}")

    logger.info("Loading mathscore model")
    # NOTE: we don't use the attribute functionality in FasttextClassifier
    mathscore_model = FasttextClassifier(model_name="open-web-math/filtering-models", attribute_name="")
    logger.info("Loaded mathscore model")

    randomized_config = OpenWebMathConfig(
        os.path.join(os.path.dirname(__file__), os.path.pardir, "open-web-math", "configs", "randomized_all.yaml")
    )

    fetched_urls = set()
    num_records_skipped = 0
    num_records_passing = 0
    num_records_saved = 0

    urls_with_scores = []
    text_with_scores = []
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

                randomized_config_sample = randomized_config.sample()
                try:
                    extraction_result = extract_text(html_decoded, randomized_config_sample, fast=True)
                except Exception:
                    num_records_skipped += 1
                    continue

                if extraction_result is None:
                    num_records_skipped += 1
                    continue

                extracted_text, extraction_metadata = extraction_result
                score = score_text(extracted_text, mathscore_model)
                found_math = extraction_metadata["found_math"]
                canonicalized_url = w3lib.url.canonicalize_url(record_url)

                if found_math and score > 0.15:
                    # If the URL has LaTeX, the threshold is 0.15.
                    num_records_passing += 1
                elif not found_math and score > 0.8:
                    # If the URL doesn't have LaTeX, the threshold is 0.8.
                    num_records_passing += 1

                urls_with_scores.append(
                    {
                        "url": record_url,
                        "canonicalized_url": canonicalized_url,
                        "score": score,
                        "found_math": found_math,
                    }
                )
                record_id = record.rec_headers.get_header("WARC-Record-ID")
                assert record_id
                record_date = record.rec_headers.get_header("WARC-Date")
                assert record_date
                text_with_scores.append(
                    asdict(
                        DolmaFormattedOpenWebMathRecord(
                            id=record_id,
                            source=data_source,
                            format="text",
                            text=extracted_text,
                            metadata={
                                "url": record_url,
                                "canonicalized_url": canonicalized_url,
                                "date": record_date,
                                "file_path": warc_path,
                                "score": score,
                                "found_math": found_math,
                            },
                        )
                    )
                )
                num_records_saved += 1

    # Count the number of URLs that weren't fetched
    unfetched_urls = urls - fetched_urls
    logger.info(f"Out of {len(urls)} URLs to fetch, {len(unfetched_urls)} were not successfully fetched")
    # As a sanity check, count the number of fetched_urls that aren't in the original set. This should hopefully be 0.
    logger.info(f"Out of {len(fetched_urls)} fetched_urls, {len(fetched_urls - urls)} were not in the input set of URLs")
    logger.info(f"{num_records_passing} URLs passed the quality filtering pipeline")
    # Write examples from this shard to parquet
    write_examples_to_parquet(urls_with_scores, urls_and_scores_output_path)
    write_examples_to_parquet(text_with_scores, text_with_scores_output_path)
    logger.info(f"Saved {num_records_saved} records from WARC, skipped {num_records_skipped} records")

    with fsspec.open(success_path, "w") as fout:
        json.dump(
            {
                "total_urls": len(urls),
                "total_urls_fetched": len(fetched_urls),
                "total_urls_passing": num_records_passing,
            },
            fout,
        )

    return (
        len(urls),
        len(fetched_urls),
        num_records_passing,
    )


def write_examples_to_parquet(examples: list[dict], output_path: str):
    table = pa.Table.from_pylist(examples)
    output_fs, output_path_in_fs = fsspec.core.url_to_fs(output_path)
    pq.write_table(table, output_path_in_fs, filesystem=output_fs, compression="snappy")


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def get_shard_indices_to_process(urls_input_directory: str) -> list[int]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".parquet").removeprefix("links."))
        for path in fsspec_glob(os.path.join(urls_input_directory, "links.*.parquet"))
    ]
    shard_indices = sorted(shard_indices)
    logger.info(f"Found {len(shard_indices)} shards to process")
    return shard_indices


@draccus.wrap()
def main(cfg: GetCrawlYieldConfig):
    shard_indices_to_process = ray.get(get_shard_indices_to_process.remote(cfg.urls_input_directory))
    random.shuffle(shard_indices_to_process)

    # Calculate the yield for each shard
    unfinished = []
    for shard_index in shard_indices_to_process:
        urls_path = os.path.join(cfg.urls_input_directory, f"links.{shard_index}.parquet")
        warc_path = os.path.join(cfg.crawl_input_directory, f"links.{shard_index}.warc.gz")
        robots_path = os.path.join(cfg.crawl_input_directory, f"links.{shard_index}_robots.json.gz")
        errors_path = os.path.join(cfg.crawl_input_directory, f"links.{shard_index}_errors.json.gz")
        urls_and_scores_output_path = os.path.join(
            cfg.urls_and_scores_output_directory, f"links.{shard_index}_urls_and_scores.parquet"
        )
        text_and_scores_output_path = os.path.join(
            cfg.text_output_directory, f"links.{shard_index}_text_and_scores.parquet"
        )
        unfinished.append(
            get_shard_yield.remote(
                urls_path,
                warc_path,
                robots_path,
                errors_path,
                cfg.data_source,
                text_and_scores_output_path,
                urls_and_scores_output_path,
            )
        )

    # Wait for jobs to finish
    total_urls = 0
    total_urls_fetched = 0
    total_urls_passing = 0
    while unfinished:
        finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
        try:
            results = ray.get(finished)
            for shard_urls, shard_urls_fetched, shard_urls_passing in results:
                total_urls += shard_urls
                total_urls_fetched += shard_urls_fetched
                total_urls_passing += shard_urls_passing
        except Exception as e:
            logger.exception(f"Error processing shard: {e}")
            raise
    logger.info(
        f"Total URLs: {total_urls}\n"
        f"Total URLs fetched: {total_urls_fetched}\n"
        f"Total URLs passing: {total_urls_passing}\n"
    )
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
    consolidated_parquet_path = os.path.join(cfg.urls_and_scores_output_directory, "urls_and_scores.parquet")
    _ = ray.get(consolidate_parquet.remote(parquet_paths, consolidated_parquet_path))


if __name__ == "__main__":
    main()
