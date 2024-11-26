#!/usr/bin/env python3
"""
Given a pattern of files with records containing URLs and their scores,
split the records into train and test by domain and resample the train set
to ensure that the distribution of scores is uniform.

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/open-web-math/resample_openwebmath_urls_by_quality_score.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-cc/CC*/*_urls_and_quality_classifier_scores.jsonl.gz", "gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math/*_urls_and_quality_classifier_scores.jsonl.gz"]' \
    --resample True \
    --train_output_path gs://marin-us-central2/scratch/nfliu/datasets/url_scoring/open-web-math/train.parquet \
    --test_output_path gs://marin-us-central2/scratch/nfliu/datasets/url_scoring/open-web-math/test.parquet
```
"""
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse

import draccus
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ResamplingConfig:
    input_patterns: list[str]
    train_output_path: str
    test_output_path: str
    resample: bool
    test_size: float = 0.2


@ray.remote(memory=256 * 1024 * 1024 * 1024, num_cpus=8)
def resample_urls_remote(
    input_patterns: list[str], train_output_path: str, test_output_path: str, resample: bool, test_size: float = 0.2
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Set the random seed for reproducibility
    random.seed(0)

    train_success_path = train_output_path + ".SUCCESS"
    test_success_path = test_output_path + ".SUCCESS"
    # Skip if we've already written success files for the training and test datasets.
    if fsspec_exists(train_success_path) and fsspec_exists(test_success_path):
        return

    input_filepaths = set()
    logger.info(f"Reading from input patterns: {input_patterns}")
    for input_pattern in input_patterns:
        input_filepaths.update(fsspec_glob(input_pattern))
    logger.info(f"Found {len(input_filepaths)} input files with URLs and scores")

    # Load all examples into memory
    all_examples = []
    seen_urls = set()
    num_skipped = 0
    for filepath in tqdm(input_filepaths, desc="Reading input filepaths"):
        with fsspec.open(filepath, "rt", compression="infer") as f:
            # Use readlines to encourage fsspec to load everything into memory
            # at once, rather than making multiple requests to GCS.
            for line in tqdm(f.readlines(), desc="Reading input filepath"):
                parsed_line = json.loads(line)
                if parsed_line["score"] > 5 or parsed_line["score"] < 0:
                    num_skipped += 1
                    continue
                if parsed_line["url"] in seen_urls:
                    num_skipped += 1
                    continue
                all_examples.append(parsed_line)
                seen_urls.add(parsed_line["url"])

    # Delete the set of all URLs, since we won't need it past this point
    del seen_urls
    logger.info(
        f"Read {len(all_examples)} deduplicated examples, " f"{num_skipped} skipped (due to invalid score or duplicate)"
    )

    # Build a mapping from domain to examples
    domain_examples = defaultdict(list)
    for example in tqdm(all_examples, desc="Building domain to examples mapping"):
        url = example["url"]
        domain = urlparse(url).netloc
        domain_examples[domain].append(example)
    logger.info(f"Found {len(domain_examples)} unique domains in the input examples")

    # Split domains into train and test sets
    logger.info("Splitting domains into train and test")
    domains = list(domain_examples.keys())
    logger.info("Shuffling domains")
    random.shuffle(domains)
    logger.info("Shuffled domains")
    test_domains_count = int(len(domains) * test_size)
    test_domains = set(domains[:test_domains_count])
    train_domains = set(domains[test_domains_count:])
    logger.info("Split domains into train and test")

    # Assign examples to train and test sets based on domains
    train_examples = []
    test_examples = []
    logger.info("Building train dataset")
    for domain in train_domains:
        train_examples.extend(domain_examples[domain])
    logger.info("Built train dataset")
    logger.info("Building test dataset")
    for domain in test_domains:
        test_examples.extend(domain_examples[domain])
    logger.info("Built test dataset")

    # Function to bucket and resample examples
    def bucket_and_resample(examples):
        # Each example has `url`, `canonicalized_url`, `score`, and `found_math`.
        # Scores range from 0.0 to 1.0.
        # Fixed bucketing logic to bucket examples in increments of 0.1
        buckets = defaultdict(list)
        for example in examples:
            score = example["score"]
            # Compute the bucket index based on score in increments of 0.1
            # Ensure that score=1.0 falls into the last bucket
            bucket_index = min(int(score / 0.1), 9)
            buckets[bucket_index].append(example)

        logger.info("Bucketing complete. Counts per bucket:")
        for bucket_index in sorted(buckets):
            bucket_range_start = bucket_index * 0.1
            bucket_range_end = bucket_range_start + 0.1
            logger.info(f"Bucket {bucket_range_start:.1f}-{bucket_range_end:.1f}: {len(buckets[bucket_index])}")

        # Resample to ensure even distribution across buckets
        max_samples_per_bucket = min(len(bucket) for bucket in buckets.values())
        resampled_examples = []
        for bucket in buckets.values():
            resampled_examples.extend(random.sample(bucket, k=max_samples_per_bucket))
        logger.info(f"Resampling complete, got {len(resampled_examples)} examples in total")
        return resampled_examples

    # Resample train examples
    if resample:
        resampled_train_examples = bucket_and_resample(train_examples)
    else:
        resampled_train_examples = train_examples

    # Convert train examples to a PyArrow Table
    train_table = pa.Table.from_pylist(resampled_train_examples)
    # Get filesystem and path for the train output
    fs_train, train_path_in_fs = fsspec.core.url_to_fs(train_output_path)
    # Write the resampled train examples to the output path as Parquet
    logger.info(f"Writing {len(resampled_train_examples)} train examples")
    pq.write_table(train_table, train_path_in_fs, filesystem=fs_train, compression="snappy")
    logger.info(f"Wrote {len(resampled_train_examples)} train examples")

    # Convert all test examples to a PyArrow Table
    test_table = pa.Table.from_pylist(test_examples)
    # Get filesystem and path for the test output
    fs_test, test_path_in_fs = fsspec.core.url_to_fs(test_output_path)
    # Write all test examples to the output path as Parquet (without resampling)
    logger.info(f"Writing {len(test_examples)} test examples")
    pq.write_table(test_table, test_path_in_fs, filesystem=fs_test, compression="snappy")
    logger.info(f"Wrote {len(test_examples)} test examples")

    # Write success files indicating the number of examples written
    with fsspec.open(train_success_path, "wt", compression="infer") as f:
        f.write(json.dumps({"num_examples": len(resampled_train_examples)}))
    with fsspec.open(test_success_path, "wt", compression="infer") as f:
        f.write(json.dumps({"num_examples": len(test_examples)}))


@draccus.wrap()
def resample_urls(cfg: ResamplingConfig):
    _ = ray.get(
        resample_urls_remote.remote(
            cfg.input_patterns, cfg.train_output_path, cfg.test_output_path, cfg.resample, cfg.test_size
        )
    )


if __name__ == "__main__":
    resample_urls()
