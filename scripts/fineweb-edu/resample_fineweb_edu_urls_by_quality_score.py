#!/usr/bin/env python3
"""
Given a CC dump, randomly sample the specified number of WARCs from this
dump. Then, extract the URL and text of each record and score the text with the
FineWeb-Edu quality classifier.

```
for fineweb_edu_dump_html_path in $(gcloud storage ls gs://marin-us-central2/documents/fineweb-edu/html); do
    dump_name=$(basename -- ${fineweb_edu_dump_html_path})
    python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/fineweb-edu/resample_fineweb_edu_urls_by_quality_score.py \
    --input_pattern gs://marin-us-central2/scratch/nfliu/urls_and_scores/fineweb-edu*/CC*/*_urls_and_quality_classifier_scores.jsonl.gz
    --output_path gs://marin-us-central2/scratch/nfliu/urls_and_scores/fineweb-edu-cc/${dump_name}
done
```
"""
from collections import defaultdict
import json
import logging
from dataclasses import dataclass
import random
from urllib.parse import urlparse


import draccus
import fsspec
import ray
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_glob, fsspec_exists

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ResamplingConfig:
    input_pattern: str
    output_path: str


@ray.remote(memory=64 * 1024 * 1024 * 1024, num_cpus=8)
def resample_urls_remote(input_pattern: str, train_output_path: str, test_output_path: str, test_size: float = 0.2):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set the random seed for reproducibility
    random.seed(0)

    train_success_path = train_output_path + ".SUCCESS"
    test_success_path = test_output_path + ".SUCCESS"
    # Skip if we've already written success files for the training and test datasets.
    if fsspec_exists(train_success_path) and fsspec_exists(test_success_path):
        return

    input_filepaths = fsspec_glob(input_pattern)
    logger.info(f"Found {len(input_filepaths)} input files with URLs and scores")

    # Load all examples into memory
    all_examples = []
    for filepath in tqdm(input_filepaths, desc="Reading input filepaths"):
        with fsspec.open(filepath, "rt", compression="infer") as f:
            # Use readlines to encourage fsspec to load everything into memory
            # at once, rather than making multiple requests to GCS.
            for line in f.readlines():
                all_examples.append(json.loads(line))

    # Build a mapping from domain to examples
    domain_examples = defaultdict(list)
    for example in tqdm(all_examples, desc="Building domain to examples mapping"):
        url = example["url"]
        domain = urlparse(url).netloc
        domain_examples[domain].append(example)

    # Split domains into train and test sets
    domains = list(domain_examples.keys())
    random.shuffle(domains)
    test_domains_count = int(len(domains) * test_size)
    test_domains = set(domains[:test_domains_count])
    train_domains = set(domains[test_domains_count:])

    # Assign examples to train and test sets based on domains
    train_examples = []
    test_examples = []
    for domain in train_domains:
        train_examples.extend(domain_examples[domain])
    for domain in test_domains:
        test_examples.extend(domain_examples[domain])

    # Function to bucket and resample examples
    def bucket_and_resample(examples):
        # Each example has `url`, `canonicalized_url`, and `score`.
        # Scores range from 0 - 5. We want to resample the examples such that there's
        # an even number of examples for each bucket. Buckets are defined
        # in increments of 0.5 (i.e., 0-0.5, 0.5-1.0, 1.0-1.5, etc.)
        bucket_size = 0.5
        max_score = 5.0
        num_buckets = int(max_score / bucket_size)
        buckets = {i: [] for i in range(num_buckets)}
        for example in examples:
            score = example["score"]
            bucket_index = min(int(score / bucket_size), num_buckets - 1)
            buckets[bucket_index].append(example)

        logger.info("Bucketing complete. Counts per bucket:")
        for bucket_index in sorted(buckets):
            logger.info(f"Bucket {bucket_index}: {len(buckets[bucket_index])}")

        # Resample to ensure even distribution
        max_samples_per_bucket = min(len(bucket) for bucket in buckets.values())
        resampled_examples = []
        for bucket_index, bucket in buckets.items():
            resampled_examples.extend(random.sample(bucket, k=max_samples_per_bucket))
        logger.info(f"Resampling complete, got {len(resampled_examples)} examples in total")
        return resampled_examples

    # Resample train examples
    resampled_train_examples = bucket_and_resample(train_examples)

    # Write the resampled train examples to the output path
    with fsspec.open(train_output_path, "wt", compression="infer") as f:
        for example in tqdm(resampled_train_examples, desc="Writing train output"):
            f.write(json.dumps(example) + "\n")

    # Write all test examples to the output path (without resampling)
    with fsspec.open(test_output_path, "wt", compression="infer") as f:
        for example in tqdm(test_examples, desc="Writing test output"):
            f.write(json.dumps(example) + "\n")

    with fsspec.open(train_success_path, "wt", compression="infer") as f:
        f.write({"num_examples": len(resampled_train_examples)})
    with fsspec.open(test_success_path, "wt", compression="infer") as f:
        f.write({"num_examples": len(test_examples)})


@draccus.wrap()
def resample_urls(cfg: ResamplingConfig):
    _ = ray.get(resample_urls_remote.remote(cfg.input_pattern, cfg.output_path))


if __name__ == "__main__":
    resample_urls()
