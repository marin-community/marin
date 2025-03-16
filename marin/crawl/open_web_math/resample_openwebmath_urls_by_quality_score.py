#!/usr/bin/env python3
"""
Given a pattern of files with records containing URLs and their scores,
split the records into train and test by domain and resample the train set
to ensure that the distribution of scores is uniform.

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/open-web-math/resample_openwebmath_urls_by_quality_score.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-cc/CC*/*_urls_and_quality_classifier_scores.jsonl.gz", "gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math/*_urls_and_quality_classifier_scores.jsonl.gz"]' \
    --cc_prefix 'gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-cc/' \
    --train_output_path gs://marin-us-central2/scratch/nfliu/datasets/url_scoring/open-web-math/train.parquet \
    --cc_test_output_path gs://marin-us-central2/scratch/nfliu/datasets/url_scoring/open-web-math/test_cc.parquet \
    --balanced_test_output_path gs://marin-us-central2/scratch/nfliu/datasets/url_scoring/open-web-math/test_balanced.parquet
```
"""  # noqa: E501
import json
import logging
import math
import random
from collections import Counter, defaultdict
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


@dataclass(frozen=True)
class ResamplingConfig:
    input_patterns: list[str]
    cc_prefix: str
    train_output_path: str
    cc_test_output_path: str
    balanced_test_output_path: str
    test_size: float = 0.2


def clip(minimum, x, maximum):
    return max(minimum, min(x, maximum))


@ray.remote(memory=256 * 1024 * 1024 * 1024, num_cpus=8)
def resample_urls(
    input_patterns: list[str],
    cc_prefix: str,
    train_output_path: str,
    cc_test_output_path: str,
    balanced_test_output_path: str,
    test_size: float = 0.2,
):
    """This function resamples and split records into training and test sets.
    Each record contains a str URL and a continuous float score (the mathscore
    classifier's score, in range [0, 1]). OpenWebMath adaptively changes the
    filtering thresholds based on whether LaTeX was detected in a page. If the
    URL has LaTeX, the threshold is 0.15. If the URL doesn't have LaTeX, the
    threshold is 0.8. To unify these two into a single score, we add a constant
    value of 0.65 to the math classifier score of all URLs that contain LaTeX
    (the "regex-adjusted score"). These regex-adjusted scores are clipped
    between 0.0 and 1.0 .

    To resample the data, we discretize the regex-adjusted scores into 10
    buckets (with `min(int(parsed_line["regex_adjusted_score"] / 0.1), 9)`). The
    training dataset is balanced, i.e., the resultant dataset has an equal # of
    examples in each of these bins. We also generate two test datasets:

        - The cc-distributed test set follows the same quality distribution as
          CC (computed on a sample, where CC files are denoted by `cc_prefix`).
        - The balanced test set has an equal # of examples in each bin.

    In addition, the train and test sets are split s.t. domains that appear in
    train do not appear in test. On the other hand, there may be overlap (in
    either domains or URLs) between the CC-distributed test set and the balanced
    test set.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Set the random seed for reproducibility
    random.seed(0)

    train_success_path = train_output_path + ".SUCCESS"
    cc_test_success_path = cc_test_output_path + ".SUCCESS"
    balanced_test_success_path = balanced_test_output_path + ".SUCCESS"
    # Skip if we've already written success files for the training and test datasets.
    if (
        fsspec_exists(train_success_path)
        and fsspec_exists(cc_test_success_path)
        and fsspec_exists(balanced_test_success_path)
    ):
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
    num_cc_filepaths = 0
    cc_discrete_labels_to_counts = Counter()

    for filepath in tqdm(input_filepaths, desc="Reading input filepaths"):
        with fsspec.open(filepath, "rt", compression="infer") as f:
            # Use readlines to encourage fsspec to load everything into memory
            # at once, rather than making multiple requests to GCS.
            for line in tqdm(f.readlines(), desc="Reading input filepath"):
                parsed_line = json.loads(line)
                parsed_line["score"] = clip(0.0, parsed_line["score"], 1.0)
                if parsed_line["url"] in seen_urls:
                    num_skipped += 1
                    continue

                parsed_line["regex_adjusted_score"] = (
                    parsed_line["score"] + 0.65 if parsed_line["found_math"] else parsed_line["score"]
                )
                parsed_line["regex_adjusted_score"] = clip(0.0, parsed_line["regex_adjusted_score"], 1.0)

                all_examples.append(parsed_line)
                seen_urls.add(parsed_line["url"])
                if filepath.startswith(cc_prefix):
                    discrete_label = min(int(parsed_line["regex_adjusted_score"] / 0.1), 9)
                    cc_discrete_labels_to_counts[discrete_label] += 1
        if filepath.startswith(cc_prefix):
            num_cc_filepaths += 1

    # Delete the set of all URLs, since we won't need it past this point
    del seen_urls
    logger.info(
        f"Read {len(all_examples)} deduplicated examples, "
        f"{num_skipped} skipped (due to invalid score or duplicate), "
        f"{num_cc_filepaths} CC filepaths used to calculate CC label distribution"
    )

    total_cc_examples = sum(cc_discrete_labels_to_counts.values())
    for label in sorted(cc_discrete_labels_to_counts.keys()):
        rel_count = cc_discrete_labels_to_counts[label] / total_cc_examples
        logger.info(f"Label {label}: {cc_discrete_labels_to_counts[label]} samples, relative={rel_count:.4f}")

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
    def bucket_and_resample(examples, weights: dict[int, int] | None = None):
        # Each example has `url`, `canonicalized_url`, `score`, and `found_math`.
        # Scores range from 0.0 to 1.0.
        # Fixed bucketing logic to bucket examples in increments of 0.1
        # We discretize the labels into buckets from 0-9 with `min(int(score / 0.1), 9)`
        buckets = defaultdict(list)
        for example in examples:
            score = example["regex_adjusted_score"]
            # Bucket by the example's corresponding discrete label.
            label = min(int(score / 0.1), 9)
            buckets[label].append(example)

        logger.info("Bucketing complete. Counts per label:")
        for label in sorted(buckets):
            logger.info(f"Label {label}: {len(buckets[label])}")

        if not weights:
            # If no weights provided, resample to ensure even distribution across labels
            max_samples_per_label = min(len(bucket) for bucket in buckets.values())
            resampled_examples = []
            for bucket in buckets.items():
                resampled_examples.extend(random.sample(bucket, k=max_samples_per_label))
        else:
            # Resample according to provided weights
            logger.info(f"Got label weights: {weights}")
            # Compute scaling factor S
            sum_weights = sum(weights.values())
            # Avoid division by zero if sum_weights == 0
            if sum_weights == 0:
                logger.warning("All weights are zero, will return empty sample.")
                return []

            # For each label, determine the maximum S that doesn't exceed that bucket's size
            # We want to maximize S subject to S * weights[label] <= len(buckets[label])
            # => S <= len(buckets[label]) / weights[label]
            # If weights[label] = 0, that label should get no samples, so skip that constraint.
            feasible_S_values = []
            for label, w in weights.items():
                if w > 0:
                    max_S = len(buckets[label]) / w
                    feasible_S_values.append(max_S)
                else:
                    # If weight is zero, no samples needed from that bucket.
                    # This doesn't constrain S, but we won't pick any from it.
                    pass

            if not feasible_S_values:
                logger.warning("No positive weights provided, will return empty sample.")
                return []

            S = min(feasible_S_values)

            # Now compute the number of samples for each label
            resampled_examples = []
            sample_counts = {}
            for label, bucket in buckets.items():
                w = weights.get(label, 0)
                # Number of samples from this bucket
                count = math.floor(S * w) if w > 0 else 0
                chosen = random.sample(bucket, k=count)
                resampled_examples.extend(chosen)
                sample_counts[label] = count

            # Log the resulting distribution after sampling
            total_samples = sum(sample_counts.values())
            logger.info("Resulting sampled label distributions:")
            for label in sorted(sample_counts.keys()):
                rel_count = sample_counts[label] / total_samples
                logger.info(f"Label {label}: {sample_counts[label]} samples, relative={rel_count:.4f}")

        logger.info(f"Resampling complete, got {len(resampled_examples)} examples in total")
        return resampled_examples

    # Resample train examples
    resampled_train_examples = bucket_and_resample(train_examples)
    # Resample test examples to match CC distribution
    cc_resampled_test_examples = bucket_and_resample(test_examples, cc_discrete_labels_to_counts)
    # Resample test examples to be balanced
    balanced_resampled_test_examples = bucket_and_resample(test_examples)

    # Write train set
    write_examples_to_parquet(resampled_train_examples, train_output_path, train_success_path)

    # Write CC-distributed test set
    write_examples_to_parquet(cc_resampled_test_examples, cc_test_output_path, cc_test_success_path)

    # Write balanced test set
    write_examples_to_parquet(balanced_resampled_test_examples, balanced_test_output_path, balanced_test_success_path)


def write_examples_to_parquet(examples: list[dict], output_path: str, output_success_path: str):
    table = pa.Table.from_pylist(examples)
    # Get filesystem and path for the train output
    output_fs, output_path_in_fs = fsspec.core.url_to_fs(output_path)
    # Write the resampled train examples to the output path as Parquet
    logger.info(f"Writing {len(examples)} train examples")
    pq.write_table(table, output_path_in_fs, filesystem=output_fs, compression="snappy")
    logger.info(f"Wrote {len(examples)} train examples")
    # Write success files indicating the number of examples written
    with fsspec.open(output_success_path, "wt", compression="infer") as f:
        f.write(json.dumps({"num_examples": len(examples)}))


@draccus.wrap()
def resample_and_split_urls(cfg: ResamplingConfig):
    ray.get(
        resample_urls.remote(
            cfg.input_patterns,
            cfg.cc_prefix,
            cfg.train_output_path,
            cfg.cc_test_output_path,
            cfg.balanced_test_output_path,
            cfg.test_size,
        )
    )


if __name__ == "__main__":
    resample_urls()
