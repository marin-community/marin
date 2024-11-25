#!/usr/bin/env python3
"""
Given Dolma-format FineWeb-Edu examples containing HTML, get the URL of each page and
its associated quality classifier score.

Running on FineWeb-Edu:

```
for fineweb_edu_dump_html_path in $(gcloud storage ls gs://marin-us-central2/documents/fineweb-edu/html); do
    dump_name=$(basename -- ${fineweb_edu_dump_html_path})
    ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python scripts/fineweb-edu/get_urls_and_scores_from_fineweb_edu_html.py \
    --html_input_path ${fineweb_edu_dump_html_path} \
    --prefix fineweb_edu \
    --output_path gs://marin-us-central2/scratch/nfliu/urls_and_scores/fineweb-edu/${dump_name}
done
```
"""

import json
import logging
import os
import pathlib
from dataclasses import dataclass
from marin.utils import remove_tpu_lockfile_on_exit
from ray.runtime_env import RuntimeEnv
import math

import draccus
import fsspec
import ray
from tqdm_loggable.auto import tqdm

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class UrlsAndScoresExtractionConfig:
    html_input_path: str
    prefix: str
    output_path: str


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


@ray.remote(
    memory=16 * 1024 * 1024 * 1024,
    num_cpus=8,
    resources={"TPU": 4, "TPU-v4-8-head": 1},
    runtime_env=RuntimeEnv(pip="scripts/fineweb-edu/get_urls_and_scores_from_fineweb_edu_html_requirements.txt"),
)
@remove_tpu_lockfile_on_exit
@cached_or_construct_output(
    success_suffix="SUCCESS", verbose=False
)  # We use this decorator to make this function idempotent
def process_one_batch(input_path: str, output_path: str):
    """
    Takes in an input file, extracts the outlinks, and writes them to output_path.

    Args:
    input_path (str): Path of HTML file (Dolma-format JSONL) to extract outlinks from.
    output_path (str): Path to write JSONL file with outlinks.
    """
    from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
    import trafilatura
    from trafilatura import extract
    import w3lib.url

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info(f"Using trafilatura version {trafilatura.__version__}")
    logger.info("Loading quality classifier...")
    # Load the quality classifier
    model = FlaxAutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
    logger.info("Loaded quality classifier...")

    with fsspec.open(input_path, "rt", compression="gzip") as fin:
        input_lines = fin.readlines()

    num_examples_skipped = 0
    examples_to_classify = []
    for line in tqdm(input_lines, desc="Extracting text"):
        record = json.loads(line)
        html_str = record.get("html", "").strip()
        url = record.get("metadata", {}).get("url", "")

        if not html_str or not url:
            num_examples_skipped += 1
            continue

        canonicalized_url = w3lib.url.canonicalize_url(url)

        extracted_text = extract(
            html_str,
            favor_precision=True,
            include_comments=False,
            deduplicate=True,
        )
        if not extracted_text:
            num_examples_skipped += 1
            continue

        examples_to_classify.append(
            {"url": url, "canonicalized_url": canonicalized_url, "extracted_text": extracted_text}
        )
    logger.info(
        f"Got {len(examples_to_classify)} examples to classify."
        f"{len(input_lines)} examples in total, "
        f"{num_examples_skipped} examples skipped due to failed extraction"
    )

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


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def get_shards_indices_to_process(shard_path: str, prefix: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".jsonl.gz").removeprefix(f"{prefix}_"))
        for path in fsspec_glob(os.path.join(shard_path, f"{prefix}_*.jsonl.gz"))
    ]
    shard_indices = sorted(shard_indices)
    logger.info(f"Found {len(shard_indices)} shards to process")
    return shard_indices


@draccus.wrap()
def get_urls_and_scores_from_html(cfg: UrlsAndScoresExtractionConfig):
    shard_indices = ray.get(get_shards_indices_to_process.remote(cfg.html_input_path, cfg.prefix))

    refs = []
    for i, html_shard_index in enumerate(shard_indices):
        input_path = os.path.join(cfg.html_input_path, f"{cfg.prefix}_{html_shard_index}.jsonl.gz")
        output_path = os.path.join(cfg.output_path, f"{i}_urls_and_quality_classifier_scores.jsonl.gz")
        refs.append(process_one_batch.remote(input_path, output_path))
    logger.info(f"Submitted {len(refs)} tasks")

    # Wait for the tasks to finish
    _ = ray.get(refs)


if __name__ == "__main__":
    get_urls_and_scores_from_html()
