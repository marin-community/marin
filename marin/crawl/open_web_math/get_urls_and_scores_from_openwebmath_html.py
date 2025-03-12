#!/usr/bin/env python3
"""
Given Dolma-format OpenWebMath examples containing HTML, get the URL of each page and
its associated quality classifier score.

Running on open-web-math:

```
python marin/run/ray_run.py --no_wait \
    --pip_deps 'resiliparse,fasttext,lxml,py-asciimath,tabulate,w3lib' \
    -- \
    python marin/crawl/open-web-math/get_urls_and_scores_from_openwebmath_html.py \
    --html_input_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/ \
    --prefix openwebmath \
    --output_path gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math/
```
"""
import json
import logging
import os
import pathlib
from dataclasses import asdict, dataclass

import draccus
import fsspec
import ray
import w3lib.url
from tqdm_loggable.auto import tqdm

from marin.core.runtime import cached_or_construct_output
from marin.processing.classification.classifier import FasttextClassifier
from marin.processing.open_web_math.extract import extract_text
from marin.processing.open_web_math.text_normalizer import normalize
from marin.processing.open_web_math.utils import Config as OpenWebMathConfig
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UrlsAndScoresExtractionConfig:
    html_input_path: str
    prefix: str
    output_path: str


@dataclass(frozen=True)
class OpenWebMathUrlWithScore:
    url: str
    canonicalized_url: str
    score: float
    found_math: bool


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


def score_line(line, mathscore_model, randomized_config):
    record = json.loads(line)
    html_str = str(record.get("html", "")).strip()
    url = record.get("metadata", {}).get("url", "")

    if not html_str or not url:
        return None

    randomized_config_sample = randomized_config.sample()
    try:
        extraction_result = extract_text(html_str, randomized_config_sample, fast=True)
    except Exception:
        return None

    if extraction_result is None:
        return None

    extracted_text, extraction_metadata = extraction_result
    score = score_text(extracted_text, mathscore_model)
    canonicalized_url = w3lib.url.canonicalize_url(url)
    found_math = extraction_metadata["found_math"]
    return asdict(
        OpenWebMathUrlWithScore(url=url, canonicalized_url=canonicalized_url, score=score, found_math=found_math)
    )


@ray.remote(
    memory=32 * 1024 * 1024 * 1024,
    num_cpus=8,
)
@cached_or_construct_output(success_suffix="SUCCESS", verbose=False)
def process_one_batch(input_path: str, output_path: str):
    """
    Takes in an input file, get the URLs and the quality classifier scores from the text,
    and writes them to output_path.

    Args:
    input_path (str): Path of HTML file (Dolma-format JSONL) to extract outlinks from.
    output_path (str): Path to write JSONL file with outlinks.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Reading input path with HTML")
    with fsspec.open(input_path, "rt", compression="gzip") as fin:
        input_lines = fin.readlines()
    logger.info("Finished reading input path with HTML")

    logger.info("Loading mathscore model")
    # Load the model
    # NOTE: we don't use the attribute functionality in FasttextClassifier
    mathscore_model = FasttextClassifier(model_name="open-web-math/filtering-models", attribute_name="")
    logger.info("Loaded mathscore model")

    num_examples_written = 0
    num_examples_skipped = 0

    randomized_config = OpenWebMathConfig(os.path.join(os.path.dirname(__file__), "configs", "randomized_all.yaml"))
    with fsspec.open(output_path, "w", compression="gzip") as fout:
        for result in tqdm(
            map(lambda line: score_line(line, mathscore_model, randomized_config), input_lines),
            total=len(input_lines),
            desc="Scoring",
        ):
            if result is None:
                num_examples_skipped += 1
                continue
            fout.write(json.dumps(result) + "\n")
            num_examples_written += 1

    logger.info(
        f"Got {num_examples_written} (url, score) pairs. "
        f"{len(input_lines)} examples in total, "
        f"{num_examples_skipped} examples skipped due to failed extraction"
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
    logger.info(f"Submitted {len(refs)} tasks to score URLs")

    # Wait for the tasks to finish
    ray.get(refs)


if __name__ == "__main__":
    get_urls_and_scores_from_html()
