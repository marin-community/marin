#!/usr/bin/env python3
"""
Given a CC dump, randomly sample the specified number of WARCs from this
dump. Then, extract the URL and text of each record and score the text with the
open-web-math quality classifier.

```
for dump_name in CC-MAIN-2022-05 CC-MAIN-2022-21 CC-MAIN-2022-27 CC-MAIN-2022-33 CC-MAIN-2022-40 CC-MAIN-2022-49 CC-MAIN-2023-06 CC-MAIN-2023-14; do
for dump_name in CC-MAIN-2022-05; do
    python marin/run/ray_run.py \
    --pip_deps 'resiliparse,fasttext,lxml,py-asciimath,tabulate,w3lib,warcio' \
    --no_wait -- \
    python scripts/open-web-math/get_urls_and_openwebmath_scores_from_warcs.py \
    --cc_dump ${dump_name} \
    --num_warcs_to_sample 500 \
    --output_path gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-cc/${dump_name}
done
```
"""
import gzip
import json
import logging
import os
from dataclasses import dataclass
import random
from io import BytesIO

import draccus
import fsspec
import ray
import requests
import w3lib.url
from tqdm_loggable.auto import tqdm
from warcio import ArchiveIterator
from resiliparse.parse.encoding import detect_encoding, bytes_to_str

from marin.core.runtime import cached_or_construct_output
from marin.processing.open_web_math.extract import extract_text
from marin.processing.open_web_math.utils import Config as OpenWebMathConfig
from marin.processing.open_web_math.text_normalizer import normalize
from marin.processing.classification.classifier import FasttextClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CCUrlsAndScoresExtractionConfig:
    cc_dump: str
    num_warcs_to_sample: int
    output_path: str


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


@ray.remote(
    memory=32 * 1024 * 1024 * 1024,
    num_cpus=8,
)
@cached_or_construct_output(success_suffix="SUCCESS", verbose=False)
def process_one_batch(warc_path: str, output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    response = requests.get(warc_path)
    response.raise_for_status()
    # Load the response content into memory
    file_in_memory = BytesIO(response.content)

    logger.info("Loading mathscore model")
    # Load the model
    # NOTE: we don't use the attribute functionality in FasttextClassifier
    mathscore_model = FasttextClassifier(model_name="open-web-math/filtering-models", attribute_name="")
    logger.info("Loaded mathscore model")

    randomized_config = OpenWebMathConfig(os.path.join(os.path.dirname(__file__), "configs", "randomized_all.yaml"))

    # Iterate through the WARC, decompressing as we go.
    num_records_saved = 0
    num_records_skipped = 0
    with gzip.open(file_in_memory, "rb") as file_stream, fsspec.open(output_path, "w", compression="gzip") as fout:
        for record in tqdm(ArchiveIterator(file_stream)):
            if record.rec_type == "response":
                record_url = record.rec_headers.get_header("WARC-Target-URI")
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

                extracted_text, _ = extraction_result
                score = score_text(extracted_text, mathscore_model)
                canonicalized_url = w3lib.url.canonicalize_url(record_url)

                fout.write(
                    json.dumps({"url": record_url, "canonicalized_url": canonicalized_url, "score": score}) + "\n"
                )
                num_records_saved += 1
    logger.info(f"Saved {num_records_saved} records from WARC, skipped {num_records_skipped} records")


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def get_warc_paths_to_process(cc_dump: str, num_warcs_to_sample):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Download the WARC paths for the specified CC dump
    warc_paths = []
    with fsspec.open(f"https://data.commoncrawl.org/crawl-data/{cc_dump}/warc.paths.gz", "rt", compression="infer") as f:
        for line in f:
            warc_paths.append(line.strip())
    # Seed for reproducibility
    random.seed(0)
    # Shuffle and slice instead of using random.sample(), so we can modify the
    # number of WARCs to sample while still using the previously-sampled WARCs.
    random.shuffle(warc_paths)
    return [f"https://data.commoncrawl.org/{path}" for path in warc_paths[:num_warcs_to_sample]]


@draccus.wrap()
def get_urls_and_scores_from_warcs(cfg: CCUrlsAndScoresExtractionConfig):
    warc_paths = ray.get(get_warc_paths_to_process.remote(cfg.cc_dump, cfg.num_warcs_to_sample))

    refs = []
    for warc_path in warc_paths:
        warc_name = os.path.basename(warc_path)
        output_path = os.path.join(cfg.output_path, f"{warc_name}_urls_and_quality_classifier_scores.jsonl.gz")
        refs.append(process_one_batch.remote(warc_path, output_path))
    logger.info(f"Submitted {len(refs)} tasks to score URLs from WARCs")

    # Wait for the tasks to finish
    _ = ray.get(refs)


if __name__ == "__main__":
    get_urls_and_scores_from_warcs()
