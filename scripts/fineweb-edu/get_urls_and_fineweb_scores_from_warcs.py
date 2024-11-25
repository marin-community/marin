#!/usr/bin/env python3
"""
Given a CC dump, randomly sample the specified number of WARCs from this
dump. Then, extract the URL and text of each record and score the text with the
FineWeb-Edu quality classifier.

```
for fineweb_edu_dump_html_path in $(gcloud storage ls gs://marin-us-central2/documents/fineweb-edu/html); do
    dump_name=$(basename -- ${fineweb_edu_dump_html_path})
    python marin/run/ray_run.py \
    --pip_deps '--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html,w3lib,trafilatura,jax[tpu],flax,transformers,requests,warcio,resiliparse' \
    --no_wait -- \
    python scripts/fineweb-edu/get_urls_and_fineweb_scores_from_warcs.py \
    --cc_dump ${dump_name} \
    --num_warcs_to_sample 100 \
    --output_path gs://marin-us-central2/scratch/nfliu/urls_and_scores/fineweb-edu-cc/${dump_name}
done
```
"""
import gzip
import json
import logging
import math
import os
from dataclasses import dataclass
import random
from io import BytesIO

import draccus
import fsspec
import ray
import requests
import trafilatura
import w3lib.url
from tqdm_loggable.auto import tqdm
from trafilatura import extract
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
from warcio import ArchiveIterator
from resiliparse.parse.encoding import detect_encoding, bytes_to_str

from marin.core.runtime import cached_or_construct_output
from marin.utils import remove_tpu_lockfile_on_exit

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


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


@ray.remote(
    memory=32 * 1024 * 1024 * 1024,
    num_cpus=120,
)
@cached_or_construct_output(success_suffix="SUCCESS", verbose=False)
def extract_text_from_warc(warc_path: str, output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(f"Using trafilatura version {trafilatura.__version__}")

    response = requests.get(warc_path)
    response.raise_for_status()
    # Load the response content into memory
    file_in_memory = BytesIO(response.content)

    # Iterate through the WARC, decompressing as we go.
    num_records_saved = 0
    num_records_skipped = 0
    with gzip.open(file_in_memory, "rb") as file_stream, fsspec.open(output_path, "w", compression="gzip") as fout:
        for record in tqdm(ArchiveIterator(file_stream)):
            if record.rec_type == "response":
                record_url = record.rec_headers.get_header("WARC-Target-URI")
                content = record.content_stream().read()
                html_decoded: str | None = decode_html(content)
                if not html_decoded:
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

                fout.write(
                    json.dumps(
                        {"url": record_url, "canonicalized_url": canonicalized_url, "extracted_text": extracted_text}
                    )
                    + "\n"
                )
                num_records_saved += 1
    logger.info(f"Saved {num_records_saved} records from WARC, skipped {num_records_skipped} records")


@ray.remote(
    memory=32 * 1024 * 1024 * 1024,
    num_cpus=120,
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
    with fsspec.open(input_path, "rt", compression="gzip") as fin:
        examples_to_classify = json.load(fin)
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
        output_path = os.path.join(cfg.output_path, f"{warc_name}_extracted_text.jsonl.gz")
        refs.append(extract_text_from_warc.remote(warc_path, output_path))
    logger.info(f"Submitted {len(refs)} tasks to extract text")

    # Wait for the tasks to finish
    _ = ray.get(refs)

    refs = []
    for warc_path in warc_paths:
        warc_name = os.path.basename(warc_path)
        input_path = os.path.join(cfg.output_path, f"{warc_name}_extracted_text.json.gz")
        output_path = os.path.join(cfg.output_path, f"{warc_name}_urls_and_quality_classifier_scores.jsonl.gz")
        refs.append(process_one_batch.remote(input_path, output_path))
    logger.info(f"Submitted {len(refs)} tasks to run quality classifier")

    # Wait for the tasks to finish
    _ = ray.get(refs)


if __name__ == "__main__":
    get_urls_and_scores_from_warcs()
