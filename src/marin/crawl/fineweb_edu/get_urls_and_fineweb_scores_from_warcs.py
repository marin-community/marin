#!/usr/bin/env python3
"""
Given a CC dump, randomly sample the specified number of WARCs from this
dump. Then, extract the URL and text of each record and score the text with the
FineWeb-Edu quality classifier.

```
python marin/run/ray_run.py \
    --pip_deps '--find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html,w3lib,trafilatura,jax[tpu],flax,transformers,requests,warcio,resiliparse' \
    --no_wait -- \
    python marin/crawl/fineweb_edu/get_urls_and_fineweb_scores_from_warcs.py \
    --cc_dumps '["CC-MAIN-2022-05", "CC-MAIN-2022-21", "CC-MAIN-2022-27", "CC-MAIN-2022-33", "CC-MAIN-2022-40", "CC-MAIN-2022-49", "CC-MAIN-2023-06", "CC-MAIN-2023-14", "CC-MAIN-2023-23", "CC-MAIN-2023-40", "CC-MAIN-2023-50", "CC-MAIN-2024-10"]' \
    --num_warcs_to_sample 500 \
    --output_path gs://marin-us-central2/scratch/nfliu/urls_and_scores/fineweb-edu-cc/
done
```
"""  # noqa: E501
import gzip
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
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

from marin.core.runtime import cached_or_construct_output
from marin.crawl.common.utils import decode_html
from marin.utils import remove_tpu_lockfile_on_exit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CCUrlsAndScoresExtractionConfig:
    cc_dumps: list[str]
    num_warcs_to_sample: int
    output_path: str


@dataclass(frozen=True)
class FineWebEduExtractedText:
    url: str
    canonicalized_url: str
    extracted_text: str


@dataclass(frozen=True)
class FineWebEduUrlWithScore:
    url: str
    canonicalized_url: str
    score: float


def batched(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


@ray.remote(
    memory=64 * 1024 * 1024 * 1024,
    num_cpus=4,
)
@cached_or_construct_output(success_suffix="SUCCESS", verbose=False)
def extract_text_from_warc(warc_path: str, output_path: str):
    """
    Given a path to a WARC, extract text from its responses.

    Args:
    warc_path (str): path to input WARC
    output_path (str): path to write extracted text records as
                       JSONL-formatted `FineWebEduExtractedText` records.
    """
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
                        asdict(
                            FineWebEduExtractedText(
                                url=record_url, canonicalized_url=canonicalized_url, extracted_text=extracted_text
                            )
                        )
                    )
                    + "\n"
                )
                num_records_saved += 1
    logger.info(f"Saved {num_records_saved} records from WARC, skipped {num_records_skipped} records")


@ray.remote(
    memory=32 * 1024 * 1024 * 1024,
    num_cpus=8,
    resources={"TPU": 4, "TPU-v4-8-head": 1},
)
@remove_tpu_lockfile_on_exit
@cached_or_construct_output(success_suffix="SUCCESS", verbose=False)
def score_extracted_text(input_path: str, output_path: str):
    """
    Given an input path to extracted text (JSONL-formatted `FineWebEduUrlWithScore`),
    run the FineWeb-Edu classifier to get a quality score for the extracted text.
    The output is written to output_path, as JSONL-formatted `FineWebEduUrlWithScore`
    records.

    Args:
    input_path (str): Path to JSONL-serialized `FineWebEduExtractedText` records
    output_path (str): Path to write JSONL-serialized `FineWebEduUrlWithScore` records
    """
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
        ) in zip(examples_to_classify, examples_scores, strict=True):
            fout.write(
                json.dumps(
                    asdict(
                        FineWebEduUrlWithScore(
                            url=example["url"],
                            canonicalized_url=example["canonicalized_url"],
                            score=example_score,
                        )
                    )
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


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def get_urls_and_scores_from_warcs(cc_dump: str, output_path: str, num_warcs_to_sample: int):
    warc_paths = ray.get(get_warc_paths_to_process.remote(cc_dump, num_warcs_to_sample))

    refs = []
    for warc_path in warc_paths:
        warc_name = os.path.basename(warc_path)
        output_path = os.path.join(output_path, f"{warc_name}_extracted_text.jsonl.gz")
        refs.append(extract_text_from_warc.remote(warc_path, output_path))
    logger.info(f"Submitted {len(refs)} tasks to extract text")
    ray.get(refs)

    refs = []
    for warc_path in warc_paths:
        warc_name = os.path.basename(warc_path)
        input_path = os.path.join(output_path, f"{warc_name}_extracted_text.jsonl.gz")
        output_path = os.path.join(output_path, f"{warc_name}_urls_and_quality_classifier_scores.jsonl.gz")
        refs.append(score_extracted_text.remote(input_path, output_path))
    logger.info(f"Submitted {len(refs)} tasks to run quality classifier")
    ray.get(refs)


@draccus.wrap()
def get_urls_and_scores_from_dumps(cfg: CCUrlsAndScoresExtractionConfig):
    refs = []
    logger.info(f"Got {len(cfg.cc_dumps)} CC dumps to process")
    for cc_dump in cfg.cc_dumps:
        dump_output_path = os.path.join(cfg.output_path, cc_dump)
        refs.append(get_urls_and_scores_from_warcs.remote(cc_dump, dump_output_path, cfg.num_warcs_to_sample))
    ray.get(refs)


if __name__ == "__main__":
    get_urls_and_scores_from_warcs()
