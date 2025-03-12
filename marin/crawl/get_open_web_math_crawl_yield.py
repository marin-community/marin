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

Running on OpenWebMath-10M:

```
python marin/run/ray_run.py \
    --pip_deps 'resiliparse,fasttext,lxml,py-asciimath,tabulate,warcio[all],w3lib,cchardet,kenlm' \
    --no_wait -- \
    python marin/crawl/get_open_web_math_crawl_yield.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/ \
    --crawl_input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/ \
    --data_source open-web-math-fde8ef8-10M \
    --text_output_directory gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/ \
    --urls_and_scores_output_directory gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-fde8ef8-10M/ \
    --statistics_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M/yield_statistics.json.gz
```

Running on OpenWebMath-10M-cc-deduplicated:

```
python marin/run/ray_run.py \
    --pip_deps 'resiliparse,fasttext,lxml,py-asciimath,tabulate,warcio[all],w3lib,cchardet,kenlm' \
    --no_wait -- \
    python marin/crawl/get_open_web_math_crawl_yield.py \
    --urls_input_directory gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --crawl_input_directory gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --data_source open-web-math-fde8ef8-10M-cc-deduplicated \
    --text_output_directory gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --urls_and_scores_output_directory gs://marin-us-central2/scratch/nfliu/urls_and_scores/open-web-math-fde8ef8-10M-cc-deduplicated/ \
    --statistics_output_path gs://marin-us-central2/scratch/nfliu/fetched_outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/yield_statistics.json.gz
```
"""  # noqa: E501
import atexit
import hashlib
import json
import logging
import os
import pathlib
import random
import re
import shutil
from dataclasses import asdict, dataclass
from typing import Any

import draccus
import fsspec
import kenlm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import w3lib.url
from filelock import FileLock
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import bytes_to_str, detect_encoding
from resiliparse.parse.html import HTMLTree
from tqdm_loggable.auto import tqdm
from warcio import ArchiveIterator

from marin.processing.classification.classifier import FasttextClassifier
from marin.processing.open_web_math.extract import extract_text
from marin.processing.open_web_math.manual_filter import manual_url_filter
from marin.processing.open_web_math.text_normalizer import normalize
from marin.processing.open_web_math.utils import Config as OpenWebMathConfig
from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MATH_KEYWORDS = [
    "MathJax",
    "mathjax",
    "<math",
    "math-container",
    "katex.min.css",
    "latex.php",
    "codecogs",
    "tex.cgi",
    'class="tex"',
    "class='tex'",
]
latex_math_commands = [
    "\\end",
    "\\begin",
    "\\ref",
    "\\frac",
    "\\label",
    "\\bf",
    "\\right",
    "\\left",
    "\\rm",
    "\\alpha",
    "\\mu",
    "\\def",
    "\\it",
    "\\pi",
    "\\sigma",
    "\\sum",
    "\\lambda",
    "\\beta",
    "\\nu",
    "\\partial",
    "\\int",
    "\\delta",
    "\\rho",
    "\\phi",
    "\\gamma",
    "\\omega",
    "\\over",
    "\\nonumber",
    "\\bar",
    "\\sqrt",
    "\\theta",
    "\\tau",
    "\\em",
    "\\rangle",
    "\\hat",
    "\\tilde",
    "\\cal",
    "\\hline",
    "\\item",
    "\\psi",
    "\\vec",
    "\\langle",
    "\\epsilon",
    "\\eta",
    "\\cdot",
    "\\in",
    "\\xi",
    "\\infty",
    "\\quad",
    "\\mathcal",
    "\\times",
    "\\emph",
    "\\mathbf",
    "\\prime",
    "\\be",
    "\\mathrm",
    "\\ee",
    "\\vspace",
    "\\pm",
    "\\chi",
    "\\ell",
    "\\text",
    "\\qquad",
    "\\noindent",
    "\\to",
    "\\varphi",
    "\\hspace",
    "\\leq",
    "\\cos",
    "\\eqref",
    "\\overline",
    "\\sin",
    "\\kappa",
    "\\hbox",
    "\\rightarrow",
    "\\varepsilon",
    "\\textit",
    "\\dagger",
    "\\big",
    "\\otimes",
    "\\equiv",
    "\\zeta",
    "\\dot",
    "\\ln",
]
latex_regex = re.compile("\\\\[a-z]{2,}")
original_regex = re.compile("|".join(MATH_KEYWORDS))


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


def is_english(text, lid_model):
    normalized_text = normalize(text).replace("\n", " ")
    # Request all labels/probabilities (k=-1)
    labels, probs = lid_model.predict(normalized_text)
    # Create a dictionary {label: probability}
    label_probs = dict(zip(labels, probs, strict=True))
    # Get the probability of English.
    # Sometimes, "__label__en" doesn't show up in the predicted labels
    # (e.g., if the input text is all Chinese characters). In that case, we'll
    # just default to a probability of zero.
    en_prob = label_probs.get("__label__en", 0.0)
    # NOTE: This is not a typo, the original open-web-math
    # code release checks that both:
    # (1) the highest-scoring label is 'en'
    # (2) the probability is >= 0.5
    is_en = en_prob >= 0.5
    return is_en, en_prob


def document_perplexity(text, lm):
    text = normalize(text)
    score = lm.score(text)
    return 10 ** (-score / len(text.split()))


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


def contains_math_prefilter(data, score_model):
    original_match = original_regex.search(data)
    if original_match:
        return True
    latex_match = latex_regex.search(data)
    text = ""
    if latex_match:
        data = data.replace("<template", "<div")
        data = data.replace("</template", "</div")
        tree = HTMLTree.parse(data)
        text = extract_plain_text(tree, main_content=True, alt_texts=False)
        for term in latex_math_commands:
            if term in text:
                return True
        score = score_text(text, score_model)
        if score > 0.8 and len(text) > 500:
            return True

    return False


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
    passing_urls_and_scores_output_path: str,
    failing_urls_and_scores_output_path: str,
    passing_text_and_scores_output_path: str,
    failing_text_and_scores_output_path: str,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    success_path = passing_urls_and_scores_output_path + ".SUCCESS"
    if fsspec_exists(success_path):
        logger.info(f"Success path {success_path} already exists, skipping...")
        with fsspec.open(success_path, block_size=1 * 1024 * 1024 * 1024) as f:
            shard_stats = json.load(f)
        return (
            shard_stats["total_urls"],
            shard_stats["total_urls_fetched"],
            shard_stats["total_urls_passing"],
        )

    with fsspec.open(urls_path, block_size=1 * 1024 * 1024 * 1024) as f:
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

    logger.info("Loading language ID model")
    # NOTE: we don't use the attribute functionality in FasttextClassifier
    lid_model = FasttextClassifier(model_name="julien-c/fasttext-language-id", attribute_name="", k=-1)
    logger.info("Loaded language ID model")

    logger.info("Loading language model for perplexity filtering")
    # TODO(nfliu): make this more sophisticated with a filelock to prevent repeated downloads on the same host.
    LM_URL = "https://huggingface.co/open-web-math/filtering-models/resolve/main/lm-v2.binary"
    model_descriptor = hashlib.md5(LM_URL.encode()).hexdigest()
    lock_file = f"/tmp/{model_descriptor}.lock"
    success_file = f"/tmp/{model_descriptor}.success"
    local_filepath = f"/tmp/{model_descriptor}/lm-v2.binary"
    if os.path.exists(success_file) and not os.path.exists(local_filepath):
        logger.info(
            f"Warning: Success file found for {LM_URL}, but model file not found. "
            f"Removing stale success file {success_file}"
        )
        os.unlink(success_file)

    with FileLock(lock_file):
        if not os.path.exists(success_file):
            os.makedirs(f"/tmp/{model_descriptor}", exist_ok=True)
            with fsspec.open(LM_URL, "rb", block_size=1 * 1024 * 1024 * 1024) as src, open(local_filepath, "wb") as dst:
                shutil.copyfileobj(src, dst)
            atexit.register(lambda: os.unlink(local_filepath))
            logger.info(f"Downloaded model from {LM_URL} to {local_filepath}")
            with open(success_file, "w") as f:
                f.write("success")
            atexit.register(lambda: os.unlink(success_file))
        else:
            logger.info(f"Model already downloaded to {local_filepath}")

    assert os.path.exists(success_file) and os.path.exists(local_filepath), f"Model file {local_filepath} not found"
    lm = kenlm.Model(local_filepath)
    logger.info("Loaded language model for perplexity filtering")

    randomized_config = OpenWebMathConfig(
        os.path.join(os.path.dirname(__file__), "open-web-math", "configs", "randomized_all.yaml")
    )

    fetched_urls = set()
    num_records_skipped = 0
    num_records_passing = 0
    num_records_saved = 0

    passing_urls_and_scores_output_records = []
    failing_urls_and_scores_output_records = []
    passing_text_and_scores_output_records = []
    failing_text_and_scores_output_records = []
    with fsspec.open(warc_path, "rb", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as file_stream:
        for record in tqdm(ArchiveIterator(file_stream)):
            if record.rec_type == "response":
                record_url = record.rec_headers.get_header("WARC-Target-URI")
                fetched_urls.add(record_url)

                content = record.content_stream().read()
                html_decoded: str | None = decode_html(content)
                if not html_decoded or not record_url:
                    num_records_skipped += 1
                    continue

                # Apply the prefilter
                passes_prefilter = contains_math_prefilter(html_decoded, mathscore_model)

                # Extract text, randomizing whether it's markdown or plaintext
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
                if not extracted_text.strip() or not normalize(extracted_text).strip():
                    num_records_skipped += 1
                    continue

                # Apply the language ID filter
                passes_langid_filter, en_probability = is_english(extracted_text, lid_model)
                # Apply the perplexity filter
                perplexity = document_perplexity(extracted_text, lm)
                passes_perplexity_filter = perplexity <= 15_000
                # Apply the mathscore filter
                found_math = extraction_metadata["found_math"]
                score = score_text(extracted_text, mathscore_model)
                passes_mathscore_filter = (found_math and score > 0.17) or (not found_math and score > 0.8)
                # Apply the post-hoc manual open-web-math filter
                passes_manual_filter, new_text = manual_url_filter(url=record_url, original_text=extracted_text)

                canonicalized_url = w3lib.url.canonicalize_url(record_url)
                urls_and_scores_record = {
                    "url": record_url,
                    "canonicalized_url": canonicalized_url,
                    "passes_prefilter": passes_prefilter,
                    "passes_langid_filter": passes_langid_filter,
                    "en_probability": en_probability,
                    "passes_perplexity_filter": passes_perplexity_filter,
                    "passes_manual_filter": passes_manual_filter,
                    "perplexity": perplexity,
                    "found_math": found_math,
                    "score": score,
                    "passes_mathscore_filter": passes_mathscore_filter,
                }
                record_id = record.rec_headers.get_header("WARC-Record-ID")
                assert record_id
                record_date = record.rec_headers.get_header("WARC-Date")
                assert record_date

                text_and_scores_record = asdict(
                    DolmaFormattedOpenWebMathRecord(
                        id=record_id,
                        source=data_source,
                        format="text",
                        text=new_text,
                        metadata={
                            **urls_and_scores_record,
                            "date": record_date,
                            "file_path": warc_path,
                        },
                    )
                )

                passed_quality_filters = all(
                    [
                        passes_prefilter,
                        passes_langid_filter,
                        passes_perplexity_filter,
                        passes_mathscore_filter,
                    ]
                )
                if passed_quality_filters:
                    passing_urls_and_scores_output_records.append(urls_and_scores_record)
                    passing_text_and_scores_output_records.append(text_and_scores_record)
                    num_records_passing += 1
                else:
                    failing_urls_and_scores_output_records.append(urls_and_scores_record)
                    failing_text_and_scores_output_records.append(text_and_scores_record)
                num_records_saved += 1

    # Count the number of URLs that weren't fetched
    unfetched_urls = urls - fetched_urls
    logger.info(f"Out of {len(urls)} URLs to fetch, {len(unfetched_urls)} were not successfully fetched")
    # As a sanity check, count the number of fetched_urls that aren't in the original set. This should hopefully be 0.
    logger.info(f"Out of {len(fetched_urls)} fetched_urls, {len(fetched_urls - urls)} were not in the input set of URLs")
    logger.info(f"{num_records_passing} URLs passed the quality filtering pipeline")
    # Write examples from this shard to parquet
    write_examples_to_parquet(passing_urls_and_scores_output_records, passing_urls_and_scores_output_path)
    write_examples_to_parquet(passing_text_and_scores_output_records, passing_text_and_scores_output_path)
    write_examples_to_parquet(failing_urls_and_scores_output_records, failing_urls_and_scores_output_path)
    write_examples_to_parquet(failing_text_and_scores_output_records, failing_text_and_scores_output_path)

    logger.info(f"Saved {num_records_saved} records from WARC, skipped {num_records_skipped} records")

    with fsspec.open(success_path, "w", block_size=1 * 1024 * 1024 * 1024) as fout:
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
        passing_urls_and_scores_output_path = os.path.join(
            cfg.urls_and_scores_output_directory, f"links.{shard_index}_urls_and_scores.passing.parquet"
        )
        failing_urls_and_scores_output_path = os.path.join(
            cfg.urls_and_scores_output_directory, f"links.{shard_index}_urls_and_scores.failing.parquet"
        )
        passing_text_and_scores_output_path = os.path.join(
            cfg.text_output_directory, f"links.{shard_index}_text_and_scores.passing.parquet"
        )
        failing_text_and_scores_output_path = os.path.join(
            cfg.text_output_directory, f"links.{shard_index}_text_and_scores.failing.parquet"
        )
        unfinished.append(
            get_shard_yield.remote(
                urls_path,
                warc_path,
                robots_path,
                errors_path,
                cfg.data_source,
                passing_urls_and_scores_output_path,
                failing_urls_and_scores_output_path,
                passing_text_and_scores_output_path,
                failing_text_and_scores_output_path,
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
    with fsspec.open(cfg.statistics_output_path, "w", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as fout:
        json.dump(
            {
                "total_urls": total_urls,
                "total_urls_fetched": total_urls_fetched,
                "total_urls_passing": total_urls_passing,
            },
            fout,
        )


if __name__ == "__main__":
    main()
