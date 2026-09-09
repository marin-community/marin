# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned July 2026 Uncheatable Eval release for document scoring."""

import hashlib
import json
import logging
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_ID = "Jellyfish042/UncheatableEval-2026-07"
SOURCE_REVISION = "65889535d56aa38d448ce7e07b08e6e36c031545"
SOURCE_FILE = "data/test-00000-of-00001.parquet"
OUTPUT_PREFIX = "s3://marin-us-east-02a/evaluation/document-losses/uncheatable/2026-07-65889535d56a/data"
EXPECTED_DOCUMENTS = 7500
CATEGORIES = (
    "ao3_english",
    "ao3_nonenglish",
    "arxiv_cs",
    "arxiv_math",
    "arxiv_other",
    "arxiv_physics",
    "bbc_news",
    "biorxiv_all",
    "github_cpp",
    "github_javascript",
    "github_markdown",
    "github_other",
    "github_python",
    "wikipedia_english",
    "wikipedia_nonenglish",
)
logger = logging.getLogger(__name__)


def prepare_manifest(source_path: str, output_path: str) -> dict:
    """Split benchmark content by category, preserving stable source row IDs."""
    counts = Counter()
    dates = {category: [] for category in CATEGORIES}
    with ExitStack() as stack:
        streams = {
            category: stack.enter_context(fsspec.open(f"{output_path}/{category}.jsonl.gz", "wt", compression="gzip"))
            for category in CATEGORIES
        }
        source_index = 0
        for batch in pq.ParquetFile(source_path).iter_batches(columns=["content", "category", "date", "url"]):
            for row in batch.to_pylist():
                category = row["category"]
                if category not in streams:
                    raise ValueError(f"Unknown benchmark category: {category!r}")
                if not isinstance(row["content"], str):
                    raise ValueError(f"Non-text content at source row {source_index}")
                record = {
                    "id": f"{DATASET_ID}#test:{source_index}",
                    "text": row["content"],
                    "corpus_id": category,
                    "source_url": row["url"],
                }
                streams[category].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[category] += 1
                dates[category].append(row["date"])
                source_index += 1
    if sum(counts.values()) != EXPECTED_DOCUMENTS or any(counts[c] != 500 for c in CATEGORIES):
        raise ValueError(f"Expected 500 documents in each of 15 categories; got {dict(counts)}")
    with open(source_path, "rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "source_repository": f"https://huggingface.co/datasets/{DATASET_ID}",
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": source_hash,
        "text_field": "content",
        "split": "test",
        "subsets": [
            {
                "name": category,
                "input_path": f"{output_path}/{category}.jsonl.gz",
                "expected_documents": counts[category],
                "start_date": min(dates[category]),
                "end_date": max(dates[category]),
            }
            for category in CATEGORIES
        ],
    }


def main():
    logging.basicConfig(level=logging.INFO)
    source = hf_hub_download(DATASET_ID, SOURCE_FILE, repo_type="dataset", revision=SOURCE_REVISION)
    manifest = prepare_manifest(source, OUTPUT_PREFIX)
    manifest_path = f"{OUTPUT_PREFIX}/manifest.json"
    contents = json.dumps(manifest, indent=2) + "\n"
    with fsspec.open(manifest_path, "wt") as stream:
        stream.write(contents)
    Path(os.environ["IRIS_OUTPUT_DIR"], "manifest.json").write_text(contents)
    logger.info("UNCHEATABLE_MANIFEST_READY %s %s", manifest_path, json.dumps(manifest))


if __name__ == "__main__":
    main()
