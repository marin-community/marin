# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lean-Workbook dataset: math problems with Lean 4 formalizations.

Downloads internlm/Lean-Workbook from HuggingFace, concatenates
natural_language_statement and formal_statement into a single text field,
and tokenizes.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec
from levanter.data.text import TextLmDatasetFormat
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from zephyr import Dataset, ZephyrContext

from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConcatFieldsConfig:
    input_path: str
    output_path: str


def _concat_records(input_path: str) -> Iterator[dict]:
    """Read lean_workbook.json and yield records with concatenated text."""
    with fsspec.open(input_path, "r") as f:
        records = json.load(f)

    for record in records:
        nl = record.get("natural_language_statement", "")
        formal = record.get("formal_statement", "")
        text = f"{nl}\n\n{formal}".strip()
        if text:
            yield {"text": text}


def concat_fields(config: ConcatFieldsConfig) -> None:
    """Concat natural_language_statement and formal_statement into text, write JSONL."""
    pipeline = (
        Dataset.from_list([config.input_path])
        .flat_map(_concat_records)
        .write_jsonl(f"{config.output_path}/lean_workbook-{{shard:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="lean-workbook-concat")
    ctx.execute(pipeline)


# ============================================================================
# PIPELINE STEPS
# ============================================================================

download_lean_workbook = ExecutorStep(
    name="raw/lean-workbook",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="internlm/Lean-Workbook",
        revision=versioned("2e066e3"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="raw/lean-workbook",
)

concat_lean_workbook = ExecutorStep(
    name="documents/lean-workbook",
    fn=concat_fields,
    config=ConcatFieldsConfig(
        input_path=download_lean_workbook.cd("lean_workbook.json"),
        output_path=this_output_path(),
    ),
)

tokenized_lean_workbook = ExecutorStep(
    name="tokenized/lean-workbook",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[concat_lean_workbook],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
        format=TextLmDatasetFormat(),
    ),
)
