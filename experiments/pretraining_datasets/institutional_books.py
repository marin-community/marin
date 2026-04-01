# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Institutional Books dataset definitions for the pretraining dataset CLI."""

import dataclasses

from levanter.data.text import TextLmDatasetFormat

from experiments.marin_models import marin_tokenizer
from experiments.long_context_datasets.institutional_books import institutional_books_raw
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from zephyr import Dataset, ZephyrContext, load_parquet

institutional_books_download = institutional_books_raw.step


@dataclasses.dataclass(frozen=True)
class PrepareInstitutionalBooksConfig:
    input_path: str
    output_path: str


def prepare_institutional_books(config: PrepareInstitutionalBooksConfig):
    def concat_pages(record: dict) -> dict | None:
        pages = record.get("text_by_page_gen")
        if not pages:
            return None
        text = "\n\n".join(p for p in pages if p)
        if not text.strip():
            return None
        return {"text": text}

    pipeline = (
        Dataset.from_files(f"{config.input_path}/data/**/*.parquet")
        .flat_map(load_parquet)
        .map(concat_pages)
        .filter(lambda r: r is not None)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="prepare-institutional-books", resources=ResourceConfig(cpu=2, ram="20g"))
    ctx.execute(pipeline)


institutional_books_prepared = ExecutorStep(
    name="documents/institutional_books",
    fn=prepare_institutional_books,
    config=PrepareInstitutionalBooksConfig(
        input_path=institutional_books_download.as_input_name(),
        output_path=this_output_path(),
    ),
)

institutional_books_tokenized = ExecutorStep(
    name="tokenized/institutional_books",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[institutional_books_prepared / "**/*.jsonl.gz"],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(marin_tokenizer),
        format=TextLmDatasetFormat(),
        worker_resources=ResourceConfig(ram="40g", disk="10g"),
    ),
)
