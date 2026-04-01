# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Institutional Books dataset definitions for the pretraining dataset CLI."""

import dataclasses

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig
from marin.datakit.download.institutional_books import download_institutional_books_step
from marin.execution.executor import ExecutorStep, this_output_path
from zephyr import Dataset, ZephyrContext, load_parquet

institutional_books_download = download_institutional_books_step().as_executor_step()


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

institutional_books_tokenized = default_tokenize(
    "institutional_books",
    institutional_books_prepared / "**/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    worker_resources=ResourceConfig(ram="40g", disk="10g"),
)
