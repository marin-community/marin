# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NuminaMath dataset definitions for the pretraining dataset CLI."""

import dataclasses

from experiments.defaults import default_download, default_tokenize
from experiments.marin_models import marin_tokenizer
from marin.execution.executor import ExecutorStep, this_output_path
from zephyr import Dataset, ZephyrContext, load_parquet
from fray.cluster import ResourceConfig

numinamath_download = default_download(
    name="numinamath_1_5",
    hf_dataset_id="AI-MO/NuminaMath-1.5",
    revision="1b05109",
).step


@dataclasses.dataclass(frozen=True)
class PrepareNuminaMathConfig:
    input_path: str
    output_path: str


def prepare_numinamath(config: PrepareNuminaMathConfig):
    def format_record(record: dict) -> dict | None:
        problem = record.get("problem", "")
        solution = record.get("solution", "")
        if not problem:
            return None
        return {"text": f"{problem}\n\n{solution}"}

    pipeline = (
        Dataset.from_files(f"{config.input_path}/data/*.parquet")
        .flat_map(load_parquet)
        .map(format_record)
        .filter(lambda r: r is not None)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="prepare-numinamath", resources=ResourceConfig(cpu=1, ram="4g"))
    ctx.execute(pipeline)


numinamath_prepared = ExecutorStep(
    name="documents/numinamath_1_5",
    fn=prepare_numinamath,
    config=PrepareNuminaMathConfig(
        input_path=numinamath_download.as_input_name(),
        output_path=this_output_path(),
    ),
)

numinamath_tokenized = default_tokenize(
    name="numinamath_1_5",
    dataset=numinamath_prepared / "**/*.jsonl.gz",
    tokenizer=marin_tokenizer,
)
