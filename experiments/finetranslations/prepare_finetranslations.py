# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare FineTranslations for pretraining: concat English translation + original text.

Each document becomes: translated_text (English) followed by og_full_text (original language).
This gives the model parallel text exposure — English comprehension of the content followed
by the same content in the original language.

Source: https://huggingface.co/datasets/HuggingFaceFW/finetranslations
"The world's knowledge in 1+1T tokens of parallel text" (~2T tokens combined).
License: ODC-By 1.0

Example Usage:
    uv run python experiments/finetranslations/prepare_finetranslations.py
"""

import dataclasses

from fray.cluster import ResourceConfig
from marin.datakit.download.finetranslations import download_finetranslations_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from zephyr import Dataset, ZephyrContext, load_parquet

finetranslations_raw = download_finetranslations_step().as_executor_step().as_input_name()


@dataclasses.dataclass(frozen=True)
class PrepareFinetranslationsConfig:
    input_path: str
    output_path: str


def prepare_finetranslations(config: PrepareFinetranslationsConfig):
    """Concatenate translated_text + og_full_text into a single text field."""

    def concat_parallel(record: dict) -> dict | None:
        translated = record.get("translated_text", "")
        original = record.get("og_full_text", "")
        if not translated and not original:
            return None
        return {"text": f"{translated}\n\n{original}"}

    pipeline = (
        Dataset.from_files(f"{config.input_path}/data/**/*.parquet")
        .flat_map(load_parquet)
        .map(concat_parallel)
        .filter(lambda r: r is not None)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(
        name="prepare-finetranslations",
        resources=ResourceConfig(cpu=2, ram="16g"),
    )
    ctx.execute(pipeline)


finetranslations_prepared = ExecutorStep(
    name="documents/finetranslations_parallel",
    fn=prepare_finetranslations,
    config=PrepareFinetranslationsConfig(
        input_path=finetranslations_raw,
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[finetranslations_prepared])
