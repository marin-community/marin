# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FinePDFs dataset definitions for the pretraining dataset CLI."""

from levanter.data.text import TextLmDatasetFormat

from experiments.marin_models import marin_tokenizer
from experiments.long_context_datasets.finepdfs import (
    FINEPDFS_EXTRA_LANGS,
    finepdfs_by_language,
    finepdfs_eng_raw,
    finepdfs_extra_by_language,
    finepdfs_extra_raw,
)
from experiments.reshard_parquet import ReshardConfig, reshard_parquet
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize


def _tokenize_step(name: str, train_paths: list, *, worker_ram: str = "10g") -> ExecutorStep:
    kwargs = {}
    if worker_ram != "10g":
        kwargs["worker_resources"] = ResourceConfig(ram=worker_ram, disk="10g")
    return ExecutorStep(
        name=f"tokenized/{name}",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=train_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(marin_tokenizer),
            format=TextLmDatasetFormat(),
            **kwargs,
        ),
    )


finepdfs_downloads: dict[str, ExecutorStep] = {
    "eng_Latn": finepdfs_eng_raw.step,
    **{lang: raw.step for lang, raw in finepdfs_extra_raw.items()},
}

finepdfs_tokenized: dict[str, ExecutorStep] = {}

_english_resharded = ExecutorStep(
    name="resharded/finepdfs_eng_Latn",
    fn=reshard_parquet,
    config=ReshardConfig(
        input_path=finepdfs_by_language["eng_Latn"],
        output_path=this_output_path(),
        input_glob="",
        filter_null_text=True,
    ),
)
finepdfs_tokenized["eng_Latn"] = _tokenize_step(
    "finepdfs_eng_Latn",
    [_english_resharded / "**/*.jsonl.gz"],
    worker_ram="80g",
)

for _lang in FINEPDFS_EXTRA_LANGS:
    _resharded = ExecutorStep(
        name=f"resharded/finepdfs_{_lang}",
        fn=reshard_parquet,
        config=ReshardConfig(
            input_path=finepdfs_extra_by_language[_lang],
            output_path=this_output_path(),
            input_glob="",
            filter_null_text=True,
        ),
    )
    finepdfs_tokenized[_lang] = _tokenize_step(
        f"finepdfs_{_lang}",
        [_resharded / "**/*.jsonl.gz"],
        worker_ram="80g",
    )
