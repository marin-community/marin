# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
FinePDFs and FinePDFs-edu dataset definitions.

Downloads come from marin.datakit.download.finepdfs; this module owns the
InputNames, reshard steps, tokenize steps, and path dicts used by both
long-context and pretraining experiments.
"""

from levanter.data.text import TextLmDatasetFormat

from experiments.defaults import default_download
from experiments.marin_models import marin_tokenizer
from experiments.reshard_parquet import ReshardConfig, reshard_parquet
from fray.cluster import ResourceConfig
from marin.datakit.download.finepdfs import FINEPDFS_EXTRA_LANGS, download_finepdfs_step
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

# --- finepdfs downloads ---

finepdfs_eng_raw = download_finepdfs_step("eng_Latn").as_executor_step().as_input_name()

finepdfs_by_language = {
    "eng_Latn": finepdfs_eng_raw / "data/eng_Latn/train/*.parquet"
}
finepdfs_validation_by_language = {
    "eng_Latn": finepdfs_eng_raw.cd("data/eng_Latn/test/*.parquet")
}

# ~206,917,202 docs * ~3,600 tokens/doc from manual audit ≈ 7.45e11 tokens
finepdfs_token_counts = {
    "eng_Latn": 7.45e11,
}

finepdfs_extra_raw = {
    lang: download_finepdfs_step(lang).as_executor_step().as_input_name() for lang in FINEPDFS_EXTRA_LANGS
}

finepdfs_extra_by_language = {lang: dl / f"data/{lang}/train/*.parquet" for lang, dl in finepdfs_extra_raw.items()}

finepdfs_downloads: dict[str, ExecutorStep] = {
    "eng_Latn": finepdfs_eng_raw.step,
    **{lang: raw.step for lang, raw in finepdfs_extra_raw.items()},
}

# --- finepdfs tokenization ---


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

# --- finepdfs-edu (higher quality educational PDFs) ---

finepdfs_edu_eng_raw = default_download(
    name="finepdfs_edu_eng_Latn",
    hf_dataset_id="HuggingFaceFW/finepdfs-edu",
    revision="9cfabe2",
    hf_urls_glob=["data/eng_Latn/train/*.parquet"],
    override_output_path="finepdfs_edu_eng_Latn",
)

finepdfs_edu_by_language = {"eng_Latn": finepdfs_edu_eng_raw / "data/eng_Latn/train/*.parquet"}
# ~140B tokens for English
finepdfs_edu_token_counts = {"eng_Latn": 140e9}
