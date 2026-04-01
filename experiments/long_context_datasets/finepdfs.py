# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
FinePDFs and FinePDFs-edu dataset definitions.

Downloads come from marin.datakit.download.finepdfs; this module owns the
InputNames, reshard steps, tokenize steps, and path dicts used by both
long-context and pretraining experiments.
"""

import dataclasses

from experiments.defaults import default_download, default_tokenize
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig
from marin.datakit.download.finepdfs import FINEPDFS_EXTRA_LANGS, download_finepdfs_step
from marin.execution.executor import ExecutorStep, this_output_path
from zephyr import Dataset, ZephyrContext, load_parquet

# --- finepdfs downloads ---

finepdfs_eng_raw = download_finepdfs_step("eng_Latn").as_executor_step().as_input_name()

finepdfs_by_language = {"eng_Latn": finepdfs_eng_raw / "data/eng_Latn/train/*.parquet"}
finepdfs_validation_by_language = {"eng_Latn": finepdfs_eng_raw.cd("data/eng_Latn/test/*.parquet")}

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

# --- finepdfs reshard + tokenization ---


@dataclasses.dataclass(frozen=True)
class ReshardConfig:
    input_path: str
    output_path: str
    input_glob: str = "**/*.parquet"
    filter_null_text: bool = False


def reshard_parquet(config: ReshardConfig):
    """Read parquet files and rewrite as smaller JSONL shards."""
    pattern = f"{config.input_path}/{config.input_glob}" if config.input_glob else str(config.input_path)
    ds = Dataset.from_files(pattern).flat_map(load_parquet)
    if config.filter_null_text:
        ds = ds.filter(lambda r: r.get("text") is not None and len(r.get("text", "")) > 0)
    pipeline = ds.write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    ctx = ZephyrContext(
        name="reshard-parquet",
        # 120g needed because load_parquet decompresses entire parquet files in memory;
        # SFT-Math has 8.7GB parquets that expand to ~50GB+ in memory during read.
        resources=ResourceConfig(cpu=2, ram="120g"),
    )
    ctx.execute(pipeline)


def _reshard_step(lang: str, input_path) -> ExecutorStep:
    return ExecutorStep(
        name=f"resharded/finepdfs_{lang}",
        fn=reshard_parquet,
        config=ReshardConfig(
            input_path=input_path,
            output_path=this_output_path(),
            input_glob="",
            filter_null_text=True,
        ),
    )


_worker_resources = ResourceConfig(ram="80g", disk="10g")

finepdfs_tokenized: dict[str, ExecutorStep] = {}

_english_resharded = _reshard_step("eng_Latn", finepdfs_by_language["eng_Latn"])
finepdfs_tokenized["eng_Latn"] = default_tokenize(
    "finepdfs_eng_Latn",
    _english_resharded / "**/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    worker_resources=_worker_resources,
)

for _lang in FINEPDFS_EXTRA_LANGS:
    _resharded = _reshard_step(_lang, finepdfs_extra_by_language[_lang])
    finepdfs_tokenized[_lang] = default_tokenize(
        f"finepdfs_{_lang}",
        _resharded / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        worker_resources=_worker_resources,
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
