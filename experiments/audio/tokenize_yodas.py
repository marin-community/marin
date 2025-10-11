"""Tokenize each YODAS2 language split for Marin audio experiments."""

from collections.abc import Mapping

from levanter.data.text import LMMixtureDatasetConfig

from experiments.defaults import default_tokenize
from marin.download.huggingface.download import DownloadConfig, download as storage_transfer_download
from marin.execution import executor_main, versioned
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize.data_configs import TokenizerStep, lm_mixture_data_config

_YODAS2_LANGUAGE_WEIGHTS = {
    "en": 0.355727909201,
    "th": 0.000970575517,
    "ar": 0.00059,
    "de": 0.023203517508,
    "es": 0.073786617338,
    "fr": 0.034971397976,
    "hi": 0.000784368275,
    "zh": 0.00050167398,
}

_YODAS2_REVISION_STR = "8eda080a5fd6dfc070dd306c1e6446ab7c5b5f17"
YODAS2_REVISION = versioned(_YODAS2_REVISION_STR)
YODAS2_TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"

_YODAS2_SPLIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("yodas2/en", "en{{000..007},{100..129}}/*.parquet"),
    ("yodas2/th", "th{000,100}/*.parquet"),
    ("yodas2/ar", "ar000/*.parquet"),
    ("yodas2/de", "de{000,{100..102}}/*.parquet"),
    ("yodas2/es", "es{000,{100..108}}/*.parquet"),
    ("yodas2/fr", "fr{000,{100..103}}/*.parquet"),
    ("yodas2/hi", "hi{000,100}/*.parquet"),
    ("yodas2/zh", "zh000/*.parquet"),
)

_YODAS2_DOWNLOAD_NAME = "raw/yodas2-mm-pretrain"


def _yodas_download_step() -> ExecutorStep[DownloadConfig]:
    """Use Google Storage Transfer Service for faster YODAS2 downloads."""
    return ExecutorStep(
        name=_YODAS2_DOWNLOAD_NAME,
        fn=storage_transfer_download,
        config=DownloadConfig(
            hf_dataset_id="potsawee/yodas2-mm-pretrain",
            revision=YODAS2_REVISION,
            gcs_output_path=this_output_path(),
            hf_urls_glob=["**/*.parquet"],
            wait_for_completion=True,
        ),
    ).with_output_path(_YODAS2_DOWNLOAD_NAME)


def tokenize_yodas_steps() -> Mapping[str, ExecutorStep]:
    """Create tokenization steps for each YODAS2 language split."""
    download_step = _yodas_download_step()
    revision_root = (
        download_step / _YODAS2_REVISION_STR
    ) / "huggingface.co/datasets/potsawee/yodas2-mm-pretrain/resolve/8eda080a5fd6dfc070dd306c1e6446ab7c5b5f17"

    steps: dict[str, ExecutorStep] = {_YODAS2_DOWNLOAD_NAME: download_step}
    for name, pattern in _YODAS2_SPLIT_PATTERNS:
        steps[name] = default_tokenize(
            name=name,
            dataset=revision_root / pattern,
            tokenizer=YODAS2_TOKENIZER,
            enforce_bos=False,
            enforce_eos=False,
        )
    return steps


def yodas2_tokenized_steps() -> dict[str, TokenizerStep]:
    """Return only the tokenization steps for the YODAS2 language splits."""
    steps = tokenize_yodas_steps()
    language_names = {f"yodas2/{lang}" for lang in _YODAS2_LANGUAGE_WEIGHTS}
    return {name: step for name, step in steps.items() if name in language_names}


def yodas2_mixture_config() -> LMMixtureDatasetConfig:
    """Create an lm_mixture_dataset config for the tokenized YODAS2 splits."""
    tokenized = yodas2_tokenized_steps()
    weights = {f"yodas2/{lang}": weight for lang, weight in _YODAS2_LANGUAGE_WEIGHTS.items()}
    return lm_mixture_data_config(components=tokenized, weights=weights)


if __name__ == "__main__":
    executor_main(steps=list(tokenize_yodas_steps().values()))
