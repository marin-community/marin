"""Tokenize each YODAS2 language split for Marin audio experiments."""

from collections.abc import Mapping

from experiments.defaults import default_tokenize
from marin.download.huggingface.download import DownloadConfig, download as storage_transfer_download
from marin.execution import executor_main, versioned
from marin.execution.executor import ExecutorStep, this_output_path

_YODAS2_REVISION_STR = "8eda080a5fd6dfc070dd306c1e6446ab7c5b5f17"
YODAS2_REVISION = versioned(_YODAS2_REVISION_STR)
YODAS2_TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"

_YODAS2_SPLIT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("yodas2/all", "*/*.parquet"),
    ("yodas2/en", "en*/*.parquet"),
    ("yodas2/th", "th*/*.parquet"),
    ("yodas2/ar", "ar*/*.parquet"),
    ("yodas2/de", "de*/*.parquet"),
    ("yodas2/es", "es*/*.parquet"),
    ("yodas2/fr", "fr*/*.parquet"),
    ("yodas2/hi", "hi*/*.parquet"),
    ("yodas2/zh", "zh*/*.parquet"),
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
    revision_root = download_step / _YODAS2_REVISION_STR

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


if __name__ == "__main__":
    executor_main(steps=list(tokenize_yodas_steps().values()))
