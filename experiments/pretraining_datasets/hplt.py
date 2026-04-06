# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HPLT v3.0 dataset definitions and tokenization.

HPLT v3.0 English data filtered to non-Common Crawl sources only (WIDE, survey crawls),
with register-based quality filtering. This avoids redundancy with Nemotron CC while
adding ~450B unique tokens from European web crawls.
"""

import os.path

from fray.v2 import ResourceConfig
from marin.datakit.download.hplt import download_hplt_v3_step
from marin.execution.executor import ExecutorStep, InputName, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep


def hplt_v3_download() -> ExecutorStep:
    return download_hplt_v3_step().as_executor_step()


HPLT_DATASETS = {
    "all": ["*.jsonl.zst"],
}


_HPLT_V3_DATA_PATH = InputName.hardcoded("raw/hplt_v3_2a08d6f3")


def tokenize_hplt_v3(
    *,
    tokenizer: str | None = None,
    max_workers: int = 4096,
) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for the HPLT v3 dataset."""
    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    output_path = os.path.join("tokenized", "hplt_v3", "all")
    step = ExecutorStep(
        name=output_path,
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[_HPLT_V3_DATA_PATH / "*.jsonl.zst"],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
            max_workers=max_workers,
            worker_resources=ResourceConfig(ram="20g", disk="5g"),
        ),
    )

    return {"hplt_v3/all": step}
