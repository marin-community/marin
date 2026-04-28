# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HPLT v3.0 dataset definitions and tokenization.

HPLT v3.0 English data filtered to non-Common Crawl sources only (WIDE, survey crawls),
with register-based quality filtering. This avoids redundancy with Nemotron CC while
adding ~612.7B unique tokens from European web crawls.
"""

import os.path

from fray import ResourceConfig
from marin.datakit.download.hplt import download_hplt_v3_step, normalize_hplt_v3_step
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

HPLT_DATASETS = {
    "all": ["*.parquet"],
}

_hplt_v3_download_spec = download_hplt_v3_step()
hplt_v3_download = _hplt_v3_download_spec.as_executor_step()
hplt_v3_normalized = normalize_hplt_v3_step(_hplt_v3_download_spec).as_executor_step()


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
            train_paths=[output_path_of(hplt_v3_normalized, "outputs/main/*.parquet")],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
            max_workers=max_workers,
            worker_resources=ResourceConfig(ram="20g", disk="5g"),
        ),
    )

    return {"hplt_v3/all": step}
