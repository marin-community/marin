# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ClimbLab-Ja pre-training dataset tokenization.

Download/normalize definitions live in marin.datakit.download.climblab_ja.
This file wires the normalized output into a tokenize step for experiment
pipelines.
"""

import os.path

from experiments.marin_models import marin_tokenizer
from marin.datakit.download.climblab_ja import climblab_ja_normalize_steps
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

_download_spec, _normalize_spec = climblab_ja_normalize_steps()

download: ExecutorStep = _download_spec.as_executor_step()
normalized: ExecutorStep = _normalize_spec.as_executor_step()


def tokenize_climblab_ja(*, tokenizer: str = marin_tokenizer) -> TokenizerStep:
    return ExecutorStep(
        name=os.path.join("tokenized", "climblab-ja"),
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[normalized / "outputs/main/*.parquet"],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )


if __name__ == "__main__":
    executor_main(steps=[tokenize_climblab_ja()])
