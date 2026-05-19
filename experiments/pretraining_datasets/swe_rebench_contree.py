# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-rebench ConTree trace dataset normalization and tokenization."""

from marin.datakit.download.swe_rebench_contree import swe_rebench_contree_normalize_steps
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_models import marin_tokenizer

swe_rebench_contree_normalized = swe_rebench_contree_normalize_steps()[-1].as_executor_step()

swe_rebench_contree_tokenized = ExecutorStep(
    name="tokenized/swe-rebench-contree",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[output_path_of(swe_rebench_contree_normalized, "outputs/main/*.parquet")],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(marin_tokenizer),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[swe_rebench_contree_tokenized])
