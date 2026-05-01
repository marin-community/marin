# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MASSIVE → function-calling dataset tokenization.

Mirrors the ``nsf_awards.py`` pattern: take the terminal normalize StepSpec,
adapt it to an ExecutorStep, then wire a tokenize ExecutorStep that reads its
``outputs/main/*.parquet`` shards through the standard Marin tokenizer.
"""

from marin.datakit.download.massive import massive_normalize_steps
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_models import marin_tokenizer

# massive_normalize_steps returns (stage, transform, normalize); the terminal
# normalize step is what consumers tokenize off.
*_, _massive_normalized_step = massive_normalize_steps()
massive_function_calling_download = _massive_normalized_step.as_executor_step()

massive_function_calling_tokenized = ExecutorStep(
    name="tokenized/massive_function_calling",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[output_path_of(massive_function_calling_download, "outputs/main/*.parquet")],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(marin_tokenizer),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[massive_function_calling_tokenized])
