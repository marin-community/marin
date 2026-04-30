# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Molmo2-Cap dataset tokenization."""

from fray import ResourceConfig

from marin.datakit.download.molmo2_cap import molmo2_cap_normalize_steps
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

molmo2_cap_normalized = molmo2_cap_normalize_steps()[-1].as_executor_step()


def tokenize_molmo2_cap(*, tokenizer: str | None = None) -> TokenizerStep:
    """Tokenize the normalized Molmo2-Cap captions."""
    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    return ExecutorStep(
        name="tokenized/molmo2_cap",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[molmo2_cap_normalized.as_input_name() / "outputs/main/*.parquet"],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
            worker_resources=ResourceConfig(ram="16g", disk="5g", preemptible=True),
        ),
    )


if __name__ == "__main__":
    executor_main(steps=[tokenize_molmo2_cap()])
