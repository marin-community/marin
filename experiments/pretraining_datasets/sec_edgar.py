# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TeraflopAI/SEC-EDGAR dataset tokenization."""

from fray import ResourceConfig
from marin.datakit.download.sec_edgar import sec_edgar_normalize_steps
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

sec_edgar_normalized = sec_edgar_normalize_steps()[-1].as_executor_step()


def tokenize_sec_edgar(*, tokenizer: str | None = None) -> TokenizerStep:
    """Tokenize the normalized SEC-EDGAR shards.

    Bumps RAM over the default because the ``content`` column carries
    multi-MB filings; a single batch's worth of raw text can spike memory
    well above the 16 GiB default.
    """
    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    return ExecutorStep(
        name="tokenized/sec-edgar",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[sec_edgar_normalized.as_input_name() / "outputs/main/*.parquet"],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
            worker_resources=ResourceConfig(cpu=4, ram="32g", disk="10g", preemptible=True),
        ),
    )


if __name__ == "__main__":
    executor_main(steps=[tokenize_sec_edgar()])
