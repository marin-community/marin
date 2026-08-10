# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize normalized TeraflopAI/SEC-EDGAR filings."""

from fray.types import ResourceConfig
from marin.datakit.download.sec_edgar import sec_edgar_normalize_steps
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from rigging.filesystem import prefix_join

from experiments.marin_tokenizer import marin_tokenizer


def tokenize_sec_edgar(*, tokenizer: str = marin_tokenizer) -> StepSpec:
    """Return the SEC-EDGAR tokenization step."""
    normalized = sec_edgar_normalize_steps()[-1]
    return StepSpec(
        name="tokenized/sec-edgar",
        deps=[normalized],
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[prefix_join(normalized.output_path, "outputs/main/*.parquet")],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=tokenizer,
                worker_resources=ResourceConfig(cpu=4, ram="32g", disk="10g", preemptible=True),
            )
        ),
        hash_attrs={"tokenizer": tokenizer},
    )


def build_steps() -> list[StepSpec]:
    tokenized = tokenize_sec_edgar()
    normalized = tokenized.deps[0]
    return [*normalized.deps, normalized, tokenized]


if __name__ == "__main__":
    StepRunner().run(build_steps())
