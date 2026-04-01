# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""nvidia/Nemotron-Terminal-Corpus rollout dataset.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py -- python experiments/rollout_data/nemotron_terminal.py
"""

from fray.v2 import ResourceConfig
from marin.datakit.download.nemotron_terminal import download_nemotron_terminal_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from experiments.marin_models import marin_tokenizer


def build_steps() -> list[StepSpec]:
    processed = download_nemotron_terminal_step()

    tokenized = StepSpec(
        name="tokenized/nemotron-terminal-corpus",
        deps=[processed],
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[processed.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=marin_tokenizer,
                num_shards=64,
                worker_resources=ResourceConfig(ram="48g", disk="5g"),
            )
        ),
        hash_attrs={"tokenizer": marin_tokenizer},
    )

    return [*processed.deps, processed, tokenized]


if __name__ == "__main__":
    StepRunner().run(build_steps())
