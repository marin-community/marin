# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""lambda/hermes-agent-reasoning-traces tokenization terminal.

Usage:
    uv run iris --cluster=marin job run --preemptible --region=us-central1 \
      --cpu=1 --memory=2G \
      -- python -m experiments.rollout_data.hermes_agent_reasoning_traces
"""

from marin.datakit.download.hermes_agent_reasoning_traces import (
    download_hermes_agent_reasoning_traces_step,
)
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from experiments.marin_models import marin_tokenizer


def build_steps() -> list[StepSpec]:
    processed = download_hermes_agent_reasoning_traces_step()

    tokenized = StepSpec(
        name="tokenized/hermes-agent-reasoning-traces",
        deps=[processed],
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[processed.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=marin_tokenizer,
            )
        ),
        hash_attrs={"tokenizer": marin_tokenizer},
    )

    return [*processed.deps, processed, tokenized]


if __name__ == "__main__":
    StepRunner().run(build_steps())
