# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""penfever/*-minimax-m27-131k-traces rollout dataset family.

Materializes the 12 MiniMax-M2.7 @ 131k cohorts registered for tracking
issue marin-community/marin#6191. Each cohort runs the full
``(download → transform → normalize → tokenize)`` chain so it can land in
the datakit mixture alongside other agentic rollout sources.

Usage:
    uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \\
      -- python -m experiments.rollout_data.penfever_minimax_m27_traces
"""

from fray import ResourceConfig
from marin.datakit.download.penfever_minimax_m27_traces import (
    penfever_minimax_m27_traces_normalize_steps,
)
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_models import marin_tokenizer


def build_steps() -> list[StepSpec]:
    chains = penfever_minimax_m27_traces_normalize_steps()
    steps: list[StepSpec] = []
    for marin_name, chain in chains.items():
        processed = chain[0]
        # ``marin_name`` looks like ``penfever-traces/minimax-m27-131k/<slug>``;
        # mirror that under ``tokenized/`` so the cache path is unambiguous.
        tokenized = StepSpec(
            name=f"tokenized/{marin_name}",
            deps=[processed],
            fn=lambda output_path, _p=processed: tokenize(
                TokenizeConfig(
                    train_paths=[_p.output_path],
                    validation_paths=[],
                    cache_path=output_path,
                    tokenizer=marin_tokenizer,
                    worker_resources=ResourceConfig(ram="48g", disk="5g"),
                )
            ),
            hash_attrs={"tokenizer": marin_tokenizer},
        )
        steps.extend([*processed.deps, processed, tokenized])
    return steps


if __name__ == "__main__":
    StepRunner().run(build_steps())
