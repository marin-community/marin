# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""marin-community/glm-5.2-kernelgym-rollouts GPU kernel-optimization rollouts.

Usage:
    uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
      -- python -m experiments.rollout_data.glm_kernelgym_rollouts
"""

from marin.datakit.download.glm_kernelgym_rollouts import download_glm_kernelgym_rollouts_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_tokenizer import marin_tokenizer


def build_steps() -> list[StepSpec]:
    processed = download_glm_kernelgym_rollouts_step()

    tokenized = StepSpec(
        name="tokenized/glm-5.2-kernelgym-rollouts",
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
