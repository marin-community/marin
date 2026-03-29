# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_base_eval, default_long_context_eval
from experiments.tootsie.exp2062_long_context_8b import (
    STARLING_WARMSTART_STEP,
    phase1_final_checkpoint,
    phase2_final_checkpoint,
    phase3_final_checkpoint,
)
from experiments.tootsie.exp600_tootsie import tootsie_8b_sensible_starling
from marin.execution.executor import InputName, executor_main

BASE_EVAL_RESOURCES = ResourceConfig.with_tpu("v6e-8")
LONG_CONTEXT_RESOURCES = ResourceConfig.with_tpu("v5p-8")

# Fill this in once the frozen natural long-context manifest is ready.
# When left as None, the runner evaluates only the synthetic long-context tasks.
FINEPDF_MANIFEST_PATH: str | None = None


@dataclass(frozen=True)
class CheckpointEvalSpec:
    checkpoint: InputName
    lengths: tuple[int, ...]


CHECKPOINTS = (
    CheckpointEvalSpec(
        checkpoint=tootsie_8b_sensible_starling.cd(f"hf/step-{STARLING_WARMSTART_STEP}"),
        lengths=(4096,),
    ),
    CheckpointEvalSpec(
        checkpoint=phase1_final_checkpoint,
        lengths=(4096,),
    ),
    CheckpointEvalSpec(
        checkpoint=phase2_final_checkpoint,
        lengths=(4096, 16384, 32768),
    ),
    CheckpointEvalSpec(
        checkpoint=phase3_final_checkpoint,
        lengths=(4096, 16384, 32768, 65536),
    ),
)


if __name__ == "__main__":
    eval_steps = []
    for checkpoint in CHECKPOINTS:
        eval_steps.extend(
            default_base_eval(
                checkpoint.checkpoint,
                resource_config=BASE_EVAL_RESOURCES,
                discover_latest_checkpoint=False,
            )
        )
        eval_steps.append(
            default_long_context_eval(
                checkpoint.checkpoint,
                lengths=checkpoint.lengths,
                finepdf_manifest_path=FINEPDF_MANIFEST_PATH,
                resource_config=LONG_CONTEXT_RESOURCES,
                discover_latest_checkpoint=False,
            )
        )

    executor_main(steps=eval_steps)
