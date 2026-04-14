# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RULER/NIAH evaluations for the exp2062 8B long-context checkpoints.

These checkpoints are base models, so RULER prompts are sent as plain completions
without chat-template wrapping. Use ``apply_chat_template=True`` only for
instruction-tuned checkpoints whose tokenizer chat template matches training.
"""

from dataclasses import dataclass

from fray.cluster import ResourceConfig

from experiments.evals.evals import default_ruler_eval
from experiments.tootsie.exp2062_long_context_8b import (
    STARLING_WARMSTART_STEP,
    phase1_final_checkpoint,
    phase2_final_checkpoint,
    phase3_final_checkpoint,
)
from experiments.tootsie.exp600_tootsie import tootsie_8b_sensible_starling
from marin.execution.executor import InputName, executor_main

RULER_RESOURCES = ResourceConfig.with_tpu("v5p-8")
RULER_TOKENIZER = "stanford-crfm/marin-tokenizer"


@dataclass(frozen=True)
class RulerCheckpointSpec:
    checkpoint: InputName
    lengths: tuple[int, ...]


CHECKPOINTS = (
    RulerCheckpointSpec(
        checkpoint=tootsie_8b_sensible_starling.cd(f"hf/step-{STARLING_WARMSTART_STEP}").nonblocking(),
        lengths=(4096,),
    ),
    RulerCheckpointSpec(
        checkpoint=phase1_final_checkpoint.nonblocking(),
        lengths=(4096,),
    ),
    RulerCheckpointSpec(
        checkpoint=phase2_final_checkpoint.nonblocking(),
        lengths=(4096, 8192, 16384, 32768),
    ),
    RulerCheckpointSpec(
        checkpoint=phase3_final_checkpoint.nonblocking(),
        lengths=(4096, 8192, 16384, 32768, 65536),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            default_ruler_eval(
                checkpoint.checkpoint,
                lengths=checkpoint.lengths,
                tokenizer=RULER_TOKENIZER,
                resource_config=RULER_RESOURCES,
                apply_chat_template=False,
                discover_latest_checkpoint=False,
            )
            for checkpoint in CHECKPOINTS
        ]
    )
