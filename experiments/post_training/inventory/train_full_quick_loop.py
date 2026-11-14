# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a short Snowball SFT from the pre-trained base, then the quick RL loop from its export.

The RL stage is train_quick_loop's, with the SFT export in place of the pinned checkpoint.
"""

from dataclasses import replace

import click
from levanter.utils.mesh import MeshConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactHfModel

from experiments.post_training.inventory.train_quick_loop import (
    ARTIFACT_NAME,
    DEFAULT_SELECTION_URI,
    DEFAULT_STEPS,
    MODEL_ID,
    MODEL_REVISION,
    build_run,
    selection_step,
)
from experiments.sft.configs.snowball_h100 import SPEC as SNOWBALL_SFT
from experiments.sft.launcher import resources_from_accelerator, sft_step

SFT_ARTIFACT_NAME = "checkpoints/snowball-inventory-sft"
DEFAULT_SFT_STEPS = 10
SFT_NODE_ACCELERATOR = "8xH100"
SFT_NODES = 4
# Experts shard across the eight GPUs of a node and context across nodes. With context inside a
# node, the 67B weights and Adam moments shard over only eight GPUs and exhaust their memory.
SFT_MESH = MeshConfig(
    axes={"data": 1, "replica": 1, "model": 1, "expert": 8},
    compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
    dcn_axes={"replica_dcn": 1, "context": SFT_NODES},
)
STAGES = ("sft", "rl")


@click.command(help=__doc__)
@click.option(
    "--selection-uri", default=DEFAULT_SELECTION_URI, show_default=True, help="Published task-selection artifact root."
)
@click.option(
    "--stage",
    type=click.Choice(STAGES),
    default="rl",
    show_default=True,
    help="Terminal stage to plan or run; rl includes sft.",
)
@click.option("--sft-steps", type=click.IntRange(min=1), default=DEFAULT_SFT_STEPS, show_default=True)
@click.option("--steps", type=click.IntRange(min=1), default=DEFAULT_STEPS, show_default=True, help="RL updates.")
@click.option("--export-hf", is_flag=True, help="Export the final Hugging Face model after RL.")
@click.option(
    "--reproducible-serving",
    is_flag=True,
    help="Use batch-invariant serving with one sequence per rank and greedy evaluation.",
)
@rl_build_options
def main(
    selection_uri: str, stage: str, sft_steps: int, steps: int, export_hf: bool, reproducible_serving: bool
) -> ArtifactStep:
    version = resolve_version(user_owned_name(ARTIFACT_NAME), None)
    sft = sft_step(
        replace(
            SNOWBALL_SFT,
            name=user_owned_name(SFT_ARTIFACT_NAME),
            version=version,
            num_train_steps=sft_steps,
            mesh=SFT_MESH,
            # One conversation per sequence rejects any conversation longer than the sequence;
            # packing truncates it instead.
            pack=True,
        ),
        replace(resources_from_accelerator(SFT_NODE_ACCELERATOR), replicas=SFT_NODES),
    )
    rl = build_run(
        version,
        selection=selection_step(selection_uri),
        # Without a relative path, RL reads the SFT run's terminal HF export.
        model=ArtifactHfModel(step=sft, tokenizer_uri=MODEL_ID, tokenizer_revision=MODEL_REVISION),
        steps=steps,
        export_hf=export_hf,
        reproducible_serving=reproducible_serving,
    )
    if stage == "rl":
        return rl
    # The RL CLI routes its coordinator from the RL step's execution, so the SFT-only stage carries it
    # too, without the RL-only credentials that execution requires at submission.
    routing = {name: replace(execution, required_env_vars=()) for name, execution in rl.runtime_args.items()}
    return replace(sft, runtime_args={**sft.runtime_args, **routing})


if __name__ == "__main__":
    main()
