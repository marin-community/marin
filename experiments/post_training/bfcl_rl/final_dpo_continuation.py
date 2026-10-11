# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the frozen final DPO run after its separately verified step-24 evaluation."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import skyrl_step

from experiments.post_training.bfcl_rl.checkpoint_restore import CHECKPOINT
from experiments.post_training.bfcl_rl.final_dpo import DPO_EXECUTION, RunScale, final_dpo_spec


def final_dpo_continuation_step() -> ArtifactStep:
    original = final_dpo_spec("2026.10.08.320", RunScale.FULL)
    recipe = yaml.safe_load(original.config_yaml)
    trainer = recipe["trainer"]
    trainer.update(
        resume_mode="from_path",
        resume_path=CHECKPOINT,
        restore_dataloader_state=True,
        reset_global_step_on_resume=False,
        eval_before_train=False,
    )
    for callback in trainer["callbacks"]:
        if callback["type"] == "evaluation":
            # The separate restore check already evaluates these exact weights.
            # The next canonical evaluation must be step28, not a repeated24.
            callback["eval_before_train"] = False
    name = user_owned_name("models/bfcl-rl-final-native-dpo-continuation")
    spec = replace(
        original,
        name=name,
        version=resolve_version(name, None),
        config_yaml=yaml.safe_dump(recipe, sort_keys=False),
    )
    return skyrl_step(spec, DPO_EXECUTION, export_hf=False)


@click.command(help=__doc__)
@rl_build_options
def main() -> ArtifactStep:
    return final_dpo_continuation_step()


if __name__ == "__main__":
    main()
