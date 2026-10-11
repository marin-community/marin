# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore the banked final-DPO checkpoint and evaluate it without another update."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import skyrl_step

from experiments.post_training.bfcl_rl.final_dpo import DPO_EXECUTION, RunScale, final_dpo_spec

CHECKPOINT = (
    "s3://marin-us-east-02a/tmp/ttl=14d/skyrl/marin/users/benfeuer/models/"
    "bfcl-rl-final-native-dpo-full/2026.10.10.357/checkpoints/global_step_24"
)


def checkpoint_restore_step() -> ArtifactStep:
    original = final_dpo_spec("2026.10.08.320", RunScale.FULL)
    recipe = yaml.safe_load(original.config_yaml)
    trainer = recipe["trainer"]
    trainer.update(
        max_steps=24,
        resume_mode="from_path",
        resume_path=CHECKPOINT,
        restore_dataloader_state=True,
        reset_global_step_on_resume=False,
    )
    # The source checkpoint is already retained. This diagnostic must not copy
    # its training state or select a winner independently of the full run chain.
    trainer["callbacks"] = [
        callback for callback in trainer["callbacks"] if callback["type"] in ("logging", "evaluation")
    ]
    name = user_owned_name("models/bfcl-rl-final-native-dpo-restore-check")
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
    return checkpoint_restore_step()


if __name__ == "__main__":
    main()
