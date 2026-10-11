# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run native Qwen/student DPO from an explicit recovered Snowball checkpoint."""

from dataclasses import replace

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION
from experiments.post_training.bfcl_rl.launch import recovered_model
from experiments.post_training.bfcl_rl.optimize import RecoveryOptimization, recovery_optimizer_step
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache


def native_optimizer_step(
    preference_version: str,
    preference_name: str,
    recovery_version: str,
    policy_export_version: str,
    policy_checkpoint_step: int,
    num_train_steps: int,
    learning_rate: float,
    hf_save_steps: int,
) -> ArtifactStep:
    name = user_owned_name(preference_name)
    cache = ArtifactStep.adopt(
        f"{name}-input", preference_version, f"{name}/{preference_version}", kind=RecoveryPreferenceCache
    )
    policy = replace(
        recovered_model(recovery_version, policy_export_version), relative_path=f"hf/step-{policy_checkpoint_step}"
    )
    optimization = RecoveryOptimization(
        num_train_steps=num_train_steps,
        batch_size=16,
        beta=0.1,
        learning_rate=learning_rate,
        hf_save_steps=hf_save_steps,
        num_nodes=16,
        expert_axis=8,
        context_axis=16,
        jax_memory_fraction=0.90,
    )
    step = recovery_optimizer_step(cache, initial_policy=policy, selection_name="native", optimization=optimization)
    return replace(step, runtime_args={"execution": COLLECTION_EXECUTION})


@click.command(help=__doc__)
@click.option("--preference-version", required=True)
@click.option("--preference-name", required=True, help="Preference cache artifact name without version.")
@click.option("--recovery-version", required=True)
@click.option("--policy-export-version", required=True)
@click.option("--policy-checkpoint-step", type=click.IntRange(min=0), required=True)
@click.option("--num-train-steps", type=click.IntRange(min=1), required=True)
@click.option("--learning-rate", type=click.FloatRange(min=0, min_open=True), required=True)
@click.option("--hf-save-steps", type=click.IntRange(min=1), required=True)
@rl_build_options
def main(
    preference_version: str,
    preference_name: str,
    recovery_version: str,
    policy_export_version: str,
    policy_checkpoint_step: int,
    num_train_steps: int,
    learning_rate: float,
    hf_save_steps: int,
) -> ArtifactStep:
    return native_optimizer_step(
        preference_version,
        preference_name,
        recovery_version,
        policy_export_version,
        policy_checkpoint_step,
        num_train_steps,
        learning_rate,
        hf_save_steps,
    )


if __name__ == "__main__":
    main()
