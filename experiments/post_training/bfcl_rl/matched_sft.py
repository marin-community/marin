# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the audited chosen-only exposure for a matched SFT comparison."""

from dataclasses import replace

import click
from fray.types import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, complement_data_step
from experiments.post_training.bfcl_rl.matched_sft_data import MatchedChosenCache, MatchedChosenConfig, run_matched_chosen
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache

def matched_chosen_step(source: ArtifactStep[RecoveryPreferenceCache], *, seed: int, presentations: int) -> ArtifactStep:
    data = complement_data_step()
    name = user_owned_name("data/bfcl-rl-matched-chosen")

    def build_config(ctx: StepContext) -> MatchedChosenConfig:
        return MatchedChosenConfig(ctx.artifact_path(source), ctx.artifact_path(data), seed, presentations, ctx.output_path)

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=MatchedChosenCache,
        run=remote(run_matched_chosen, resources=ResourceConfig.with_cpu(cpu=4, ram="32Gi", disk="64Gi")),
        build_config=build_config,
        deps=(source, data),
    )


@click.command(help=__doc__)
@click.option("--preference-name", required=True)
@click.option("--preference-version", required=True)
@click.option("--seed", type=click.IntRange(min=0), required=True)
@click.option("--presentations", type=click.IntRange(min=1), required=True)
@rl_build_options
def main(preference_name: str, preference_version: str, seed: int, presentations: int) -> ArtifactStep:
    name = user_owned_name(preference_name)
    source = ArtifactStep.adopt(name + "-input", preference_version, f"{name}/{preference_version}", kind=RecoveryPreferenceCache)
    step = matched_chosen_step(source, seed=seed, presentations=presentations)
    return replace(step, runtime_args={"execution": COLLECTION_EXECUTION})


if __name__ == "__main__":
    main()
