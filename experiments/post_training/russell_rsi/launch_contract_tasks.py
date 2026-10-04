# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the fixed Russell contract preparation and admission artifacts."""

import json
import os
from dataclasses import replace

import click
from fray.types import ResourceConfig
from iris.client.context_state import has_current_context
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.cli import build_options

from experiments.post_training.russell_rsi.contract_tasks import (
    RENDERED_METHOD,
    TEACHER_METHOD,
    ContractTasksConfig,
    run_contract_tasks_in_project,
)
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV


def run_contract_tasks(config: ContractTasksConfig) -> None:
    remote(
        run_contract_tasks_in_project,
        resources=ResourceConfig.with_cpu(cpu=32, ram="128GB", disk="64GB"),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]} if config.method == TEACHER_METHOD else {},
    )(config)


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@build_options
def main(config_uri: str, config_sha256: str) -> ArtifactStep[Artifact]:
    if click.get_current_context().params.get("do_run") and not (
        has_current_context() or os.environ.get("IRIS_TASK_ID")
    ):
        raise click.UsageError("Run this CPU coordinator inside the CW02 Iris context")
    version = resolve_version("russell-rsi-contracts", None)
    config = ContractTasksConfig(**json.loads(pinned_bytes(config_uri, config_sha256)))
    if config.method == RENDERED_METHOD and not all(
        (
            config.observation_manifest_uri,
            config.observation_manifest_sha256,
            config.observation_source_manifest_uri,
            config.observation_source_manifest_sha256,
        )
    ):
        raise click.UsageError("Rendered construction requires pinned original observations and source inputs")
    if config.stage not in {"prepare", "admit"}:
        raise click.UsageError("The fixed-contract stage must be prepare or admit")
    if config.stage == "admit" and (not config.statement_review_uri or not config.statement_review_sha256):
        raise click.UsageError("Admission requires the exact reviewed statements and their digest")
    if config.stage == "admit" and (not config.prepared_manifest_uri or not config.prepared_manifest_sha256):
        raise click.UsageError("Admission requires the complete pinned preparation evidence")
    prepared = (
        ()
        if config.stage == "prepare"
        else (
            ArtifactStep.adopt(
                "documents/russell-rsi-reviewed-preparation",
                version,
                config.prepared_manifest_uri,
                config={"sha256": config.prepared_manifest_sha256},
            ),
        )
    )
    if (
        config.method == TEACHER_METHOD
        and click.get_current_context().params.get("do_run")
        and not os.environ.get(GLM_TOKEN_ENV)
    ):
        raise click.UsageError(f"Contract construction requires {GLM_TOKEN_ENV}")
    return ArtifactStep(
        name=f"documents/russell-rsi-contracts-{config.method}-{config.stage}",
        version=version,
        artifact_type=Artifact,
        deps=prepared,
        build_config=lambda ctx: replace(config, output_path=ctx.output_path),
        run=run_contract_tasks,
    )


if __name__ == "__main__":
    main()
