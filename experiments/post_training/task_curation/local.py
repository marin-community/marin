# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute complete curation artifact graphs sequentially on a local Zephyr context."""

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import click
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import (
    CampaignArtifact,
    CampaignFailed,
    CampaignRuntime,
    CampaignStatus,
    OutcomeStatus,
    SourceOutcome,
    error_chain,
    write_campaign_report,
)


@contextmanager
def _local_artifact_prefix(prefix: str) -> Iterator[None]:
    # lazy.lower resolves dependency paths from MARIN_PREFIX at execution time.
    # QUICK is sequential; reviewed campaigns never enter this scope.
    previous = os.environ.get("MARIN_PREFIX")
    os.environ["MARIN_PREFIX"] = prefix
    try:
        yield
    finally:
        if previous is None:
            del os.environ["MARIN_PREFIX"]
        else:
            os.environ["MARIN_PREFIX"] = previous


def run_local_steps(
    steps: Mapping[str, ArtifactStep[CampaignArtifact]],
    output_root: Path,
    *,
    runtime: CampaignRuntime,
    max_workers: int,
    download_cache: Path,
) -> tuple[SourceOutcome, ...]:
    """Run each complete graph, retaining peer results when a source fails."""
    output_root, download_cache = output_root.resolve(), download_cache.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "campaign.json"
    outcomes = {name: SourceOutcome(name, str(output_root / name), OutcomeStatus.QUEUED) for name in steps}

    def report(status: CampaignStatus) -> None:
        write_campaign_report(
            str(report_path), status, mode=SourceProcessingMode.QUICK, outcomes=list(outcomes.values())
        )

    report(CampaignStatus.RUNNING)
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(
                client=client,
                max_workers=max_workers,
                resources=ResourceConfig(cpu=1, ram="4g"),
                chunk_storage_prefix=str(output_root / ".zephyr"),
                name="task-curation-quick",
            ) as context,
            runtime.activate(context),
        ):
            for name, step in steps.items():
                path = str(output_root / name)
                outcomes[name] = SourceOutcome(name, path, OutcomeStatus.RUNNING)
                report(CampaignStatus.RUNNING)
                try:
                    with _local_artifact_prefix(str(download_cache)):
                        # Mutable terminal versions rerun through the ordinary executor;
                        # dependency versions and cache addresses remain unchanged.
                        terminal = replace(step, version="dev")
                        spec = replace(terminal.lower(), override_output_path=path)
                        StepRunner().run([spec], max_concurrent=1)
                        artifact = step.artifact_type.raw_load(path)
                except Exception as error:
                    outcomes[name] = SourceOutcome(name, path, OutcomeStatus.FAILED, error_chain(error))
                    click.echo(json.dumps(asdict(outcomes[name])), err=True)
                else:
                    outcomes[name] = SourceOutcome(name, path, artifact.status, result=artifact.result)
                    click.echo(json.dumps(asdict(outcomes[name])))
                report(CampaignStatus.RUNNING)
    finally:
        client.shutdown()
    failed = any(outcome.status == OutcomeStatus.FAILED for outcome in outcomes.values())
    report(CampaignStatus.FAILED if failed else CampaignStatus.COMPLETED)
    if failed:
        raise CampaignFailed(f"Quick conversion failed for some sources; see {report_path}")
    return tuple(outcomes.values())
