# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute complete curation artifact graphs sequentially on a local Zephyr context."""

import json
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
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
from experiments.post_training.task_curation.curation_catalog import CATALOG_DIRECTORY


@contextmanager
def artifact_prefix(prefix: str) -> Iterator[None]:
    """Bind planning and execution to one storage root for the entire campaign.

    MARIN_PREFIX is process global, so source threads share this enclosing scope.
    """
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
    worker_resources: ResourceConfig,
    cache_dir: Path,
    source_ids: Mapping[str, str],
) -> tuple[SourceOutcome, ...]:
    """Run each complete graph, retaining peer results when a source fails."""
    output_root, cache_dir = output_root.resolve(), cache_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "campaign.json"
    outcomes = {name: SourceOutcome(name, str(output_root / name), OutcomeStatus.QUEUED) for name in steps}
    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC).isoformat()

    def report(status: CampaignStatus) -> None:
        write_campaign_report(
            str(report_path),
            status,
            mode=SourceProcessingMode.QUICK,
            outcomes=list(outcomes.values()),
            run_id=run_id,
            started_at=started_at,
            source_ids=source_ids,
            execution={"max_workers": max_workers},
            catalog_root=str(output_root / CATALOG_DIRECTORY),
        )

    report(CampaignStatus.RUNNING)
    client = LocalClient()
    try:
        with ExitStack() as cleanup:
            cleanup.callback(client.shutdown)
            with (
                set_current_client(client),
                ZephyrContext(
                    client=client,
                    max_workers=max_workers,
                    resources=worker_resources,
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
                        with artifact_prefix(str(cache_dir)):
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
    except BaseException:
        report(CampaignStatus.FAILED)
        raise
    failed = any(outcome.status == OutcomeStatus.FAILED for outcome in outcomes.values())
    report(CampaignStatus.FAILED if failed else CampaignStatus.COMPLETED)
    if failed:
        raise CampaignFailed(f"Quick conversion failed for some sources; see {report_path}")
    return tuple(outcomes.values())
