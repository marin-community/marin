# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the bounded SFT-only comparison from a local foreground coordinator."""

import importlib.metadata
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import click
from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, run
from marin.experiment.cli import BuildResult, build_options_with_runner
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_transport_replacement import (
    PROTOCOL as CODING_REPLACEMENT_PROTOCOL,
)
from experiments.post_training.russell_rsi.coding_transport_replacement import (
    prepare_coding_replacement,
    replacement_selection_stages,
)
from experiments.post_training.russell_rsi.completed_coding_analysis import (
    PROTOCOL as COMPLETED_ANALYSIS_PROTOCOL,
)
from experiments.post_training.russell_rsi.completed_coding_analysis import (
    completed_coding_analysis_stages,
)
from experiments.post_training.russell_rsi.completed_sft_selection import (
    EXTRACTION_PROTOCOL,
    completed_coding_extraction_stages,
    completed_sft_selection_stages,
)
from experiments.post_training.russell_rsi.completed_sft_selection import PROTOCOL as COMPLETED_SELECTION_PROTOCOL
from experiments.post_training.russell_rsi.interrupted_calibration import (
    BRANCH_PACKAGES,
    OUTPUT_PROTOCOL,
    BoundedIrisClient,
)
from experiments.post_training.russell_rsi.launch import CLUSTER
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.retention_continuation import (
    HASH_REPAIR_PROTOCOL,
    prepare_retention_continuation,
    prepare_retention_hash_repair,
    retention_continuation_stages,
)
from experiments.post_training.russell_rsi.retention_continuation import (
    PROTOCOL as RETENTION_CONTINUATION_PROTOCOL,
)
from experiments.post_training.russell_rsi.settings import IRIS_TASK_ID_ENV
from experiments.post_training.russell_rsi.teacher_chat_study import chat_study_post_workflow


def require_reviewed_source(uri: str, sha256: str) -> dict:
    """Require the clean reviewed source and the actual installed science runtime."""
    review = json.loads(pinned_bytes(uri, sha256))
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if review["status"] != "approved" or review["source_path"] != str(root) or review["source_head"] != head or dirty:
        raise ValueError("Foreground coordinator requires the reviewed clean source commit")
    for name in BRANCH_PACKAGES:
        module = __import__(name)
        expected = root / "lib" / name / "src" / name
        if list(module.__path__) != [str(expected)]:
            raise ValueError(f"Foreground coordinator imported {name} outside its reviewed source")
    distribution = importlib.metadata.distribution(MARIN_SKYRL.distribution)
    direct_url = json.loads(distribution.read_text("direct_url.json") or "null")
    if not isinstance(direct_url, dict) or direct_url.get("vcs_info", {}).get("commit_id") != MARIN_SKYRL.commit:
        raise ValueError("Foreground coordinator must use the installed f124 science runtime")
    return {"source_head": head, "review_sha256": sha256, "skyrl": direct_url}


def foreground_runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
    if os.environ.get(IRIS_TASK_ID_ENV):
        raise ValueError("Interrupted SFT evaluation requires a local foreground coordinator")
    root = Path(__file__).resolve().parents[3]
    with open_iris_client(cluster_name=CLUSTER, workspace=root) as raw_client:
        client = cast(IrisClient, BoundedIrisClient(raw_client))
        with iris_ctx_scope(IrisContext(job_id=None, client=client)):
            with set_current_client(FrayIrisClient.from_iris_client(client)):
                run(*handles, max_concurrent=max_concurrent, force_run_failed=False)


def foreground_build_options(fn: Callable[..., BuildResult]) -> Callable[..., None]:
    return build_options_with_runner(fn, foreground_runner)


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@click.option("--source-review-uri", required=True)
@click.option("--source-review-sha256", required=True)
@click.option(
    "--stage",
    type=click.Choice(
        [
            "evaluate-interrupted",
            "retain",
            "repair-retention",
            "select",
            "replace-coding",
            "select-replacement",
            "select-completed",
            "extract-completed-coding",
            "analyze-completed-coding",
        ]
    ),
    required=True,
)
@foreground_build_options
def main(
    config_uri: str, config_sha256: str, source_review_uri: str, source_review_sha256: str, stage: str
) -> list[ArtifactStep]:
    configure_coreweave_s3()
    require_reviewed_source(source_review_uri, source_review_sha256)
    config = json.loads(pinned_bytes(config_uri, config_sha256))
    if stage == "analyze-completed-coding":
        if resolve_version(COMPLETED_ANALYSIS_PROTOCOL, None) != config["version"]:
            raise click.UsageError("Completed coding analysis version differs from its frozen configuration")
        extraction = PinnedFile(**config["extraction_config"]).read_json()
        source = PinnedFile(**extraction["source_config"]).read_json()
        original = chat_study_post_workflow(source, "evaluate-interrupted")
        return [completed_coding_analysis_stages(config, PinnedFile(config_uri, config_sha256), original)["terminal"]]
    if stage in {"select-completed", "extract-completed-coding"}:
        protocol = COMPLETED_SELECTION_PROTOCOL if stage == "select-completed" else EXTRACTION_PROTOCOL
        if resolve_version(protocol, None) != config["version"]:
            raise click.UsageError("Completed selection version differs from its frozen configuration")
        source = PinnedFile(**config["source_config"]).read_json()
        original = chat_study_post_workflow(source, "evaluate-interrupted")
        stages = completed_sft_selection_stages if stage == "select-completed" else completed_coding_extraction_stages
        return [stages(config, PinnedFile(config_uri, config_sha256), original)["terminal"]]
    if stage in {"replace-coding", "select-replacement"}:
        if resolve_version(CODING_REPLACEMENT_PROTOCOL, None) != config["version"]:
            raise click.UsageError("Coding replacement version differs from its frozen amendment")
        coding_pin = (
            PinnedFile(config_uri, config_sha256)
            if stage == "replace-coding"
            else PinnedFile(config["coding_config_uri"], config["coding_config_sha256"])
        )
        coding_config = coding_pin.read_json()
        amendment = PinnedFile(
            coding_config["transport_amendment_uri"], coding_config["transport_amendment_sha256"]
        ).read_json()
        retention_pin = PinnedFile(**amendment["retention"]["config"])
        retention_config = retention_pin.read_json()
        source = PinnedFile(retention_config["source_config_uri"], retention_config["source_config_sha256"]).read_json()
        original = chat_study_post_workflow(source, "evaluate-interrupted")
        retention = prepare_retention_continuation(retention_config, source, original, retention_pin)
        replacement = prepare_coding_replacement(coding_config, coding_pin, retention)
        if stage == "replace-coding":
            return [replacement.coding]
        return [replacement_selection_stages(config, replacement)["terminal"]]
    if stage in {"retain", "repair-retention", "select"}:
        protocol = HASH_REPAIR_PROTOCOL if stage == "repair-retention" else RETENTION_CONTINUATION_PROTOCOL
        if resolve_version(protocol, None) != config["version"]:
            raise click.UsageError("Retention continuation version differs from its frozen amendment")
        if stage == "retain":
            retention_pin = PinnedFile(config_uri, config_sha256)
        elif stage == "repair-retention":
            retention_pin = PinnedFile(config["predecessor_config_uri"], config["predecessor_config_sha256"])
        else:
            retention_pin = PinnedFile(config["retention_config_uri"], config["retention_config_sha256"])
        retention_config = retention_pin.read_json()
        source = PinnedFile(retention_config["source_config_uri"], retention_config["source_config_sha256"]).read_json()
        if source["runtime_commit"] != MARIN_SKYRL.commit:
            raise click.UsageError("Retention continuation changed the original science runtime")
        original = chat_study_post_workflow(source, "evaluate-interrupted")
        prepared = prepare_retention_continuation(retention_config, source, original, retention_pin)
        if stage == "repair-retention":
            return [prepare_retention_hash_repair(config, PinnedFile(config_uri, config_sha256), prepared)]
        if stage == "retain":
            return [prepared.step]
        return [retention_continuation_stages(config, prepared)["terminal"]]
    if resolve_version(OUTPUT_PROTOCOL, None) != config["version"] or config["runtime_commit"] != MARIN_SKYRL.commit:
        raise click.UsageError("Interrupted evaluation version or runtime differs from the frozen study")
    return [chat_study_post_workflow(config, stage)["terminal"]]


if __name__ == "__main__":
    main()
