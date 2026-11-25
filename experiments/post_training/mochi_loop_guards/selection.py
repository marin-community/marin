# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the frozen 823-trial panel as a task selection for the loop-guard stages.

Lowers the pinned Harbor cohorts (TB2 89, OT-TBLite 100, SWE-bench-100) and the 128
held-out swesmith-ho rows from `cohorts/manifest.json` into native TaskRolloutSpecs, one
parquet per cohort, with each cohort's frozen protocol. Runs CPU-only, in region, before
S0; the stages adopt the published selection by name and version:

    uv run --package marin-core python -m experiments.post_training.mochi_loop_guards.selection \\
        --manifest cohorts/manifest.json --output <SELECTION_ROOT>/users/benfeuer/mochi-loop-guards/panel/2026.11.25.1
"""

import argparse
import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply, resolve
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.agents.pi.agent import PI_HARNESS
from rolloutengine.agents.terminus.agent import TERMINUS_HARNESS
from rolloutengine.models import TASK_ROLLOUT_COLUMN, MachineRuntimeSpec, RolloutLimits, TaskRolloutSpec
from shellbox.machine import Backend, NetworkPolicy
from taskcompendium.importers.harbor.config import read_task_config
from taskcompendium.importers.harbor.package import JobSettings, lower_package
from taskcompendium.models import Source, TaskSpec, grader_environment
from zephyr.writers import write_parquet_file

from experiments.evaluation.benchmarks.agentic import OT_TBLITE, SWEBENCH_VERIFIED_100, TERMINAL_BENCH_2
from experiments.evaluation.benchmarks.harbor import image_lock
from experiments.post_training.mochi_loop_guards.panel import SWESMITH_SOURCE_URI, _row_hash
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.harbor_packages import ARCHIVE_SUFFIX, extract_package
from experiments.post_training.task_curation.pipeline import download_step
from experiments.post_training.task_curation.swesmith_training import (
    SWESMITH_DATASET,
    SWESMITH_MACHINE,
    lower_swesmith_task,
)

logger = logging.getLogger(__name__)

SELECTION_VERSION = "2026.11.25.1"
CLEANUP_TIMEOUT = 120
HARBOR_COHORTS = {"tb2": TERMINAL_BENCH_2, "tblite": OT_TBLITE, "swe100": SWEBENCH_VERIFIED_100}
TRIALS = {"tb2": 3, "tblite": 2, "swe100": 1, "swesmith_ho": 2}
COHORT_FILES = {"tb2": "tb2", "tblite": "tblite", "swe100": "swe100", "swesmith_ho": "swesmith_ho"}
MANIFEST_COHORTS = {"tb2": "tb2", "tblite": "tblite", "swe100": "swe100", "swesmith_ho": "swesmith-ho"}


def _machine(environment: Any, network: NetworkPolicy) -> MachineRuntimeSpec:
    return MachineRuntimeSpec(
        backend=Backend.DAYTONA,
        network=network,
        cpus=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        gpus=0,
        user=None,
        startup_timeout=environment.build_timeout_sec,
        cleanup_timeout=CLEANUP_TIMEOUT,
    )


def _rollout(
    task: TaskSpec,
    *,
    harness: str,
    task_machine: MachineRuntimeSpec,
    verifier_machine: MachineRuntimeSpec | None,
    protocol: dict[str, Any],
) -> TaskRolloutSpec:
    return TaskRolloutSpec(
        task=task,
        harness=harness,
        task_machine=task_machine,
        verifier_machine=verifier_machine,
        limits=RolloutLimits(
            max_turns=protocol.get("max_turns"),
            max_total_tokens=None,
            model_turn_timeout=protocol["model_turn_timeout_seconds"],
            tool_turn_timeout=None,
            total_turn_timeout=protocol["agent_timeout_seconds"],
            attempt_timeout=protocol["attempt_timeout_seconds"],
            verifier_timeout=protocol["verifier_timeout_seconds"],
            cleanup_timeout=CLEANUP_TIMEOUT,
        ),
    )


def _harbor_rows(
    cohort: str, manifest: dict[str, Any], downloaded: str, image_digests: dict[str, str]
) -> list[dict[str, str]]:
    """Lower every pinned package of one Harbor cohort under the frozen protocol."""
    dataset = HARBOR_COHORTS[cohort]
    protocol = manifest["cohorts"][MANIFEST_COHORTS[cohort]]["protocol"]
    root = StoragePath(downloaded)
    packages = sorted((root / "tasks").ls(), key=lambda path: path.name)
    if len(packages) != len(manifest["cohorts"][MANIFEST_COHORTS[cohort]]["rows"]):
        raise ValueError(f"{cohort}: pinned package census differs from frozen source-order panel")
    rows = []
    for index, entry in enumerate(manifest["cohorts"][MANIFEST_COHORTS[cohort]]["rows"]):
        archive = packages[int(entry["task_id"].rpartition(":")[2])]
        package = archive.name.removesuffix(ARCHIVE_SUFFIX)
        provenance = Source(
            dataset=dataset.source.name,
            revision=dataset.source.revision,
            row=f"tasks/{package}{ARCHIVE_SUFFIX}:0",
            importer_revision=SELECTION_VERSION,
        )
        with tempfile.TemporaryDirectory(prefix=f"panel-{cohort}-") as temp:
            directory = Path(temp)
            extract_package(archive.read_bytes(), directory)
            config = read_task_config(directory)
            lowered = lower_package(
                directory,
                task_id=f"{cohort}/{package}",
                source=provenance,
                rule=dataset.rule(directory),
                image_digests=image_digests,
                job=JobSettings(
                    agent_timeout=protocol["agent_timeout_seconds"],
                    verifier_timeout=protocol["verifier_timeout_seconds"],
                ),
            )
        verifier_machine = None
        if grader_environment(lowered.task.grader) is not None:
            verifier = config.verifier_environment
            verifier_machine = _machine(verifier, NetworkPolicy.ALLOW if verifier.allow_internet else NetworkPolicy.DENY)
        rollout = _rollout(
            lowered.task,
            harness=TERMINUS_HARNESS,
            task_machine=_machine(config.environment, lowered.episode.network),
            verifier_machine=verifier_machine,
            protocol=protocol,
        )
        rows.append({TASK_ROLLOUT_COLUMN: rollout.model_dump_json()})
        if (index + 1) % 25 == 0:
            logger.info(
                "lowered %d/%d %s packages",
                index + 1,
                len(manifest["cohorts"][MANIFEST_COHORTS[cohort]]["rows"]),
                cohort,
            )
    return rows


def _swesmith_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Bind the 128 held-out swesmith-ho TaskSpecs to the RL harness and the RL protocol."""
    cohort = manifest["cohorts"]["swesmith-ho"]
    protocol = cohort["protocol"]
    machine = SWESMITH_MACHINE.model_copy(update={"memory_mb": protocol["memory_mb"]})
    limits = {
        **protocol,
        "model_turn_timeout_seconds": protocol["agent_timeout_seconds"],
        "attempt_timeout_seconds": protocol["trial_timeout_seconds"],
    }
    digests = image_lock(SWESMITH_DATASET)
    table = pq.read_table(io.BytesIO(StoragePath(SWESMITH_SOURCE_URI).read_bytes())).to_pylist()
    selected = {row["index"]: row for row in cohort["rows"]}
    rows = []
    for index, row in enumerate(table):
        entry = selected.get(index)
        if entry is None:
            continue
        if _row_hash(row) != entry["row_hash"]:
            raise ValueError(f"swesmith row {index} hash differs from the frozen manifest")
        task = TaskSpec.model_validate_json(row["task_spec"])
        if task.id != entry["task_id"]:
            raise ValueError(f"swesmith row {index} task id differs from the frozen manifest")
        task = lower_swesmith_task(task, SWESMITH_MACHINE, digests)
        rollout = _rollout(
            task,
            harness=PI_HARNESS,
            task_machine=machine,
            verifier_machine=None,
            protocol=limits,
        )
        rows.append({TASK_ROLLOUT_COLUMN: rollout.model_dump_json()})
    return rows


def publish_panel(manifest_path: str, output: str) -> dict[str, Any]:
    """Lower every cohort and publish the panel selection with its per-file summary."""
    manifest_bytes = StoragePath(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    root = StoragePath(output)
    root.mkdirs()
    summary: dict[str, Any] = {"grader_kind_counts": {"train": {}}, "selection_version": SELECTION_VERSION}
    for cohort in ("tb2", "tblite", "swe100", "swesmith_ho"):
        if cohort == "swesmith_ho":
            rows = _swesmith_rows(manifest)
        else:
            dataset = HARBOR_COHORTS[cohort]
            digests = image_lock(dataset.image_lock_name)
            downloaded = resolve(download_step(dataset.source, CampaignRuntime())).path
            rows = _harbor_rows(cohort, manifest, downloaded, digests)
        expected = manifest["cohorts"][MANIFEST_COHORTS[cohort]]["trials"] // TRIALS[cohort]
        if len(rows) != expected:
            raise ValueError(f"{cohort}: lowered {len(rows)} tasks, the manifest freezes {expected}")
        write_parquet_file(rows, str(root / f"{COHORT_FILES[cohort]}.parquet"))
        stop_rows = []
        for row in rows:
            spec = TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN])
            spec = spec.model_copy(update={"limits": spec.limits.model_copy(update={"repeat_stall_batches": 3})})
            stop_rows.append({TASK_ROLLOUT_COLUMN: spec.model_dump_json()})
        write_parquet_file(stop_rows, str(root / f"{COHORT_FILES[cohort]}_stop.parquet"))
        summary[f"{cohort}_rows"] = len(rows)
        summary[f"{cohort}_stop_rows"] = len(rows)
        summary["grader_kind_counts"][COHORT_FILES[cohort]] = {}
        summary["grader_kind_counts"][f"{COHORT_FILES[cohort]}_stop"] = {}
        logger.info("published %s: %d tasks x %d trials", cohort, len(rows), TRIALS[cohort])
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def panel_selection_step() -> ArtifactStep[Artifact]:
    """The panel selection as an artifact step, for graphs that depend on it."""
    return apply(
        "data/rl/mochi-loop-guards/panel",
        lambda output: publish_panel("cohorts/manifest.json", output),
        version=SELECTION_VERSION,
        output=OUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="cohorts/manifest.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_coreweave_s3()
    print(json.dumps(publish_panel(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
