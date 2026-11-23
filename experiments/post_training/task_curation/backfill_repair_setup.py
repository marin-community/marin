# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair the backfilled variants' setup commands: drop Dockerfile replays the pinned image provides.

The converter replayed the archive Dockerfile's RUN steps as environment setup commands, so
every machine start re-runs ``pip install pytest`` even though the pinned image was built by
that very step and already ships the package. On the network-denied local tier that replay
fails and the machine never starts. This repair removes a setup command only when it is a
verbatim replay of a RUN step in the task's own Dockerfile — the pinned image's build recipe
— and fails closed otherwise: anything not proven redundant stays, keeping the task's
declared behavior complete.
"""

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any

import pyarrow.parquet as pq
from marin.execution.artifact import write_artifact
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import task_spec_json
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import (
    RENDERER_VERSION,
    write_text_population,
)

LOGGER = logging.getLogger(__name__)

RUN_LINE = re.compile(r"^\s*RUN\s+(.+?)\s*$", re.MULTILINE)
CD_WRAPPER = re.compile(r"^\(\s*cd\s+\S+\s+&&\s+(.+?)\s*\)$", re.DOTALL)


def normalized(command: str) -> str:
    return " ".join(command.split())


def replayed_run_lines(setup_command: str, run_lines: list[str]) -> bool:
    """Whether the setup command replays one of the Dockerfile's RUN steps verbatim."""
    candidates = [normalized(setup_command)]
    wrapped = CD_WRAPPER.match(normalized(setup_command))
    if wrapped:
        candidates.append(normalized(wrapped.group(1)))
    return any(candidate in {normalized(line) for line in run_lines} for candidate in candidates)


def repair_task(task: TaskSpec) -> tuple[TaskSpec, list[str], list[str]]:
    """Drop setup commands the pinned image provides; return the task, the dropped and the kept commands."""
    dockerfile = next(
        (resource for resource in task.resources.oracle if resource.path.endswith("Dockerfile")),
        None,
    )
    if dockerfile is None:
        return task, [], list(task.environment_requirements.setup_commands)
    recipe = resource_bytes(dockerfile).decode("utf-8", errors="replace")
    run_lines = RUN_LINE.findall(recipe)
    if not run_lines:
        return task, [], list(task.environment_requirements.setup_commands)
    dropped, kept = [], []
    for command in task.environment_requirements.setup_commands:
        if replayed_run_lines(command, run_lines):
            dropped.append(command)
        else:
            kept.append(command)
    if not dropped:
        return task, [], kept
    environment = task.environment_requirements.model_copy(update={"setup_commands": tuple(kept)})
    return task.model_copy(update={"environment_requirements": environment}), dropped, kept


def spec_rows(population: str) -> list[tuple[str, str, str]]:
    rows = []
    shards = sorted(str(path) for path in (StoragePath(population) / "specs/*.parquet").glob())
    for shard in shards:
        with StoragePath(shard).open("rb") as stream:
            table = pq.read_table(stream, columns=["id", "task_spec", "runtime"])
        rows.extend(
            zip(
                table.column("id").to_pylist(),
                table.column("task_spec").to_pylist(),
                table.column("runtime").to_pylist(),
                strict=True,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--parent-population", required=True, help="The backfilled population artifact")
    parser.add_argument("--parent-task-specs-sha256", required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--output", required=True, help="The repaired population's versioned artifact prefix")
    parser.add_argument("--converter-version", required=True, help="The repaired variant's bumped version")
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    configure_coreweave_s3()

    digest = hashlib.sha256()
    rows = spec_rows(args.parent_population)
    for _, encoded, _ in rows:
        digest.update(encoded.encode())
    if digest.hexdigest() != args.parent_task_specs_sha256:
        raise ValueError(
            f"The parent population's task specs changed: {digest.hexdigest()} != {args.parent_task_specs_sha256}"
        )
    if len(rows) != args.task_count:
        raise ValueError(f"The parent population holds {len(rows)} tasks, not the declared {args.task_count}")

    counts: Counter[str] = Counter()
    documents = []
    bindings = []
    parent_hashes = {}
    repaired_hashes = {}
    for task_id, encoded, runtime in rows:
        counts["task:total"] += 1
        task = TaskSpec.model_validate_json(encoded)
        parent_hash = hashlib.sha256(encoded.encode()).hexdigest()
        parent_hashes[task_id] = parent_hash
        repaired, dropped, kept = repair_task(task)
        counts["setup_commands_dropped"] += len(dropped)
        counts["setup_commands_kept"] += len(kept)
        counts["tasks_repaired"] += 1 if dropped else 0
        serialized = task_spec_json(repaired)
        repaired_hash = hashlib.sha256(serialized.encode()).hexdigest()
        repaired_hashes[task_id] = repaired_hash
        if dropped:
            bindings.append(
                {
                    "task_id": task_id,
                    "parent_task_sha256": parent_hash,
                    "repaired_task_sha256": repaired_hash,
                    "dropped_setup_commands": dropped,
                    "kept_setup_commands": kept,
                }
            )
        documents.append((json.loads(serialized), json.loads(runtime) if runtime else {}))

    files = write_text_population(documents, args.output)
    if files.task_count != len(documents):
        raise ValueError(f"The repaired population holds {files.task_count} of {len(documents)} tasks")
    LOGGER.info("Wrote the repaired population: %s (task_specs_sha256 %s)", args.output, files.task_specs_sha256)

    record = SourceTextPopulation(
        path=args.output,
        source_id=args.source_id,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        source_version=args.converter_version,
        converter_revision=args.converter_revision,
        text_renderer_version=RENDERER_VERSION,
        task_count=files.task_count,
        text_sha256=files.text_sha256,
        task_specs_sha256=files.task_specs_sha256,
        main_output_dir=files.main_output_dir,
        specs_output_dir=files.specs_output_dir,
        converter_rejections={},
        admission_rejections={},
        within_source_dedup={
            "raw_count": files.raw_task_count,
            "unique_count": files.task_count,
            "within_source_duplicates": files.within_source_duplicates,
            "drop_manifest_uri": files.drop_manifest_uri,
        },
    )
    write_artifact(record.model_dump(mode="json", exclude={"path"}), args.output)
    (StoragePath(args.output) / "population.json").write_text(record.model_dump_json(indent=2) + "\n")

    receipt: dict[str, Any] = {
        "source_id": args.source_id,
        "parent_population": args.parent_population,
        "parent_task_specs_sha256": args.parent_task_specs_sha256,
        "population": args.output,
        "task_specs_sha256": files.task_specs_sha256,
        "task_count": files.task_count,
        "converter_version": args.converter_version,
        "converter_revision": args.converter_revision,
        "repair": {
            "rule": (
                "drop environment.setup_commands that verbatim-replay a RUN step of the task's own "
                "Dockerfile (the pinned image was built by that step and ships the package); "
                "fail closed on anything else"
            ),
            "tasks_repaired": counts["tasks_repaired"],
            "setup_commands_dropped": counts["setup_commands_dropped"],
            "setup_commands_kept": counts["setup_commands_kept"],
            "bindings": len(bindings),
        },
    }
    output = StoragePath(args.output)
    (output / "repair-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "repair-bindings.json").write_text(json.dumps(bindings, indent=1) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
