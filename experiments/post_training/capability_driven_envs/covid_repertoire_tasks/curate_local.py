# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize a local Taskforge release with the TaskCompendium curation pipeline.

The output is the normalized input consumed by the existing TaskTrove/Harbor
exporter. QUICK mode intentionally performs no model review or additional grader
execution; those remain release gates recorded by the Taskforge manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec, VerifyitGrader, verifyit_spec
from taskcompendium.pipeline.controls import answer_reply
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import Controls, IntendedUse, RawRow, Reply, SourceRecipe
from taskcompendium.pipeline.source_processing import SourceProcessingMode, run_source_pipeline
from verifyit.spec import StructuredExactSpec
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.environments import VERIFYIT_PACKAGE
from experiments.post_training.task_curation.images.build import BASE_IMAGE
from experiments.post_training.task_curation.tasktrove import harbor_export
from experiments.post_training.task_curation.tasktrove.harbor_export import HarborSourceMetadata

SOURCE_NAME = "covid-repertoire-taskforge"
SOURCE_VERSION = "1"
SOURCE_FILE = "accepted_tasks.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(row: RawRow, _context: ConversionContext) -> TaskSpec:
    """Validate one already-authored TaskSpec without changing its contract."""
    task = TaskSpec.model_validate(dict(row.data))
    if task.id != row.data.get("id"):
        raise ValueError(f"row task id changed while decoding: {task.id!r}")
    return task.model_copy(
        update={
            "id": row.id,
            "source": row.source,
            "tags": (*task.tags, f"taskforge-id:{task.id}"),
        }
    )


def golden(task: TaskSpec) -> Reply | None:
    """Return the structured grader's exact reference as a typed submission."""
    grader = task.grader
    if not isinstance(grader, VerifyitGrader):
        return None
    spec = verifyit_spec(grader)
    if not isinstance(spec, StructuredExactSpec):
        return None
    return answer_reply(task, json.dumps(spec.expected))


def negative(task: TaskSpec) -> Reply:
    """Return a schema-valid but incomplete severe-negative submission."""
    return answer_reply(task, "{}")


def recipe(release: Path) -> SourceRecipe:
    """Build the local source recipe pinned to the accepted release bytes."""
    source_path = release / SOURCE_FILE
    return SourceRecipe(
        name=SOURCE_NAME,
        version=SOURCE_VERSION,
        source=SourceFiles(
            dataset=SOURCE_NAME,
            revision=sha256_file(source_path),
            patterns=(SOURCE_FILE,),
            format=SourceFormat.JSONL,
        ),
        convert=convert,
        rubric=None,
        controls=Controls(golden=golden, negative=negative),
        intended_use=IntendedUse.TRAIN,
    )


def run(release: Path, output: Path) -> object:
    """Run the existing local Zephyr curation pipeline in QUICK mode."""
    source_recipe = recipe(release)
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(output.parent / "curation-chunks")) as context:
        return run_source_pipeline(
            source_recipe,
            context,
            str(release),
            str(output),
            config=None,
            mode=SourceProcessingMode.QUICK,
            canonical_source=source_recipe.name,
        )


def export_tasktrove(normalized: Path, output: Path) -> dict[str, object]:
    """Lower normalized tasks with the existing TaskTrove/Harbor exporter."""
    return harbor_export.export_harbor(
        StoragePath(str(normalized)),
        StoragePath(str(output)),
        grader_image=BASE_IMAGE,
        source=HarborSourceMetadata(
            SOURCE_NAME,
            "stephenson2021-covid-repertoire-objective-v1",
            "scientific-analysis",
        ),
        fallback_actor_image=BASE_IMAGE,
        verifyit_package_root=VERIFYIT_PACKAGE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--normalized-output", type=Path, required=True)
    parser.add_argument("--tasktrove-output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.release / SOURCE_FILE).is_file():
        parser.error(f"--release must contain {SOURCE_FILE}")
    result = run(args.release, args.normalized_output)
    exported = export_tasktrove(args.normalized_output, args.tasktrove_output)
    summary = {key: value for key, value in exported.items() if key != "rejections"}
    print(f"normalized {SOURCE_NAME} at {result}")
    print(f"exported TaskTrove compatibility data: {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
