# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise ten pinned task sources with complete Parquet audit outputs."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.inference.openai_batch import OpenAIBatchClient
from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.datasets import calendar_tasks, executable_tasks, nemo_actions, qa_tasks, reasoning_tasks
from taskcompendium.pipeline.models import DatasetRecipe, RawRow
from taskcompendium.pipeline.records import read_jsonl
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.runner import run_pipeline

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation_executable import convert_snapshot
from experiments.post_training.task_curation_sampling import CONFIGS


def source_recipe(name: str, sources: Path, image: str) -> DatasetRecipe:
    snapshot = sources / name / "sample.jsonl"
    if name in executable_tasks.CONFIGS:
        converted = snapshot.with_name("converted.jsonl")
        convert_snapshot(snapshot, converted, name)
        return executable_tasks.recipe(name, converted, image, timeout=120.0, memory_mb=512)
    factories = {
        "calendar": calendar_tasks.recipe,
        "reasoning_gym": reasoning_tasks.reasoning_recipe,
        "all_puzzles": reasoning_tasks.puzzle_recipe,
        "nemo_actions": nemo_actions.snapshot_recipe,
        "knowledge_openqa": qa_tasks.knowledge_recipe,
        "science_openqa": qa_tasks.science_recipe,
    }
    return factories[name](snapshot)


def checked_recipe(recipe: DatasetRecipe, rows: list[dict], workers: int) -> DatasetRecipe:
    """Run the same executable controls concurrently before normal pipeline deduplication."""
    suite = recipe.check_suite
    assert suite is not None
    tasks = []
    for index, data in enumerate(rows):
        source = Source(
            dataset=recipe.source.dataset,
            revision=recipe.source.revision,
            row=f"{recipe.source.config}:{recipe.source.split}:{index}",
            importer_revision=recipe.version,
        )
        task_id = f"{recipe.name}-{canonical_sha256(source.model_dump())}"
        task = recipe.normalize(RawRow(task_id, source, data))
        if isinstance(task, TaskSpec):
            tasks.append(task)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        reports = dict(zip((task.id for task in tasks), executor.map(suite.run, tasks), strict=True))
    return replace(recipe, check_suite=replace(suite, run=lambda task: reports[task.id]))


def combined_table(output: Path, names: list[str], filename: str) -> pa.Table:
    tables = []
    for name in names:
        table = pq.read_table(output / name / filename)
        tables.append(table.append_column("source_name", pa.array([name] * table.num_rows, type=pa.string())))
    return pa.concat_tables(tables)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--image", required=True, help="Existing immutable local Docker image ID")
    parser.add_argument("--source", action="append", choices=(*CONFIGS, "nemo_actions"))
    parser.add_argument("--check-workers", type=int, default=4)
    parser.add_argument("--review-max-characters", type=int, default=64000)
    args = parser.parse_args()
    names = args.source or [*CONFIGS, "nemo_actions"]
    args.output.mkdir(parents=True, exist_ok=True)
    reviewer = BatchReviewer(
        OpenAIBatchClient(args.base_url, os.environ[GLM_BULK_TOKEN_ENV]),
        GLM_MODEL,
        args.model_revision,
        max_tokens=4096,
        max_prompt_characters=args.review_max_characters,
    )
    recipes = {name: source_recipe(name, args.sources, args.image) for name in names}
    # Freeze all criteria and partitions before reviewing any reserved holdout rows.
    plan = {
        name: {
            "rubric": asdict(recipe.rubric),
            "sample": json.loads((args.sources / name / "sample-manifest.json").read_text()),
        }
        for name, recipe in recipes.items()
    }
    frozen = json.dumps(plan, sort_keys=True, indent=2)
    freeze_path = args.output / "frozen-plan.json"
    if freeze_path.exists() and freeze_path.read_text() != frozen:
        raise ValueError("Sample partitions or rubrics differ from the frozen run plan")
    freeze_path.write_text(frozen)
    print(json.dumps({"frozen_plan_sha256": hashlib.sha256(frozen.encode()).hexdigest(), "sources": names}), flush=True)
    manifests = {}
    for name, recipe in recipes.items():
        rows = read_jsonl(Path(recipe.source.path))
        if name in executable_tasks.CONFIGS and not (args.output / name / "checks.jsonl").exists():
            print(json.dumps({"source": name, "stage": "grader_controls", "rows": len(rows)}), flush=True)
            recipe = checked_recipe(recipe, rows, args.check_workers)
        manifests[name] = run_pipeline(recipe, rows, output_path=args.output / name, limit=len(rows), reviewer=reviewer)
        print(json.dumps({"source": name, "dispositions": manifests[name]["dispositions"]}), flush=True)
    audit = combined_table(args.output, names, "audit.parquet")
    pq.write_table(audit, args.output / "audit.parquet", compression="zstd")
    accepted = combined_table(args.output, names, "accepted.parquet")
    pq.write_table(accepted, args.output / "accepted.parquet", compression="zstd")
    (args.output / "summary.json").write_text(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
