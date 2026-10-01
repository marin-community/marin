# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curate the next ten pinned Atlas sources with binary quality decisions."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
from marin.inference.openai_batch import OpenAIBatchClient
from taskcompendium.pipeline.datasets import (
    advanced_calculations,
    arc_inductive,
    arc_transductive,
    atlas_code,
    code_contests,
    codenet,
    indirect_injection,
    knowledge_mcqa,
    math_openreasoning,
    qa_abstention,
    web_search_mcqa,
)
from taskcompendium.pipeline.models import DatasetRecipe, FilterPolicy
from taskcompendium.pipeline.records import read_jsonl
from taskcompendium.pipeline.review import BASE_RUBRIC, BatchReviewer
from taskcompendium.pipeline.runner import run_pipeline

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL
from experiments.post_training.task_curation_next_code import convert_snapshot
from experiments.post_training.task_curation_sampling import NEXT_CONFIGS
from experiments.post_training.task_curation_ten import checked_recipe, combined_table


def source_recipe(name: str, sources: Path, image: str) -> DatasetRecipe:
    snapshot = sources / name / "sample.jsonl"
    if name in atlas_code.CONFIGS:
        converted = snapshot.with_name("converted.jsonl")
        convert_snapshot(snapshot, converted, name)
        factories = {"code_contests": code_contests.recipe, "codenet": codenet.recipe}
        return factories[name](converted, image, timeout=120.0, memory_mb=512)
    factories = {
        "math_openreasoning": math_openreasoning.recipe,
        "advanced_calculations": advanced_calculations.recipe,
        "knowledge_mcqa": knowledge_mcqa.recipe,
        "web_search_mcqa": web_search_mcqa.recipe,
        "qa_abstention": qa_abstention.recipe,
        "arc_transductive": arc_transductive.recipe,
        "arc_inductive": arc_inductive.recipe,
        "indirect_injection": indirect_injection.recipe,
    }
    return factories[name](snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", action="append", choices=tuple(NEXT_CONFIGS))
    parser.add_argument("--check-workers", type=int, default=4)
    parser.add_argument("--review-max-characters", type=int, default=128000)
    args = parser.parse_args()
    names = args.source or list(NEXT_CONFIGS)
    args.output.mkdir(parents=True, exist_ok=True)
    reviewer = BatchReviewer(
        OpenAIBatchClient(args.base_url, os.environ[GLM_BULK_TOKEN_ENV]),
        GLM_MODEL,
        args.model_revision,
        max_tokens=4096,
        max_prompt_characters=args.review_max_characters,
    )
    policy = FilterPolicy()
    recipes = {name: source_recipe(name, args.sources, args.image) for name in names}
    plan = {
        "base_rubric": BASE_RUBRIC,
        "base_rubric_sha256": hashlib.sha256(BASE_RUBRIC.encode()).hexdigest(),
        "policy": asdict(policy),
        "reviewer": reviewer.identity,
        "sources": {
            name: {
                "rubric": asdict(recipe.rubric),
                "sample": json.loads((args.sources / name / "sample-manifest.json").read_text()),
            }
            for name, recipe in recipes.items()
        },
    }
    frozen = json.dumps(plan, sort_keys=True, indent=2)
    path = args.output / "frozen-plan.json"
    if path.exists() and path.read_text() != frozen:
        raise ValueError("Sample, rubric, model or policy differs from the frozen run plan")
    path.write_text(frozen)
    print(json.dumps({"frozen_plan_sha256": hashlib.sha256(frozen.encode()).hexdigest(), "sources": names}), flush=True)
    manifests = {}
    for name, recipe in recipes.items():
        rows = read_jsonl(Path(recipe.source.path))
        if name in atlas_code.CONFIGS and not (args.output / name / "checks.jsonl").exists():
            print(json.dumps({"source": name, "stage": "grader_controls", "rows": len(rows)}), flush=True)
            recipe = checked_recipe(recipe, rows, args.check_workers)
        manifests[name] = run_pipeline(
            recipe, rows, output_path=args.output / name, limit=len(rows), reviewer=reviewer, policy=policy
        )
        print(json.dumps({"source": name, "dispositions": manifests[name]["dispositions"]}), flush=True)
    for filename in ("audit.parquet", "accepted.parquet"):
        pq.write_table(combined_table(args.output, names, filename), args.output / filename, compression="zstd")
    (args.output / "summary.json").write_text(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
