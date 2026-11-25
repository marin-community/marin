# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron abstention questions as self-distillation problems, with every AbstentionBench question removed.

Only the bare question is kept: Nemotron's system prompt and its ``\\boxed{[IDK]}`` answer format are RL
environment conventions, not SFT targets.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from huggingface_hub import HfApi, hf_hub_download
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_parquet

from experiments.post_training.curriculum_sft.generation import MANIFEST_FILENAME, PROBLEMS_FILENAME, write_table

NEMOTRON_DATASET = "nvidia/Nemotron-RL-QA-Abstention-v1"
NEMOTRON_REVISION = "7c2ead1dcfd549874c106c0da8a24554cc4dc746"
NEMOTRON_FILE = "data/train.jsonl"
ABSTENTIONBENCH_DATASET = "ymerkli/AbstentionBench"
ABSTENTIONBENCH_REVISION = "693e80c53a7f6aa19db815561b7dc2b3412a02a7"
# Nemotron rows without a domain are its HotpotQA questions.
HOTPOTQA_DOMAIN = "hotpotqa"
PROBLEMS_VERSION = "2026.11.25"

PROBLEM_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("problem", pa.string(), nullable=False),
        pa.field("answer", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("accepted", pa.bool_(), nullable=False),
    ]
)

_NON_WORD = re.compile(r"\W+")


def normalized_question(text: str) -> str:
    return _NON_WORD.sub(" ", text.lower()).strip()


def nemotron_problems(rows: Iterable[dict[str, Any]], eval_questions: set[str]) -> list[dict[str, Any]]:
    """Keep each Nemotron question whose normalized text is not an evaluation question."""
    return [
        {
            "request_id": row["id"],
            "problem": row["question"],
            "answer": row["answer"],
            "domain": row.get("domain") or HOTPOTQA_DOMAIN,
            "accepted": True,
        }
        for row in rows
        if normalized_question(row["question"]) not in eval_questions
    ]


def _abstentionbench_questions() -> set[str]:
    """Every question in every Parquet split of the pinned AbstentionBench snapshot."""
    files = HfApi().list_repo_files(ABSTENTIONBENCH_DATASET, repo_type="dataset", revision=ABSTENTIONBENCH_REVISION)
    return {
        normalized_question(row["question"])
        for file in files
        if file.startswith("data/") and file.endswith(".parquet")
        for row in load_parquet(
            hf_hub_download(ABSTENTIONBENCH_DATASET, file, repo_type="dataset", revision=ABSTENTIONBENCH_REVISION)
        )
    }


@dataclass(frozen=True)
class ProblemsConfig:
    output_path: str


def nemotron_rows() -> list[dict[str, Any]]:
    path = hf_hub_download(NEMOTRON_DATASET, NEMOTRON_FILE, repo_type="dataset", revision=NEMOTRON_REVISION)
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def load_problems() -> list[dict[str, Any]]:
    """The pinned Nemotron questions without any pinned AbstentionBench question."""
    return nemotron_problems(nemotron_rows(), _abstentionbench_questions())


def _write_problems(config: ProblemsConfig) -> Artifact:
    rows = nemotron_rows()
    problems = nemotron_problems(rows, _abstentionbench_questions())
    output = StoragePath(config.output_path)
    output.mkdirs()
    write_table(output, PROBLEMS_FILENAME, problems, PROBLEM_SCHEMA)
    manifest = {
        "source": f"{NEMOTRON_DATASET}@{NEMOTRON_REVISION}",
        "decontaminated_against": f"{ABSTENTIONBENCH_DATASET}@{ABSTENTIONBENCH_REVISION}",
        "rows": len(rows),
        "kept": len(problems),
        "dropped_overlap": len(rows) - len(problems),
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return Artifact(path=config.output_path)


def problems_step() -> ArtifactStep[Artifact]:
    return ArtifactStep(
        name=user_owned_name("documents/abstention-sft/problems"),
        version=PROBLEMS_VERSION,
        artifact_type=Artifact,
        run=_write_problems,
        build_config=lambda ctx: ProblemsConfig(output_path=ctx.output_path),
    )
