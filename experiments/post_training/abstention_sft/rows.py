# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn labelled questions into chat rows: the student's own answer, or a written abstain reply.

Answer rows reuse the student's first correct sample, reasoning included, so they keep its style and
teach no new fact. Abstain rows carry the written reply with no reasoning. Held-out questions are
written to a separate file and never trained on, so a trained model can be checked on them later.
"""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from openai import OpenAI
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from zephyr.readers import load_parquet

from experiments.post_training.abstention_sft.abstain import abstain_reply
from experiments.post_training.abstention_sft.knowledge import Label
from experiments.post_training.abstention_sft.probe import LABELS_FILENAME, SAMPLES_FILENAME
from experiments.post_training.curriculum_sft.generation import (
    MANIFEST_FILENAME,
    PROBLEMS_FILENAME,
    REASONING_CHAT_SCHEMA,
    reasoning_chat_row,
    write_table,
)
from experiments.post_training.curriculum_sft.self_distill import CONCURRENT_REQUESTS, split_thinking

TRAIN_FILENAME = "train/part-00000-of-00001.parquet"
HELD_OUT_FILENAME = "held_out/part-00000-of-00001.parquet"


def chat_rows(
    problems: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    abstain_replies: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (train, held_out) chat rows; dropped questions get no row."""
    by_id = {problem["request_id"]: problem for problem in problems}
    first_correct: dict[str, dict[str, Any]] = {}
    for sample in sorted(samples, key=lambda sample: sample["sample"]):
        if sample["correct"]:
            first_correct.setdefault(sample["request_id"], sample)
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for label in labels:
        request_id = label["request_id"]
        problem = by_id[request_id]["problem"]
        if label["label"] == Label.ANSWER:
            split = split_thinking(first_correct[request_id]["output"])
            assert split is not None, f"a correct sample has a closed think block: {request_id}"
            reasoning, content = split
            row = reasoning_chat_row(request_id, problem, content, reasoning)
        elif label["label"] == Label.ABSTAIN:
            row = reasoning_chat_row(request_id, problem, abstain_replies[request_id], "")
        else:
            continue
        (held if label["held_out"] else train).append(row)
    return train, held


@dataclass(frozen=True)
class RowsConfig:
    problems_path: str
    probe_path: str
    output_path: str
    writer_endpoint: str
    writer_api_key_env: str
    writer_model: str


def build_rows(
    problems: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    client: OpenAI,
    writer_model: str,
    output_path: str,
) -> None:
    """Write abstain replies, then the train and held-out chat rows and a manifest of label counts."""
    by_id = {problem["request_id"]: problem for problem in problems}
    abstaining = [label["request_id"] for label in labels if label["label"] == Label.ABSTAIN]
    with ThreadPoolExecutor(CONCURRENT_REQUESTS) as pool:
        replies = list(
            pool.map(
                lambda request_id: abstain_reply(
                    by_id[request_id]["problem"], by_id[request_id]["answer"], client, writer_model
                ),
                abstaining,
            )
        )
    train, held = chat_rows(problems, samples, labels, dict(zip(abstaining, replies, strict=True)))
    output = StoragePath(output_path)
    output.mkdirs()
    write_table(output, TRAIN_FILENAME, train, REASONING_CHAT_SCHEMA)
    write_table(output, HELD_OUT_FILENAME, held, REASONING_CHAT_SCHEMA)
    per_domain = Counter((by_id[label["request_id"]]["domain"], label["label"]) for label in labels)
    manifest = {
        "writer_model": writer_model,
        "train_rows": len(train),
        "held_out_rows": len(held),
        "labels_by_domain": {f"{domain}/{label}": count for (domain, label), count in sorted(per_domain.items())},
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")


def _write_rows(config: RowsConfig) -> Artifact:
    build_rows(
        list(load_parquet(str(StoragePath(config.problems_path) / PROBLEMS_FILENAME))),
        list(load_parquet(str(StoragePath(config.probe_path) / SAMPLES_FILENAME))),
        list(load_parquet(str(StoragePath(config.probe_path) / LABELS_FILENAME))),
        OpenAI(base_url=config.writer_endpoint, api_key=require_secret(config.writer_api_key_env)),
        config.writer_model,
        config.output_path,
    )
    return Artifact(path=config.output_path)


def rows_step(
    problems: ArtifactStep[Artifact],
    probe: ArtifactStep[Artifact],
    *,
    writer_endpoint: str,
    writer_api_key_env: str,
    writer_model: str,
    version: str,
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> RowsConfig:
        return RowsConfig(
            problems_path=ctx.artifact_path(problems),
            probe_path=ctx.artifact_path(probe),
            output_path=ctx.output_path,
            writer_endpoint=writer_endpoint,
            writer_api_key_env=writer_api_key_env,
            writer_model=writer_model,
        )

    return ArtifactStep(
        name=user_owned_name("documents/abstention-sft/rows"),
        version=version,
        artifact_type=Artifact,
        run=_write_rows,
        build_config=build_config,
        deps=(problems, probe),
    )
