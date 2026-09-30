# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Gym source rows to tasks with private verifier inputs."""

from collections.abc import Iterator, Mapping
from typing import Any

import fsspec
import pyarrow.parquet as pq

from taskcompendium.chat import chat_input
from taskcompendium.environment import EnvironmentKind, EnvironmentSpec, ExternalVerifierSpec
from taskcompendium.models import AnswerType, Source, EnvironmentRequirements, TaskSpec, VerifierKind, VerifierSpec
from taskcompendium.parquet import PARQUET_BATCH_SIZE

GYM_INTERACTION = "skyrl_gym"


def gym_task(
    prompt: list[dict[str, Any]],
    environment: str,
    extras: dict[str, Any],
    config: dict[str, Any],
    source: Source,
) -> TaskSpec:
    """Convert a source row to a self-contained task with private grading inputs."""
    verifier = ExternalVerifierSpec(name=environment, parameters={"extras": extras, "config": config})
    return TaskSpec(
        id=f"{source.dataset}:{source.row}",
        context=chat_input(prompt),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=VerifierSpec(kind=VerifierKind.EXTERNAL, parameters_json=verifier.model_dump_json()),
        environment=EnvironmentSpec(kind=EnvironmentKind.NULL, interaction=GYM_INTERACTION),
        source=source,
        metadata={"teacher_route": extras["teacher_route"]} if "teacher_route" in extras else {},
    )


def read_gym_tasks(
    path: str,
    *,
    dataset: str,
    revision: str,
    environment_configs: Mapping[str, dict[str, Any]],
) -> Iterator[TaskSpec]:
    """Convert Parquet rows in bounded batches with explicit source provenance."""
    with fsspec.open(path, "rb") as source:
        parquet = pq.ParquetFile(source)
        index = 0
        for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_SIZE):
            for row in batch.to_pylist():
                prompt = row.pop("prompt")
                environment = row.pop("env_class")
                yield gym_task(
                    prompt,
                    environment,
                    row,
                    environment_configs[environment],
                    Source(dataset=dataset, revision=revision, row=str(index), importer_revision="1"),
                )
                index += 1
