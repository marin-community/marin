# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic procedural parquet for the CatCountCanary environment."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from rigging.filesystem.storage_path import prefix_join
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

TRAIN_FILENAME = "train.parquet"
VALIDATION_FILENAME = "validation.parquet"
DEFAULT_TRAIN_NS = (1, 2, 4, 7, 10, 20)
HELDOUT_NS = (3, 5, 13, 16)


@dataclass(frozen=True)
class CatCountDataConfig:
    output_path: str
    train_ns: tuple[int, ...]
    train_rows: int
    seed: int

    def __post_init__(self) -> None:
        if not self.train_ns or len(set(self.train_ns)) != len(self.train_ns) or min(self.train_ns) <= 0:
            raise ValueError("train_ns must contain distinct positive integers")
        if set(self.train_ns) & set(HELDOUT_NS):
            raise ValueError("held-out N must not appear in the training split")
        if self.train_rows <= 0:
            raise ValueError("train_rows must be positive")


def cat_count_record(n: int, split: str, index: int) -> dict[str, object]:
    return {
        "data_source": f"cat_count_n{n}",
        "prompt": [
            {
                "role": "user",
                "content": f"Reply with the word cat exactly {n} times, separated by single spaces. Nothing else.",
            }
        ],
        "env_class": "cat_count",
        "reward_spec": {"method": "rule", "ground_truth": n},
        "extra_info": {"n": n, "split": split, "index": index},
    }


def cat_count_rows(config: CatCountDataConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(config.seed)
    schedule: list[int] = []
    while len(schedule) < config.train_rows:
        cycle = list(config.train_ns)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    train = [cat_count_record(n, "train", index) for index, n in enumerate(schedule[: config.train_rows])]
    eval_ns = (*config.train_ns, *HELDOUT_NS)
    validation = [cat_count_record(n, "validation", index) for index, n in enumerate(eval_ns)]
    return train, validation


def write_cat_count_parquet(config: CatCountDataConfig) -> None:
    train, validation = cat_count_rows(config)
    for filename, records in ((TRAIN_FILENAME, train), (VALIDATION_FILENAME, validation)):
        destination = prefix_join(config.output_path, filename)
        write_parquet_file(records, destination)
        logger.info("Wrote %d CatCountCanary rows to %s", len(records), destination)


def cat_count_data_step(
    name: str, version: str, *, train_ns: tuple[int, ...], train_rows: int, seed: int
) -> ArtifactStep[Artifact]:
    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=remote(write_cat_count_parquet, resources=ResourceConfig.with_cpu(cpu=2, ram="8g", disk="8g")),
        build_config=lambda ctx: CatCountDataConfig(
            output_path=ctx.output_path,
            train_ns=train_ns,
            train_rows=train_rows,
            seed=seed,
        ),
    )
