# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared recipe metadata and generated-instance contracts."""

import csv
import io
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from experiments.post_training.bio_tasks.contract import Contract


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class Instance:
    instruction: str
    inputs: dict[str, str]
    contract: Contract
    mutations: dict[str, list[dict]]


@dataclass(frozen=True)
class Recipe:
    id: str
    version: str
    difficulty: Difficulty
    skills: tuple[str, ...]
    formats: tuple[str, ...]
    sources: tuple[str, ...]
    generate: Callable[[int], Instance]
    domain: str = ""
    repositories: tuple[str, ...] = ()


def csv_text(rows: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
