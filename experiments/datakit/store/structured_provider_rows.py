# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind original structured provider rows to their production text transforms."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from marin.datakit.download import (
    molmo2_cap,
    numinamath_tir,
    numinamath_v1_5,
    rollout_transforms,
    superior_reasoning,
    swe_rebench_openhands,
)


@dataclass(frozen=True)
class OriginalRowAdapter:
    transform: str
    columns: tuple[str, ...]
    row_to_docs: Callable[[dict], list[dict]]
    modules: tuple[ModuleType, ...]

    def digest(self) -> str:
        """Pin the projection, production helper, and shared text renderer."""
        components = []
        for module in self.modules:
            assert module.__file__ is not None
            components.append((module.__name__, hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()))
        binding = [self.transform, self.columns, self.row_to_docs.__qualname__, components]
        return hashlib.sha256(json.dumps(binding, separators=(",", ":")).encode()).hexdigest()


ADAPTERS = {
    "molmo2-cap": OriginalRowAdapter(
        "molmo2_cap",
        ("merged_caption", "frame_timestamps", "frame_captions"),
        molmo2_cap.row_to_doc,
        (molmo2_cap, rollout_transforms),
    ),
    "numinamath-1.5": OriginalRowAdapter(
        "numinamath_v1_5",
        (
            "problem",
            "solution",
            "problem_is_valid",
            "solution_is_valid",
            "source",
            "answer",
            "problem_type",
            "question_type",
            "synthetic",
        ),
        numinamath_v1_5.row_to_doc,
        (numinamath_v1_5,),
    ),
    "numinamath-tir": OriginalRowAdapter(
        "numinamath_tir", ("messages",), numinamath_tir.row_to_doc, (numinamath_tir, rollout_transforms)
    ),
    "superior-reasoning": OriginalRowAdapter(
        "superior_reasoning",
        ("input", "output"),
        superior_reasoning.row_to_doc,
        (superior_reasoning, rollout_transforms),
    ),
    "swe-rebench-openhands": OriginalRowAdapter(
        "swe_openhands",
        ("trajectory", "resolved"),
        swe_rebench_openhands.row_to_doc,
        (swe_rebench_openhands, rollout_transforms),
    ),
}


def source_adapter(name: str) -> OriginalRowAdapter:
    return ADAPTERS[name]


def swe_openhands_provider_paths(paths: list[str]) -> list[str]:
    """Match the frozen producer's Parquet-only input glob, excluding tools metadata."""
    selected = sorted(path for path in paths if path.endswith(".parquet"))
    if not selected:
        raise ValueError("SWE OpenHands original provider inventory contains no Parquet inputs")
    return selected
