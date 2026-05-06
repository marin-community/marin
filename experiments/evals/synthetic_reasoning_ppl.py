# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF-backed synthetic reasoning PPL validation slices for issue #4148."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from enum import StrEnum

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

EPIC_5005 = 5005
SYNTHETIC_REASONING_ISSUE = 4148
ISSUE_5052 = 5052
SYNTHETIC_REASONING_HF_DATASET_ID = "marin-community/synth-bootstrap-trial"
SYNTHETIC_REASONING_SOURCE_COMMIT = "c4a59c3e1"
DEV_SEED_BASE = 1 << 30
EXAMPLES_PER_SUBCORPUS = 1000

STEPMATH_TASKS: tuple[str, ...] = (
    "arithmetic",
    "algebra_linear_equation",
)

STEPMATH_ICL_TASKS: tuple[tuple[str, int], ...] = (
    ("arithmetic_5shot_icl", 5),
    ("algebra_linear_equation_3shot_icl", 3),
)

NATIVE_TASKS: tuple[str, ...] = (
    "bfs_shortest_path",
    "binary_search",
    "coin_change_dp",
    "connected_components",
    "dijkstra_shortest_path",
    "edit_distance_dp",
    "euclid_gcd",
    "floyd_warshall_apsp",
    "insertion_sort",
    "interval_scheduling",
    "kmp_string_search",
    "knapsack_01_dp",
    "lcs_dp",
    "lis_dp",
    "n_queens_backtracking",
    "parentheses_balance",
    "prim_mst",
    "propositional_entailment",
    "topological_sort",
    "union_find_connectivity",
)

NATIVE_ICL_TASKS: tuple[tuple[str, int], ...] = (
    ("binary_search_5shot_icl", 5),
    ("euclid_gcd_5shot_icl", 5),
    ("parentheses_balance_5shot_icl", 5),
    ("edit_distance_dp_1shot_icl", 1),
    ("interval_scheduling_1shot_icl", 1),
    ("kmp_string_search_1shot_icl", 1),
    ("knapsack_01_dp_1shot_icl", 1),
    ("lcs_dp_1shot_icl", 1),
    ("n_queens_backtracking_1shot_icl", 1),
    ("union_find_connectivity_1shot_icl", 1),
)

CLRS_STYLE_TASKS: tuple[str, ...] = (
    "clrs_bfs",
    "clrs_binary_search",
    "clrs_connected_components",
    "clrs_dijkstra",
    "clrs_floyd_warshall",
    "clrs_insertion_sort",
    "clrs_lcs_length",
    "clrs_lis",
    "clrs_mst_prim",
    "clrs_topological_sort",
)


class SyntheticReasoningFamily(StrEnum):
    STEPMATH = "stepmath"
    NATIVE = "native"
    CLRS_STYLE = "clrs_style"


@dataclass(frozen=True)
class SyntheticReasoningPplSlice:
    family: SyntheticReasoningFamily
    task_name: str
    hf_config_name: str
    icl_shots: int = 0

    @property
    def registry_key(self) -> str:
        return posixpath.join("synthetic_reasoning_ppl", self.family.value, self.task_name)

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_reasoning_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{SYNTHETIC_REASONING_ISSUE}",
            f"issue:{ISSUE_5052}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"seed:{DEV_SEED_BASE}",
            f"examples:{EXAMPLES_PER_SUBCORPUS}",
            f"source_commit:{SYNTHETIC_REASONING_SOURCE_COMMIT}",
            *((f"icl_shots:{self.icl_shots}",) if self.icl_shots else ()),
        )

    def to_raw_text_dataset(self, *, hf_dataset_id: str) -> RawTextEvaluationDataset:
        return raw_text_dataset(
            HfDatasetSpec(id=hf_dataset_id, name=self.hf_config_name),
            text_key="text",
            split="validation",
            tags=self.tags,
        )


def _slice(
    *,
    family: SyntheticReasoningFamily,
    task_name: str,
    icl_shots: int = 0,
) -> SyntheticReasoningPplSlice:
    return SyntheticReasoningPplSlice(
        family=family,
        task_name=task_name,
        hf_config_name=f"{family.value}_{task_name}",
        icl_shots=icl_shots,
    )


SYNTHETIC_REASONING_PPL_SLICES: tuple[SyntheticReasoningPplSlice, ...] = (
    *(_slice(family=SyntheticReasoningFamily.STEPMATH, task_name=task) for task in STEPMATH_TASKS),
    *(
        _slice(family=SyntheticReasoningFamily.STEPMATH, task_name=task, icl_shots=icl_shots)
        for task, icl_shots in STEPMATH_ICL_TASKS
    ),
    *(_slice(family=SyntheticReasoningFamily.NATIVE, task_name=task) for task in NATIVE_TASKS),
    *(
        _slice(family=SyntheticReasoningFamily.NATIVE, task_name=task, icl_shots=icl_shots)
        for task, icl_shots in NATIVE_ICL_TASKS
    ),
    *(_slice(family=SyntheticReasoningFamily.CLRS_STYLE, task_name=task) for task in CLRS_STYLE_TASKS),
)


def synthetic_reasoning_raw_validation_sets(
    *, hf_dataset_id: str = SYNTHETIC_REASONING_HF_DATASET_ID
) -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(hf_dataset_id=hf_dataset_id)
        for slice_ in SYNTHETIC_REASONING_PPL_SLICES
    }
