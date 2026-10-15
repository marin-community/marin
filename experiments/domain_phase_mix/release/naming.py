# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public bucket and task ids, their paper names, and maps from every internal spelling."""

import re
from functools import cache

import pandas as pd

from experiments.domain_phase_mix.release import paths, tables

HERE = paths.REPO / "experiments/domain_phase_mix/release"
BUCKET_NAMES = HERE / "bucket_names.csv"
TASK_METADATA = HERE / "task_metadata.csv"
OBJECTIVE_WEIGHTS = paths.MIXSEL / "data/objectives.csv"


@cache
def bucket_table() -> pd.DataFrame:
    return tables.read_csv(paths.require(BUCKET_NAMES))


@cache
def task_table() -> pd.DataFrame:
    return tables.read_csv(paths.require(TASK_METADATA))


BUCKETS: list[str] = list(bucket_table().bucket_id)
TASKS: list[str] = list(task_table().task_id)
_BUCKET_BY_INTERNAL = dict(zip(bucket_table().internal_name, bucket_table().bucket_id, strict=True))
# objectives.csv spellings, including the bare MMLU bucket names (`mmlu_stem`) that no pattern below covers.
_TASK_BY_INTERNAL = dict(zip(task_table().internal_metric, task_table().task_id, strict=True))
_TASK_PATTERNS = [
    (re.compile(r"^(?:olmo_base_eval/easy_bpb|olmo_base_easy/table9|table9)/(?P<t>[^/]+)/bpb$"), "{t}"),
    (re.compile(r"^eval/uncheatable_eval/(?P<t>[^/]+)/bpb$"), "uncheatable/{t}"),
    (re.compile(r"^uncheatable_(?P<t>.+)_bpb$"), "uncheatable/{t}"),
]


# Public names of the objective a run's mixture was proposed for: the two objectives, the subsets of their tasks that
# some proposals optimized, `none` for the scaling baselines (proportional, UniMax-8), and the held-out registry's labels
# of its baselines (`baseline`), of designs built for both objectives and of runs whose names name no objective.
OBJECTIVES = frozenset(
    {
        "uncheatable",
        "obe_easy",
        "uncheatable_worsened",
        "obe_easy_qa15",
        "obe_easy_qa19",
        "obe_easy_snr2",
        "obe_easy_snr5",
        "obe_easy_snr8",
        "none",
        "baseline",
        "uncheatable_and_obe_easy",
        "other",
    }
)
# Sources name OlmoBaseEval Easy `table9` (and its subsets `table9_<subset>`); some name an objective by its metric.
_OBJECTIVE_ALIASES = {
    "table9_macro_bpb": "obe_easy",
    "uncheatable_bpb": "uncheatable",
    "uncheatable_and_table9": "uncheatable_and_obe_easy",
}


def bucket_public(internal: str) -> str:
    return _BUCKET_BY_INTERNAL[internal]


def task_public(internal_metric: str) -> str:
    if internal_metric in _TASK_BY_INTERNAL:
        return _TASK_BY_INTERNAL[internal_metric]
    for pattern, template in _TASK_PATTERNS:
        m = pattern.match(internal_metric)
        if m:
            public = template.format(**m.groupdict())
            if public in TASKS:
                return public
    raise KeyError(f"unknown task metric {internal_metric!r}")


def objective_public(internal: str) -> str:
    """The public name of an objective from any source's spelling; raises ValueError for an unknown one."""
    public = _OBJECTIVE_ALIASES.get(internal) or re.sub(r"^table9", "obe_easy", internal)
    if public not in OBJECTIVES:
        raise ValueError(f"unknown objective {internal!r}")
    return public


@cache
def task_weights() -> dict[str, dict[str, float]]:
    objectives = tables.read_csv(paths.require(OBJECTIVE_WEIGHTS))
    unch = objectives[objectives.objective == "uncheatable"]
    t9 = objectives[objectives.objective == "table9"]
    return {
        "uncheatable_fixed": {task_public(c): float(w) for c, w in zip(unch.component, unch.weight, strict=True)},
        "obe_easy": {task_public(c): float(w) for c, w in zip(t9.component, t9.weight, strict=True)},
    }
