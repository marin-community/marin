# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public bucket and task ids, their paper names, and maps from every internal spelling."""

import re
from functools import cache

import pandas as pd

from experiments.domain_phase_mix.release import paths

HERE = paths.REPO / "experiments/domain_phase_mix/release"


@cache
def bucket_table() -> pd.DataFrame:
    return pd.read_csv(HERE / "bucket_names.csv")


@cache
def task_table() -> pd.DataFrame:
    return pd.read_csv(HERE / "task_metadata.csv")


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


@cache
def task_weights() -> dict[str, dict[str, float]]:
    objectives = pd.read_csv(paths.require(paths.MIXSEL / "data/objectives.csv"))
    unch = objectives[objectives.objective == "uncheatable"]
    t9 = objectives[objectives.objective == "table9"]
    # The bundled Uncheatable byte shares sum to 1 + 2.1e-8; normalizing moves the aggregate by about 2e-8 BPB.
    unch_weights = unch.weight / unch.weight.sum()
    t9_weights = t9.weight / t9.weight.sum()
    return {
        "uncheatable_fixed": {task_public(c): float(w) for c, w in zip(unch.component, unch_weights, strict=True)},
        "obe_easy": {task_public(c): float(w) for c, w in zip(t9.component, t9_weights, strict=True)},
    }
