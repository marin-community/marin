# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Column conventions shared by every run table, and the CSV reader every exporter uses."""

from pathlib import Path

import pandas as pd

# A recorded aggregate that its source scored with the fixed rule agrees with the fixed-rule sum of the run's components
# to this; the sources that did so differ from that sum by at most 1.6e-7 BPB.
FIXED_RULE_TOLERANCE = 2e-6


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """`pd.read_csv` with exact float parsing, so exported values are bit-identical to their sources.

    pandas' default float parser misreads some 17-digit values by one ulp.
    """
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def order_columns(df: pd.DataFrame, buckets: list[str], tasks: list[str]) -> pd.DataFrame:
    w = [f"w__{b}" for b in buckets if f"w__{b}" in df]
    w0 = [f"w0__{b}" for b in buckets if f"w0__{b}" in df]
    w1 = [f"w1__{b}" for b in buckets if f"w1__{b}" in df]
    pf = [f"pf__{b}" for b in buckets if f"pf__{b}" in df]
    bpb = [f"bpb__{t}" for t in tasks if f"bpb__{t}" in df]
    agg = [c for c in df if c.startswith("bpb__") and c not in bpb]
    acc = [c for c in df if c.startswith("acc__")]
    meta = [c for c in df if c not in set(w + w0 + w1 + pf + bpb + agg + acc)]
    return df[meta + w + w0 + w1 + pf + bpb + agg + acc]


def add_aggregates(
    df: pd.DataFrame,
    task_weights: dict[str, dict[str, float]],
    recorded: pd.DataFrame | None = None,
    fixed_rule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`df` with `bpb__<objective>` for each objective of `task_weights`, which maps objectives to their task weights.

    An aggregate is the weighted sum of the objective's components wherever the row has all of them. Elsewhere it is
    the row's recorded aggregate, `recorded[objective]`, or null where the source recorded none. `fixed_rule[objective]`
    says whether the source scored that value with the same weights. A recorded value under other weights is never
    used: a row that would need one raises ValueError, and so does a fixed-rule value that differs from the row's
    components by more than FIXED_RULE_TOLERANCE. `recorded` and `fixed_rule` are aligned with `df` and hold the same
    objectives.
    """
    out = df.copy()
    for objective, weights in task_weights.items():
        computed = sum(out[f"bpb__{t}"] * w for t, w in weights.items())  # null wherever a component is
        if recorded is not None and objective in recorded:
            computed = _fill_recorded(computed, recorded[objective], fixed_rule[objective], objective)
        out[f"bpb__{objective}"] = computed
    return out


def _fill_recorded(computed: pd.Series, recorded: pd.Series, fixed_rule: pd.Series, objective: str) -> pd.Series:
    """`computed`, with the recorded fixed-rule aggregate wherever a component is missing."""
    checked = computed.notna() & recorded.notna() & fixed_rule
    off = (computed[checked] - recorded[checked]).abs() > FIXED_RULE_TOLERANCE
    if off.any():
        raise ValueError(
            f"{off.sum()} recorded {objective} aggregates said to use the fixed rule differ from their components by "
            f"more than {FIXED_RULE_TOLERANCE}"
        )
    needed = computed.isna() & recorded.notna()
    other_weights = needed & ~fixed_rule
    if other_weights.any():
        raise ValueError(
            f"{other_weights.sum()} rows lack {objective} components, and their sources recorded the aggregate under "
            "other weights than the fixed rule"
        )
    return computed.mask(needed, recorded)
