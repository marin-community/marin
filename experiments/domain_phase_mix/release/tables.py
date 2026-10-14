# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Column conventions shared by every run table."""

import pandas as pd


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


def add_aggregates(df: pd.DataFrame, task_weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for name, weights in task_weights.items():
        cols = {f"bpb__{t}": w for t, w in weights.items()}
        out[f"bpb__{name}"] = sum(out[c] * w for c, w in cols.items())
    return out


def weight_columns(df: pd.DataFrame, prefix: str = "w__") -> list[str]:
    return [c for c in df if c.startswith(prefix)]
