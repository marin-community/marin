# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-table consistency of the release, and the table of run-set memberships.

A physical training run has one observation id in every table that holds it: the 39 Llama 160M single-bucket deletions
are rows of `swarm_llama_160m_1p2b` and of `heldout`, 117 validation runs are rows of `heldout`, and the ladder's
3e18 rung copies validation and swarm rows. `consistency` checks that every copy of a run carries the same weights,
pool fractions and BPB, and that every content id hashes its own row.
"""

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import keys, paths

TOLERANCE = 1e-9
SHARED_PREFIXES = ("bpb__", "w__", "pf__")
SWARMS = ("swarm_qwen3_360m_1p6b", "swarm_llama_160m_1p2b", "swarm_llama_200m_6b")
# Configs whose every run forms one set, whichever of their tables and id columns names it.
RUN_GROUPS = ("starcoder", "domain_sweeps")


def consistency(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Every problem as `<table>: <observation_id>: <column> = <value> but <other> has <value>`; empty when clean.

    Two tables agree on a run's shared `bpb__`, `w__` and `pf__` columns within TOLERANCE wherever both have a value;
    a value that one table lacks is no disagreement (`heldout` lacks components that the swarm tables have).
    """
    return _id_problems(tables) + _value_problems(tables)


def _id_problems(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Each `mixture_id` must hash its row's weights, each `exposure_id` its weights and pool fractions."""
    problems = []
    for name, df in tables.items():
        buckets = [c.removeprefix("w__") for c in df.columns if c.startswith("w__")]
        if not buckets:
            continue
        weights = [dict(zip(buckets, row, strict=True)) for row in df[[f"w__{b}" for b in buckets]].to_numpy()]
        if "mixture_id" in df:
            for oid, recorded, w in zip(df.observation_id, df.mixture_id, weights, strict=True):
                expected = keys.mixture_id(w)
                if recorded != expected:
                    problems.append(f"{name}: {oid}: mixture_id = {recorded} but its weights hash to {expected}")
        if "exposure_id" in df:
            fractions = df[[f"pf__{b}" for b in buckets]].to_numpy()
            for oid, recorded, w, pf in zip(df.observation_id, df.exposure_id, weights, fractions, strict=True):
                expected = keys.exposure_id(w, dict(zip(buckets, pf, strict=True)), paths.PARTITION_REVISION)
                if recorded != expected:
                    problems.append(
                        f"{name}: {oid}: exposure_id = {recorded} but its weights and pool fractions hash to {expected}"
                    )
    return problems


def _value_problems(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Each value of a run against the first value of that run and column, in the order of `tables`."""
    parts = []
    for name, df in tables.items():
        columns = [c for c in df.columns if c.startswith(SHARED_PREFIXES)]
        if not columns:
            continue
        long = df[["observation_id", *columns]].melt(id_vars="observation_id", var_name="column").dropna()
        parts.append(long.assign(table=name, value=long.value.astype(float)))
    if not parts:
        return []
    values = pd.concat(parts, ignore_index=True)
    first = values.drop_duplicates(["observation_id", "column"])
    paired = values.merge(first, on=["observation_id", "column"], suffixes=("", "_first"), validate="many_to_one")
    bad = paired[(paired.value - paired.value_first).abs() > TOLERANCE]
    return [
        f"{r.table}: {r.observation_id}: {r.column} = {r.value} but {r.table_first} has {r.value_first}"
        for r in bad.itertuples(index=False)
    ]


def run_sets(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Set memberships (`observation_id, run_set`), one row per run and set.

    Sets: `fit_<swarm table>` (rows with `fit_inclusion`) and `proportional_repeats_<swarm>` (the swarm's proportional
    repeats, its only rows outside the fit), `heldout_<cohort>`, `validation/<launch>`, `ladder`, and one set for all
    the runs of the `starcoder` and `domain_sweeps` configs. A run belongs to every set of every table that holds it.
    """
    parts = []
    for name in SWARMS:
        if name in tables:
            t = tables[name]
            repeats = f"proportional_repeats_{name.removeprefix('swarm_')}"
            parts.append(_members(t.observation_id, np.where(t.fit_inclusion, f"fit_{name}", repeats)))
    if "heldout" in tables:
        heldout = tables["heldout"]
        parts.append(_members(heldout.observation_id, ("heldout_" + heldout.cohort).to_numpy()))
    if "validation_3e18" in tables:
        validation = tables["validation_3e18"]
        parts.append(_members(validation.observation_id, ("validation/" + validation.launch).to_numpy()))
    if "ladder" in tables:
        parts.append(_members(tables["ladder"].observation_id, "ladder"))
    for group in RUN_GROUPS:
        for name, t in tables.items():
            if name == group or name.startswith(f"{group}_"):
                ids = [c for c in t.columns if c == "observation_id" or c.endswith("_observation_id")]
                parts.extend(_members(t[c].dropna(), group) for c in ids)
    members = pd.concat(parts, ignore_index=True).drop_duplicates()
    return members.sort_values(["run_set", "observation_id"]).reset_index(drop=True)


def _members(observation_ids: pd.Series, run_set: np.ndarray | str) -> pd.DataFrame:
    """Rows of (id, set), positionally: `run_set` is one set for all ids or an array aligned with them."""
    return pd.DataFrame({"observation_id": observation_ids.to_numpy(), "run_set": run_set})
