# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Will the mechanism analyzer's inventory assertions pass, BEFORE we spend the compute (ATOM-032)?

The analyzer refuses to report anything unless the observed rows exactly match the frozen manifest:
`_assert_exact_manifest_inventory` compares (row_id, target) sets, and the per-estimand paths additionally
require 24 m100a seeds, 24 full seeds, 8 m100b seeds, 24 H3 seed pairs, and 16 H5 paired seeds per target.
Every one of those raises AFTER the 932 rows have been computed.

That is the wrong time to find out. This reproduces each assertion from the manifest alone, by predicting
what the runtime WOULD emit for each row rather than by reading results:

  a row contributes (row_id, target) for every target in its target list, and only if the primary source
  contrast is constructible -- `_contrast_pairs` emits `starcoder_excluded_global__minus__nemotron_aggregate`
  only when BOTH sources appear in that row's source list, so a row missing either one silently produces no
  primary-contrast statistic and shows up as a missing inventory key.

This is the check that the original probe failure would have needed in reverse: it cannot tell you the
statistic is the right one, but it can tell you the rows you are about to pay for will actually satisfy
the analysis.

Usage: ``uv run python ...``
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pandas as pd  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    freeze_starcoder_wsd80_gradient_mechanism_repair_20260818 as freeze,
)

GLOBAL_STARCODER = freeze.GLOBAL_STARCODER
NEMOTRON = freeze.NEMOTRON
SUPPORT_STARCODER = freeze.SUPPORT_STARCODER
CONTRACT = freeze.ANALYSIS_CONTRACT["estimands"]


def manifest() -> pd.DataFrame:
    return pd.read_csv(freeze.FULL_MANIFEST_PATH)


def has_primary_contrast(row) -> bool:
    """`_contrast_pairs` emits the primary contrast only when both of its sources are present."""
    sources = set(json.loads(row["source_distribution_ids_json"]))
    return {GLOBAL_STARCODER, NEMOTRON} <= sources


def has_support_contrast(row) -> bool:
    sources = set(json.loads(row["source_distribution_ids_json"]))
    return {SUPPORT_STARCODER, GLOBAL_STARCODER} <= sources


def check_alignment_inventory(frame: pd.DataFrame, roles: set[str], labels: set[str], name: str) -> list[str]:
    """Reproduce `_assert_exact_manifest_inventory` for an alignment path."""
    selected = frame[frame["analysis_role"].isin(roles) & frame["checkpoint_label"].isin(labels)]
    problems = []
    missing_contrast = [row["row_id"] for _, row in selected.iterrows() if not has_primary_contrast(row)]
    if missing_contrast:
        problems.append(
            f"{name}: {len(missing_contrast)} rows cannot build the primary contrast "
            f"(first: {missing_contrast[:3]}) -- the analyzer would report them as missing inventory"
        )
    targets_per_row = {len(json.loads(row["target_distribution_ids_json"])) for _, row in selected.iterrows()}
    if len(targets_per_row) > 1:
        problems.append(f"{name}: rows carry differing target counts {sorted(targets_per_row)}")
    return problems


def main() -> None:
    frame = manifest()
    print(f"frozen full manifest: {len(frame)} rows, {frame['analysis_role'].nunique()} roles")
    print(f"stage split: {dict(frame['launch_stage'].value_counts().sort_index())}\n")

    problems: list[str] = []

    h2_labels = {label for states in CONTRACT["h2"]["states"].values() for label in states}
    h2_roles = {"h2_primary", "h3_full_support_pair", "h3_second_pool_sensitivity"}
    problems += check_alignment_inventory(frame, h2_roles, h2_labels, "H2/H3 alignment")

    h5_labels = {label for states in CONTRACT["h5_profile"]["periods"].values() for label in states}
    problems += check_alignment_inventory(frame, {"h5_preregistered_profile"}, h5_labels, "H5 profile alignment")

    h1_labels = set(CONTRACT["h1"]["states"])
    problems += check_alignment_inventory(frame, {"h1_trajectory_extension"}, h1_labels, "H1")

    # Seed-count assertions, each of which raises inside the analyzer after the compute.
    h2 = frame[frame["analysis_role"].isin(h2_roles) & frame["checkpoint_label"].isin(h2_labels)]
    observed = h2.groupby("support_id")["training_seed"].nunique().to_dict()
    expected = {"m100a": 24, "full": 24, "m100b": 8}
    print(f"H2/H3 seeds per support: observed={observed} expected={expected}")
    if observed != expected:
        problems.append(f"H2/H3 seed inventory will fail: {observed} != {expected}")

    pairs = h2[h2["support_id"].isin(("m100a", "full"))].groupby("training_seed")["support_id"].nunique()
    complete_pairs = int((pairs == 2).sum())
    print(f"H3 complete m100a/full seed pairs: {complete_pairs} (analyzer requires 24)")
    if complete_pairs != 24:
        problems.append(f"H3 requires 24 seed pairs, manifest supports {complete_pairs}")

    h5 = frame[frame["analysis_role"].eq("h5_preregistered_profile") & frame["checkpoint_label"].isin(h5_labels)]
    policies = set(CONTRACT["h5_profile"]["policies"])
    paired = h5[h5["policy_role"].isin(policies)].groupby("training_seed")["policy_role"].nunique()
    complete = int((paired == 2).sum())
    print(f"H5 complete beta0.60/beta0.85 seed pairs: {complete} (analyzer requires 16)")
    if complete != 16:
        problems.append(f"H5 requires 16 paired seeds, manifest supports {complete}")
    unexpected = set(h5["policy_role"]) - policies
    if unexpected:
        print(f"  note: H5 rows also carry policy roles {sorted(unexpected)}, which the analyzer filters out")

    # Every period must be populated for every seed, or the pivot leaves NaN and the analyzer raises.
    for name, roles, labels, contract_key, index in (
        ("H2", h2_roles, h2_labels, "h2", ["training_seed", "support_id"]),
        ("H5", {"h5_preregistered_profile"}, h5_labels, "h5_profile", ["training_seed", "policy_role"]),
    ):
        periods = CONTRACT[contract_key]["states" if contract_key == "h2" else "periods"]
        by_state = {state: period for period, states in periods.items() for state in states}
        subset = frame[frame["analysis_role"].isin(roles) & frame["checkpoint_label"].isin(labels)].copy()
        subset["period"] = subset["checkpoint_label"].map(by_state)
        counts = subset.groupby([*index, "period"]).size().unstack("period")
        incomplete = counts[counts.isna().any(axis=1)]
        print(f"{name} cells missing a period: {len(incomplete)} of {len(counts)}")
        if len(incomplete):
            problems.append(f"{name} period inventory incomplete for {len(incomplete)} cells")

    print()
    if problems:
        print("PREFLIGHT FAILED — these would raise AFTER the compute:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("PREFLIGHT PASSED: every analyzer inventory assertion is satisfiable by the frozen manifest.")


if __name__ == "__main__":
    main()
