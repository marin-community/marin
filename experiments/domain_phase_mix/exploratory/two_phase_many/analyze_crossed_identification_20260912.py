# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Describe crossed-prefix interaction and noise using frozen local BRW inputs."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
from prepare_fixed_checkpoint_branch_20260907 import load_inputs, sha256
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "reference_outputs/fixed_checkpoint_branch_wspu_20260907/data"
OUTPUT = SCRIPT_DIR / "reference_outputs/two_phase_mariner_transfer_20260912/identification"
PRIMARY = (
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
    "cap4_shared_bounded_ensemble_kl0",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)
REPLICAS = {
    "cap4_shared_bounded_ensemble_kl0__seed1": ("cap4_shared_bounded_ensemble_kl0", "prefix_seed_repeat"),
    "shared_bounded_ensemble_kl0p05__v6e_bridge": ("shared_bounded_ensemble_kl0p05", "hardware_bridge"),
}
SELECTED = "observed_cap10_best"
PANELS = ("crossed_broad", "crossed_local")
CODE_BUCKETS = ("dolma3_stack_edu", "dolmino_stack_edu_fim", "dolmino_synth_code")
EXPECTED_HASHES = {
    "rows.csv": "d2a2140eb9507c9201f3e9b5130859a14cd75d6ad3869f34630eb83797dab65d",
    "arrays.npz": "60bbc3e213294fbb2065991f09eab4300aef390eae50f8825ca1bde573d06050",
    "action_aliases.csv": "21a61f2fe8b629c42f57d43203461a17f0b2ba1b72e76b7dd449872b8d44e631",
}
TOLERANCE = 1e-10


@dataclass(frozen=True)
class Panel:
    name: str
    states: tuple[str, ...]
    actions: tuple[str, ...]
    indices: np.ndarray
    losses: np.ndarray


def double_center(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True) - matrix.mean(axis=1, keepdims=True) + matrix.mean()


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def state_role(state: str) -> str:
    if state in PRIMARY:
        return "ordinary_primary"
    if state in REPLICAS:
        return REPLICAS[state][1]
    if state == SELECTED:
        return "outcome_selected_descriptive"
    raise ValueError(f"Unknown crossed state: {state}")


def spectrum(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > max(float(singular[0]), 1.0) * TOLERANCE))
    return singular, rank


def matrix_panel(rows: pd.DataFrame, losses: np.ndarray, name: str, states: tuple[str, ...]) -> Panel:
    frame = rows[rows.panel.eq(name) & rows.fit_budget & rows.state_id.isin(states)]
    action_sets = [set(frame.loc[frame.state_id.eq(state), "coordinate_hash"]) for state in states]
    actions = tuple(sorted(set.intersection(*action_sets)))
    assert actions and all(set(actions) == group for group in action_sets), "Incomplete common action crossing"
    positions = []
    for state in states:
        subset = frame[frame.state_id.eq(state)].set_index("coordinate_hash")
        assert subset.index.is_unique
        positions.append(subset.loc[list(actions), "row"].to_numpy(int))
    indices = np.asarray(positions)
    return Panel(name, states, actions, indices, losses[indices])


def write_groups(rows: pd.DataFrame, arrays: dict[str, np.ndarray], output: Path) -> dict[str, object]:
    crossed = rows[rows.panel.isin(PANELS)].copy()
    assert set(crossed.state_id) == set(PRIMARY) | set(REPLICAS) | {SELECTED}
    crossed["state_role"] = crossed.state_id.map(state_role)
    crossed["prefix_family"] = crossed.prefix
    assert all(crossed.loc[crossed.state_id.eq(state), "prefix"].eq(base).all() for state, (base, _) in REPLICAS.items())
    state_columns = [
        "panel",
        "state_id",
        "prefix_family",
        "state_role",
        "prefix_repeat_seed",
        "prefix_checkpoint_uri",
        "prefix_provenance_sha256",
        "historical_confirmatory_state",
    ]
    crossed[state_columns].drop_duplicates().to_csv(output / "state_groups.csv", index=False)
    metadata = rows.drop(
        columns=[
            "target",
            "anchor",
            "effect",
            "anchor_std_bpb",
            "training_anchor",
            "training_anchor_sd",
        ]
    ).copy()
    metadata["prefix_family"] = metadata.prefix
    metadata["action_group"] = metadata.coordinate_hash
    metadata["state_role"] = [
        state_role(row.state_id) if row.panel in PANELS else "historical" for row in rows.itertuples()
    ]
    metadata.to_csv(output / "row_groups.csv", index=False)
    aliases = []
    fit = crossed[crossed.fit_budget]
    for coordinate, group in rows[rows.coordinate_hash.isin(fit.coordinate_hash)].groupby("coordinate_hash"):
        weights = arrays["phase1_weight"][group.row.to_numpy(int)]
        assert np.max(np.abs(weights - weights[0])) < 1e-11
        aliases.append(
            {
                "coordinate_hash": coordinate,
                "action_ids": ";".join(sorted(set(group.action_id))),
                "panels": ";".join(sorted(set(group.panel))),
                "rows": len(group),
                "row_ids_json": json.dumps(group.row_id.tolist()),
            }
        )
    pd.DataFrame(aliases).to_csv(output / "action_groups.csv", index=False)
    counts = crossed.groupby(["panel", "state_role", "fit_budget", "role"]).size().rename("rows").reset_index()
    counts.to_csv(output / "membership_counts.csv", index=False)
    action_sets = {name: set(fit.loc[fit.panel.eq(name), "coordinate_hash"]) for name in PANELS}
    return {
        "primary_states": list(PRIMARY),
        "state_roles": {s: state_role(s) for s in sorted(set(crossed.state_id))},
        "crossed_rows": len(crossed),
        "primary_fit_rows": int((crossed.fit_budget & crossed.state_id.isin(PRIMARY)).sum()),
        "actions_per_panel": {name: len(group) for name, group in action_sets.items()},
        "broad_local_action_overlap": len(set.intersection(*action_sets.values())),
        "excluded_incomplete_actions": [],
        "action_holdout_key": "coordinate_hash across all rows, including controls and historical aliases",
        "prefix_holdout_key": "prefix family, grouping original, seed replica, hardware bridge, and historical copies",
    }


def decompose(panel: Panel, names: tuple[str, ...], rows: pd.DataFrame, tables: dict[str, list[dict]]) -> None:
    for component, name in enumerate(names):
        values = panel.losses[:, :, component]
        residual = double_center(values)
        state_effect = values.mean(axis=1) - values.mean()
        action_effect = values.mean(axis=0) - values.mean()
        state_ss = len(panel.actions) * np.sum(state_effect**2)
        action_ss = len(panel.states) * np.sum(action_effect**2)
        interaction_ss = np.sum(residual**2)
        total_ss = np.sum((values - values.mean()) ** 2)
        assert np.isclose(state_ss + action_ss + interaction_ss, total_ss, rtol=1e-12, atol=1e-14)
        singular, rank = spectrum(residual)
        tables["decomposition"].append(
            {
                "panel": panel.name,
                "component": name,
                "states": len(panel.states),
                "actions": len(panel.actions),
                "state_ss": state_ss,
                "action_ss": action_ss,
                "interaction_ss": interaction_ss,
                "total_ss": total_ss,
                "interaction_share_total": interaction_ss / total_ss,
                "interaction_share_within_state": interaction_ss / (interaction_ss + action_ss),
                "interaction_rms_bpb": rms(residual),
                "action_main_rms_bpb": rms(action_effect),
                "interaction_rank": rank,
                "rank1_interaction_share": singular[0] ** 2 / interaction_ss,
                "rank2_interaction_share": np.sum(singular[:2] ** 2) / interaction_ss,
            }
        )
        for index, value in enumerate(singular):
            tables["interaction_spectrum"].append(
                {
                    "panel": panel.name,
                    "component": name,
                    "index": index + 1,
                    "singular_value_bpb": value,
                    "normalized_singular_rms_bpb": value / np.sqrt(values.size),
                    "energy_share": value**2 / interaction_ss,
                }
            )
        for state_index, state in enumerate(panel.states):
            for action_index, action in enumerate(panel.actions):
                row_index = panel.indices[state_index, action_index]
                tables["residuals"].append(
                    {
                        "panel": panel.name,
                        "component": name,
                        "state_id": state,
                        "coordinate_hash": action,
                        "action_id": rows.iloc[row_index].action_id,
                        "row": row_index,
                        "endpoint_bpb": values[state_index, action_index],
                        "interaction_bpb": residual[state_index, action_index],
                    }
                )


def choice_metrics(panel: Panel, names: tuple[str, ...], rows: pd.DataFrame, tables: dict[str, list[dict]]) -> None:
    for component, name in enumerate(names):
        values = panel.losses[:, :, component]
        for source, target in itertools.permutations(range(len(panel.states)), 2):
            selected = int(np.argmin(values[source]))
            left, right = np.triu_indices(len(panel.actions), 1)
            signs1 = np.sign(values[source, left] - values[source, right])
            signs2 = np.sign(values[target, left] - values[target, right])
            tables["prefix_transfer"].append(
                {
                    "panel": panel.name,
                    "component": name,
                    "source_state": panel.states[source],
                    "target_state": panel.states[target],
                    "spearman": stats.spearmanr(values[source], values[target]).statistic,
                    "kendall_tau": stats.kendalltau(values[source], values[target]).statistic,
                    "action_pair_agreement": np.mean(signs1 == signs2),
                    "same_choice": selected == int(np.argmin(values[target])),
                    "selected_action": rows.iloc[panel.indices[target, selected]].action_id,
                    "regret_bpb": values[target, selected] - values[target].min(),
                }
            )
        for target, state in enumerate(panel.states):
            other = np.arange(len(panel.states)) != target
            selected = int(np.argmin(values[other].mean(axis=0)))
            tables["leave_prefix_out_choice"].append(
                {
                    "panel": panel.name,
                    "component": name,
                    "state_id": state,
                    "selected_action": rows.iloc[panel.indices[target, selected]].action_id,
                    "oracle_action": rows.iloc[panel.indices[target, int(np.argmin(values[target]))]].action_id,
                    "regret_bpb": values[target, selected] - values[target].min(),
                    "range_bpb": np.ptp(values[target]),
                    "selected_rank": int(stats.rankdata(values[target], method="min")[selected]),
                }
            )
    for state_index, state in enumerate(panel.states):
        for first, second in itertools.combinations(range(len(names)), 2):
            left = panel.losses[state_index, :, first]
            right = panel.losses[state_index, :, second]
            tables["component_agreement"].append(
                {
                    "panel": panel.name,
                    "state_id": state,
                    "component_1": names[first],
                    "component_2": names[second],
                    "spearman": stats.spearmanr(left, right).statistic,
                    "same_choice": int(np.argmin(left)) == int(np.argmin(right)),
                    "component2_regret_at_component1_choice_bpb": right[np.argmin(left)] - right.min(),
                }
            )


def control_metrics(
    panel: Panel,
    rows: pd.DataFrame,
    losses: np.ndarray,
    names: tuple[str, ...],
    tables: dict[str, list[dict]],
) -> None:
    controls = rows[rows.panel.eq(panel.name) & rows.state_id.isin(PRIMARY) & ~rows.fit_budget & ~rows.is_tied_control]
    hashes = sorted(set(controls.coordinate_hash))
    assert len(hashes) == 2
    original, repeated = [], []
    for state in PRIMARY:
        original_rows, repeat_rows = [], []
        for coordinate in hashes:
            main = rows[
                rows.panel.eq(panel.name)
                & rows.state_id.eq(state)
                & rows.coordinate_hash.eq(coordinate)
                & rows.fit_budget
            ]
            repeat = controls[controls.state_id.eq(state) & controls.coordinate_hash.eq(coordinate)]
            assert len(main) == len(repeat) == 1
            main_row, repeat_row = main.iloc[0], repeat.iloc[0]
            assert main_row.prefix_checkpoint_uri == repeat_row.prefix_checkpoint_uri
            assert main_row.trainer_seed == repeat_row.trainer_seed and main_row.data_seed != repeat_row.data_seed
            original_rows.append(int(main_row.row))
            repeat_rows.append(int(repeat_row.row))
            for component, name in enumerate(names):
                tables["control_differences"].append(
                    {
                        "panel": panel.name,
                        "component": name,
                        "state_id": state,
                        "action_id": main_row.action_id,
                        "coordinate_hash": coordinate,
                        "main_row": int(main_row.row),
                        "repeat_row": int(repeat_row.row),
                        "main_data_seed": main_row.data_seed,
                        "repeat_data_seed": repeat_row.data_seed,
                        "repeat_minus_main_bpb": losses[repeat_row.row, component] - losses[main_row.row, component],
                    }
                )
        original.append(original_rows)
        repeated.append(repeat_rows)
    first, second = losses[np.asarray(original)], losses[np.asarray(repeated)]
    for component, name in enumerate(names):
        values1, values2 = first[:, :, component], second[:, :, component]
        difference = values2 - values1
        residual1, residual2 = double_center(values1), double_center(values2)
        noise = double_center(difference) / np.sqrt(2)
        tables["control_noise"].append(
            {
                "panel": panel.name,
                "component": name,
                "states": 6,
                "repeated_actions": 2,
                "paired_difference_rms_bpb": rms(difference),
                "conditional_single_run_raw_noise_rms_bpb": rms(difference) / np.sqrt(2),
                "conditional_interaction_noise_rms_bpb": rms(noise),
                "matched_main_interaction_rms_bpb": rms(residual1),
                "matched_repeat_interaction_rms_bpb": rms(residual2),
                "main_repeat_interaction_correlation": np.corrcoef(residual1.ravel(), residual2.ravel())[0, 1],
                "matched_main_to_noise_rms_ratio": rms(residual1) / rms(noise),
                "same_pair_contrast_sign_fraction": np.mean(
                    np.sign(values1[:, 0] - values1[:, 1]) == np.sign(values2[:, 0] - values2[:, 1])
                ),
            }
        )
        for index, value in enumerate(np.linalg.svd(noise, compute_uv=False)):
            tables["control_noise_spectrum"].append(
                {
                    "panel": panel.name,
                    "component": name,
                    "index": index + 1,
                    "conditional_noise_singular_bpb": value,
                    "conditional_noise_normalized_singular_rms_bpb": value / np.sqrt(noise.size),
                }
            )


def auxiliary_metrics(
    rows: pd.DataFrame,
    losses: np.ndarray,
    names: tuple[str, ...],
    tables: dict[str, list[dict]],
) -> None:
    for panel_name in PANELS:
        for state, (base, role) in REPLICAS.items():
            pair = matrix_panel(rows, losses, panel_name, (base, state))
            for component, name in enumerate(names):
                first, second = pair.losses[:, :, component]
                difference = second - first
                record = {
                    "panel": panel_name,
                    "component": name,
                    "comparison": role,
                    "base_state": base,
                    "other_state": state,
                    "actions": len(pair.actions),
                    "mean_shift_bpb": np.mean(difference),
                    "raw_difference_rms_bpb": rms(difference),
                    "centered_difference_rms_bpb": rms(difference - difference.mean()),
                    "spearman": stats.spearmanr(first, second).statistic,
                    "same_choice": int(np.argmin(first)) == int(np.argmin(second)),
                    "base_choice_regret_on_other_bpb": second[np.argmin(first)] - second.min(),
                    "other_choice_regret_on_base_bpb": first[np.argmin(second)] - first.min(),
                }
                if role == "prefix_seed_repeat":
                    record["conditional_single_run_centered_noise_rms_bpb"] = rms(
                        difference - difference.mean()
                    ) / np.sqrt(2)
                tables["replica_bridge_comparison"].append(record)
        selected = matrix_panel(rows, losses, panel_name, (*PRIMARY, SELECTED))
        for component, name in enumerate(names):
            ordinary = selected.losses[:-1, :, component].mean(axis=0)
            outcome = selected.losses[-1, :, component]
            tables["outcome_selected_prefix"].append(
                {
                    "panel": panel_name,
                    "component": name,
                    "spearman_vs_ordinary_mean": stats.spearmanr(ordinary, outcome).statistic,
                    "ordinary_choice_regret_bpb": outcome[np.argmin(ordinary)] - outcome.min(),
                    "same_choice": int(np.argmin(ordinary)) == int(np.argmin(outcome)),
                }
            )
    for state in PRIMARY:
        indices = [
            rows.loc[rows.panel.eq(panel) & rows.state_id.eq(state) & rows.is_tied_control, "row"].to_numpy(int)
            for panel in PANELS
        ]
        assert all(len(group) == 1 for group in indices)
        for component, name in enumerate(names):
            tables["cross_panel_tied_controls"].append(
                {
                    "state_id": state,
                    "component": name,
                    "broad_row": indices[0][0],
                    "local_row": indices[1][0],
                    "local_minus_broad_bpb": losses[indices[1][0], component] - losses[indices[0][0], component],
                }
            )


def feature_metrics(panel: Panel, arrays: dict[str, np.ndarray], tables: dict[str, list[dict]]) -> None:
    state_indices = panel.indices[:, 0]
    action_indices = panel.indices[0]
    state_weights = arrays["phase0_weight"][state_indices]
    state_epochs = arrays["phase0_epochs"][state_indices]
    action_weights = arrays["phase1_weight"][action_indices]
    action_epochs = arrays["phase1_epochs"][action_indices]
    for state_row in panel.indices:
        assert np.allclose(arrays["phase0_epochs"][state_row], arrays["phase0_epochs"][state_row[0]], atol=1e-12)
        assert np.allclose(arrays["phase1_epochs"][state_row], action_epochs, atol=1e-12)
    bucket_names = arrays["bucket_names"].tolist()
    code = [bucket_names.index(name) for name in CODE_BUCKETS]
    overlap = double_center(np.log1p(state_epochs) @ action_weights.T)
    code_product = double_center(np.outer(state_weights[:, code].sum(axis=1), action_weights[:, code].sum(axis=1)))
    features = np.column_stack([overlap.ravel(), code_product.ravel()])
    feature_rms = np.sqrt(np.mean(features**2, axis=0))
    assert (feature_rms > 0).all()
    matrices = {
        "prefix_weight": state_weights - state_weights.mean(axis=0),
        "prefix_epochs": state_epochs - state_epochs.mean(axis=0),
        "prefix_log_epochs": np.log1p(state_epochs) - np.log1p(state_epochs).mean(axis=0),
        "action_weight": action_weights - action_weights.mean(axis=0),
        "action_epochs": action_epochs - action_epochs.mean(axis=0),
        "prespecified_interactions_rms_scaled": features / feature_rms,
    }
    for label, matrix in matrices.items():
        singular, rank = spectrum(matrix)
        energy = singular**2 / np.sum(singular**2)
        tables["feature_span"].append(
            {
                "panel": panel.name,
                "features": label,
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "rank": rank,
                "rank1_energy_share": energy[0],
                "rank2_energy_share": energy[:2].sum(),
                "rank_for_95pct_energy": int(np.searchsorted(np.cumsum(energy), 0.95) + 1),
                "condition_nonzero": singular[0] / singular[rank - 1],
            }
        )
        for index, value in enumerate(singular):
            tables["feature_spectrum"].append(
                {
                    "panel": panel.name,
                    "features": label,
                    "index": index + 1,
                    "singular_value": value,
                    "energy_share": energy[index],
                }
            )
    for state_index, state in enumerate(panel.states):
        for action_index, action in enumerate(panel.actions):
            tables["prespecified_features"].append(
                {
                    "panel": panel.name,
                    "state_id": state,
                    "coordinate_hash": action,
                    "repetition_overlap_centered": overlap[state_index, action_index],
                    "code_product_centered": code_product[state_index, action_index],
                }
            )
        other = np.arange(len(panel.states)) != state_index
        training = state_epochs[other]
        center = training.mean(axis=0)
        _, singular, right = np.linalg.svd(training - center, full_matrices=False)
        rank = int(np.sum(singular > max(float(singular[0]), 1.0) * TOLERANCE))
        basis = right[:rank]
        held = state_epochs[state_index] - center
        residual = held - held @ basis.T @ basis
        tables["leave_prefix_out_geometry"].append(
            {
                "panel": panel.name,
                "state_id": state,
                "train_affine_rank": rank,
                "held_epoch_span_residual_fraction": np.linalg.norm(residual) / np.linalg.norm(held),
                "prefix_code_token_fraction": state_weights[state_index, code].sum(),
                "token_weighted_prefix_epochs": state_weights[state_index] @ state_epochs[state_index],
            }
        )


def make_report(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    decomposition = tables["decomposition"]
    macro = decomposition[decomposition.component.eq("macro")].set_index("panel")
    noise = tables["control_noise"]
    macro_noise = noise[noise.component.eq("macro")].set_index("panel")
    transfers = tables["prefix_transfer"]
    choices = tables["leave_prefix_out_choice"]
    overview, repeats, components = [], [], []
    for name in PANELS:
        row = macro.loc[name]
        ranking = transfers[transfers.panel.eq(name) & transfers.component.eq("macro")]
        regret = choices[choices.panel.eq(name) & choices.component.eq("macro")].regret_bpb
        overview.append(
            f"| {name.removeprefix('crossed_')} | {row.interaction_rms_bpb:.6f} | "
            f"{row.interaction_share_within_state:.2%} | "
            f"{row.rank1_interaction_share:.1%} / {row.rank2_interaction_share:.1%} | "
            f"{ranking.spearman.mean():.3f} | {regret.mean():.6f} / {regret.max():.6f} |"
        )
        row = macro_noise.loc[name]
        repeats.append(
            f"| {name.removeprefix('crossed_')} | "
            f"{row.matched_main_interaction_rms_bpb:.6f} / {row.matched_repeat_interaction_rms_bpb:.6f} | "
            f"{row.conditional_interaction_noise_rms_bpb:.6f} | {row.main_repeat_interaction_correlation:.3f} |"
        )
    for name in decomposition.component.unique():
        if name == "macro":
            continue
        subset = decomposition[decomposition.component.eq(name)].set_index("panel")
        broad, local = subset.loc[PANELS[0]], subset.loc[PANELS[1]]
        broad_rank = transfers[transfers.panel.eq(PANELS[0]) & transfers.component.eq(name)].spearman.mean()
        local_rank = transfers[transfers.panel.eq(PANELS[1]) & transfers.component.eq(name)].spearman.mean()
        components.append(
            f"| {name} | {broad.interaction_rms_bpb:.6f} | {local.interaction_rms_bpb:.6f} | "
            f"{broad_rank:.3f} | {local_rank:.3f} |"
        )
    overview_table, repeat_table, component_table = ("\n".join(group) for group in (overview, repeats, components))
    report = dedent(
        """\
        Every ordinary prefix chooses the same best observed aggregate action within each panel:
        `fit_maximin_19` among the 50 broad actions and `local_plus_32` among the ten local actions.
        A common ranking learned from the other five prefixes therefore incurs zero observed aggregate
        regret at each excluded prefix. Broad rankings transfer closely; local rankings vary more.
        The aggregate interaction on the two repeated actions is smaller than its conditional noise
        scale and its pattern does not reproduce across continuation seeds. This panel provides little
        replicated aggregate evidence for a transferable state interaction, while some component
        interactions remain visible on the narrow repeat subsets.

        Six ordinary prefix configurations form the primary population. A prefix-seed replica, a
        hardware bridge, and an outcome-selected prefix are kept separate. The analysis removes
        prefix and action main effects from endpoint bits per byte (BPB; lower is better).
        The interaction is R(s,a) = Y(s,a) - mean_a Y(s,a) - mean_s Y(s,a) + mean Y.
        Its within-prefix share divides interaction sum of squares by interaction plus action sum
        of squares. Singular energy shares describe this residual matrix, including its training noise.
        All results are development evidence; no new response model was fitted.

        | Panel | Interaction RMS | Within-prefix share | SVD energy 1 / 2 | Mean rank rho | Regret mean / max |
        |---|---:|---:|---:|---:|---:|
        OVERVIEW_TABLE

        Interaction and regret are in BPB. Rank rho is the mean Spearman correlation across prefix
        pairs. The excluded-prefix choice uses the action with the lowest mean endpoint over the other
        five ordinary prefixes. It tests transfer of a common action ranking; it does not test an unseen
        action. The many prefix pairs and action pairs are descriptive comparisons and do not increase
        the six state configurations. The broad aggregate action range is 0.07526-0.08725 BPB within
        a prefix; the local range is 0.000835-0.001189. Local interaction is a larger fraction of a
        much smaller action signal.

        | Two-action repeat subset | Main / repeat interaction RMS | Conditional noise RMS | Interaction correlation |
        |---|---:|---:|---:|
        REPEAT_TABLE

        Each repeat subset crosses two actions with the six ordinary prefixes. Broad repeats cover
        `fit_maximin_00` and `fit_maximin_26`; local repeats cover `local_anchor_fit079` and `local_plus_24`.
        The repeated action changes the recorded continuation data seed within the same frozen prefix,
        trainer seed, and mixture. The conditional single-run interaction noise scale is the RMS of the
        double-centered repeat-minus-main matrix divided by the square root of two. That scale assumes
        comparable variance and independent seed realizations. It is not calibrated for the other
        48 broad or eight local actions. Two actions allow at most one nonzero interaction direction.
        No significance test or noise subtraction is used.

        | Component | Broad interaction RMS | Local interaction RMS | Broad rank rho | Local rank rho |
        |---|---:|---:|---:|---:|
        COMPONENT_TABLE

        Broad component rankings transfer across prefixes with mean Spearman 0.983-0.997. Six of
        seven components keep the same best action across all six prefixes; physics has two winners.
        Local components have one to three observed winners. Their mean excluded-prefix regrets are
        0-0.000216 BPB. Local macro rankings correlate positively with code-component rankings
        (mean rho 0.687 for C++ and 0.675 for Python) and negatively with every non-code component
        (-0.390 to -0.071). This is an aggregate objective tradeoff, not seven independent replications.
        On the two-action controls, local Wikipedia interaction has repeat correlation 0.880 and
        main-interaction/noise RMS ratio 2.21; broad arXiv CS, physics, and BBC correlations are
        0.640-0.653 with ratios 1.27-1.84. These narrow component patterns do not establish aggregate
        transfer or justify selecting a component-specific response head.

        The cap-4 KL0 prefix-seed replica changes mean aggregate BPB by -0.004003 on broad actions
        and -0.002523 on local actions. After removing those offsets, paired-difference RMS is
        0.001263 broad and 0.000342 local. The broad winner survives; the local winner changes and
        the original-prefix choice incurs 0.000424 BPB regret on the replica. The hardware bridge
        likewise keeps the broad winner but changes the local winner (0.000194 BPB transfer regret).
        Neither auxiliary state increases the number of ordinary prefix configurations. The
        outcome-selected prefix keeps both primary aggregate winners and remains descriptive.

        Broad and local fit actions share no coordinate and use different continuation data seeds
        (970000 versus 974000). Panel differences combine action support and a seed change. All
        crossed trainer seeds equal zero. Each primary state-action cell has one endpoint, so the
        interaction residual cannot distinguish systematic interaction from training noise by itself.
        The bridge comparison also changes hardware and is not an independent seed-noise estimate.

        Strict action holdouts group `coordinate_hash` over every panel and role. All 50 broad
        coordinates also occur in the historical cap-10 broad panel. The local anchor aliases
        proportional `fit_079` and its four frontier repeats. Noise controls alias scored actions.
        `control_tied` has one label but seven distinct coordinates across the nine states.
        Strict prefix transfer groups original checkpoints, seed replicas, hardware bridges, and
        historical copies by prefix family. Full mappings are in `row_groups.csv`, `action_groups.csv`,
        and `state_groups.csv`. No incomplete primary action was excluded: both crossings are complete.

        Prefix weight and exposure matrices have centered rank five. Two directions explain 97.2%
        of prefix-weight energy and 94.2% of prefix-epoch energy. Broad action weights have rank 38;
        local weights have rank six. Thus the local panel probes only a small part of the 38-dimensional
        mixture simplex. In each prefix holdout, five training states span an affine space of rank four;
        5.6-26.8% of the held state's centered epoch-vector norm lies outside it. These are geometric
        support measurements, not outcome fits or uncertainty estimates.

        [FEATURE_SPEC.md](FEATURE_SPEC.md) fixed two scalar interactions before endpoint computation:
        continuation-weighted log prefix exposure and prefix-code fraction times continuation-code
        fraction. Their double-centered, RMS-scaled design has rank two with condition number 1.08
        on broad actions and 1.30 on local actions. These are separately identifiable feature
        directions on this input matrix, but their response coefficients and transfer value are
        unmeasured. Test them with shared coefficients and prefix-family/action-alias holdouts before
        adding response flexibility. The audit fits no coefficients, chooses no signs, and performs no
        outcome-dependent feature search. See `feature_span.csv` and `leave_prefix_out_geometry.csv`.

        Small interaction on these actions does not show that two-phase training has no advantage.
        Prefix and action jointly determine total aggregate exposure, and the panel lacks an
        aggregate-matched tied endpoint for every crossed cell. This decomposition cannot isolate
        causal phase-order benefit. Singular energy concentration also does not establish transfer
        outside the observed prefix and action support.

        Reproduce from the repository root with:

        ```sh
        cd experiments/domain_phase_mix/exploratory/two_phase_many
        uv run --offline analyze_crossed_identification_20260912.py
        ```

        [PROTOCOL.md](PROTOCOL.md) records the pre-measurement plan. `manifest.json` pins inputs,
        protocol, feature specification, script, and output hashes; `sources/` contains source snapshots.
        `component_weights.csv` gives the frozen aggregate weights. `VALIDATION.json` records reconstruction
        and membership checks. No remote data was accessed and no training, evaluation, registry,
        logbook, or Fieldbook state was changed.
        """
    )
    report = report.replace("OVERVIEW_TABLE", overview_table).replace("REPEAT_TABLE", repeat_table)
    (output / "REPORT.md").write_text(report.replace("COMPONENT_TABLE", component_table))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = {name: sha256(args.input_dir / name) for name in EXPECTED_HASHES}
    assert inputs == EXPECTED_HASHES, "Frozen BRW inputs changed"
    sources = {
        str(Path(__file__).resolve()): sha256(Path(__file__)),
        str(Path(load_inputs.__code__.co_filename).resolve()): sha256(Path(load_inputs.__code__.co_filename)),
    }
    protocols = {name: sha256(output / name) for name in ("PROTOCOL.md", "FEATURE_SPEC.md")}
    identity = {"inputs": inputs, "sources": sources, "protocols": protocols}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] == identity and all(
            (output / path).is_file() and sha256(output / path) == digest for path, digest in previous["outputs"].items()
        ):
            print("Existing diagnostic verified; no recomputation needed.")
            return
    rows, arrays = load_inputs(args.input_dir)
    crossed = rows.panel.isin(PANELS).to_numpy()
    assert np.isfinite(arrays["component_bpb"][crossed]).all()
    aggregate = arrays["component_bpb"] @ arrays["component_weights"]
    error = float(np.max(np.abs(aggregate[crossed] - arrays["target"][crossed])))
    assert error < 5e-7
    names = ("macro", *arrays["component_names"].tolist())
    losses = np.column_stack([aggregate, arrays["component_bpb"]])
    groups = write_groups(rows, arrays, output)
    table_names = (
        "decomposition",
        "interaction_spectrum",
        "residuals",
        "prefix_transfer",
        "leave_prefix_out_choice",
        "component_agreement",
        "control_differences",
        "control_noise",
        "control_noise_spectrum",
        "replica_bridge_comparison",
        "outcome_selected_prefix",
        "cross_panel_tied_controls",
        "feature_span",
        "feature_spectrum",
        "prespecified_features",
        "leave_prefix_out_geometry",
    )
    tables: dict[str, list[dict]] = {name: [] for name in table_names}
    panels = [matrix_panel(rows, losses, panel_name, PRIMARY) for panel_name in PANELS]
    for panel in panels:
        feature_metrics(panel, arrays, tables)
    for panel in panels:
        decompose(panel, names, rows, tables)
        choice_metrics(panel, names, rows, tables)
        control_metrics(panel, rows, losses, names, tables)
    auxiliary_metrics(rows, losses, names, tables)
    frames = {name: pd.DataFrame(records) for name, records in tables.items()}
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    component_weights = pd.DataFrame(
        {"component": arrays["component_names"], "canonical_weight": arrays["component_weights"]}
    )
    component_weights.to_csv(output / "component_weights.csv", index=False)
    make_report(output, frames)
    validation = {
        "status": "passed",
        "input_hashes_verified": True,
        "complete_primary_crossings": True,
        "within_panel_state_action_uniqueness": True,
        "canonical_aggregate_max_error_bpb": error,
        "auxiliary_states_kept_out_of_primary": True,
        "response_heads_fitted": 0,
        "outcome_independent_feature_specification": True,
        "remote_calls": 0,
        "table_rows": {name: len(frame) for name, frame in frames.items()},
        **groups,
    }
    (output / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    snapshot = output / "sources"
    snapshot.mkdir(exist_ok=True)
    for path in sources:
        shutil.copyfile(path, snapshot / Path(path).name)
    outputs = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(json.dumps({"identity": identity, "outputs": outputs}, indent=2) + "\n")
    print(json.dumps({"output": str(output), "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
