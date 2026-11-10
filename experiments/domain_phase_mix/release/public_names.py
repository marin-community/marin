# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public names for the release's internal values, and the record of every rename.

The exporters keep the names their sources use, because their joins depend on them. `publish` renames once, after
every table is built:

- each swarm's deletion run labels, `pctrl_del_<internal bucket>`, to `deletion_<bucket id>` with the bucket id's `/`
  written `_`, the convention of `mariner.design`; the bucket is the one the row's weights zero;
- each swarm's ten proportional repeats, `proportional_repeat_<letter or trainer seed>`, to `proportional_repeat_00`
  through `proportional_repeat_09`, in the order of the old labels;
- the development campaigns that `heldout.proposal_method` and `fits_learning_curves.stratum` name by their archive
  folders (`archive::<folder>`) to the names in CAMPAIGNS;
- launches and reliability sources named after an old paper table or the internal name of the Qwen3 360M/1.6B
  setting ("table9" is OlmoBaseEval Easy, "Delphi" the setting, "table5" an earlier draft's baseline table).

Every run keeps its observation id, its W&B run id. `publish` returns the renamed tables and the aliases, one row per
renamed value: the table, column, observation id (run labels only), legacy value and public value. The export writes
them to ALIASES, which the repository keeps as the internal record of the old names; the public dataset holds only the
new ones.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.release import paths

ALIASES = paths.REPO / "experiments/domain_phase_mix/release/public_name_aliases.csv"
SWARMS = ("swarm_qwen3_360m_1p6b", "swarm_llama_160m_1p2b", "swarm_llama_200m_6b")
DELETION_PREFIX = "deletion_"
REPEAT_PREFIX = "proportional_repeat_"
REPEATS_PER_SWARM = 10
ARCHIVE_PREFIX = "archive::"
CAMPAIGN_COLUMNS = (("heldout", "proposal_method"), ("fits_learning_curves", "stratum"))
CAMPAIGNS = {
    "archive::delphi_3e18_adversarial_stress_panel_20260716": "adversarial_stress_panel",
    "archive::delphi_3e18_frontier_phase_fiber_20260719": "frontier_phase_fiber",
    "archive::delphi_3e18_hybrid_phase_ordering_validation_20260720": "hybrid_phase_ordering_validation",
    "archive::delphi_augmented_profile_validation_20260705": "augmented_profile_validation",
    "archive::delphi_baseline_mixtures_issue6607_20260623": "baseline_mixtures",
    "archive::delphi_best_phase_model_validation_20260710": "best_phase_model_validation",
    "archive::delphi-corrective-hpr-280-tied-controls-3e18": "corrective_tied_controls",
    "archive::delphi_decoupled_phase_information_validation_3e18_20260712": "decoupled_phase_information_validation",
    "archive::delphi_gamma_capped_bowl_validation_20260704": "gamma_capped_bowl_validation",
    "archive::delphi_one_phase_olmix_kl_sweep_3e18_20260705": "olmix_kl_sweep",
    "archive::delphi_one_phase_table9_dsp_kl_sweep_3e18_20260628": "dsp_kl_sweep_obe_easy",
    "archive::delphi_one_phase_table9_olmix_scaling_20260628": "olmix_scaling_obe_easy",
    "archive::delphi_one_phase_uncheatable_validation_20260629": "uncheatable_validation",
    "archive::delphi_original_style_matched_sepheads_ablation_3e18_20260712": "matched_separate_heads_ablation",
    "archive::delphi_sep_frontier_tied_validation_20260710": "separate_heads_frontier_tied_validation",
    "archive::delphi_sufficiency_floored_validation_20260705": "sufficiency_floored_validation",
    "archive::delphi_symmetric_sepheads_geometry_frontier_3e18_20260711": "symmetric_separate_heads_frontier",
    "archive::delphi_table9_controlled_tilt_validation_20260705": "controlled_tilt_validation_obe_easy",
    "archive::delphi_table9_fresh_anneal_v2_20260705": "fresh_anneal_obe_easy_v2",
    "archive::delphi_table9_fresh_anneal_validation_20260705": "fresh_anneal_validation_obe_easy",
    "archive::delphi_table9_value_room_20260705": "value_room_obe_easy",
    "archive::delphi_uncheatable_optimized_mixtures_20260625": "uncheatable_optimized_mixtures",
    "archive::delphi_winner_neighborhood_validation_20260705": "winner_neighborhood_validation",
    "archive::hpr_300m_to_3e18_optimum_validation_panel_20260720": "optimum_transfer_from_llama_200m_6b",
    "archive::hpr_3e18_to_3e18_optimum_validation_panel_20260720": "optimum_validation_panel",
}
LAUNCH_COLUMNS = (("validation_3e18", "launch"), ("fits_proposal_predictions", "launch"))
LAUNCHES = {
    "table5_repeats": "baseline_proposal_repeats",
    "table9_qa_subset": "obe_easy_qa_subset",
    "table9_snr_subset": "obe_easy_snr_subset",
}
SOURCE_COLUMNS = (("fits_reliability", "source_table"),)
SOURCES = {
    "snr_fit_components_delphi": "snr_fit_components",
    "snr_fit_tasks_delphi": "snr_fit_tasks",
    "snr_fit_uncheatable_delphi": "snr_fit_uncheatable",
}


def publish(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """The tables with public names, and the aliases of every renamed value."""
    out = dict(tables)
    aliases = []
    for name in SWARMS:
        out[name], renamed = _swarm_labels(name, tables[name])
        aliases.append(renamed)
    for columns, mapping, strict in (
        (CAMPAIGN_COLUMNS, CAMPAIGNS, True),
        (LAUNCH_COLUMNS, LAUNCHES, False),
        (SOURCE_COLUMNS, SOURCES, False),
    ):
        for table, column in columns:
            out[table], renamed = _values(table, column, tables[table], mapping, strict)
            aliases.append(renamed)
    return out, pd.concat(aliases, ignore_index=True)


def deletion_label(bucket_id: str) -> str:
    """`deletion_<bucket id>` with the bucket id's `/` written `_`, as `mariner.design` names a deletion run."""
    return f"{DELETION_PREFIX}{bucket_id.replace('/', '_')}"


def _swarm_labels(name: str, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = table[[c for c in table.columns if c.startswith("w__")]]
    labels = table.run_label.copy()
    deletion = table.design_kind == "deletion"
    zeroed = weights[deletion] == 0
    if not (zeroed.sum(axis=1) == 1).all():
        raise ValueError(f"{name}: every deletion row must zero exactly one bucket")
    labels[deletion] = [deletion_label(column.removeprefix("w__")) for column in zeroed.idxmax(axis=1)]
    repeat = table.design_kind == "repeat"
    if repeat.sum() != REPEATS_PER_SWARM:
        raise ValueError(f"{name}: {repeat.sum()} proportional repeats, not {REPEATS_PER_SWARM}")
    order = np.argsort(table.run_label[repeat].to_numpy(), kind="stable")
    labels[table.index[repeat.to_numpy()][order]] = [f"{REPEAT_PREFIX}{i:02d}" for i in range(REPEATS_PER_SWARM)]
    if not labels.is_unique:
        raise ValueError(f"{name}: public run labels are not unique")
    changed = labels != table.run_label
    aliases = pd.DataFrame(
        {
            "table": name,
            "column": "run_label",
            "observation_id": table.observation_id[changed].to_numpy(),
            "legacy": table.run_label[changed].to_numpy(),
            "public": labels[changed].to_numpy(),
        }
    )
    return table.assign(run_label=labels), aliases


def _values(
    table: str, column: str, df: pd.DataFrame, mapping: dict[str, str], strict: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`df` with `column` renamed by `mapping`; with `strict`, every archive folder must have a public name."""
    values = df[column]
    if strict:
        unnamed = sorted(v for v in values.dropna().unique() if v.startswith(ARCHIVE_PREFIX) and v not in mapping)
        if unnamed:
            raise ValueError(f"{table}.{column}: archive folders without a public name: {unnamed}")
    present = sorted(set(values.dropna().unique()) & mapping.keys())
    clash = sorted({mapping[v] for v in present} & set(values.dropna().unique()))
    if clash:
        raise ValueError(f"{table}.{column}: public names already in use: {clash}")
    aliases = pd.DataFrame(
        {
            "table": table,
            "column": column,
            "observation_id": "",
            "legacy": present,
            "public": [mapping[v] for v in present],
        }
    )
    return df.assign(**{column: values.replace(mapping)}), aliases


def write_aliases(aliases: pd.DataFrame, path: Path) -> None:
    """The aliases, sorted, so that a rebuild that renames the same values rewrites the file unchanged."""
    aliases.sort_values(["table", "column", "legacy", "observation_id"]).to_csv(path, index=False)
