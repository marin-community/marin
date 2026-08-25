# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase0_harsh_cap_validation_20260825 as materialize,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    prepare_delphi_phase0_harsh_cap_candidates_20260825 as prepare,
)


def _write_cap_artifacts(directory: Path, cap: int) -> None:
    candidate_ids = (
        *(f"shared_bounded_ensemble_{label}" for label in prepare.KL_LABELS),
        f"observed_cap{cap}_best",
        "proportional_control",
    )
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"phase_0_epoch_cap": cap}))
    pd.DataFrame({"candidate_id": candidate_ids}).to_csv(directory / "candidate_summary.csv", index=False)
    rows = []
    for position, candidate_id in enumerate(candidate_ids):
        if candidate_id.startswith("observed_cap"):
            counts = (800, 1_248)
        elif candidate_id == "proportional_control":
            counts = (1_024, 1_024)
        else:
            first_count = cap * 100 + position
            counts = (first_count, prepare.MIXTURE_BLOCK_SIZE - first_count)
        for bucket, count in zip(("a", "b"), counts, strict=True):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "bucket": bucket,
                    "phase_0_weight": count / prepare.MIXTURE_BLOCK_SIZE,
                    "phase_0_count": count,
                    "phase_0_materialized_epochs": count / prepare.MIXTURE_BLOCK_SIZE,
                }
            )
    pd.DataFrame(rows).to_csv(directory / "candidate_weights.csv", index=False)
    pd.DataFrame({"candidate_id": candidate_ids}).to_csv(directory / "optimization_audit.csv", index=False)
    pd.DataFrame({"candidate_id": candidate_ids}).to_csv(directory / "partition_stability.csv", index=False)


def test_freeze_candidates_deduplicates_only_exact_runtime_mixtures(tmp_path: Path) -> None:
    cap4 = tmp_path / "cap4"
    cap6 = tmp_path / "cap6"
    _write_cap_artifacts(cap4, 4)
    _write_cap_artifacts(cap6, 6)

    alias_summary, _, aliases, training_weights, _ = prepare.freeze_candidates({4: cap4, 6: cap6})

    assert len(alias_summary) == 12
    assert aliases.selection_eligible.sum() == 8
    assert training_weights.candidate_id.nunique() == 10
    cap6_observed = aliases[aliases.alias_id.eq("cap6_observed_cap6_best")].iloc[0]
    cap6_proportional = aliases[aliases.alias_id.eq("cap6_proportional_control")].iloc[0]
    assert cap6_observed.canonical_candidate_id == "cap4_observed_cap4_best"
    assert cap6_proportional.canonical_candidate_id == "cap4_proportional_control"


def test_select_aliases_uses_boundary_mean_and_lower_kl_tie_break(tmp_path: Path) -> None:
    cap4 = tmp_path / "cap4"
    cap6 = tmp_path / "cap6"
    _write_cap_artifacts(cap4, 4)
    _write_cap_artifacts(cap6, 6)
    _, _, aliases, training_weights, _ = prepare.freeze_candidates({4: cap4, 6: cap6})
    scores = {candidate_id: 1.2 for candidate_id in training_weights.candidate_id.unique()}
    scores["cap4_shared_bounded_ensemble_kl0"] = 1.0
    scores["cap4_shared_bounded_ensemble_kl0p05"] = 1.0
    scores["cap6_shared_bounded_ensemble_kl0p2"] = 0.9
    summary = pd.DataFrame(
        {
            "canonical_candidate_id": list(scores),
            "uncheatable_mean": list(scores.values()),
            "uncheatable_sd": 0.01,
            "github_cpp_mean": 1.0,
            "github_cpp_sd": 0.01,
            "uncheatable_sem": 0.01,
        }
    )

    _, selected = materialize.select_aliases(aliases, summary)

    selected_by_cap = selected.set_index("cap_epochs")
    assert selected_by_cap.loc[4, "candidate_id"] == "shared_bounded_ensemble_kl0"
    assert selected_by_cap.loc[6, "candidate_id"] == "shared_bounded_ensemble_kl0p2"
