# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from experiments.domain_phase_mix.release import public_names


def _swarm(deletion_zeroes: str = "cc/art_and_design_high") -> pd.DataFrame:
    """A swarm table with one sampled run, one deletion that zeroes `deletion_zeroes`, and ten lettered repeats."""
    buckets = ["arxiv", "cc/art_and_design_high"]
    rows = [{"observation_id": "s", "run_label": "run_00000", "design_kind": "sampled", "w__arxiv": 0.5}]
    rows.append(
        {"observation_id": "d", "run_label": "pctrl_del_dolma3_cc_art_and_design_high", "design_kind": "deletion"}
    )
    rows[-1].update({f"w__{b}": (0.0 if b == deletion_zeroes else 1.0) for b in buckets})
    rows += [
        {"observation_id": f"r{letter}", "run_label": f"proportional_repeat_{letter}", "design_kind": "repeat"}
        for letter in "jihgfedcba"  # out of order, so the public numbers must follow the old labels' order
    ]
    table = pd.DataFrame(rows)
    return table.fillna({f"w__{b}": 0.5 for b in buckets})


def _tables(**overrides: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {name: _swarm() for name in public_names.SWARMS}
    tables["heldout"] = pd.DataFrame({"proposal_method": ["archive::delphi_table9_value_room_20260705", "factorial"]})
    tables["fits_learning_curves"] = pd.DataFrame({"stratum": ["pooled"]})
    tables["validation_3e18"] = pd.DataFrame({"launch": ["table9_snr_subset", "factorial"]})
    tables["fits_proposal_predictions"] = pd.DataFrame({"launch": ["factorial"]})
    tables["fits_reliability"] = pd.DataFrame({"source_table": ["snr_fit_components_delphi"]})
    tables.update(overrides)
    return tables


def test_publish_renames_labels_and_records_every_alias():
    published, aliases = public_names.publish(_tables())
    swarm = published["swarm_qwen3_360m_1p6b"].set_index("observation_id").run_label
    assert swarm["s"] == "run_00000"
    assert swarm["d"] == "deletion_cc_art_and_design_high"
    assert [swarm[f"r{letter}"] for letter in "abcdefghij"] == [f"proportional_repeat_{i:02d}" for i in range(10)]
    assert list(published["heldout"].proposal_method) == ["value_room_obe_easy", "factorial"]
    assert list(published["validation_3e18"].launch) == ["obe_easy_snr_subset", "factorial"]
    assert list(published["fits_reliability"].source_table) == ["snr_fit_components"]
    deletion = aliases[(aliases.table == "swarm_qwen3_360m_1p6b") & (aliases.observation_id == "d")]
    assert deletion[["legacy", "public"]].values.tolist() == [
        ["pctrl_del_dolma3_cc_art_and_design_high", "deletion_cc_art_and_design_high"]
    ]


def test_the_deletion_label_follows_the_zeroed_bucket_not_the_old_label():
    published, _ = public_names.publish(_tables(swarm_llama_200m_6b=_swarm(deletion_zeroes="arxiv")))
    labels = published["swarm_llama_200m_6b"].set_index("observation_id").run_label
    assert labels["d"] == "deletion_arxiv"


def test_an_archive_folder_without_a_public_name_fails():
    heldout = pd.DataFrame({"proposal_method": ["archive::delphi_new_campaign_20261109"]})
    with pytest.raises(ValueError, match="archive folders without a public name"):
        public_names.publish(_tables(heldout=heldout))
