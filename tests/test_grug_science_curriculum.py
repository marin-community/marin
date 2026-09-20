# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.datasets.science_curricula import science_curriculum_datasets
from experiments.grug_sft.science_mix import MIX_BUDGETS, TOKENS, ScienceMix, run_id


def test_science_mix_budgets_are_size_controlled():
    assert TOKENS == 100_059_316_224
    for budget in MIX_BUDGETS.values():
        assert math.isclose(math.fsum(budget.as_dict().values()), 100.0)
        assert budget.science == 14.5
        assert budget.curriculum == 0.35
        assert budget.replay < 8.0


def test_science_mix_identities_are_distinct_and_versioned():
    identities = {run_id(mix, "v1") for mix in ScienceMix}

    assert len(identities) == len(ScienceMix)
    with pytest.raises(ValueError, match="Version must contain"):
        run_id(ScienceMix.BALANCED, "../replace")


def test_science_text_catalog_has_each_approved_family():
    datasets = science_curriculum_datasets()

    assert len(datasets) == 16
    assert "biocollection/free_text_stream" in datasets
    assert "swallow-math-v2/qa" in datasets
    assert "ultradata-math/l2" in datasets
    assert "openstax/biology" in datasets
    assert "mit-ocw/physics/classical-mechanics" in datasets
