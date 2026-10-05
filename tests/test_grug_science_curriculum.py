# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.grug_sft.science_mix import (
    BATCH,
    MIX_BUDGETS,
    MIXTURE_BLOCK_SIZE,
    STEPS,
    ScienceMix,
    run_id,
)


def test_science_mix_budgets_are_size_controlled():
    assert STEPS > 0
    assert MIXTURE_BLOCK_SIZE < 2**16
    assert STEPS * BATCH % MIXTURE_BLOCK_SIZE == 0
    for budget in MIX_BUDGETS.values():
        assert math.isclose(math.fsum(budget.as_dict().values()), 100.0)
        assert budget.replay < 8.0


def test_science_mix_identities_are_distinct_and_versioned():
    identities = {run_id(mix, "v1") for mix in ScienceMix}

    assert len(identities) == len(ScienceMix)
    with pytest.raises(ValueError, match="Version must contain"):
        run_id(ScienceMix.BALANCED, "../replace")
