# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from experiments.grug_sft.science_mix import (
    BASE_STEP,
    BATCH,
    FINAL_STEP,
    MIX_BUDGETS,
    MIXTURE_BLOCK_SIZE,
    START_STEP,
    STEPS,
    TOKENS,
    ScienceMix,
    run_id,
)


def test_science_mix_budgets_are_size_controlled():
    assert BASE_STEP == 157_000
    assert START_STEP == 0
    assert FINAL_STEP == STEPS == 1_491
    assert TOKENS == 100_059_316_224
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
