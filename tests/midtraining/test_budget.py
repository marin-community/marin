# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CPT budget model."""

import pytest
from marin.midtraining.budget import (
    BudgetKind,
    BudgetPolicy,
    default_budget_label,
    resolve_cpt_budget,
)

DEFAULT_SEQ_LEN = 4096


def test_fixed_tokens_budget_resolves_old_10b_steps():
    policy = BudgetPolicy.fixed_tokens(10_000_000_000, label="10b")
    resolved = resolve_cpt_budget(
        policy,
        base_flops_key="1e21",
        base_pretrain_tokens=46_300_000_000,
        batch_size=512,
        seq_len=DEFAULT_SEQ_LEN,
    )
    assert resolved.num_train_steps == round(10_000_000_000 / (512 * DEFAULT_SEQ_LEN))
    assert resolved.label == "10b"
    assert resolved.requested_tokens == 10_000_000_000


def test_pretrain_fraction_budget_resolves_k020_steps():
    policy = BudgetPolicy.pretrain_fraction(0.20)
    resolved = resolve_cpt_budget(
        policy,
        base_flops_key="1e21",
        base_pretrain_tokens=46_300_000_000,
        batch_size=512,
        seq_len=DEFAULT_SEQ_LEN,
    )
    assert resolved.requested_tokens == round(0.20 * 46_300_000_000)
    assert resolved.num_train_steps == 4416
    assert resolved.pretrain_fraction_actual == pytest.approx(0.20, rel=0.005)


def test_fixed_steps_budget_renders_smoke_runs():
    policy = BudgetPolicy.fixed_steps(20, label="smoke20")
    resolved = resolve_cpt_budget(
        policy,
        base_flops_key="3e20",
        base_pretrain_tokens=18_600_000_000,
        batch_size=128,
        seq_len=DEFAULT_SEQ_LEN,
    )
    assert resolved.num_train_steps == 20
    assert resolved.label == "smoke20"


def test_pretrain_fraction_validates_range():
    with pytest.raises(ValueError, match="PRETRAIN_FRACTION requires"):
        BudgetPolicy.pretrain_fraction(0.0)
    with pytest.raises(ValueError, match="PRETRAIN_FRACTION requires"):
        BudgetPolicy.pretrain_fraction(1.5)


def test_budget_policy_rejects_overspecified_fields():
    with pytest.raises(ValueError, match="must not also set"):
        BudgetPolicy(kind=BudgetKind.FIXED_TOKENS, tokens=1_000_000, fraction=0.1)


def test_default_budget_label_for_round_billion():
    assert default_budget_label(10_000_000_000) == "10b"


def test_default_budget_label_for_fractional_billion():
    assert default_budget_label(9_250_000_000) == "9p25b"
