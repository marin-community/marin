# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fast tests for the midtraining mix framework (Layers 1, 2 of the safety model).

Layer 3 (val/train disjointness, partition fingerprint) lives in
:mod:`experiments.midtrain_data_safety` and requires a real Levanter cache;
those checks are invoked at sweep launch and are not exercised here.

These tests guard the invariants documented in
``.agents/logbooks/midtraining_delphi.md`` § "2026-05-01 21:00 UTC":

* Heuristic budget rule: ``midtrain_tokens = pretrain_tokens * K`` with
  ``K = MIDTRAIN_BUDGET_FRACTION``.
* Spec-time validation: rejects bad shares, duplicates, name collisions,
  and weight-sum mismatches.
* Built-config validation: weights sum to 1.0, val carve-outs reference
  real components, ``shuffle_before_trainval_split=True``,
  ``shuffle != True`` (full Feistel on training reintroduces the I/O
  cost regression that PR #5246 fixed), ``auto_build_caches=False``.
* Registry integrity: every registered mix passes Layer 2.
"""

import dataclasses

import pytest
from levanter.data.text import BlockShuffleConfig

from experiments.midtraining_mixes import (
    DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT,
    FULL_HIGHQUALITY_NEMO_MATH_NAME,
    HIGHQUALITY_NEMO_MATH_KEY,
    MIDTRAIN_BUDGET_FRACTION,
    MIDTRAIN_SPECS,
    MIDTRAINING_MIXES,
    PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME,
    PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME,
    MidtrainComponent,
    MidtrainMixSpec,
    _highquality_nemo_math_step,
    assert_lm_data_config_safe,
    build_midtrain_lm_data_config,
    midtrain_token_budget,
    midtraining_mix_by_name,
    validate_midtrain_spec,
)
from experiments.pretraining_datasets.nemotron import nemotron_mix

# ─── Heuristic ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pretrain, fraction, expected",
    [
        # 1e20 (3e20-iso d2048-L21): 47,064 * 128 * 4096
        (24_671_944_704, 0.20, 4_934_388_940),
        # 1e21-v5: 22,057 * 512 * 4096
        (46_272_479_232, 0.20, 9_254_495_846),
        # 1e22-v5: 38,235 * 1024 * 4096
        (160_367_738_880, 0.20, 32_073_547_776),
    ],
)
def test_midtrain_token_budget_canonical(pretrain, fraction, expected):
    assert midtrain_token_budget(pretrain_tokens=pretrain, fraction=fraction) == expected


def test_midtrain_token_budget_default_uses_global_constant():
    pretrain = 24_671_944_704
    assert midtrain_token_budget(pretrain_tokens=pretrain) == int(pretrain * MIDTRAIN_BUDGET_FRACTION)


@pytest.mark.parametrize("bad", [-1, 0])
def test_midtrain_token_budget_rejects_non_positive_pretrain(bad):
    with pytest.raises(ValueError):
        midtrain_token_budget(pretrain_tokens=bad)


def test_midtrain_token_budget_rejects_non_int_pretrain():
    with pytest.raises(TypeError):
        midtrain_token_budget(pretrain_tokens=1.5e10)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_fraction", [-0.1, 0.0, 1.1])
def test_midtrain_token_budget_rejects_bad_fraction(bad_fraction):
    with pytest.raises(ValueError):
        midtrain_token_budget(pretrain_tokens=10_000, fraction=bad_fraction)


# ─── Spec validation (Layer 1) ─────────────────────────────────────────────


def _math(weight: float = 0.33, val_sequences: int | None = None) -> MidtrainComponent:
    if val_sequences is None:
        return MidtrainComponent(
            name=HIGHQUALITY_NEMO_MATH_KEY,
            step=_highquality_nemo_math_step,
            weight=weight,
        )
    return MidtrainComponent(
        name=HIGHQUALITY_NEMO_MATH_KEY,
        step=_highquality_nemo_math_step,
        weight=weight,
        val_sequences=val_sequences,
    )


def test_spec_accepts_valid_67_33_mix():
    spec = MidtrainMixSpec(
        name="ok",
        pretrain_base=nemotron_mix,
        pretrain_share=0.67,
        midtrain=(_math(0.33),),
    )
    validate_midtrain_spec(spec)


@pytest.mark.parametrize("bad_share", [-0.1, 0.0, 1.0, 1.1])
def test_spec_rejects_pretrain_share_out_of_range(bad_share):
    with pytest.raises(ValueError, match="pretrain_share"):
        MidtrainMixSpec(
            name="bad",
            pretrain_base=nemotron_mix,
            pretrain_share=bad_share,
            midtrain=(_math(0.5),),
        )


def test_spec_rejects_empty_midtrain():
    with pytest.raises(ValueError, match="at least one midtrain"):
        MidtrainMixSpec(
            name="bad",
            pretrain_base=nemotron_mix,
            pretrain_share=0.5,
            midtrain=(),
        )


def test_spec_rejects_share_sum_mismatch():
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        MidtrainMixSpec(
            name="bad",
            pretrain_base=nemotron_mix,
            pretrain_share=0.7,
            midtrain=(_math(0.5),),  # 0.7 + 0.5 = 1.2
        )


def test_spec_rejects_duplicate_midtrain_names():
    c = _math(0.15)
    c2 = MidtrainComponent(
        name=HIGHQUALITY_NEMO_MATH_KEY,
        step=_highquality_nemo_math_step,
        weight=0.18,
    )
    with pytest.raises(ValueError, match="duplicate"):
        MidtrainMixSpec(
            name="bad",
            pretrain_base=nemotron_mix,
            pretrain_share=0.67,
            midtrain=(c, c2),
        )


def test_spec_rejects_overlap_with_pretrain_replay():
    # nemotron_mix has "starcoderdata" as a replay component; if a midtrain
    # component reuses that name, it would silently merge weights.
    pretrain_name = next(iter(nemotron_mix.components))
    overlap = MidtrainComponent(
        name=pretrain_name,
        step=_highquality_nemo_math_step,
        weight=0.33,
    )
    with pytest.raises(ValueError, match="overlap with pretrain replay"):
        MidtrainMixSpec(
            name="bad",
            pretrain_base=nemotron_mix,
            pretrain_share=0.67,
            midtrain=(overlap,),
        )


def test_spec_rejects_zero_weight_component():
    with pytest.raises(ValueError, match="must be in"):
        MidtrainMixSpec(
            name="bad",
            pretrain_base=nemotron_mix,
            pretrain_share=1.0 - 1e-9,
            midtrain=(_math(0.0),),
        )


def test_spec_rejects_negative_val_sequences():
    bad = MidtrainComponent(
        name=HIGHQUALITY_NEMO_MATH_KEY,
        step=_highquality_nemo_math_step,
        weight=0.33,
        val_sequences=-1,
    )
    with pytest.raises(ValueError, match="val_sequences"):
        MidtrainMixSpec(
            name="bad",
            pretrain_base=nemotron_mix,
            pretrain_share=0.67,
            midtrain=(bad,),
        )


def test_spec_accepts_disabled_val_carveout():
    """val_sequences=0 is valid (rare debug knob)."""
    spec = MidtrainMixSpec(
        name="no_val",
        pretrain_base=nemotron_mix,
        pretrain_share=0.67,
        midtrain=(_math(0.33, val_sequences=0),),
    )
    cfg = build_midtrain_lm_data_config(spec)
    assert cfg.num_validation_sequences is None


def test_spec_supports_multiple_midtrain_components():
    """Generalization: multi-component midtraining mix (Recipe-B style)."""
    # Use the same step under two different registry names — purely a config
    # test (we'd never do this in production); the spec just has to accept it.
    spec = MidtrainMixSpec(
        name="multi",
        pretrain_base=nemotron_mix,
        pretrain_share=0.67,
        midtrain=(
            MidtrainComponent("math_a", _highquality_nemo_math_step, 0.18, val_sequences=1000),
            MidtrainComponent("math_b", _highquality_nemo_math_step, 0.15, val_sequences=2000),
        ),
    )
    cfg = build_midtrain_lm_data_config(spec)
    assert cfg.num_validation_sequences == {"math_a": 1000, "math_b": 2000}
    assert cfg.train_weights["math_a"] == pytest.approx(0.18)
    assert cfg.train_weights["math_b"] == pytest.approx(0.15)


# ─── Built-config validation (Layer 2) ──────────────────────────────────────


def test_assert_lm_data_config_safe_accepts_registered_mixes():
    for cfg in MIDTRAINING_MIXES.values():
        assert_lm_data_config_safe(cfg)


def test_assert_lm_data_config_safe_rejects_full_feistel_train_shuffle():
    cfg = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
    bad = dataclasses.replace(cfg, shuffle=True)
    with pytest.raises(ValueError, match="reintroduces"):
        assert_lm_data_config_safe(bad)


def test_assert_lm_data_config_safe_accepts_block_shuffle():
    cfg = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
    custom_block = dataclasses.replace(
        cfg,
        shuffle=BlockShuffleConfig(io_block_size=128, window_blocks=256),
    )
    assert_lm_data_config_safe(custom_block)


def test_assert_lm_data_config_safe_accepts_no_shuffle():
    cfg = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
    no_shuffle = dataclasses.replace(cfg, shuffle=False)
    assert_lm_data_config_safe(no_shuffle)


def test_assert_lm_data_config_safe_rejects_disabled_pre_split_shuffle():
    cfg = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
    bad = dataclasses.replace(cfg, shuffle_before_trainval_split=False)
    with pytest.raises(ValueError, match="shuffle_before_trainval_split"):
        assert_lm_data_config_safe(bad)


def test_assert_lm_data_config_safe_rejects_unknown_val_component():
    cfg = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
    bad = dataclasses.replace(cfg, num_validation_sequences={"not_a_real_component": 100})
    with pytest.raises(ValueError, match="not a registered component"):
        assert_lm_data_config_safe(bad)


def test_assert_lm_data_config_safe_rejects_non_unit_weight_sum():
    cfg = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
    half = {k: v / 2 for k, v in cfg.train_weights.items()}
    bad = dataclasses.replace(cfg, train_weights=half)
    with pytest.raises(ValueError, match="weights sum"):
        assert_lm_data_config_safe(bad)


# ─── Registry integrity ─────────────────────────────────────────────────────


def test_all_registered_mixes_pass_safety_check():
    for cfg in MIDTRAINING_MIXES.values():
        assert_lm_data_config_safe(cfg)


def test_67p_33m_mix_carves_out_math_val_only():
    cfg = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
    assert cfg.num_validation_sequences == {
        HIGHQUALITY_NEMO_MATH_KEY: DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT,
    }


def test_33p_67m_mix_carves_out_math_val_only():
    cfg = MIDTRAINING_MIXES[PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME]
    assert cfg.num_validation_sequences == {
        HIGHQUALITY_NEMO_MATH_KEY: DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT,
    }


def test_full_math_mix_also_has_val_carveout():
    cfg = MIDTRAINING_MIXES[FULL_HIGHQUALITY_NEMO_MATH_NAME]
    assert cfg.num_validation_sequences == {
        HIGHQUALITY_NEMO_MATH_KEY: DEFAULT_VAL_SEQUENCES_PER_MIDTRAIN_COMPONENT,
    }


def test_mix_weights_sum_to_one():
    for name, cfg in MIDTRAINING_MIXES.items():
        weights = cfg.train_weights
        assert isinstance(weights, dict), name
        assert abs(sum(weights.values()) - 1.0) < 1e-6, (name, sum(weights.values()))


def test_67p_33m_share_split_correct():
    cfg = MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]
    weights = cfg.train_weights
    assert weights[HIGHQUALITY_NEMO_MATH_KEY] == pytest.approx(0.33)
    pretrain_total = sum(v for k, v in weights.items() if k != HIGHQUALITY_NEMO_MATH_KEY)
    assert pretrain_total == pytest.approx(0.67)


def test_33p_67m_share_split_correct():
    cfg = MIDTRAINING_MIXES[PRETRAIN_33P_MATH_67P_HIGHQUALITY_NEMO_MATH_NAME]
    weights = cfg.train_weights
    assert weights[HIGHQUALITY_NEMO_MATH_KEY] == pytest.approx(0.67)
    pretrain_total = sum(v for k, v in weights.items() if k != HIGHQUALITY_NEMO_MATH_KEY)
    assert pretrain_total == pytest.approx(0.33)


def test_auto_build_caches_disabled():
    for name, cfg in MIDTRAINING_MIXES.items():
        assert cfg.auto_build_caches is False, name


def test_block_shuffle_default_preserved():
    # The Levanter default after #5246 is BlockShuffleConfig; our build path
    # must not silently switch back to full Feistel or no-shuffle.
    for name, cfg in MIDTRAINING_MIXES.items():
        assert isinstance(cfg.shuffle, BlockShuffleConfig), (name, type(cfg.shuffle).__name__)


def test_shuffle_before_trainval_split_is_true():
    for name, cfg in MIDTRAINING_MIXES.items():
        assert cfg.shuffle_before_trainval_split is True, name


def test_midtraining_mix_by_name_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown midtraining mix"):
        midtraining_mix_by_name("does_not_exist")


def test_midtraining_mix_by_name_returns_registered():
    cfg = midtraining_mix_by_name(PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME)
    assert cfg is MIDTRAINING_MIXES[PRETRAIN_67P_MATH_33P_HIGHQUALITY_NEMO_MATH_NAME]


def test_registered_specs_match_registered_mixes():
    """Every spec should appear in MIDTRAINING_MIXES; the full-math mix is
    spec-less but still in the registry."""
    for name in MIDTRAIN_SPECS:
        assert name in MIDTRAINING_MIXES
    # Full-math is registered without a spec; make sure it's still present.
    assert FULL_HIGHQUALITY_NEMO_MATH_NAME in MIDTRAINING_MIXES
