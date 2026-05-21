# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guard that the small-base K=0.20 sweep produces the SAME math val partition
as the 1e21/1e22 K=0.20 sweep.

This is the load-bearing test for cross-scale loss comparability. If it
ever fails, the math val set has drifted between scales and any val-loss
plot mixing 3e18..2e20 with 1e21/1e22 is meaningless.
"""

import json
from pathlib import Path

import pytest
from marin.midtraining import (
    CPT_DEFAULT_DECAY,
    CPT_DEFAULT_WARMUP_FRACTION,
    resolve_midtrain_spec,
    validate_midtrain_spec,
)
from marin.midtraining.levanter_config import render_train_lm_config

from experiments.midtrain_specs import (
    DELPHI_MIDTRAIN_MIXES,
    MATH_CACHE_DIR,
    MATH_VAL_SEQUENCES,
    load_legacy_data_section,
)
from experiments.midtrain_specs.delphi_small_cpt_k020 import build_spec

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "experiments" / "midtrain_specs" / "data_sections"

VAL_DETERMINING_KEYS = (
    "components",  # includes math cache_dir and split=validation
    "num_validation_sequences",
    "shuffle",
    "shuffle_before_trainval_split",
    "permutation_type",
    "block_cross_document_attention",
    "mixture_block_size",
    "enforce_eos",
    "cache_options",
    "tokenizer",
    "vocab_size",
    "stop_strategy",
    "train_weights",
)


@pytest.mark.parametrize("mix", DELPHI_MIDTRAIN_MIXES)
def test_reference_data_section_carries_math_val_carve_out(mix: str):
    """The captured 1e21 reference must declare exactly 12500 math val sequences."""
    ref = json.loads((REFERENCE_DIR / f"{mix}.json").read_text(encoding="utf-8"))
    assert ref["num_validation_sequences"] == {"nemotron_cc_math_v1/4plus": MATH_VAL_SEQUENCES}
    assert ref["shuffle_before_trainval_split"] is True
    assert ref["shuffle"]["perm_type"] == "feistel"
    assert ref["permutation_type"] == "feistel"
    assert ref["components"]["nemotron_cc_math_v1/4plus"]["cache_dir"] == MATH_CACHE_DIR
    assert ref["components"]["nemotron_cc_math_v1/4plus"]["split"] == "validation"


def test_three_mixes_share_identical_val_carve_out():
    """All three mixes must point at the same math cache + same val carve-out."""
    ref_p33 = load_legacy_data_section("p33m67")
    ref_p50 = load_legacy_data_section("p50m50")
    ref_p67 = load_legacy_data_section("p67m33")
    for key in (
        "num_validation_sequences",
        "shuffle",
        "shuffle_before_trainval_split",
        "permutation_type",
        "block_cross_document_attention",
        "mixture_block_size",
        "enforce_eos",
        "cache_options",
        "tokenizer",
        "vocab_size",
        "stop_strategy",
    ):
        assert (
            ref_p33[key] == ref_p50[key] == ref_p67[key]
        ), f"Mixes differ on {key!r}: {ref_p33.get(key)} / {ref_p50.get(key)} / {ref_p67.get(key)}"
    # Math component identity (cache_dir + split) must match across mixes.
    for ref in (ref_p33, ref_p50, ref_p67):
        comp = ref["components"]["nemotron_cc_math_v1/4plus"]
        assert comp["cache_dir"] == MATH_CACHE_DIR
        assert comp["split"] == "validation"


@pytest.mark.parametrize("mix", DELPHI_MIDTRAIN_MIXES)
@pytest.mark.parametrize("base_key", ["3e18", "9e18", "2e19", "3e19", "9e19", "2e20", "3e20"])
def test_small_base_rendered_data_section_bit_identical_to_reference(base_key: str, mix: str):
    """For every (base, mix) cell, the rendered ``data:`` block must equal the reference."""
    spec = build_spec(base_key=base_key, mix=mix, lr_factor=0.5)
    resolved = resolve_midtrain_spec(spec)
    validate_midtrain_spec(resolved)
    rendered = render_train_lm_config(resolved)
    ref = json.loads((REFERENCE_DIR / f"{mix}.json").read_text(encoding="utf-8"))
    assert (
        rendered["data"] == ref
    ), f"Rendered data section for base={base_key!r} mix={mix!r} drifted from 1e21 reference."


def test_small_base_cpt_schedule_is_triangular_not_wsd():
    """CPT should match legacy warmup -> decay, with no stable LR plateau."""
    spec = build_spec(base_key="3e18", mix="p33m67", lr_factor=0.5)
    resolved = resolve_midtrain_spec(spec)
    validate_midtrain_spec(resolved)
    rendered = render_train_lm_config(resolved)
    optimizer = rendered["optimizer"]

    assert optimizer["warmup"] == CPT_DEFAULT_WARMUP_FRACTION
    assert optimizer["decay"] is CPT_DEFAULT_DECAY
    assert optimizer["lr_schedule"] == "linear"
    assert rendered["trainer"]["num_train_steps"] == 7400


def test_small_base_build_spec_rejects_disallowed_tpu():
    """The per-base TPU allowlist must protect imported drivers, not just CLI calls."""
    with pytest.raises(RuntimeError, match="not in the allowlist"):
        build_spec(base_key="3e18", mix="p33m67", lr_factor=0.5, tpu_type="v5p-64")


def test_9e18_allows_v6e_benchmark_tpus_with_named_suffix():
    spec = build_spec(
        base_key="9e18",
        mix="p33m67",
        lr_factor=0.5,
        tpu_type="v6e-4",
        run_suffix="bench-v6e4",
    )

    assert spec.compute.tpu_type == "v6e-4"
    assert spec.run.run_id == "delphi-9e18-p33m67-k0p20-lr50-bench-v6e4-a001"


def test_probe_mode_uses_fixed_steps_and_visible_tags():
    spec = build_spec(
        base_key="3e19",
        mix="p67m33",
        lr_factor=0.5,
        tpu_type="v6e-8",
        run_suffix="probe-v6e8-20s",
        probe_steps=20,
    )
    resolved = resolve_midtrain_spec(spec)
    validate_midtrain_spec(resolved)
    rendered = render_train_lm_config(resolved)

    assert spec.compute.tpu_type == "v6e-8"
    assert spec.run.run_id == "delphi-3e19-p67m33-k0p20-lr50-probe-v6e8-20s-a001"
    assert rendered["trainer"]["num_train_steps"] == 20
    assert "probe:throughput-hbm" in rendered["trainer"]["tracker"]["tags"]
    assert "probe_steps:20" in rendered["trainer"]["tracker"]["tags"]
    assert "do_not_compare:quality" in rendered["trainer"]["tracker"]["tags"]


def test_probe_mode_requires_probe_suffix():
    with pytest.raises(ValueError, match="run_suffix starting with 'probe-'"):
        build_spec(
            base_key="2e19",
            mix="p67m33",
            lr_factor=0.5,
            tpu_type="v6e-4",
            run_suffix="bench-v6e4",
            probe_steps=20,
        )


def test_run_suffix_rejects_attempt_like_suffix():
    with pytest.raises(ValueError, match="must not end with an attempt suffix"):
        build_spec(base_key="9e18", mix="p33m67", lr_factor=0.5, run_suffix="bench-a001")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"base_key": "1e20", "mix": "p33m67", "lr_factor": 0.5}, "Unknown base"),
        ({"base_key": "3e18", "mix": "p99m01", "lr_factor": 0.5}, "Unknown mix"),
        ({"base_key": "3e18", "mix": "p33m67", "lr_factor": 0.42}, "Unknown lr_factor"),
    ],
)
def test_small_base_build_spec_validates_selectors(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        build_spec(**kwargs)


@pytest.mark.parametrize("mix", DELPHI_MIDTRAIN_MIXES)
def test_train_weights_distinguish_only_mix(mix: str):
    """Mix identity should differ ONLY in train_weights, not in val-determining fields."""
    ref = json.loads((REFERENCE_DIR / f"{mix}.json").read_text(encoding="utf-8"))
    canonical = json.loads((REFERENCE_DIR / "p33m67.json").read_text(encoding="utf-8"))
    for key in VAL_DETERMINING_KEYS:
        if key == "train_weights":
            # Only this is allowed to differ.
            continue
        assert ref[key] == canonical[key], f"Mix {mix!r} differs from p33m67 on val-determining key {key!r}"


def test_spec_rejects_data_section_override_without_provenance():
    """data_section_override requires data_section_provenance for audit."""
    from marin.midtraining import (
        LLAMA3_TOKENIZER,
        BudgetPolicy,
        CheckpointSourceKind,
        ComputeProfile,
        CptInit,
        CptMode,
        MidtrainSpec,
        build_run_identity,
    )

    from tests.midtraining._fixtures import FAKE_1E21, make_model_config, make_optimizer_config

    section = load_legacy_data_section("p33m67")
    run = build_run_identity(
        logical_cell_id="x",
        attempt=1,
        output_region_name="us-east5",
        wandb_project="delphi-midtraining",
    )
    with pytest.raises(ValueError, match="data_section_provenance"):
        MidtrainSpec(
            base=FAKE_1E21,
            run=run,
            compute=ComputeProfile(tpu_type="v5p-8", batch_size=8, regions=("us-east5",)),
            mode=CptMode(
                init=CptInit(
                    source_kind=CheckpointSourceKind.HF_WEIGHTS,
                    hf_repo=FAKE_1E21.hf_repo,
                    hf_revision=FAKE_1E21.hf_revision,
                ),
                budget=BudgetPolicy.pretrain_fraction(0.20),
            ),
            tokenizer=LLAMA3_TOKENIZER,
            model_config=make_model_config(FAKE_1E21),
            optimizer_config=make_optimizer_config(),
            data_section_override=section,
            # data_section_provenance missing — should fail
        )


def test_spec_rejects_both_data_sources():
    """data_manifest_uri and data_section_override are mutually exclusive."""
    from marin.midtraining import (
        LLAMA3_TOKENIZER,
        BudgetPolicy,
        CheckpointSourceKind,
        ComputeProfile,
        CptInit,
        CptMode,
        MidtrainSpec,
        build_run_identity,
    )

    from tests.midtraining._fixtures import FAKE_1E21, make_model_config, make_optimizer_config

    section = load_legacy_data_section("p33m67")
    run = build_run_identity(
        logical_cell_id="x",
        attempt=1,
        output_region_name="us-east5",
        wandb_project="delphi-midtraining",
    )
    with pytest.raises(ValueError, match="exactly one"):
        MidtrainSpec(
            base=FAKE_1E21,
            run=run,
            compute=ComputeProfile(tpu_type="v5p-8", batch_size=8, regions=("us-east5",)),
            mode=CptMode(
                init=CptInit(
                    source_kind=CheckpointSourceKind.HF_WEIGHTS,
                    hf_repo=FAKE_1E21.hf_repo,
                    hf_revision=FAKE_1E21.hf_revision,
                ),
                budget=BudgetPolicy.pretrain_fraction(0.20),
            ),
            tokenizer=LLAMA3_TOKENIZER,
            model_config=make_model_config(FAKE_1E21),
            optimizer_config=make_optimizer_config(),
            data_manifest_uri="gs://marin-us-east5/midtrain-manifests/data/p33m67/abc.json",
            data_section_override=section,
            data_section_provenance="legacy:test",
        )
