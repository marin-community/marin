# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""The frozen-swarm launcher takes seeds and pool fractions from the design table and skips reused rows."""

import hashlib
import json

import pytest

from experiments.domain_phase_mix import launch_delphi_apriori_swarm_3e18 as launcher
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import DOMAIN_NAMES, PHASE_NAMES

SPEC_KWARGS = dict(
    train_steps=1000,
    batch_size=128,
    model_hidden_dim=1024,
    model_layers=12,
    non_embedding_params=100_000_000,
    total_trainable_params=110_000_000,
    tensor_parallel_size=1,
    tpu_type="v5p-8",
    tpu_region="us-east5",
    tpu_zone="us-east5-a",
)


def _row(
    name: str,
    *,
    source: str = "new",
    wave: str = "pilot",
    data_seed: int = 662_009,
    subset_seed: int | None = None,
    fractions: dict[str, float] | None = None,
) -> dict[str, str]:
    weights = {domain: 1.0 / len(DOMAIN_NAMES) for domain in DOMAIN_NAMES}
    row = {
        "run_name": name,
        "block": "test",
        "source": source,
        "wave": wave,
        "data_seed": str(data_seed),
        "trainer_seed": "0",
        "subset_seed": str(data_seed if subset_seed is None else subset_seed),
    }
    for phase in PHASE_NAMES:
        for domain in DOMAIN_NAMES:
            row[f"{phase}_{domain}"] = repr(weights[domain])
    for domain in DOMAIN_NAMES:
        row[f"pool_fraction_{domain}"] = repr((fractions or {}).get(domain, 1.0))
    return row


def test_specs_carry_paired_seeds_and_pool_fractions_and_skip_reused_rows():
    rows = [
        _row("reused", source="reused_panel", data_seed=7_141_000),
        _row("full_support"),
        _row("half_pool", fractions={"dolmino_synth_qa": 0.5}, data_seed=662_010),
        _row("second_wave", wave="full"),
    ]
    pilot = launcher.run_specs_from_rows(rows, wave="pilot", **SPEC_KWARGS)
    assert [spec.run_name for spec in pilot] == ["full_support", "half_pool"]
    assert pilot[0].simulated_epoch_pool_fractions is None
    assert pilot[1].simulated_epoch_pool_fractions == {"dolmino_synth_qa": 0.5}
    assert pilot[1].data_seed == pilot[1].simulated_epoch_subset_seed == 662_010
    assert pilot[0].trainer_seed == 0 and pilot[0].realized_train_tokens == 1000 * 128 * launcher.swarm.SEQ_LEN_DELPHI
    assert pilot[0].phase_weights[PHASE_NAMES[0]] == pilot[0].phase_weights[PHASE_NAMES[1]]
    everything = launcher.run_specs_from_rows(rows, wave="full", **SPEC_KWARGS)
    assert [spec.run_name for spec in everything] == ["full_support", "half_pool", "second_wave"]
    by_name = {spec.run_name: spec for spec in everything}
    assert all(by_name[spec.run_name] == spec for spec in pilot)  # identities do not depend on the wave
    assert [spec.run_id for spec in everything] == [
        launcher.RUN_ID_BASE + 1,
        launcher.RUN_ID_BASE + 2,
        launcher.RUN_ID_BASE + 3,
    ]


def test_only_the_frozen_table_is_accepted(tmp_path):
    table = tmp_path / "swarm_mixtures.csv"
    table.write_text("run_name\nx\n")
    with pytest.raises(FileNotFoundError, match=r"manifest\.json"):
        launcher.frozen_design_sha256(table)
    (tmp_path / "manifest.json").write_text(json.dumps({"mixtures_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="does not match the frozen manifest"):
        launcher.frozen_design_sha256(table)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"mixtures_sha256": hashlib.sha256(table.read_bytes()).hexdigest()})
    )
    assert launcher.frozen_design_sha256(table) == hashlib.sha256(table.read_bytes()).hexdigest()


def test_unpaired_seeds_and_bad_fractions_are_rejected():
    with pytest.raises(ValueError, match="paired seeds"):
        launcher.run_specs_from_rows([_row("unpaired", subset_seed=1)], wave="pilot", **SPEC_KWARGS)
    with pytest.raises(ValueError, match="outside"):
        launcher.run_specs_from_rows([_row("bad", fractions={"dolmino_synth_qa": 1.5})], wave="pilot", **SPEC_KWARGS)
