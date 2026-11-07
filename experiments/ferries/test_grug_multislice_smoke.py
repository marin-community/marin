# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from levanter.store.cache import CacheLedger
from marin.execution.lazy import materialized_config

from experiments.ferries import grug_multislice_smoke as smoke

LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
NEMOTRON_PIN = "tokenized/nemotron_cc/hq_actual-5af4cc"
STARCODER_PIN = "tokenized/starcoderdata-12f018/"
ALL_PINS = {
    "nemotron_cc/hq_actual-llama3": NEMOTRON_PIN,
    "nemotron_cc/hq_synth-llama3": "tokenized/nemotron_cc/hq_synth-3525e2",
    "nemotron_cc/medium_high-llama3": "tokenized/nemotron_cc/medium_high-d21701",
    "nemotron_cc/medium-llama3": "tokenized/nemotron_cc/medium-d86506",
    "nemotron_cc/medium_low-llama3": "tokenized/nemotron_cc/medium_low-0fdb07",
    "nemotron_cc/low_actual-llama3": "tokenized/nemotron_cc/low_actual-cb3f2c",
    "nemotron_cc/low_synth-llama3": "tokenized/nemotron_cc/low_synth-3c57b3",
    "starcoderdata-llama3": STARCODER_PIN,
    "proofpile_2-llama3": "tokenized/proofpile_2-4a35c7/",
}
LEGACY_INFO = {"name": "tokenized/cache", "config": {"tokenizer": LLAMA3_TOKENIZER}}
STEP_RUNNER_STUB = {"executor_version": "step_runner", "config": {"fingerprint": "abc"}}


def _write_cache(root, pin, *, status=None, info=None, train_payload=False, finished=True):
    path = root / pin.removesuffix("/")
    path.mkdir(parents=True)
    if status is not None:
        (path / ".executor_status").write_text(status)
    if info is not None:
        (path / ".executor_info").write_text(json.dumps(info))
    if train_payload:
        (path / "train" / "input_ids").mkdir(parents=True)
        (path / "train" / "shard_ledger.json").write_text(
            CacheLedger(total_num_rows=0, shard_rows={}, is_finished=finished).to_json()
        )


def _write_failed_local_and_healthy_remote(east5, central2, pin):
    _write_cache(east5, pin, status="FAILED", info=STEP_RUNNER_STUB)
    _write_cache(central2, pin, info=LEGACY_INFO, train_payload=True)


def test_failed_local_pin_resolves_to_healthy_remote_copy(tmp_path):
    east5, central2 = tmp_path / "east5", tmp_path / "central2"
    _write_failed_local_and_healthy_remote(east5, central2, NEMOTRON_PIN)

    resolved = smoke.resolve_available_pin(NEMOTRON_PIN, [str(east5), str(central2)])

    assert resolved == f"{central2}/{NEMOTRON_PIN}"


def test_successful_local_copy_wins_and_trailing_slash_pin_matches(tmp_path):
    east5, central2 = tmp_path / "east5", tmp_path / "central2"
    _write_cache(east5, STARCODER_PIN, status="SUCCESS", info=LEGACY_INFO)
    _write_cache(central2, STARCODER_PIN, info=LEGACY_INFO, train_payload=True)

    resolved = smoke.resolve_available_pin(STARCODER_PIN, [str(east5), str(central2)])

    assert resolved == f"{east5}/tokenized/starcoderdata-12f018"


def test_failed_and_stub_only_copies_are_never_chosen(tmp_path):
    east5, central2 = tmp_path / "east5", tmp_path / "central2"
    _write_cache(east5, NEMOTRON_PIN, info=STEP_RUNNER_STUB, train_payload=True)
    _write_cache(central2, NEMOTRON_PIN, status="FAILED", info=LEGACY_INFO, train_payload=True)

    with pytest.raises(FileNotFoundError, match=NEMOTRON_PIN):
        smoke.resolve_available_pin(NEMOTRON_PIN, [str(east5), str(central2)])


def test_build_adopts_each_training_cache_from_its_usable_copy(tmp_path, monkeypatch):
    east5, central2 = tmp_path / "east5", tmp_path / "central2"
    for pin in ALL_PINS.values():
        _write_failed_local_and_healthy_remote(east5, central2, pin)
    monkeypatch.setenv("MARIN_PREFIX", str(east5))
    monkeypatch.delenv("GRUG_MULTISLICE_OUTPUT_PATH", raising=False)
    monkeypatch.setattr(smoke, "_pin_candidate_prefixes", lambda: [str(east5), str(central2)])

    checkpoint = smoke.build()
    data = materialized_config(checkpoint, str(east5)).data

    expected_sources = {name: f"{central2}/{pin.removesuffix('/')}" for name, pin in ALL_PINS.items()}
    assert {dep.name: dep.adopt_source for dep in checkpoint.deps} == expected_sources
    assert all(dep.override_path is None and dep.version == "2026.06.28" for dep in checkpoint.deps)
    assert {name: component.cache_dir for name, component in data.components.items()} == expected_sources
    assert set(data.train_weights) == set(ALL_PINS)
    assert data.tokenizer == LLAMA3_TOKENIZER


@pytest.mark.parametrize("ledger_present", [False, True])
def test_incomplete_local_cache_falls_back_to_finished_copy(tmp_path, ledger_present):
    east5, central2 = tmp_path / "east5", tmp_path / "central2"
    _write_cache(east5, NEMOTRON_PIN, info=LEGACY_INFO, train_payload=True, finished=False)
    if not ledger_present:
        (east5 / NEMOTRON_PIN / "train" / "shard_ledger.json").unlink()
    _write_cache(central2, NEMOTRON_PIN, info=LEGACY_INFO, train_payload=True)

    assert smoke.resolve_available_pin(NEMOTRON_PIN, [str(east5), str(central2)]) == f"{central2}/{NEMOTRON_PIN}"


@pytest.mark.parametrize("filename", [".artifact.json", ".executor_info"])
@pytest.mark.parametrize("contents", ['{"config":', '{"name": 123, "config": "invalid"}'])
def test_corrupt_local_metadata_falls_back_to_healthy_copy(tmp_path, filename, contents):
    east5, central2 = tmp_path / "east5", tmp_path / "central2"
    _write_cache(east5, NEMOTRON_PIN, info=LEGACY_INFO, train_payload=True)
    (east5 / NEMOTRON_PIN / filename).write_text(contents)
    _write_cache(central2, NEMOTRON_PIN, info=LEGACY_INFO, train_payload=True)

    assert smoke.resolve_available_pin(NEMOTRON_PIN, [str(east5), str(central2)]) == f"{central2}/{NEMOTRON_PIN}"
