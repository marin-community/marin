# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from pathlib import Path

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmExample
from levanter.trainer import TrainerConfig
from levanter.utils.multihost import gather_host_scalar
from marin.training import chat_token_accounting
from marin.training.chat_token_accounting import RowAccounting, RowIdentity, TurnTokens
from marin.training.frozen_batch_audit import FrozenBatchAudit, audit_training_iterator, checked_update


def fixture_stream():
    config = TrainLmConfig(data_seed=17, train_seq_len=4, trainer=TrainerConfig(train_batch_size=2))
    tokens = np.array([[10, 0, 0, 0], [20, 0, 0, 0]], np.int32)
    masks = np.array([[1, 0, 0, 0], [1, 1, 1, 0]], np.int32)
    rows, fingerprints = [], {}
    for index, label in enumerate(("selected", "preserve")):
        token_sha = hashlib.sha256(tokens[index].astype("<i4").tobytes()).hexdigest()
        mask_sha = hashlib.sha256(masks[index].astype("<i4").tobytes()).hexdigest()
        identifier = str(index)
        fingerprints[token_sha, mask_sha] = identifier
        rows.append(
            RowAccounting(
                RowIdentity(identifier, identifier, identifier, label, label),
                4,
                "template-pin",
                token_sha,
                mask_sha,
                (TurnTokens(1, label, int(masks[index].sum())),),
            )
        )
    first = {
        "occurrences": 2,
        "unique_ids": 2,
        "family_rows": {"selected": 1, "preserve": 1},
        "family_tokens": {"selected": 1, "preserve": 3},
        "class_tokens": {"selected": 1, "preserve": 3},
        "supervised_tokens": 4,
        "selected_tokens": 1,
        "selected_fraction": {"numerator": 1, "denominator": 4, "value": 0.25},
    }
    second = {
        "occurrences": 2,
        "unique_ids": 1,
        "family_rows": {"selected": 2},
        "family_tokens": {"selected": 2},
        "class_tokens": {"selected": 2},
        "supervised_tokens": 2,
        "selected_tokens": 2,
        "selected_fraction": {"numerator": 2, "denominator": 2, "value": 1.0},
    }
    pooled = {
        "occurrences": 4,
        "unique_ids": 2,
        "family_rows": {"selected": 3, "preserve": 1},
        "family_tokens": {"selected": 3, "preserve": 3},
        "class_tokens": {"selected": 3, "preserve": 3},
        "supervised_tokens": 6,
        "selected_tokens": 3,
        "selected_fraction": {"numerator": 3, "denominator": 6, "value": 0.5},
    }
    prefixes = {
        "2": first | {"mean_update_selected_ce_coefficient_mass": {"numerator": 1, "denominator": 4, "value": 0.25}},
        "4": pooled | {"mean_update_selected_ce_coefficient_mass": {"numerator": 5, "denominator": 8, "value": 0.625}},
    }
    frozen = FrozenBatchAudit(
        freeze={"updates": [first, second], "realized_prefixes": prefixes},
        data_config=draccus.encode(config.data),
        ordered=[{"id": i, "supervised_tokens": n} for i, n in (("0", 1), ("1", 3), ("0", 1), ("0", 1))],
        rows=tuple(rows),
        fingerprints=fingerprints,
        batch_size=2,
        sequence_length=4,
        data_seed=17,
        selected_classes=frozenset({"selected"}),
        source_pins={"fixture": "caller-verified-source-pin"},
        accounting_module_sha256=hashlib.sha256(Path(chat_token_accounting.__file__).read_bytes()).hexdigest(),
    )

    def batch(ids):
        return LmExample(
            hax.named(jnp.asarray(tokens[ids]), ("batch", "position")),
            hax.named(jnp.asarray(masks[ids], dtype=jnp.float32), ("batch", "position")),
        )

    return config, frozen, batch([0, 1]), batch([0, 0])


def test_actual_prefix_replays_same_loader_batches_and_distinguishes_ce_mass(tmp_path):
    config, frozen, first, second = fixture_stream()
    sentinel = dataclasses.replace(first)
    original = iter([first, second, sentinel])
    path = tmp_path / "audit.json"
    replay, report = audit_training_iterator(
        original, config=config, completed_updates=0, frozen=frozen, receipt_path=str(path)
    )
    for index, expected in enumerate((first, second)):
        consumed = next(replay)
        assert consumed is expected
        assert checked_update(consumed, index, frozen)["ids"] == (["0", "1"] if index == 0 else ["0", "0"])
    with pytest.raises(ValueError, match="Dataset audit failed"):
        checked_update(second, -1, frozen)
    assert next(replay) is sentinel
    with pytest.raises(StopIteration):
        next(replay)
    saved = json.loads(path.read_text())
    assert saved["prefixes"] == frozen.freeze["realized_prefixes"]
    assert saved["prefixes"]["4"]["selected_fraction"]["value"] == 0.5
    assert saved["prefixes"]["4"]["mean_update_selected_ce_coefficient_mass"]["value"] == 0.625
    assert report["same_iterator_batches_buffered"] == 2 and report["global_batch_size"] == 2


@pytest.mark.parametrize("change", ["order", "mask", "family", "overflow"])
def test_corrupted_native_batch_or_frozen_accounting_fails_before_receipt(tmp_path, change):
    config, frozen, first, second = fixture_stream()
    if change == "order":
        first = dataclasses.replace(
            first,
            tokens=hax.named(first.tokens.array[::-1], ("batch", "position")),
            loss_weight=hax.named(first.loss_weight.array[::-1], ("batch", "position")),
        )
    elif change == "mask":
        first = dataclasses.replace(first, loss_weight=hax.named(jnp.zeros((2, 4)), ("batch", "position")))
    elif change == "overflow":
        with jax.enable_x64(True):
            changed = first.tokens.array.astype(jnp.int64) + 2**32
        first = dataclasses.replace(first, tokens=hax.named(changed, ("batch", "position")))
    else:
        frozen.freeze["updates"][0]["family_tokens"]["selected"] = 2
    path = tmp_path / "audit.json"
    with pytest.raises(ValueError, match="Dataset audit failed"):
        audit_training_iterator(
            iter([first, second]), config=config, completed_updates=0, frozen=frozen, receipt_path=str(path)
        )
    assert not path.exists()


def test_peer_failure_is_synchronized_at_collective_io_before_loading(tmp_path, monkeypatch):
    config, frozen, first, second = fixture_stream()

    def peer_failure(value, *, tiled):
        assert tiled
        return np.concatenate([value, np.array([[1, 0]], np.int32)], axis=0)

    monkeypatch.setattr(multihost_utils, "process_allgather", peer_failure)
    batches = iter([first, second])
    with pytest.raises(ValueError, match="at least one rank"):
        audit_training_iterator(
            batches, config=config, completed_updates=0, frozen=frozen, receipt_path=str(tmp_path / "audit.json")
        )
    assert next(batches) is first
    assert not (tmp_path / "audit.json").exists()


def test_different_plan_digest_stops_before_mismatched_collective_horizons(tmp_path, monkeypatch):
    config, frozen, first, second = fixture_stream()

    def different_plan(value, *, tiled):
        peer = np.array(value, copy=True)
        if peer.dtype == np.uint8:
            peer.flat[0] ^= 1
        return np.concatenate([value, peer], axis=0)

    monkeypatch.setattr(multihost_utils, "process_allgather", different_plan)
    batches = iter([first, second])
    with pytest.raises(ValueError, match="plan differs across ranks"):
        audit_training_iterator(
            batches, config=config, completed_updates=0, frozen=frozen, receipt_path=str(tmp_path / "audit.json")
        )
    assert next(batches) is first
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize("value", [np.int64(2**40 + 17), np.int64(-(2**40) + 17), np.float64(2**40 + 0.25)])
def test_64bit_host_values_survive_a_jax_collective_with_x64_disabled(monkeypatch, value):
    def wire(values, *, tiled):
        # Simulate two ranks crossing the real JAX dtype boundary. Without the
        # word codec, jnp.asarray would narrow int64/float64 when x64 is off.
        transported = np.asarray(jnp.asarray(values))
        return np.concatenate([transported, transported], axis=0)

    monkeypatch.setattr(multihost_utils, "process_allgather", wire)
    with jax.enable_x64(False):
        result = gather_host_scalar(value)
    np.testing.assert_array_equal(result, np.array([value, value]))
    assert result.dtype == value.dtype
