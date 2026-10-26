# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.models.lm_model import LmExample
from marin.training.chat_token_accounting import RowAccounting, RowIdentity, TurnTokens

from experiments.post_training.identity_context_distill import dataset_audit
from experiments.post_training.identity_context_distill.dataset_audit import (
    DatasetAuditInputs,
    audit_training_iterator,
    checked_update,
)
from experiments.post_training.identity_context_distill.sft import identity_sft_config
from experiments.post_training.identity_context_distill.test_sft import source_arguments


def fixture_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(jax, "device_count", lambda: 32)
    config = identity_sft_config(
        hosts=1,
        moe_capacity_factor=4.0,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation="torch_order",
        dataset_uri="data",
        cache_uri="cache",
        output_uri=str(tmp_path),
        run_id="dataset-test",
        seed=310,
        **source_arguments(),
    )
    tokens = np.zeros((8, 8192), dtype=np.int32)
    masks = np.zeros_like(tokens)
    rows, ordered, fingerprints = [], [], {}
    for index in range(8):
        tokens[index, 0] = index + 1
        masks[index, : index + 1] = 1
        identifier = f"row-{index}"
        label = "identity_off" if index == 0 else "replay"
        token_sha = hashlib.sha256(tokens[index].astype("<i4").tobytes()).hexdigest()
        mask_sha = hashlib.sha256(masks[index].astype("<i4").tobytes()).hexdigest()
        fingerprints[token_sha, mask_sha] = identifier
        rows.append(
            RowAccounting(
                RowIdentity(identifier, identifier, identifier, label, label),
                8192,
                "template",
                token_sha,
                mask_sha,
                (TurnTokens(1, label, index + 1),),
            )
        )
        ordered.append({"id": identifier, "supervised_tokens": index + 1})

    def expected(repeats):
        return {
            "occurrences": 8 * repeats,
            "unique_ids": 8,
            "family_rows": {"identity_off": repeats, "replay": 7 * repeats},
            "family_tokens": {"identity_off": repeats, "replay": 35 * repeats},
            "class_tokens": {"identity_off": repeats, "replay": 35 * repeats},
            "supervised_tokens": 36 * repeats,
            "identity_tokens": repeats,
            "identity_fraction": {"numerator": repeats, "denominator": 36 * repeats, "value": 1 / 36},
        }

    prefixes = {}
    for size in (128, 512):
        prefixes[str(size)] = expected(size // 8) | {
            "mean_update_identity_ce_coefficient_mass": {"numerator": 1, "denominator": 36, "value": 1 / 36}
        }
    frozen = DatasetAuditInputs(
        {"updates": [expected(1)] * 64, "realized_prefixes": prefixes},
        draccus.encode(config.data),
        ordered * 64,
        tuple(rows),
        fingerprints,
    )
    batch = LmExample(
        hax.named(jnp.asarray(tokens), ("batch", "position")),
        hax.named(jnp.asarray(masks, dtype=jnp.float32), ("batch", "position")),
    )
    return config, frozen, batch


def test_same_loader_objects_replayed_and_consumed_batches_checked(tmp_path, monkeypatch):
    config, frozen, batch = fixture_stream(tmp_path, monkeypatch)
    sentinel = dataclasses.replace(batch)
    original = iter([batch] * 64 + [sentinel])
    replay, report = audit_training_iterator(
        original, config=config, completed_updates=0, frozen=frozen, receipt_path=str(tmp_path / "audit.json")
    )
    assert json.loads((tmp_path / "audit.json").read_text())["prefixes"] == frozen.freeze["realized_prefixes"]
    for update in range(64):
        consumed = next(replay)
        assert consumed is batch
        receipt = checked_update(consumed, update, frozen)
        assert receipt["ids"] == [f"row-{index}" for index in range(8)]
        assert receipt["identity_tokens"] == 1 and receipt["supervised_tokens"] == 36
    assert next(replay) is sentinel
    assert len(report["updates"]) == 64
    with pytest.raises(StopIteration):
        next(replay)


@pytest.mark.parametrize("change", ["order", "mask", "family"])
def test_corrupted_actual_stream_fails_before_receipt(tmp_path, monkeypatch, change):
    config, frozen, batch = fixture_stream(tmp_path, monkeypatch)
    if change == "order":
        batch = dataclasses.replace(batch, tokens=hax.named(batch.tokens.array[::-1], ("batch", "position")))
    elif change == "mask":
        batch = dataclasses.replace(batch, loss_weight=hax.named(jnp.zeros((8, 8192)), ("batch", "position")))
    else:
        frozen.freeze["updates"][0]["family_tokens"]["identity_off"] = 2
    with pytest.raises(ValueError, match="Dataset audit failed"):
        audit_training_iterator(
            iter([batch] * 64),
            config=config,
            completed_updates=0,
            frozen=frozen,
            receipt_path=str(tmp_path / "audit.json"),
        )
    assert not (tmp_path / "audit.json").exists()


def test_peer_failure_stops_successful_rank(tmp_path, monkeypatch):
    _, frozen, batch = fixture_stream(tmp_path, monkeypatch)
    monkeypatch.setattr(dataset_audit, "gather_host_scalar", lambda value: np.array([value, 1]))
    with pytest.raises(ValueError, match="at least one rank"):
        checked_update(batch, 0, frozen)


def test_check_exception_is_synchronized_before_receipt(tmp_path, monkeypatch):
    config, frozen, batch = fixture_stream(tmp_path, monkeypatch)
    statuses = []

    def fail_check(*args):
        raise RuntimeError("batch check failed before its comparison")

    def gather_status(value):
        statuses.append(value)
        return np.array([value, 0])

    monkeypatch.setattr(dataset_audit, "check_update", fail_check)
    monkeypatch.setattr(dataset_audit, "gather_host_scalar", gather_status)
    with pytest.raises(ValueError, match="at least one rank") as failure:
        audit_training_iterator(
            iter([batch] * 64),
            config=config,
            completed_updates=0,
            frozen=frozen,
            receipt_path=str(tmp_path / "audit.json"),
        )
    assert isinstance(failure.value.__cause__, RuntimeError)
    assert statuses == [0, 0, 1]
    assert not (tmp_path / "audit.json").exists()
