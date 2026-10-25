# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the frozen mixture on the actual iterator, before and during learning."""

import hashlib
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import draccus
import numpy as np
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmExample
from marin.training import chat_token_accounting
from marin.training.chat_token_accounting import (
    RowAccounting,
    RowIdentity,
    ScopeAccounting,
    TurnTokens,
    account_ordered_occurrences,
)
from rigging.filesystem.storage_path import StoragePath

from experiments.interp.snowball_identity.multihost import gather_global, gather_host_scalar

FREEZE_SHA256 = "91916521e16fff72a509f8564007b5721464ff07a78b49e2bc5bcf44983efd84"
INVENTORY_SHA256 = "3127b9a914769ffd4ed3c75d23a0afc0da0a1a2098d75f0a3e1fcadb82c32a6a"
ORDER_SHA256 = "911101395800c967cd5d9de8cd4cea68de14d5efbc4d46c20d901530da5e15a9"
LOCAL_MANIFEST_SHA256 = "d8ebbaaa07cd99452183381742959b03966fc63f31fefc59898caaf26f4c98d1"
ACCOUNTING_MODULE_SHA256 = "bb089af3009bbbdd6adbf3e07dc7444f9970164d030cf7ef6531982227340c21"
BATCH_SIZE = 8
SEQUENCE_LENGTH = 8192
AUDIT_UPDATES = 64


@dataclass(frozen=True)
class DatasetAuditInputs:
    freeze: dict
    data_config: dict
    ordered: list[dict]
    rows: tuple[RowAccounting, ...]
    fingerprints: dict[tuple[str, str], str]


def _checked_bytes(path: StoragePath, expected: str) -> bytes:
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError(f"Dataset audit input hash mismatch: {path.name}")
    return contents


def _raise_on_any_rank(error: Exception | None) -> None:
    if np.any(gather_host_scalar(int(error is not None))):
        raise ValueError("Dataset audit failed on at least one rank") from error


def load_dataset_audit_inputs(freeze_root: str, inventory_path: str) -> DatasetAuditInputs:
    """Read pinned data evidence on every rank; propagate any read failure collectively."""
    error = None
    result = None
    try:
        if hashlib.sha256(Path(chat_token_accounting.__file__).read_bytes()).hexdigest() != ACCOUNTING_MODULE_SHA256:
            raise ValueError("Public accounting implementation differs from the pinned source")
        root = StoragePath(freeze_root)
        freeze = json.loads(_checked_bytes(root / "freeze.json", FREEZE_SHA256))

        def read(name: str) -> bytes:
            return _checked_bytes(root / name, freeze["files"][name]["sha256"])

        if freeze["original_ordered512_sha256"] != ORDER_SHA256:
            raise ValueError("Unexpected frozen order")
        if freeze["original_local_manifest_sha256"] != LOCAL_MANIFEST_SHA256:
            raise ValueError("Unexpected local measurement")
        manifest = json.loads(read("original-local-manifest.json"))
        if INVENTORY_SHA256 not in manifest["source_files"].values():
            raise ValueError("Inventory is not pinned by the local measurement")
        ordered = [json.loads(line) for line in read("occurrences-512.jsonl").splitlines()]
        native = json.loads(read("native-row-fingerprints.json"))
        inventory = [
            json.loads(line) for line in _checked_bytes(StoragePath(inventory_path), INVENTORY_SHA256).splitlines()
        ]
        rows = []
        for entry in inventory:
            accounting = entry["accounting"]
            turns = tuple(
                TurnTokens(
                    turn["message_index"],
                    "identity_on_positive" if turn["label"] == "identity_on" else turn["label"],
                    turn["supervised_tokens"],
                )
                for turn in accounting["turns"]
            )
            counts: dict[str, int] = {}
            for turn in turns:
                counts[turn.label] = counts.get(turn.label, 0) + turn.supervised_tokens
            if counts != native[entry["id"]]["class_tokens"]:
                raise ValueError("Native turn accounting differs from frozen classifications")
            rows.append(
                RowAccounting(
                    RowIdentity(**accounting["identity"]),
                    accounting["rendered_tokens"],
                    accounting["template_sha256"],
                    accounting["input_ids_sha256"],
                    accounting["target_mask_sha256"],
                    turns,
                )
            )
        fingerprints = {
            (row["tokens_sha256"], row["target_mask_sha256"]): identifier for identifier, row in native.items()
        }
        if len(fingerprints) != 767 or len(rows) != 767 or len(ordered) != 512:
            raise ValueError("Frozen row coverage differs")
        result = DatasetAuditInputs(freeze, json.loads(read("data-config.json")), ordered, tuple(rows), fingerprints)
    except Exception as exc:
        error = exc
    _raise_on_any_rank(error)
    assert result is not None
    return result


def _summary(scope: ScopeAccounting, rows: tuple[RowAccounting, ...], identifiers: list[str]) -> dict:
    classes = {name: value.supervised_tokens for name, value in scope.classes.items() if value.supervised_tokens}
    families: dict[str, int] = {}
    by_id = {row.identity.occurrence_id: row for row in rows}
    for identifier in identifiers:
        row = by_id[identifier]
        family = row.identity.final_label
        families[family] = families.get(family, 0) + sum(turn.supervised_tokens for turn in row.turns)
    identity = sum(count for name, count in classes.items() if name.startswith("identity_"))
    return {
        "occurrences": scope.occurrences,
        "unique_ids": scope.unique_occurrences,
        "family_rows": scope.final_class_rows,
        "family_tokens": families,
        "class_tokens": classes,
        "supervised_tokens": scope.supervised_tokens,
        "identity_tokens": identity,
        "identity_fraction": {
            "numerator": identity,
            "denominator": scope.supervised_tokens,
            "value": identity / scope.supervised_tokens,
        },
    }


def _fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<i4").tobytes()).hexdigest()


def checked_update(batch: LmExample, update: int, frozen: DatasetAuditInputs) -> dict:
    """Propagate batch-check failures on every rank before the next training step."""
    error = None
    receipt = None
    try:
        receipt = check_update(batch, update, frozen)
    except Exception as exc:
        error = exc
    _raise_on_any_rank(error)
    assert receipt is not None
    return receipt


def check_update(batch: LmExample, update: int, frozen: DatasetAuditInputs) -> dict:
    """Gather a batch and compute its receipt; use checked_update for failure synchronization."""
    tokens = gather_global(batch.tokens.array)
    weights = gather_global(batch.loss_weight.array)
    if tokens.shape != (BATCH_SIZE, SEQUENCE_LENGTH) or weights.shape != tokens.shape:
        raise ValueError("Expected one global eight-sequence batch")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("Token IDs must be integers")
    if not np.all((weights == 0) | (weights == 1)):
        raise ValueError("Loss weights must be binary native assistant masks")
    identifiers = []
    for offset, (row, mask) in enumerate(zip(tokens, weights, strict=True)):
        fingerprint = (_fingerprint(row), _fingerprint(mask))
        identifier = frozen.fingerprints[fingerprint]
        expected = frozen.ordered[update * BATCH_SIZE + offset]
        if identifier != expected["id"] or int(mask.sum()) != expected["supervised_tokens"]:
            raise ValueError("Consumed batch differs from the frozen ordered stream")
        identifiers.append(identifier)
    accounting = account_ordered_occurrences(frozen.rows, identifiers, batch_size=BATCH_SIZE)
    summary = _summary(accounting.pooled, frozen.rows, identifiers)
    if summary != frozen.freeze["updates"][update]:
        raise ValueError("Per-family update integers differ from the freeze")
    return {"update": update, "ids": identifiers, **summary, "coefficient_mass_not_gradient_share": True}


def audit_training_iterator(
    batches: Iterator[LmExample],
    *,
    config: TrainLmConfig,
    completed_updates: int,
    frozen: DatasetAuditInputs,
    receipt_path: str,
) -> tuple[Iterator[LmExample], dict]:
    """Audit and buffer the same 64 loader batches that the training loop will consume.

    Call on every rank's main thread. The returned iterator yields the original
    batch objects first, then continues the original loader; no second sampler or
    iterator is constructed. The worker must call checked_update immediately before
    each train_step and retain those consumed-update receipts separately.
    """
    error = None
    try:
        if completed_updates != 0 or config.data_seed != 310:
            raise ValueError("Step-zero dataset audit requires step zero and data seed 310")
        if config.trainer.microbatch_size != BATCH_SIZE or config.trainer.train_batch_size != BATCH_SIZE:
            raise ValueError("Require one microbatch equal to the global batch of eight")
        if draccus.encode(config.data) != frozen.data_config or config.train_seq_len != SEQUENCE_LENGTH:
            raise ValueError("Resolved training data config differs from the frozen config")
    except Exception as exc:
        error = exc
    _raise_on_any_rank(error)
    buffered = []
    updates = []
    for update in range(AUDIT_UPDATES):
        error = None
        batch = None
        try:
            batch = next(batches)
        except Exception as exc:
            error = exc
        _raise_on_any_rank(error)
        assert batch is not None
        updates.append(checked_update(batch, update, frozen))
        buffered.append(batch)
    prefixes = {}
    error = None
    try:
        for size in (128, 512):
            identifiers = [identifier for update in updates[: size // BATCH_SIZE] for identifier in update["ids"]]
            accounting = account_ordered_occurrences(frozen.rows, identifiers, batch_size=BATCH_SIZE)
            summary = _summary(accounting.pooled, frozen.rows, identifiers)
            mean = sum(
                (
                    Fraction(value.numerator, value.denominator)
                    for name, value in accounting.mean_update_ce_coefficient_mass.items()
                    if name.startswith("identity_")
                ),
                Fraction(),
            )
            summary["mean_update_identity_ce_coefficient_mass"] = {
                "numerator": mean.numerator,
                "denominator": mean.denominator,
                "value": float(mean),
            }
            if summary != frozen.freeze["realized_prefixes"][str(size)]:
                raise ValueError("Prefix accounting differs from the freeze")
            prefixes[str(size)] = summary
    except Exception as exc:
        error = exc
    _raise_on_any_rank(error)
    report = {
        "freeze_sha256": FREEZE_SHA256,
        "ordered_512_sha256": ORDER_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "accounting_module_sha256": ACCOUNTING_MODULE_SHA256,
        "completed_updates": completed_updates,
        "global_batch_size": BATCH_SIZE,
        "microbatch_size": BATCH_SIZE,
        "prefixes": prefixes,
        "updates": updates,
        "coefficient_mass_not_gradient_share": True,
        "same_iterator_batches_buffered": AUDIT_UPDATES,
    }
    run_on_export_writer(lambda: write_export_json(StoragePath(receipt_path), report))
    return itertools.chain(buffered, batches), report
