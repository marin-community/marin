# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check a frozen loader prefix and replay the same native batches for training.

Call on every rank in matching order, before any optimizer update. A caller
loads and verifies its source freeze artifacts, supplies normalized accounting
and expectations, and records consumed batches separately with checked_update.
Auditing a buffered horizon does not authorize training through that horizon.
"""

import hashlib
import itertools
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import draccus
import numpy as np
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmExample
from levanter.utils.multihost import gather_global, gather_host_rows, gather_host_scalar
from rigging.filesystem.storage_path import StoragePath

from marin.training import chat_token_accounting
from marin.training.chat_token_accounting import RowAccounting, ScopeAccounting, account_ordered_occurrences


@dataclass(frozen=True)
class FrozenBatchAudit:
    freeze: dict
    data_config: dict
    ordered: list[dict]
    rows: tuple[RowAccounting, ...]
    fingerprints: dict[tuple[str, str], str]
    batch_size: int
    sequence_length: int
    data_seed: int
    selected_classes: frozenset[str]
    accounting_module_sha256: str
    source_pins: dict[str, str]


def _validate_plan(frozen: FrozenBatchAudit) -> bytes:
    if hashlib.sha256(Path(chat_token_accounting.__file__).read_bytes()).hexdigest() != frozen.accounting_module_sha256:
        raise ValueError("Accounting implementation differs from its source pin")
    if frozen.batch_size < 1 or frozen.sequence_length < 1 or not frozen.freeze["updates"]:
        raise ValueError("Frozen audit geometry must be nonempty")
    if len(frozen.ordered) != frozen.batch_size * len(frozen.freeze["updates"]):
        raise ValueError("Frozen order and update count differ")
    ids = {row.identity.occurrence_id for row in frozen.rows}
    if len(ids) != len(frozen.rows) or len(frozen.fingerprints) != len(ids) or set(frozen.fingerprints.values()) != ids:
        raise ValueError("Frozen row accounting and native fingerprints differ")
    for size in frozen.freeze["realized_prefixes"]:
        count = int(size)
        if not 0 < count <= len(frozen.ordered) or count % frozen.batch_size:
            raise ValueError("Frozen prefix must end on a planned batch boundary")
    payload = asdict(frozen)
    payload["fingerprints"] = sorted(
        (tokens, mask, identifier) for (tokens, mask), identifier in frozen.fingerprints.items()
    )
    payload["selected_classes"] = sorted(frozen.selected_classes)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).digest()


def _raise_on_any_rank(error: Exception | None) -> None:
    if np.any(gather_host_scalar(int(error is not None))):
        raise ValueError("Dataset audit failed on at least one rank") from error


def _summary(
    scope: ScopeAccounting, rows: tuple[RowAccounting, ...], identifiers: list[str], selected_classes: frozenset[str]
) -> dict:
    classes = {name: value.supervised_tokens for name, value in scope.classes.items() if value.supervised_tokens}
    families: dict[str, int] = {}
    by_id = {row.identity.occurrence_id: row for row in rows}
    for identifier in identifiers:
        row = by_id[identifier]
        family = row.identity.final_label
        families[family] = families.get(family, 0) + sum(turn.supervised_tokens for turn in row.turns)
    selected = sum(count for name, count in classes.items() if name in selected_classes)
    return {
        "occurrences": scope.occurrences,
        "unique_ids": scope.unique_occurrences,
        "family_rows": scope.final_class_rows,
        "family_tokens": families,
        "class_tokens": classes,
        "supervised_tokens": scope.supervised_tokens,
        "selected_tokens": selected,
        "selected_fraction": {
            "numerator": selected,
            "denominator": scope.supervised_tokens,
            "value": selected / scope.supervised_tokens,
        },
    }


def _fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<i4").tobytes()).hexdigest()


def checked_update(batch: LmExample, update: int, frozen: FrozenBatchAudit) -> dict:
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


def check_update(batch: LmExample, update: int, frozen: FrozenBatchAudit) -> dict:
    """Gather a batch and compute its receipt; use checked_update for failure synchronization."""
    tokens = gather_global(batch.tokens.array)
    weights = gather_global(batch.loss_weight.array)
    if tokens.shape != (frozen.batch_size, frozen.sequence_length) or weights.shape != tokens.shape:
        raise ValueError("Unexpected global batch geometry")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("Token IDs must be integers")
    if not np.all((weights == 0) | (weights == 1)):
        raise ValueError("Loss weights must be binary native assistant masks")
    identifiers = []
    for offset, (row, mask) in enumerate(zip(tokens, weights, strict=True)):
        fingerprint = (_fingerprint(row), _fingerprint(mask))
        identifier = frozen.fingerprints[fingerprint]
        expected = frozen.ordered[update * frozen.batch_size + offset]
        if identifier != expected["id"] or int(mask.sum()) != expected["supervised_tokens"]:
            raise ValueError("Consumed batch differs from the frozen ordered stream")
        identifiers.append(identifier)
    accounting = account_ordered_occurrences(frozen.rows, identifiers, batch_size=frozen.batch_size)
    summary = _summary(accounting.pooled, frozen.rows, identifiers, frozen.selected_classes)
    if summary != frozen.freeze["updates"][update]:
        raise ValueError("Per-family update integers differ from the freeze")
    return {"update": update, "ids": identifiers, **summary, "coefficient_mass_not_gradient_share": True}


def audit_training_iterator(
    batches: Iterator[LmExample],
    *,
    config: TrainLmConfig,
    completed_updates: int,
    frozen: FrozenBatchAudit,
    receipt_path: str,
) -> tuple[Iterator[LmExample], dict]:
    """Audit and buffer the frozen loader prefix before an optimizer update.

    Call on every rank's main thread. The returned iterator yields the original
    batch objects first, then continues the original loader; no second sampler or
    iterator is constructed. The worker must call checked_update immediately before
    each train_step and retain those consumed-update receipts separately.
    """
    error = None
    plan_digest = b""
    try:
        plan_digest = _validate_plan(frozen)
        if completed_updates != 0 or config.data_seed != frozen.data_seed:
            raise ValueError("Step-zero audit requires the frozen data seed")
        if (
            config.trainer.microbatch_size not in (None, frozen.batch_size)
            or config.trainer.train_batch_size != frozen.batch_size
        ):
            raise ValueError("Require one microbatch equal to the frozen global batch")
        if draccus.encode(config.data) != frozen.data_config or config.train_seq_len != frozen.sequence_length:
            raise ValueError("Resolved training data config differs from the frozen config")
    except Exception as exc:
        error = exc
    _raise_on_any_rank(error)
    # Fixed-size digest gather happens only after local validation succeeded on
    # every rank, so differing plan horizons cannot desynchronize later loops.
    digests = gather_host_rows(np.frombuffer(plan_digest, dtype=np.uint8))
    if not np.all(digests == digests[0]):
        raise ValueError("Frozen batch plan differs across ranks")
    buffered = []
    updates = []
    for update in range(len(frozen.freeze["updates"])):
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
        for size in sorted(int(value) for value in frozen.freeze["realized_prefixes"]):
            identifiers = [identifier for update in updates[: size // frozen.batch_size] for identifier in update["ids"]]
            accounting = account_ordered_occurrences(frozen.rows, identifiers, batch_size=frozen.batch_size)
            summary = _summary(accounting.pooled, frozen.rows, identifiers, frozen.selected_classes)
            mean = sum(
                (
                    Fraction(value.numerator, value.denominator)
                    for name, value in accounting.mean_update_ce_coefficient_mass.items()
                    if name in frozen.selected_classes
                ),
                Fraction(),
            )
            summary["mean_update_selected_ce_coefficient_mass"] = {
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
        "plan_sha256": plan_digest.hex(),
        "declared_source_pins": frozen.source_pins,
        "accounting_module_sha256": frozen.accounting_module_sha256,
        "completed_updates": completed_updates,
        "global_batch_size": frozen.batch_size,
        "microbatch_size": frozen.batch_size,
        "configured_microbatch_size": config.trainer.microbatch_size,
        "prefixes": prefixes,
        "updates": updates,
        "coefficient_mass_not_gradient_share": True,
        "same_iterator_batches_buffered": len(frozen.freeze["updates"]),
    }
    run_on_export_writer(lambda: write_export_json(StoragePath(receipt_path), report))
    return itertools.chain(buffered, batches), report
