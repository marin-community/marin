# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute missing H1/H2/H3/H5 cross-statistics from sealed checkpoints."""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import fsspec
import jax
import jax.random as jrandom
import numpy as np
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.execution.remote import remote

from experiments.domain_phase_mix import starcoder_wsd80_gradient_probe as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_mechanism_repair_20260818 as freeze,
)

SCHEMA_VERSION = "2026-08-18-gradient-mechanism-repair-v10"
ARTIFACT_VERSION = "2026.08.18.10"
TPU_TYPE = "v5p-8"
TPU_REGION = "us-central1"
TPU_ZONE = "us-central1-a"
TPU_HOST_CPU = 16
TPU_HOST_RAM = "128g"
MAX_CONCURRENT = 64
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_POST_OUTCOME_GRADIENT_MECHANISM_REPAIR"
FULL_LAUNCH_AUTHORIZATION_PATH = freeze.OUTPUT_DIR / "full_launch_authorization.json"


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _execution_observation(started_at: float, row: Mapping[str, Any]) -> dict[str, Any]:
    devices = jax.devices()
    block_counts = _json_mapping(row, "distribution_block_counts_json", int)
    return {
        "wall_seconds": time.perf_counter() - started_at,
        "peak_host_rss_bytes": _peak_rss_bytes(),
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "local_device_count": jax.local_device_count(),
        "device_kinds": sorted({str(device.device_kind) for device in devices}),
        "probe_batch_size": base.PROBE_BATCH_SIZE,
        "max_distribution_block_count": max(block_counts.values()),
        "distribution_count": len(block_counts),
    }


@dataclass(frozen=True)
class MechanismGroupConfig:
    scope: str
    group_id: str
    checkpoint_uri: str
    checkpoint_step: int
    expected_restored_state_step: int
    row: dict[str, Any]
    pod_config: Any
    output_path: str
    parent_cache_provenance_sha256: str
    release_sha256: str


def _canonical_json(value: Any) -> str:
    return freeze.canonical_json(value)


def _path_join(root: str, *parts: str) -> str:
    return "/".join((root.rstrip("/"), *(part.strip("/") for part in parts)))


def _row_path(output_path: str, row_id: str) -> str:
    return _path_join(output_path, "rows", f"{row_id}.json")


def _row_identity(row: Mapping[str, Any], release_sha256: str) -> str:
    return freeze.canonical_sha256(
        {
            "row": dict(row),
            "release_sha256": release_sha256,
            "release_version": freeze.RELEASE_VERSION,
        }
    )


def _group_identity(config: MechanismGroupConfig) -> str:
    return freeze.canonical_sha256(
        {
            "group_id": config.group_id,
            "row_identity_sha256": _row_identity(config.row, config.release_sha256),
            "release_sha256": config.release_sha256,
        }
    )


def _read_document(path: str) -> dict[str, Any] | None:
    fs, plain_path = fsspec.core.url_to_fs(path)
    if not fs.exists(plain_path):
        return None
    with fs.open(plain_path, "rb") as handle:
        document = json.load(handle)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Existing repair output has an unexpected schema: {path}")
    payload_sha256 = document.get("payload_sha256")
    if payload_sha256 != freeze.canonical_sha256({**document, "payload_sha256": ""}):
        raise RuntimeError(f"Existing repair output failed its payload hash: {path}")
    return document


def _write_create_only(path: str, payload: dict[str, Any], *, identity_sha256: str) -> str:
    document = {
        **payload,
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "payload_sha256": "",
    }
    document["payload_sha256"] = freeze.canonical_sha256(document)
    encoded = (_canonical_json(document) + "\n").encode()
    fs, plain_path = fsspec.core.url_to_fs(path)
    parent = os.path.dirname(plain_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    try:
        with fs.open(plain_path, "xb") as handle:
            handle.write(encoded)
        disposition = "created"
    except FileExistsError as error:
        existing = _read_document(path)
        assert existing is not None
        if existing.get("identity_sha256") != identity_sha256:
            raise RuntimeError(f"Existing repair output is claimed by another identity: {path}") from error
        if (_canonical_json(existing) + "\n").encode() != encoded:
            raise RuntimeError(f"Existing repair output payload differs for the same identity: {path}") from error
        disposition = "skipped_existing"
    if disposition == "created":
        fs, plain_path = fsspec.core.url_to_fs(path)
        with fs.open(plain_path, "rb") as handle:
            if handle.read() != encoded:
                raise RuntimeError(f"Create-only repair output did not persist exactly: {path}")
    return disposition


def _group_marker_payload(config: MechanismGroupConfig, row_document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "gradient_mechanism_repair_group",
        "scope": config.scope,
        "group_id": config.group_id,
        "row_count": 1,
        "checkpoint_metadata": row_document["checkpoint_metadata"],
        "runtime_summary": row_document["runtime_summary"],
        "execution_observation": row_document["execution_observation"],
        "parent_cache_provenance_sha256": row_document["parent_cache_provenance_sha256"],
        "release_sha256": config.release_sha256,
        "scientific_status": freeze.SCIENTIFIC_STATUS,
        "endpoint_metrics_read": False,
        "row_document_sha256": row_document["payload_sha256"],
    }


def _validate_existing_row(config: MechanismGroupConfig, row_document: Mapping[str, Any]) -> None:
    expected = {
        "kind": "gradient_mechanism_repair",
        "scope": config.scope,
        "group_id": config.group_id,
        "row": config.row,
        "parent_cache_provenance_sha256": config.parent_cache_provenance_sha256,
        "release_sha256": config.release_sha256,
        "scientific_status": freeze.SCIENTIFIC_STATUS,
        "endpoint_metrics_read": False,
        "identity_sha256": _row_identity(config.row, config.release_sha256),
    }
    if any(row_document.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Existing repair row semantics drifted: {config.row['row_id']}")
    required_payloads = (
        "checkpoint_metadata",
        "runtime_summary",
        "execution_observation",
        "source_pair_statistics",
        "target_source_gradient_statistics",
        "target_source_utility_statistics",
        "target_source_choice_contrasts",
    )
    if any(key not in row_document for key in required_payloads):
        raise RuntimeError(f"Existing repair row is incomplete: {config.row['row_id']}")


def _existing_group_complete(config: MechanismGroupConfig) -> bool:
    marker_path = _path_join(config.output_path, "group_complete.json")
    marker = _read_document(marker_path)
    row_document = _read_document(_row_path(config.output_path, str(config.row["row_id"])))
    if marker is None and row_document is None:
        return False
    if marker is None:
        assert row_document is not None
        _validate_existing_row(config, row_document)
        _write_create_only(
            marker_path,
            _group_marker_payload(config, row_document),
            identity_sha256=_group_identity(config),
        )
        marker = _read_document(marker_path)
        assert marker is not None
    if row_document is None:
        raise RuntimeError(f"Completed repair group is missing row {config.row['row_id']}")
    _validate_existing_row(config, row_document)
    if marker.get("identity_sha256") != _group_identity(config):
        raise RuntimeError(f"Completed repair group identity drifted: {config.group_id}")
    if row_document.get("identity_sha256") != _row_identity(config.row, config.release_sha256):
        raise RuntimeError(f"Completed repair row identity drifted: {config.row['row_id']}")
    if row_document.get("row") != config.row:
        raise RuntimeError(f"Completed repair row payload drifted: {config.row['row_id']}")
    expected_marker = {
        "kind": "gradient_mechanism_repair_group",
        "scope": config.scope,
        "group_id": config.group_id,
        "row_count": 1,
        "release_sha256": config.release_sha256,
        "scientific_status": freeze.SCIENTIFIC_STATUS,
        "endpoint_metrics_read": False,
    }
    if any(marker.get(key) != value for key, value in expected_marker.items()):
        raise RuntimeError(f"Completed repair marker semantics drifted: {config.group_id}")
    if marker.get("row_document_sha256") != row_document["payload_sha256"]:
        raise RuntimeError(f"Completed repair marker does not bind its row payload: {config.group_id}")
    if marker.get("execution_observation") != row_document.get("execution_observation"):
        raise RuntimeError(f"Completed repair marker execution observation drifted: {config.group_id}")
    return True


def _json_mapping(row: Mapping[str, Any], key: str, cast_value: Any) -> dict[str, Any]:
    return {str(name): cast_value(value) for name, value in json.loads(str(row[key])).items()}


def _json_names(row: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = json.loads(str(row[key]))
    if not isinstance(values, list):
        raise ValueError(f"{key} must encode a JSON list")
    return tuple(str(value) for value in values)


def _mean_gradient_and_updates(
    *,
    trainer: Any,
    state: Any,
    dataset: Any,
    blocks: int,
    update_draws: int,
    seed_id: str,
    gradient_fn: Any,
    update_fn: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Reproduce the v6 stochastic row while retaining its no-data update."""
    capacity = base._dataset_capacity(dataset, blocks, label=f"mechanism-repair:{seed_id}")
    iterator = iter(trainer.data_loader(dataset, batch=base.PROBE_BATCH_SIZE))
    mean_gradient = None
    mean_corrected_update = None
    mean_no_data_update = None
    losses: list[float] = []
    data_update_norms: list[float] = []
    no_data_update_norms: list[float] = []
    corrected_update_norms: list[float] = []
    prior_gradient = None
    prior_loss = None
    update_count = 0
    first_no_data_update = None
    no_data_spread = 0.0
    base_key = base._fold_in_stable(state.training_key, seed_id)
    first_gradient = None
    first_example = None
    first_loss = None
    for block in range(blocks):
        example = next(iterator)
        key = jrandom.fold_in(base_key, block)
        loss, gradient = gradient_fn(state.model, example, key)
        scalar_loss = base._loss_scalar(loss)
        if not math.isfinite(scalar_loss):
            raise RuntimeError("Non-finite mechanism-repair loss")
        losses.append(scalar_loss)
        mean_gradient = gradient if mean_gradient is None else base._tree_add(mean_gradient, gradient)
        if block == 0:
            first_example, first_gradient, first_loss = example, gradient, scalar_loss
        if block % 2 == 0:
            prior_gradient, prior_loss = gradient, scalar_loss
            continue
        if update_count >= update_draws:
            continue
        assert prior_gradient is not None and prior_loss is not None
        training_gradient = base._tree_scale(base._tree_add(prior_gradient, gradient), 0.5)
        training_loss = 0.5 * (prior_loss + scalar_loss)
        update_key = jrandom.fold_in(base_key, 100_000 + update_count)
        data_update = update_fn(state, training_gradient, training_loss, update_key)
        no_data_update = update_fn(state, base._tree_zeros(training_gradient), training_loss, update_key)
        if first_no_data_update is None:
            first_no_data_update = no_data_update
        else:
            no_data_spread = max(no_data_spread, base._tree_max_abs_diff(first_no_data_update, no_data_update))
        corrected_update = base._tree_subtract(data_update, no_data_update)
        data_update_norms.append(float(jax.device_get(base._raw_tree_norm(data_update))))
        no_data_update_norms.append(float(jax.device_get(base._raw_tree_norm(no_data_update))))
        corrected_update_norms.append(float(jax.device_get(base._raw_tree_norm(corrected_update))))
        mean_corrected_update = (
            corrected_update
            if mean_corrected_update is None
            else base._tree_add(mean_corrected_update, corrected_update)
        )
        mean_no_data_update = (
            no_data_update if mean_no_data_update is None else base._tree_add(mean_no_data_update, no_data_update)
        )
        update_count += 1
    if mean_gradient is None or first_example is None or first_gradient is None or first_loss is None:
        raise RuntimeError("Mechanism repair produced no gradients")
    mean_gradient = base._tree_scale(mean_gradient, 1.0 / blocks)
    if update_count != update_draws:
        raise RuntimeError(f"Mechanism-repair optimizer draws incomplete: {update_count} != {update_draws}")
    if update_count:
        assert mean_corrected_update is not None and mean_no_data_update is not None
        mean_corrected_update = base._tree_scale(mean_corrected_update, 1.0 / update_count)
        mean_no_data_update = base._tree_scale(mean_no_data_update, 1.0 / update_count)
    else:
        mean_corrected_update = base._tree_zeros(mean_gradient)
        mean_no_data_update = base._tree_zeros(mean_gradient)
        no_data_spread = 0.0
    repeated_loss, repeated_gradient = gradient_fn(state.model, first_example, jrandom.fold_in(base_key, 0))
    repeat_gradient_difference = base._tree_max_abs_diff(first_gradient, repeated_gradient)
    repeat_loss_difference = abs(first_loss - base._loss_scalar(repeated_loss))
    if repeat_gradient_difference > base.NUMERICAL_TOLERANCE or repeat_loss_difference > base.NUMERICAL_TOLERANCE:
        raise RuntimeError("Mechanism-repair gradient repeat is not deterministic")
    return (
        mean_gradient,
        mean_corrected_update,
        mean_no_data_update,
        {
            "replicate_block_count": blocks,
            "optimizer_update_draw_count": update_count,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            "loss_min": min(losses),
            "loss_max": max(losses),
            "no_data_update_within_source_max_abs_diff": no_data_spread,
            "repeat_gradient_max_abs_difference": repeat_gradient_difference,
            "repeat_loss_absolute_difference": repeat_loss_difference,
            "first_batch_sha256": base._tree_sha256(first_example),
            "data_update_norm_mean": float(np.mean(data_update_norms)) if data_update_norms else 0.0,
            "optimizer_memory_update_norm_mean": float(np.mean(no_data_update_norms)) if no_data_update_norms else 0.0,
            "corrected_update_norm_mean": float(np.mean(corrected_update_norms)) if corrected_update_norms else 0.0,
            "optimizer_memory_update_nonzero": bool(any(value > 0.0 for value in no_data_update_norms)),
            "data_supply": capacity,
            "stochastic_identity": seed_id,
        },
    )


def _assert_common_no_data_update(no_data_updates: Mapping[str, Any], summaries: Mapping[str, Any]) -> dict[str, Any]:
    names = sorted(no_data_updates)
    if not names:
        raise RuntimeError("Mechanism repair computed no optimizer updates")
    reference = no_data_updates[names[0]]
    cross_source = {name: base._tree_max_abs_diff(reference, no_data_updates[name]) for name in names[1:]}
    within_source = {name: float(summaries[name]["no_data_update_within_source_max_abs_diff"]) for name in names}
    maximum = max((*cross_source.values(), *within_source.values()), default=0.0)
    corrected_norm_floor = min(float(summaries[name]["corrected_update_norm_mean"]) for name in names)
    relative_maximum = maximum / max(corrected_norm_floor, np.finfo(float).tiny)
    if maximum > base.NUMERICAL_TOLERANCE:
        raise RuntimeError(f"No-data optimizer update depends on source loss or RNG key: max_abs_diff={maximum}")
    return {
        "passed": True,
        "tolerance": base.NUMERICAL_TOLERANCE,
        "max_abs_diff": maximum,
        "relative_to_min_corrected_update_norm": relative_maximum,
        "min_corrected_update_norm": corrected_norm_floor,
        "cross_source_max_abs_diff": cross_source,
        "within_source_max_abs_diff": within_source,
    }


def _verify_group_contract(config: MechanismGroupConfig) -> None:
    row = config.row
    if not config.output_path.startswith(f"{freeze.RESULT_ROOT}/{config.scope}/"):
        raise ValueError(f"Repair output is outside the frozen central1 root: {config.output_path}")
    if row["group_id"] != config.group_id or row["checkpoint_uri"] != config.checkpoint_uri:
        raise ValueError("Repair row and execution group disagree")
    if int(row["checkpoint_step"]) != config.checkpoint_step:
        raise ValueError("Repair checkpoint step drifted")
    if int(row["expected_restored_state_step"]) != config.expected_restored_state_step:
        raise ValueError("Repair restored-state step drifted")
    if config.expected_restored_state_step != base.freeze.expected_restored_state_step(config.checkpoint_step):
        raise ValueError("Repair checkpoint-step semantics are inconsistent")
    observed_config = base.freeze._config_identity(config.pod_config)["full_train_config_sha256"]
    if row["train_config_sha256"] != observed_config:
        raise ValueError("Repair train configuration drifted")
    support = base._starcoder_support_contract(config.pod_config.train_config)["support_id"]
    if config.scope == "full" and support != row["support_id"]:
        raise ValueError(f"Repair support drifted: {support} != {row['support_id']}")

    sources = _json_names(row, "source_distribution_ids_json")
    targets = _json_names(row, "target_distribution_ids_json")
    if not sources or set(sources) - base.SOURCE_DISTRIBUTIONS:
        raise ValueError(f"Repair source contract is invalid: {sources}")
    if set(targets) - base.TARGET_DISTRIBUTIONS:
        raise ValueError(f"Repair target contract is invalid: {targets}")
    requested = set(sources) | set(targets)
    blocks = _json_mapping(row, "distribution_block_counts_json", int)
    draws = _json_mapping(row, "distribution_update_draw_counts_json", int)
    sequence_sets = _json_mapping(row, "distribution_sequence_set_ids_json", str)
    probe_row_ids = _json_mapping(row, "distribution_probe_row_ids_json", str)
    if set(blocks) != requested or set(draws) != requested or set(sequence_sets) != requested:
        raise ValueError("Repair distribution contracts do not name the same inventory")
    if set(probe_row_ids) != requested or len(set(probe_row_ids.values())) != len(probe_row_ids):
        raise ValueError("Repair probe stochastic identities are incomplete or duplicated")
    if any(blocks[name] <= 0 for name in requested):
        raise ValueError("Repair block counts must be positive")
    if any(draws[name] <= 0 for name in sources) or any(draws[name] != 0 for name in targets):
        raise ValueError("Repair update-draw contract must update sources and not targets")
    if not {freeze.GLOBAL_STARCODER, freeze.NEMOTRON} <= set(sources):
        raise ValueError("Repair row omits the frozen global-heldout StarCoder versus Nemotron source contrast")


def _statistics_bundle(
    left: Any,
    right: Any,
    *,
    model: Any,
    optimizer_mask: Any,
) -> dict[str, Any]:
    return {
        "raw": base._tree_pair_statistics(
            left,
            right,
            model=model,
            optimizer_mask=optimizer_mask,
            project_muon=False,
        ),
        "projected": base._tree_pair_statistics(
            left,
            right,
            model=model,
            optimizer_mask=optimizer_mask,
            project_muon=True,
        ),
    }


def _source_pair_statistics(
    gradients: Mapping[str, Any],
    updates: Mapping[str, Any],
    *,
    model: Any,
    optimizer_mask: Any,
) -> dict[str, Any]:
    left = freeze.GLOBAL_STARCODER
    right = freeze.NEMOTRON
    return {
        "starcoder__vs__nemotron": {
            "gradient_left_source": left,
            "optimizer_update_left_source": left,
            "right_source": right,
            "gradient": _statistics_bundle(
                gradients[left],
                gradients[right],
                model=model,
                optimizer_mask=optimizer_mask,
            ),
            "optimizer_update": _statistics_bundle(
                updates[left],
                updates[right],
                model=model,
                optimizer_mask=optimizer_mask,
            ),
        }
    }


def _contrast_pairs(sources: Sequence[str]) -> tuple[tuple[str, str], ...]:
    requested = set(sources)
    candidates = (
        (freeze.GLOBAL_STARCODER, freeze.NEMOTRON),
        (freeze.SUPPORT_STARCODER, freeze.NEMOTRON),
        (freeze.SUPPORT_STARCODER, freeze.GLOBAL_STARCODER),
    )
    return tuple(pair for pair in candidates if set(pair) <= requested)


def _assert_dot_consistency(contrast: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    for geometry in ("raw", "projected"):
        for component, statistics in contrast[geometry].items():
            expected = float(left[geometry][component]["dot"]) - float(right[geometry][component]["dot"])
            observed = float(statistics["dot"])
            tolerance = 5e-5 * max(abs(expected), abs(observed), 1.0)
            if abs(expected - observed) > tolerance:
                raise RuntimeError(
                    f"Source-choice utility is inconsistent for {geometry}/{component}: {observed} != {expected}"
                )


def _target_statistics(
    targets: Mapping[str, Any],
    updates: Mapping[str, Any],
    *,
    model: Any,
    optimizer_mask: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    utilities: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for target, target_gradient in sorted(targets.items()):
        negative_target = base._tree_scale(target_gradient, -1.0)
        utilities[target] = {
            source: _statistics_bundle(
                negative_target,
                update,
                model=model,
                optimizer_mask=optimizer_mask,
            )
            for source, update in sorted(updates.items())
        }
        contrasts[target] = {}
        for left, right in _contrast_pairs(tuple(updates)):
            statistics = _statistics_bundle(
                negative_target,
                base._tree_subtract(updates[left], updates[right]),
                model=model,
                optimizer_mask=optimizer_mask,
            )
            _assert_dot_consistency(statistics, utilities[target][left], utilities[target][right])
            contrasts[target][f"{left}__minus__{right}"] = {
                "left_source": left,
                "right_source": right,
                "statistic": statistics,
                "interpretation": "dot_is_X_y_and_cosine_is_A_y",
            }
    return utilities, contrasts


def _target_gradient_statistics(
    targets: Mapping[str, Any],
    sources: Mapping[str, Any],
    *,
    model: Any,
    optimizer_mask: Any,
) -> dict[str, Any]:
    return {
        target: {
            source: _statistics_bundle(
                target_gradient,
                source_gradient,
                model=model,
                optimizer_mask=optimizer_mask,
            )
            for source, source_gradient in sorted(sources.items())
        }
        for target, target_gradient in sorted(targets.items())
    }


def run_mechanism_group(config: MechanismGroupConfig) -> None:
    _verify_group_contract(config)
    if _existing_group_complete(config):
        return
    started_at = time.perf_counter()
    metadata = base._read_checkpoint_metadata(config.checkpoint_uri, config.checkpoint_step)
    train_config = base._prepare_train_config(config.pod_config, config.checkpoint_uri, config.group_id)
    trainer, state, Pos, data_key, optimizer_mask = base._initialize_runtime(train_config)
    try:
        if int(state.step) != config.expected_restored_state_step:
            raise RuntimeError("Mechanism repair restored the wrong checkpoint state")
        runtime_summary = {
            "restoration": base._restored_optimizer_summary(
                state,
                config.checkpoint_step,
                config.expected_restored_state_step,
                allow_partial_checkpoint=train_config.trainer.allow_partial_checkpoint,
            ),
            "muon_projection": base._runtime_muon_projection_coverage(state.model, optimizer_mask),
            "optimizer_schedule": base._optimizer_schedule_summary(train_config),
            "optimizer_update_statistic": "finite_difference_Delta(g)_minus_Delta(0)_not_causal_attribution",
        }
        source_views, stream_summary = base._source_views(
            train_config,
            Pos,
            data_key,
            config.expected_restored_state_step,
        )
        runtime_summary["source_stream"] = stream_summary
        gradient_fn, update_fn, _, _ = base._gradient_functions(trainer)
        row = config.row
        source_ids = _json_names(row, "source_distribution_ids_json")
        target_ids = _json_names(row, "target_distribution_ids_json")
        block_counts = _json_mapping(row, "distribution_block_counts_json", int)
        update_draws = _json_mapping(row, "distribution_update_draw_counts_json", int)
        sequence_sets = _json_mapping(row, "distribution_sequence_set_ids_json", str)

        gradients: dict[str, Any] = {}
        updates: dict[str, Any] = {}
        no_data_updates: dict[str, Any] = {}
        update_summaries: dict[str, Any] = {}
        numerical_summaries: dict[str, Any] = {}
        probe_row_ids = _json_mapping(row, "distribution_probe_row_ids_json", str)
        for distribution in source_ids:
            dataset = base._distribution_dataset(
                distribution,
                sequence_set_id=sequence_sets[distribution],
                train_config=train_config,
                Pos=Pos,
                sources=source_views,
            )
            gradient, update, no_data_update, summary = _mean_gradient_and_updates(
                trainer=trainer,
                state=state,
                dataset=dataset,
                blocks=block_counts[distribution],
                update_draws=update_draws[distribution],
                seed_id=probe_row_ids[distribution],
                gradient_fn=gradient_fn,
                update_fn=update_fn,
            )
            gradients[distribution] = gradient
            updates[distribution] = update
            no_data_updates[distribution] = no_data_update
            update_summaries[distribution] = summary
            numerical_summaries[f"probe:{distribution}"] = summary

        target_gradients: dict[str, Any] = {}
        for distribution in target_ids:
            dataset = base._distribution_dataset(
                distribution,
                sequence_set_id=sequence_sets[distribution],
                train_config=train_config,
                Pos=Pos,
                sources=source_views,
            )
            gradient, _, _, summary = _mean_gradient_and_updates(
                trainer=trainer,
                state=state,
                dataset=dataset,
                blocks=block_counts[distribution],
                update_draws=0,
                seed_id=probe_row_ids[distribution],
                gradient_fn=gradient_fn,
                update_fn=update_fn,
            )
            target_gradients[distribution] = gradient
            numerical_summaries[f"probe:{distribution}"] = summary

        no_data_audit = _assert_common_no_data_update(no_data_updates, update_summaries)

        source_pairs = _source_pair_statistics(
            gradients,
            updates,
            model=state.model,
            optimizer_mask=optimizer_mask,
        )
        target_utilities, target_contrasts = _target_statistics(
            target_gradients,
            updates,
            model=state.model,
            optimizer_mask=optimizer_mask,
        )
        target_gradient_statistics = _target_gradient_statistics(
            target_gradients,
            gradients,
            model=state.model,
            optimizer_mask=optimizer_mask,
        )
        execution_observation = _execution_observation(started_at, row)
        row_path = _row_path(config.output_path, str(row["row_id"]))
        _write_create_only(
            row_path,
            {
                "kind": "gradient_mechanism_repair",
                "scope": config.scope,
                "group_id": config.group_id,
                "row": row,
                "checkpoint_metadata": metadata,
                "restored_state_step": int(state.step),
                "runtime_summary": runtime_summary,
                "parent_cache_provenance_sha256": config.parent_cache_provenance_sha256,
                "release_sha256": config.release_sha256,
                "scientific_status": freeze.SCIENTIFIC_STATUS,
                "numerical_summaries": numerical_summaries,
                "no_data_update_invariance": no_data_audit,
                "execution_observation": execution_observation,
                "source_pair_statistics": source_pairs,
                "target_source_gradient_statistics": target_gradient_statistics,
                "target_source_utility_statistics": target_utilities,
                "target_source_choice_contrasts": target_contrasts,
                "endpoint_metrics_read": False,
            },
            identity_sha256=_row_identity(row, config.release_sha256),
        )
        row_document = _read_document(row_path)
        assert row_document is not None
        _write_create_only(
            _path_join(config.output_path, "group_complete.json"),
            _group_marker_payload(config, row_document),
            identity_sha256=_group_identity(config),
        )
    finally:
        base._close_runtime(trainer)


def _load_release(expected_sha256: str) -> dict[str, Any]:
    release = json.loads(freeze.RELEASE_PATH.read_text())
    if release["release_sha256"] != expected_sha256:
        raise ValueError("Mechanism-repair release hash does not match the requested release")
    if freeze.canonical_sha256({**release, "release_sha256": ""}) != expected_sha256:
        raise ValueError("Mechanism-repair release document is internally inconsistent")
    for relative_path, sha256 in release["implementation_files"].items():
        if freeze.file_sha256(freeze.REPO_ROOT / relative_path) != sha256:
            raise ValueError(f"Mechanism-repair implementation drifted: {relative_path}")
    for summary in release["manifests"].values():
        if freeze.file_sha256(freeze.REPO_ROOT / summary["path"]) != summary["sha256"]:
            raise ValueError(f"Mechanism-repair manifest drifted: {summary['path']}")
    analysis_contract = release["analysis_contract"]
    if freeze.file_sha256(freeze.REPO_ROOT / analysis_contract["path"]) != analysis_contract["sha256"]:
        raise ValueError("Mechanism-repair analysis contract drifted")
    if freeze.file_sha256(freeze.PARENT_RELEASE_PATH) != release["parent_release_file_sha256"]:
        raise ValueError("Parent v6 release file drifted")
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    if parent_release["release_sha256"] != release["parent_release_sha256"]:
        raise ValueError("Parent v6 release identity drifted")
    for relative_path, sha256 in release["parent_implementation_files"].items():
        if freeze.file_sha256(freeze.REPO_ROOT / relative_path) != sha256:
            raise ValueError(f"Parent v6 implementation drifted: {relative_path}")
    for summary in parent_release["manifests"].values():
        if freeze.file_sha256(freeze.REPO_ROOT / summary["path"]) != summary["sha256"]:
            raise ValueError(f"Parent v6 manifest drifted: {summary['path']}")
    for name, sha256 in parent_release["source_design_files"].items():
        if freeze.file_sha256(base.freeze.DESIGN_DIR / name) != sha256:
            raise ValueError(f"Parent v6 design input drifted: {name}")
    if (
        release["required_region"] != TPU_REGION
        or release["required_zone"] != TPU_ZONE
        or release["required_bucket_prefix"] != freeze.MARIN_PREFIX
        or not release["result_root"].startswith(freeze.MARIN_PREFIX)
    ):
        raise ValueError("Mechanism-repair locality contract drifted")
    return release


def _read_manifest(scope: str, release: Mapping[str, Any], stage: int | None = None) -> list[dict[str, Any]]:
    summary = release["manifests"][scope]
    path = freeze.REPO_ROOT / summary["path"]
    with path.open(newline="") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    if len(rows) != summary["row_count"]:
        raise ValueError(f"Mechanism-repair {scope} manifest row count drifted")
    if scope == "full" and stage is not None:
        rows = [row for row in rows if int(row["launch_stage"]) == stage]
    return rows


def _pod_configs(scope: str) -> dict[str, Any]:
    return base.freeze._canary_configs() if scope == "canary" else base.freeze._full_configs()


def _resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(
        TPU_TYPE,
        cpu=TPU_HOST_CPU,
        ram=TPU_HOST_RAM,
        regions=(TPU_REGION,),
        zone=TPU_ZONE,
    )


def _artifact_name(scope: str, group_id: str) -> str:
    prefix = f"{freeze.MARIN_PREFIX}/"
    return f"{freeze.RESULT_ROOT.removeprefix(prefix)}/{scope}/{group_id}"


def _steps(
    scope: str,
    release: Mapping[str, Any],
    configs: Mapping[str, Any],
    stage: int | None,
) -> list[ArtifactStep[Artifact]]:
    cache_sha = json.loads(freeze.PARENT_RELEASE_PATH.read_text())["manifests"]["cache_provenance"]["sha256"]
    resources = _resources()
    steps: list[ArtifactStep[Artifact]] = []
    for row in _read_manifest(scope, release, stage):
        trajectory_id = row["trajectory_id"]
        config = MechanismGroupConfig(
            scope=scope,
            group_id=row["group_id"],
            checkpoint_uri=row["checkpoint_uri"],
            checkpoint_step=int(row["checkpoint_step"]),
            expected_restored_state_step=int(row["expected_restored_state_step"]),
            row=row,
            pod_config=configs[trajectory_id],
            output_path="",
            parent_cache_provenance_sha256=cache_sha,
            release_sha256=release["release_sha256"],
        )
        steps.append(
            ArtifactStep(
                name=_artifact_name(scope, row["group_id"]),
                version=ARTIFACT_VERSION,
                artifact_type=Artifact,
                run=remote(run_mechanism_group, resources=resources, name=row["group_id"]),
                build_config=lambda ctx, config=config: replace(config, output_path=ctx.output_path),
            )
        )
    return steps


def _checkpoint_provenance(release: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    summary = release["manifests"]["checkpoint_provenance"]
    path = freeze.REPO_ROOT / summary["path"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != summary["row_count"]:
        raise ValueError("Checkpoint provenance row count drifted")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["checkpoint_uri"], []).append(row)
    return grouped


def _parent_result_provenance(release: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    summary = release["manifests"]["parent_result_provenance"]
    path = freeze.REPO_ROOT / summary["path"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != summary["row_count"] or len({row["object_uri"] for row in rows}) != summary["object_count"]:
        raise ValueError("Parent result provenance inventory drifted")
    return {row["object_uri"]: row for row in rows}


def _read_pinned_parent_result(uri: str, expected: Mapping[str, str]) -> dict[str, Any]:
    fs, plain_path = fsspec.core.url_to_fs(uri)
    info = fs.info(plain_path)
    observed_metadata = {
        "size": str(info["size"]),
        "generation": str(info["generation"]),
        "md5_hash": str(info.get("md5Hash", "")),
        "crc32c": str(info.get("crc32c", "")),
        "etag": str(info.get("etag", "")),
    }
    expected_metadata = {key: expected[key] for key in observed_metadata}
    if observed_metadata != expected_metadata:
        raise RuntimeError(f"Parent result object generation or checksum drifted: {uri}")
    with fs.open(plain_path, "rb") as handle:
        payload = handle.read()
    if hashlib.sha256(payload).hexdigest() != expected["payload_sha256"]:
        raise RuntimeError(f"Parent result object payload drifted: {uri}")
    document = json.loads(payload)
    if document.get("identity_sha256") != expected["parent_identity_sha256"]:
        raise RuntimeError(f"Parent result row identity drifted: {uri}")
    return document


def _parent_result_readiness(
    scope: str,
    release: Mapping[str, Any],
    stage: int | None = None,
) -> dict[str, Any]:
    provenance = _parent_result_provenance(release)
    required: set[str] = set()
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    for row in _read_manifest(scope, release, stage):
        probe_row_ids = _json_mapping(row, "distribution_probe_row_ids_json", str)
        for source in _json_names(row, "source_distribution_ids_json"):
            required.add(
                _path_join(
                    parent_release["result_root"],
                    scope,
                    "probe",
                    row["parent_probe_group_id"],
                    release["parent_result_artifact_version"],
                    "rows",
                    f"{probe_row_ids[source]}.json",
                )
            )

    async def audit_one(uri: str) -> dict[str, Any]:
        try:
            expected = provenance[uri]
            await asyncio.to_thread(_read_pinned_parent_result, uri, expected)
        except Exception as error:
            return {"object_uri": uri, "ready": False, "error": repr(error)}
        return {"object_uri": uri, "ready": True, "error": ""}

    async def audit_all() -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(64)

        async def bounded(uri: str) -> dict[str, Any]:
            async with semaphore:
                return await audit_one(uri)

        return await asyncio.gather(*(bounded(uri) for uri in sorted(required)))

    unexpected = sorted(required - set(provenance))
    if unexpected:
        return {
            "expected": len(required),
            "ready": 0,
            "missing": len(unexpected),
            "failures": [{"object_uri": uri, "ready": False, "error": "not pinned"} for uri in unexpected],
        }
    results = asyncio.run(audit_all())
    failures = [result for result in results if not result["ready"]]
    return {
        "expected": len(required),
        "ready": len(results) - len(failures),
        "missing": len(failures),
        "failures": failures,
    }


def _checkpoint_readiness(
    scope: str,
    release: Mapping[str, Any],
    stage: int | None = None,
) -> dict[str, Any]:
    checkpoints = {(row["checkpoint_uri"], int(row["checkpoint_step"])) for row in _read_manifest(scope, release, stage)}
    provenance = _checkpoint_provenance(release)

    async def audit_one(uri: str, step: int) -> dict[str, Any]:
        try:
            await asyncio.to_thread(base._read_checkpoint_metadata, uri, step)
            fs, plain_path = fsspec.core.url_to_fs(uri)
            current = await asyncio.to_thread(fs.find, plain_path, detail=True)
            expected = provenance.get(uri, [])
            observed = {
                f"gs://{path}": {
                    "size": str(info["size"]),
                    "generation": str(info["generation"]),
                    "md5_hash": str(info.get("md5Hash", "")),
                    "crc32c": str(info.get("crc32c", "")),
                    "etag": str(info.get("etag", "")),
                }
                for path, info in current.items()
            }
            expected_by_path = {
                row["object_path"]: {key: row[key] for key in ("size", "generation", "md5_hash", "crc32c", "etag")}
                for row in expected
            }
            if observed != expected_by_path:
                raise RuntimeError("Checkpoint object generation or checksum inventory drifted")
        except Exception as error:
            return {"checkpoint_uri": uri, "checkpoint_step": step, "ready": False, "error": repr(error)}
        return {"checkpoint_uri": uri, "checkpoint_step": step, "ready": True, "error": ""}

    async def audit_all() -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(64)

        async def bounded(uri: str, step: int) -> dict[str, Any]:
            async with semaphore:
                return await audit_one(uri, step)

        return await asyncio.gather(*(bounded(uri, step) for uri, step in sorted(checkpoints)))

    results = asyncio.run(audit_all())
    failures = [result for result in results if not result["ready"]]
    return {
        "expected": len(checkpoints),
        "ready": len(results) - len(failures),
        "missing": len(failures),
        "failures": failures,
    }


def _assert_full_authorized(release: Mapping[str, Any], confirmation: str | None) -> None:
    if confirmation != FULL_LAUNCH_CONFIRMATION:
        raise ValueError("Full mechanism repair requires the explicit confirmation token")
    if not FULL_LAUNCH_AUTHORIZATION_PATH.exists():
        raise ValueError("Full mechanism repair is blocked pending an authorization sidecar")
    authorization = json.loads(FULL_LAUNCH_AUTHORIZATION_PATH.read_text())
    canary_audit = audit_outputs("canary", release)
    expected = {
        "full_launch_authorized": True,
        "release_sha256": release["release_sha256"],
        "confirmation": FULL_LAUNCH_CONFIRMATION,
        "canary_audit_sha256": canary_audit["audit_sha256"],
    }
    if authorization != expected:
        raise ValueError("Mechanism-repair authorization sidecar does not match the release")


def _concurrency_limit(scope: str, stage: int | None, release: Mapping[str, Any]) -> int:
    if scope == "canary":
        if stage is not None:
            raise ValueError("Canary mechanism repair does not accept --stage")
        return int(release["execution_acceptance"]["canary_max_concurrent"])
    if stage not in (1, 2, 3):
        raise ValueError("Full mechanism repair requires an explicit --stage")
    return int(release["full_launch_stages"][str(stage)]["max_concurrent"])


def _assert_concurrency(scope: str, stage: int | None, max_concurrent: int, release: Mapping[str, Any]) -> None:
    limit = _concurrency_limit(scope, stage, release)
    if max_concurrent < 1 or max_concurrent > limit:
        raise ValueError(f"max_concurrent must be in [1, {limit}] for this scope/stage")


def launch(
    scope: str,
    *,
    release_sha256: str,
    max_concurrent: int,
    confirmation: str | None,
    stage: int | None,
) -> None:
    release = _load_release(release_sha256)
    _assert_concurrency(scope, stage, max_concurrent, release)
    if scope == "full":
        _assert_full_authorized(release, confirmation)
        assert stage is not None
        for prerequisite in range(1, stage):
            audit_outputs("full", release, stage=prerequisite)
    configs = _pod_configs(scope)
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    base._audit_frozen_provenance(scope, parent_release, configs)
    readiness = _checkpoint_readiness(scope, release, stage)
    if readiness["missing"]:
        raise RuntimeError(f"Mechanism-repair checkpoint readiness failed: {readiness}")
    parent_readiness = _parent_result_readiness(scope, release, stage)
    if parent_readiness["missing"]:
        raise RuntimeError(f"Mechanism-repair parent-result readiness failed: {parent_readiness}")
    run(*_steps(scope, release, configs, stage), max_concurrent=max_concurrent, force_run_failed=True)


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _is_zero_vector_statistic(trunk: Mapping[str, Any], *, checkpoint_label: str) -> bool:
    """Is this cosine undefined because one side is the zero vector, for the one reason we accept?

    The probe defines the cosine exactly when `left_norm * right_norm > 0`, so an undefined cosine means
    one of the two vectors has no direction at all. There is exactly one state where that is expected: at
    the `final` checkpoint the schedule has decayed the learning rate to zero, so the corrected optimizer
    update `data_update - no_data_update` is identically zero and there is nothing to compare. A cosine
    between zero vectors is undefined, NOT zero, so it is recorded as missing rather than imputed --
    coding it as zero would inject a fabricated alignment into a descriptive statistic.

    Both the checkpoint AND the recorded norms are required. The norms alone are not enough: a zero
    update at a checkpoint where the learning rate is still positive has no benign explanation and is
    evidence of a fault, so accepting it on the norms alone would let exactly the failure this audit
    exists to catch pass as a real result. The canary manifest contains no `final` checkpoint at all,
    which makes the canary a genuine test of this: under the norms-only rule it could not fail.
    """
    if checkpoint_label != "final":
        return False
    if trunk.get("cosine") is not None or trunk.get("cosine_defined") is not False:
        return False
    norms = [trunk.get("left_norm"), trunk.get("right_norm")]
    if any(norm is None for norm in norms):
        return False
    return min(float(norm) for norm in norms) == 0.0


def _assert_defined_statistic(statistic: Mapping[str, Any], *, label: str, checkpoint_label: str) -> None:
    for geometry in ("raw", "projected"):
        if "trunk" not in statistic.get(geometry, {}):
            raise RuntimeError(f"Repair output omits {label} {geometry}/trunk")
        trunk = statistic[geometry]["trunk"]
        if _contains_nonfinite(trunk):
            raise RuntimeError(f"Repair output has non-finite {label} {geometry}/trunk statistic")
        if trunk.get("cosine_defined") is True and trunk.get("cosine") is not None:
            continue
        if not _is_zero_vector_statistic(trunk, checkpoint_label=checkpoint_label):
            raise RuntimeError(f"Repair output has undefined {label} {geometry}/trunk cosine at {checkpoint_label}")


def _assert_close(observed: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, (str, bool)) or isinstance(observed, (str, bool)):
        if expected != observed:
            raise RuntimeError(f"Repair did not reproduce parent {label}: {observed} != {expected}")
        return
    if expected is None or observed is None:
        if expected != observed:
            raise RuntimeError(f"Repair did not reproduce parent {label}: {observed} != {expected}")
        return
    tolerance = 5e-6 * max(abs(float(expected)), abs(float(observed)), 1.0)
    if abs(float(expected) - float(observed)) > tolerance:
        raise RuntimeError(f"Repair did not reproduce parent {label}: {observed} != {expected}")


def _assert_matches_parent_probe_statistics(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    scope: str,
    release: Mapping[str, Any],
) -> None:
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    provenance = _parent_result_provenance(release)
    probe_row_ids = _json_mapping(row, "distribution_probe_row_ids_json", str)
    for source in _json_names(row, "source_distribution_ids_json"):
        uri = _path_join(
            parent_release["result_root"],
            scope,
            "probe",
            row["parent_probe_group_id"],
            release["parent_result_artifact_version"],
            "rows",
            f"{probe_row_ids[source]}.json",
        )
        if uri not in provenance:
            raise RuntimeError(f"Parent probe result is not pinned by the release: {uri}")
        parent_document = _read_pinned_parent_result(uri, provenance[uri])
        parent_row = parent_document.get("row", {})
        if (
            parent_document.get("release_sha256") != parent_release["release_sha256"]
            or parent_row.get("row_id") != probe_row_ids[source]
            or parent_row.get("distribution_id") != source
        ):
            raise RuntimeError(f"Parent probe output identity mismatch: {uri}")
        repair_summary = document["numerical_summaries"][f"probe:{source}"]
        parent_summary = parent_document["numerical_summary"]
        for key in (
            "replicate_block_count",
            "optimizer_update_draw_count",
            "loss_mean",
            "loss_std",
            "loss_min",
            "loss_max",
            "first_batch_sha256",
            "data_update_norm_mean",
            "optimizer_memory_update_norm_mean",
            "corrected_update_norm_mean",
            "optimizer_memory_update_nonzero",
        ):
            _assert_close(repair_summary[key], parent_summary[key], label=f"{source}/numerical_summary/{key}")
        for target in _json_names(row, "target_distribution_ids_json"):
            parent_statistics = parent_document["pairwise_statistics"][target]
            repair_gradient = document["target_source_gradient_statistics"][target][source]
            repair_utility = document["target_source_utility_statistics"][target][source]
            for geometry, parent_key in (
                ("raw", "raw_gradient"),
                ("projected", "projected_gradient"),
            ):
                for component, expected in parent_statistics[parent_key].items():
                    observed = repair_gradient[geometry][component]
                    _assert_close(
                        observed["dot"], expected["dot"], label=f"{target}/{source}/{geometry}/{component}/dot"
                    )
                    _assert_close(
                        observed["left_norm"],
                        expected["right_norm"],
                        label=f"{target}/{source}/{geometry}/{component}/target_gradient_norm",
                    )
                    _assert_close(
                        observed["right_norm"],
                        expected["left_norm"],
                        label=f"{target}/{source}/{geometry}/{component}/source_gradient_norm",
                    )
                    _assert_close(
                        observed["cosine"],
                        expected["cosine"],
                        label=f"{target}/{source}/{geometry}/{component}/cosine",
                    )
            for geometry, parent_key in (
                ("raw", "raw_optimizer_update"),
                ("projected", "projected_optimizer_update"),
            ):
                for component, expected in parent_statistics[parent_key].items():
                    observed = repair_utility[geometry][component]
                    _assert_close(
                        observed["right_norm"],
                        expected["left_norm"],
                        label=f"{target}/{source}/{geometry}/{component}/source_update_norm",
                    )


def _validate_scientific_document(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    scope: str,
    release: Mapping[str, Any],
) -> None:
    if document.get("kind") != "gradient_mechanism_repair":
        raise RuntimeError("Repair row has the wrong kind")
    if document.get("endpoint_metrics_read") is not False:
        raise RuntimeError("Repair row does not prove endpoint blindness")
    if document.get("no_data_update_invariance", {}).get("passed") is not True:
        raise RuntimeError("Repair row did not pass the common no-data update audit")
    pair = document.get("source_pair_statistics", {}).get("starcoder__vs__nemotron")
    if pair is None:
        raise RuntimeError("Repair row omits H1 StarCoder-Nemotron geometry")
    checkpoint_label = str(row["checkpoint_label"])
    _assert_defined_statistic(pair["gradient"], label="H1 gradient", checkpoint_label=checkpoint_label)
    _assert_defined_statistic(pair["optimizer_update"], label="H1 optimizer update", checkpoint_label=checkpoint_label)
    targets = _json_names(row, "target_distribution_ids_json")
    sources = _json_names(row, "source_distribution_ids_json")
    expected_contrasts = {f"{freeze.GLOBAL_STARCODER}__minus__{freeze.NEMOTRON}"}
    if {freeze.SUPPORT_STARCODER, freeze.GLOBAL_STARCODER} <= set(sources):
        expected_contrasts.add(f"{freeze.SUPPORT_STARCODER}__minus__{freeze.GLOBAL_STARCODER}")
    target_contrasts = document.get("target_source_choice_contrasts", {})
    if set(target_contrasts) != set(targets):
        raise RuntimeError("Repair target inventory drifted")
    for target in targets:
        missing = expected_contrasts - set(target_contrasts[target])
        if missing:
            raise RuntimeError(f"Repair target {target} omits contrasts: {sorted(missing)}")
        for contrast in expected_contrasts:
            _assert_defined_statistic(
                target_contrasts[target][contrast]["statistic"],
                label=f"{target}/{contrast}",
                checkpoint_label=checkpoint_label,
            )
    gradient_statistics = document.get("target_source_gradient_statistics", {})
    if set(gradient_statistics) != set(targets):
        raise RuntimeError("Repair target-gradient inventory drifted")
    _assert_matches_parent_probe_statistics(document, row, scope=scope, release=release)


def _workload_shape_sha256(row: Mapping[str, Any]) -> str:
    blocks = _json_mapping(row, "distribution_block_counts_json", int)
    sources = _json_names(row, "source_distribution_ids_json")
    targets = _json_names(row, "target_distribution_ids_json")
    return freeze.canonical_sha256(
        {
            "source_block_counts": {name: blocks[name] for name in sources},
            "target_block_counts": {name: blocks[name] for name in targets},
            "probe_batch_size": base.PROBE_BATCH_SIZE,
        }
    )


def _validate_execution_document(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    gate = release["execution_acceptance"]
    observation = document.get("execution_observation", {})
    expected_blocks = _json_mapping(row, "distribution_block_counts_json", int)
    required_device_kind = str(gate["required_device_kind_substring"])
    checks = {
        "wall_seconds": float(observation.get("wall_seconds", math.inf)) <= float(gate["max_group_wall_seconds"]),
        "peak_host_rss_bytes": (
            int(observation.get("peak_host_rss_bytes", 1 << 63)) <= int(gate["max_peak_host_rss_bytes"])
        ),
        "backend": observation.get("backend") == gate["required_backend"],
        "device_kind": any(required_device_kind in str(kind) for kind in observation.get("device_kinds", ())),
        "device_count": int(observation.get("device_count", 0)) >= int(gate["minimum_device_count"]),
        "local_device_count": int(observation.get("local_device_count", 0)) >= int(gate["minimum_local_device_count"]),
        "probe_batch_size": int(observation.get("probe_batch_size", 0)) == int(gate["probe_batch_size"]),
        "max_distribution_block_count": (
            int(observation.get("max_distribution_block_count", 0)) == max(expected_blocks.values())
        ),
        "distribution_count": int(observation.get("distribution_count", 0)) == len(expected_blocks),
    }
    no_data = document.get("no_data_update_invariance", {})
    checks["no_data_absolute"] = float(no_data.get("max_abs_diff", math.inf)) <= float(
        gate["max_no_data_update_abs_diff"]
    )
    checks["no_data_relative"] = float(no_data.get("relative_to_min_corrected_update_norm", math.inf)) <= float(
        gate["max_no_data_update_relative_diff"]
    )
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"Mechanism-repair execution gate failed for {row['row_id']}: {failures}")
    return {
        **observation,
        "workload_shape_sha256": _workload_shape_sha256(row),
        "no_data_update_max_abs_diff": float(no_data["max_abs_diff"]),
        "no_data_update_relative_diff": float(no_data["relative_to_min_corrected_update_norm"]),
    }


def audit_outputs(scope: str, release: Mapping[str, Any], stage: int | None = None) -> dict[str, Any]:
    rows = _read_manifest(scope, release, stage)
    root = _path_join(freeze.RESULT_ROOT, scope)
    fs, plain_root = fsspec.core.url_to_fs(root)
    found_rows = set(fs.glob(f"{plain_root}/*/{ARTIFACT_VERSION}/rows/*.json"))
    expected_rows = {f"{plain_root}/{row['group_id']}/{ARTIFACT_VERSION}/rows/{row['row_id']}.json": row for row in rows}
    found_markers = set(fs.glob(f"{plain_root}/*/{ARTIFACT_VERSION}/group_complete.json"))
    expected_markers = {f"{plain_root}/{row['group_id']}/{ARTIFACT_VERSION}/group_complete.json": row for row in rows}
    if stage is not None:
        expected_groups = {row["group_id"] for row in rows}
        found_rows = {path for path in found_rows if path.split("/")[-4] in expected_groups}
        found_markers = {path for path in found_markers if path.split("/")[-3] in expected_groups}
    invalid = 0
    nonfinite = 0
    endpoint_metrics_read = False
    execution_observations: list[dict[str, Any]] = []
    for path in sorted(set(expected_rows) & found_rows):
        document = _read_document(f"gs://{path}")
        assert document is not None
        row = expected_rows[path]
        required = (
            document.get("kind") == "gradient_mechanism_repair"
            and document.get("scope") == scope
            and document.get("group_id") == row["group_id"]
            and document.get("row") == row
            and document.get("release_sha256") == release["release_sha256"]
            and document.get("endpoint_metrics_read") is False
            and document.get("scientific_status") == freeze.SCIENTIFIC_STATUS
            and document.get("identity_sha256") == _row_identity(row, release["release_sha256"])
            and "source_pair_statistics" in document
            and "target_source_gradient_statistics" in document
            and "target_source_choice_contrasts" in document
        )
        invalid += not required
        nonfinite += _contains_nonfinite(document)
        endpoint_metrics_read = endpoint_metrics_read or document.get("endpoint_metrics_read") is not False
        _validate_scientific_document(document, row, scope=scope, release=release)
        execution_observations.append(_validate_execution_document(document, row, release))
    for path in sorted(set(expected_markers) & found_markers):
        marker = _read_document(f"gs://{path}")
        assert marker is not None
        row = expected_markers[path]
        row_path = f"{plain_root}/{row['group_id']}/{ARTIFACT_VERSION}/rows/{row['row_id']}.json"
        row_document = _read_document(f"gs://{row_path}")
        assert row_document is not None
        config = MechanismGroupConfig(
            scope=scope,
            group_id=row["group_id"],
            checkpoint_uri=row["checkpoint_uri"],
            checkpoint_step=int(row["checkpoint_step"]),
            expected_restored_state_step=int(row["expected_restored_state_step"]),
            row=row,
            pod_config=None,
            output_path="",
            parent_cache_provenance_sha256="",
            release_sha256=release["release_sha256"],
        )
        marker_required = (
            marker.get("identity_sha256") == _group_identity(config)
            and marker.get("kind") == "gradient_mechanism_repair_group"
            and marker.get("scope") == scope
            and marker.get("group_id") == row["group_id"]
            and marker.get("row_count") == 1
            and marker.get("release_sha256") == release["release_sha256"]
            and marker.get("scientific_status") == freeze.SCIENTIFIC_STATUS
            and marker.get("endpoint_metrics_read") is False
        )
        if not marker_required:
            raise RuntimeError(f"Repair completion marker identity drifted: {path}")
        endpoint_metrics_read = endpoint_metrics_read or marker.get("endpoint_metrics_read") is not False
        if marker.get("row_document_sha256") != row_document["payload_sha256"]:
            raise RuntimeError(f"Repair completion marker does not bind row payload: {path}")
        if marker.get("execution_observation") != row_document.get("execution_observation"):
            raise RuntimeError(f"Repair completion marker execution observation drifted: {path}")
    workload_shapes = {observation["workload_shape_sha256"] for observation in execution_observations}
    expected_shapes = {_workload_shape_sha256(row) for row in rows}
    if workload_shapes != expected_shapes:
        raise RuntimeError("Mechanism-repair audited workload-shape inventory drifted")
    if scope == "canary":
        gate = release["execution_acceptance"]
        observed_max_blocks = max((item["max_distribution_block_count"] for item in execution_observations), default=0)
        observed_max_distributions = max((item["distribution_count"] for item in execution_observations), default=0)
        if observed_max_blocks < int(gate["canary_min_max_distribution_block_count"]):
            raise RuntimeError("Mechanism-repair canary omitted the frozen maximum block-count shape")
        if observed_max_distributions < int(gate["canary_min_max_distribution_count"]):
            raise RuntimeError("Mechanism-repair canary omitted the frozen maximum distribution-count shape")
    report = {
        "scope": scope,
        "stage": stage,
        "expected_rows": len(rows),
        "found_rows": len(set(expected_rows) & found_rows),
        "missing_rows": len(set(expected_rows) - found_rows),
        "unexpected_rows": len(found_rows - set(expected_rows)),
        "complete_groups": len(set(expected_markers) & found_markers),
        "missing_group_markers": len(set(expected_markers) - found_markers),
        "unexpected_group_markers": len(found_markers - set(expected_markers)),
        "invalid_documents": invalid,
        "nonfinite_documents": nonfinite,
        "endpoint_metrics_read": endpoint_metrics_read,
        "all_endpoint_metrics_unread": not endpoint_metrics_read,
        "execution_gate": {
            "passed": True,
            "workload_shape_count": len(workload_shapes),
            "max_group_wall_seconds": max((item["wall_seconds"] for item in execution_observations), default=0.0),
            "max_peak_host_rss_bytes": max((item["peak_host_rss_bytes"] for item in execution_observations), default=0),
            "max_no_data_update_abs_diff": max(
                (item["no_data_update_max_abs_diff"] for item in execution_observations), default=0.0
            ),
            "max_no_data_update_relative_diff": max(
                (item["no_data_update_relative_diff"] for item in execution_observations), default=0.0
            ),
        },
    }
    failures = {
        key: report[key]
        for key in (
            "missing_rows",
            "unexpected_rows",
            "missing_group_markers",
            "unexpected_group_markers",
            "invalid_documents",
            "nonfinite_documents",
            "endpoint_metrics_read",
        )
        if report[key]
    }
    if failures:
        raise RuntimeError(f"Mechanism-repair audit failed closed: {failures}")
    report["audit_sha256"] = freeze.canonical_sha256(report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("canary", "full"), required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--mode", choices=("readiness", "audit", "launch"), default="readiness")
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--confirm-full-launch")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    release = _load_release(args.release_sha256)
    max_concurrent = args.max_concurrent
    if args.mode == "launch":
        if args.scope == "full" and max_concurrent is None:
            raise ValueError("Full mechanism repair requires an explicit --max-concurrent")
        concurrency_limit = _concurrency_limit(args.scope, args.stage, release)
        max_concurrent = concurrency_limit if max_concurrent is None else max_concurrent
        _assert_concurrency(args.scope, args.stage, max_concurrent, release)
    if args.mode == "readiness":
        configs = _pod_configs(args.scope)
        parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
        print(
            json.dumps(
                {
                    "checkpoint_readiness": _checkpoint_readiness(args.scope, release, args.stage),
                    "parent_result_readiness": _parent_result_readiness(args.scope, release, args.stage),
                    "parent_provenance": base._audit_frozen_provenance(args.scope, parent_release, configs),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.mode == "audit":
        print(json.dumps(audit_outputs(args.scope, release, args.stage), indent=2, sort_keys=True))
        return
    launch(
        args.scope,
        release_sha256=args.release_sha256,
        max_concurrent=int(max_concurrent),
        confirmation=args.confirm_full_launch,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
