# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rerun the H4 short rollouts recording every target, not only the code one (ATOM-031).

The v6 rollouts measured `paloma_programming_languages` alone. StarCoder is code, so that loss falls
monotonically in the StarCoder weight over 512 updates: every one of the 52 parents has its argmin at
q = 1.0, and recovering the argmin of a monotone curve is free for any monotone utility. The published
"100% exact q match, zero selection regret" therefore measures nothing about whether an optimizer-aware
utility can choose a continuation.

The trade-off that the two-phase question is about is visible on the utility side, where all four targets
were recorded: over q = 0.25 to 1.0 the two code targets rise while the two text targets fall. This module
records the same four targets behaviourally, so a macro with an interior optimum exists for a utility to
be right or wrong about.

It is a separate runtime and a separate release rather than an edit to the probe because the v6 probe
release is the frozen parent of the in-flight mechanism-repair work and must keep hashing to the runtime
it was frozen against. Everything scientific is reused unchanged from that release -- the same rollout
manifests, parents, q grid, update horizon, readout steps, and frozen source streams -- via helpers
imported from the probe runtime. Only the measurement and the output root differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import fsspec
from fray.types import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.execution.remote import remote

from experiments.domain_phase_mix import starcoder_wsd80_gradient_probe as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_h4_macro_rollout_20260819 as freeze,
)

SCHEMA_VERSION = "2026-08-19-h4-macro-rollout-v1"
ARTIFACT_VERSION = "2026.08.19.1"


def _row_identity(row: Mapping[str, Any], release_sha256: str) -> str:
    return freeze.canonical_sha256(
        {
            "row_id": row["row_id"],
            "group_id": row["group_id"],
            "checkpoint_uri": row["checkpoint_uri"],
            "train_config_sha256": row["train_config_sha256"],
            "release_sha256": release_sha256,
            "release_version": freeze.RELEASE_VERSION,
        }
    )


def _group_identity(group_id: str, rows: Sequence[Mapping[str, Any]], release_sha256: str) -> str:
    return freeze.canonical_sha256(
        {
            "group_id": group_id,
            "rows": sorted(row["row_id"] for row in rows),
            "release_sha256": release_sha256,
        }
    )


def _write_create_only_json(path: str, payload: dict[str, Any], *, identity_sha256: str) -> None:
    """Create-only with this release's schema, so a rerun can never overwrite or be confused with v6."""
    document = {**payload, "schema_version": SCHEMA_VERSION, "identity_sha256": identity_sha256}
    encoded = (base._canonical_json(document) + "\n").encode()
    fs, plain_path = fsspec.core.url_to_fs(path)
    parent_directory = os.path.dirname(plain_path)
    if parent_directory:
        fs.makedirs(parent_directory, exist_ok=True)
    try:
        with fs.open(plain_path, "xb") as handle:
            handle.write(encoded)
        return
    except FileExistsError as error:
        with fs.open(plain_path, "rb") as handle:
            existing = json.load(handle)
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"Existing output has a different schema: {path}") from error
        if existing.get("identity_sha256") != identity_sha256:
            raise RuntimeError(f"Existing output is claimed by another row identity: {path}") from error


def _read_output_document(path: str) -> dict[str, Any] | None:
    fs, plain_path = fsspec.core.url_to_fs(path)
    if not fs.exists(plain_path):
        return None
    with fs.open(plain_path, "rb") as handle:
        document = json.load(handle)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Existing row has an unexpected schema: {path}")
    return document


def _row_is_complete(output_path: str, row: Mapping[str, Any], release_sha256: str) -> bool:
    document = _read_output_document(base._row_path(output_path, str(row["row_id"])))
    if document is None:
        return False
    if document.get("identity_sha256") != _row_identity(row, release_sha256):
        raise RuntimeError(f"Completed row identity drifted: {row['row_id']}")
    return True


def _group_is_complete(config: base.RolloutGroupConfig) -> bool:
    marker = _read_output_document(base._path_join(config.output_path, "group_complete.json"))
    if marker is None:
        return False
    if marker.get("identity_sha256") != _group_identity(config.group_id, config.rows, config.release_sha256):
        raise RuntimeError(f"Completed group identity drifted: {config.group_id}")
    if marker.get("row_count") != len(config.rows):
        raise RuntimeError(f"Completed group row count drifted: {config.group_id}")
    return all(_row_is_complete(config.output_path, row, config.release_sha256) for row in config.rows)


def _verify_group_contract(config: base.RolloutGroupConfig) -> None:
    """The probe's contract, re-pointed at this release's root. Everything else it checks is unchanged."""
    if not config.output_path.startswith(f"{freeze.RESULT_ROOT}/{config.scope}/"):
        raise ValueError(f"Output path is outside the frozen release root: {config.output_path}")
    support = base._starcoder_support_contract(cast(TrainLmConfig, config.pod_config.train_config))
    trajectory_id = str(config.rows[0]["parent_trajectory_id"])
    if f"_{support['support_id']}_" not in trajectory_id:
        raise ValueError(f"Trajectory {trajectory_id} does not match runtime support {support['support_id']}")
    for row in config.rows:
        if row["group_id"] != config.group_id:
            raise ValueError(f"Row {row['row_id']} belongs to a different group")
        if row["checkpoint_uri"] != config.checkpoint_uri:
            raise ValueError(f"Row {row['row_id']} points to a different checkpoint")
        if int(row["parent_checkpoint_step"]) != config.checkpoint_step:
            raise ValueError(f"Row {row['row_id']} points to a different checkpoint step")


def run_rollout_group(config: base.RolloutGroupConfig) -> None:
    """One parent state, seven StarCoder weights, every configured target measured at every readout."""
    _verify_group_contract(config)
    if _group_is_complete(config):
        return
    metadata = base._read_checkpoint_metadata(config.checkpoint_uri, config.checkpoint_step)
    train_config = base._prepare_train_config(config.pod_config, config.checkpoint_uri, config.group_id)
    trainer, state, Pos, data_key, optimizer_mask = base._initialize_runtime(train_config)
    try:
        if int(state.step) != config.expected_restored_state_step:
            raise RuntimeError("Rollout restored the wrong checkpoint step")
        runtime_summary = {
            "restoration": base._restored_optimizer_summary(
                state,
                config.checkpoint_step,
                config.expected_restored_state_step,
                allow_partial_checkpoint=train_config.trainer.allow_partial_checkpoint,
            ),
            "muon_projection": base._runtime_muon_projection_coverage(state.model, optimizer_mask),
            "optimizer_schedule": base._optimizer_schedule_summary(train_config),
        }
        sources, stream_summary = base._source_views(train_config, Pos, data_key, config.expected_restored_state_step)
        runtime_summary["source_stream"] = stream_summary
        _, _, evaluation_loss, train_step = base._gradient_functions(trainer)

        target_blocks = dict(config.target_block_counts)
        sequence_sets = dict(config.target_sequence_set_ids)
        targets = {
            name: base._distribution_dataset(
                name,
                sequence_set_id=sequence_sets[name],
                train_config=train_config,
                Pos=Pos,
                sources=sources,
            )
            for name in sorted(target_blocks)
        }

        for row in config.rows:
            if _row_is_complete(config.output_path, row, config.release_sha256):
                continue
            updates = int(row["updates"])
            readouts = sorted({int(step) for step in str(row["readout_steps"]).split("|") if int(step) <= updates})
            dataset = base._weighted_training_dataset(
                starcoder_weight=float(row["starcoder_weight"]),
                sequence_set_id=f"rollout:{config.group_id}:order{row['rollout_order_seed']}",
                train_config=train_config,
                Pos=Pos,
                sources=sources,
            )
            iterator = iter(trainer.data_loader(dataset))
            rollout_state = state
            measurements: list[dict[str, Any]] = []
            for update_index in range(1, updates + 1):
                rollout_state = train_step(rollout_state, (next(iterator),), {}).new_state
                if update_index in readouts:
                    measurements.append(
                        {
                            "updates": update_index,
                            "targets": {
                                name: base._evaluate_target(
                                    trainer=trainer,
                                    model=rollout_state.model,
                                    dataset=target_dataset,
                                    evaluation_loss=evaluation_loss,
                                    blocks=target_blocks[name],
                                )
                                for name, target_dataset in targets.items()
                            },
                        }
                    )
            if int(rollout_state.step) != config.expected_restored_state_step + updates:
                raise RuntimeError("Rollout state advanced by the wrong number of updates")
            if [measurement["updates"] for measurement in measurements] != readouts:
                raise RuntimeError("Rollout did not emit every frozen readout exactly once")
            _write_create_only_json(
                base._row_path(config.output_path, str(row["row_id"])),
                {
                    "kind": "h4_macro_rollout",
                    "scope": config.scope,
                    "group_id": config.group_id,
                    "row": row,
                    "checkpoint_metadata": metadata,
                    "restored_state_step": int(state.step),
                    "final_state_step": int(rollout_state.step),
                    "runtime_summary": runtime_summary,
                    "cache_provenance_sha256": config.cache_provenance_sha256,
                    "release_sha256": config.release_sha256,
                    "readouts": measurements,
                    "endpoint_metrics_read": False,
                },
                identity_sha256=_row_identity(row, config.release_sha256),
            )
        _write_create_only_json(
            base._path_join(config.output_path, "group_complete.json"),
            {
                "kind": "h4_macro_rollout_group",
                "scope": config.scope,
                "group_id": config.group_id,
                "row_count": len(config.rows),
                "checkpoint_metadata": metadata,
                "runtime_summary": runtime_summary,
                "cache_provenance_sha256": config.cache_provenance_sha256,
                "release_sha256": config.release_sha256,
                "endpoint_metrics_read": False,
            },
            identity_sha256=_group_identity(config.group_id, config.rows, config.release_sha256),
        )
    finally:
        base._close_runtime(trainer)


def _artifact_name(scope: str, group_id: str) -> str:
    prefix = f"{freeze.MARIN_PREFIX}/"
    if not freeze.RESULT_ROOT.startswith(prefix):
        raise ValueError(f"Result root is outside the frozen Marin prefix: {freeze.RESULT_ROOT}")
    return f"{freeze.RESULT_ROOT.removeprefix(prefix)}/{scope}/rollout/{group_id}"


def _load_release(expected_sha256: str) -> dict[str, Any]:
    release = json.loads(freeze.RELEASE_PATH.read_text())
    if release["release_sha256"] != expected_sha256:
        raise ValueError(f"Release sha256 mismatch: {release['release_sha256']} != {expected_sha256}")
    if release["release_sha256"] != freeze.canonical_sha256({**release, "release_sha256": ""}):
        raise ValueError("Release file is not self-consistent")
    for relative_path, sha256 in release["implementation_files"].items():
        if freeze.file_sha256(freeze.REPO_ROOT / relative_path) != sha256:
            raise ValueError(f"Implementation file drifted since the freeze: {relative_path}")
    return release


def _read_manifest(scope: str) -> list[dict[str, Any]]:
    path = freeze.OUTPUT_DIR / f"{scope}_rollout_manifest.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _rollout_steps(scope: str, *, release: dict[str, Any]) -> list[ArtifactStep[Artifact]]:
    rows = _read_manifest(scope)
    pod_configs = base.freeze._canary_configs() if scope == "canary" else base.freeze._full_configs()
    resources = ResourceConfig.with_tpu(
        base.TPU_TYPE, cpu=base.TPU_HOST_CPU, ram=base.TPU_HOST_RAM, regions=(base.TPU_REGION,), zone=base.TPU_ZONE
    )
    steps: list[ArtifactStep[Artifact]] = []
    for group_id, group in sorted(base._group_rows(rows).items()):
        configuration = base.RolloutGroupConfig(
            scope=scope,
            group_id=group_id,
            checkpoint_uri=group[0]["checkpoint_uri"],
            checkpoint_step=int(group[0]["parent_checkpoint_step"]),
            expected_restored_state_step=int(group[0]["expected_restored_state_step"]),
            rows=group,
            pod_config=pod_configs[group[0]["parent_trajectory_id"]],
            output_path="",
            cache_provenance_sha256=release["parent_release_sha256"],
            release_sha256=release["release_sha256"],
            target_block_counts=base._target_block_counts(release),
            target_sequence_set_ids=base._target_sequence_set_ids(group[0]),
        )
        steps.append(
            ArtifactStep(
                name=_artifact_name(scope, group_id),
                version=ARTIFACT_VERSION,
                artifact_type=Artifact,
                run=remote(run_rollout_group, resources=resources, name=group_id),
                build_config=lambda ctx, base_config=configuration: replace(base_config, output_path=ctx.output_path),
            )
        )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("canary", "full"), required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--mode", choices=("readiness", "launch"), default="readiness")
    parser.add_argument("--max-concurrent", type=int)
    args = parser.parse_args()

    release = _load_release(args.release_sha256)
    rows = _read_manifest(args.scope)
    groups = base._group_rows(rows)
    if args.mode == "readiness":
        print(
            json.dumps(
                {
                    "release_version": release["release_version"],
                    "primary_objective": release["primary_objective"],
                    "scope": args.scope,
                    "groups": len(groups),
                    "rows": len(rows),
                    "result_root": freeze.RESULT_ROOT,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    limit = freeze.CANARY_MAX_CONCURRENT if args.scope == "canary" else freeze.FULL_MAX_CONCURRENT
    max_concurrent = args.max_concurrent or limit
    if max_concurrent < 1 or max_concurrent > limit:
        raise ValueError(f"--max-concurrent must be between 1 and {limit}")
    run(*_rollout_steps(args.scope, release=release), max_concurrent=max_concurrent, force_run_failed=True)


if __name__ == "__main__":
    main()
