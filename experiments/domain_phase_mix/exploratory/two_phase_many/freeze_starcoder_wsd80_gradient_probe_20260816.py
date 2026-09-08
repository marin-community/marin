# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze detached gradient-probe manifests without reading endpoint outcomes."""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fsspec
from haliax import Axis
from levanter.utils.thread_utils import blocking_wait
from marin.execution.lazy import materialized_config
from marin.utilities.json_encoder import CustomJsonEncoder

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict as canary
from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as full

REPO_ROOT = Path(__file__).resolve().parents[4]
DESIGN_DIR = Path(__file__).with_name("reference_outputs") / "starcoder_wsd80_gradient_conflict_design_20260811_v9"
OUTPUT_DIR = Path(__file__).with_name("reference_outputs") / "starcoder_wsd80_gradient_probe_release_v6_20260816"
PROBE_RUNTIME_PATH = REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_gradient_probe.py"
MARIN_PREFIX = "gs://marin-us-central1"
RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_probe_review_v9_release_v6_20260816"
)
CANARY_TRAJECTORY_ROOT = (
    "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_20260810/trajectories"
)
FULL_TRAJECTORY_ROOT = (
    "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_review_v9_20260811/trajectories"
)
CANARY_VERSION = "2026.08.10"
FULL_VERSION = "2026.08.11.9"
PROBE_RELEASE_VERSION = "2026-08-16-detached-probe-v6"
PROBE_SEQUENCES_PER_BLOCK = 64
PRIMARY_TARGET_BLOCKS = 64
CANARY_HOLDOUT_SEED = 2_026_081_102
CANARY_HOLDOUT_PARTITION = "random_sparse_swap"
FULL_HOLDOUT_SEED = full.EXPECTED_TRAIN_HOLDOUT_SEED
FULL_HOLDOUT_PARTITION = full.EXPECTED_TRAIN_HOLDOUT_PARTITION
TRAINING_COMPONENTS = full.EXPECTED_TRAINING_COMPONENT_NAMES
TARGET_COMPONENTS = {
    "paloma_programming_languages": "paloma/dolma_100_programing_languages-llama3",
    "uncheatable_github_python": "uncheatable_eval/github_python-llama3",
    "uncheatable_wikipedia_english": "uncheatable_eval/wikipedia_english-llama3",
    "paloma_c4_en": "paloma/c4_en-llama3",
}
CANARY_ROLLOUT_WEIGHTS = (0.0, 0.25, 0.35, 0.45, 0.55, 0.75, 1.0)
CANARY_ROLLOUT_READOUTS = (128, 256, 512)
CANARY_ROLLOUT_CHECKPOINT_LABEL = "decay_minus_64"
CANARY_ROLLOUT_CHECKPOINT_STEP = 22_544


def expected_restored_state_step(checkpoint_label_step: int) -> int:
    """Return Levanter's next-update counter stored after a labeled completed step."""
    return checkpoint_label_step + 1


def canonical_json(value: Any) -> str:
    """Return the stable JSON encoding used for all release identities."""
    return json.dumps(value, cls=CustomJsonEncoder, separators=(",", ":"), sort_keys=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def target_sequence_set_ids(training_seed: int) -> dict[str, str]:
    """Return the frozen target subsets shared by every probe kind for one seed."""
    return {distribution: f"frozen:s{training_seed}:{distribution}" for distribution in sorted(TARGET_COMPONENTS)}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_uri(trajectory_id: str, step: int, *, scope: str) -> str:
    if scope == "canary":
        root, version = CANARY_TRAJECTORY_ROOT, CANARY_VERSION
    elif scope == "full":
        root, version = FULL_TRAJECTORY_ROOT, FULL_VERSION
    else:
        raise ValueError(f"Unknown scope: {scope}")
    return f"{root}/{trajectory_id}/{version}/checkpoints/step-{step}"


def probe_row_identity(row: dict[str, Any], *, scope: str) -> str:
    identity = {
        "scope": scope,
        "trajectory_id": row["trajectory_id"],
        "checkpoint_label": row["checkpoint_label"],
        "checkpoint_step": int(row["checkpoint_step"]),
        "expected_restored_state_step": int(row["expected_restored_state_step"]),
        "distribution_id": row["distribution_id"],
        "probe_sequence_set_id": row["probe_sequence_set_id"],
        "replicate_blocks": int(row["replicate_blocks"]),
        "sequences_per_block": int(row["sequences_per_block"]),
        "optimizer_update_draw_count": int(row["optimizer_update_draw_count"]),
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
    }
    return f"probe_{canonical_sha256(identity)[:24]}"


def probe_group_identity(row: dict[str, Any], *, scope: str) -> str:
    identity = {
        "scope": scope,
        "trajectory_id": row["trajectory_id"],
        "checkpoint_label": row["checkpoint_label"],
        "checkpoint_step": int(row["checkpoint_step"]),
        "expected_restored_state_step": int(row["expected_restored_state_step"]),
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
    }
    return f"probe_group_{canonical_sha256(identity)[:24]}"


def rollout_row_identity(row: dict[str, Any], *, scope: str) -> str:
    return f"rollout_{canonical_sha256({'scope': scope, **row, 'design_sha256': full.EXPECTED_DESIGN_SHA256})[:24]}"


def rollout_group_identity(row: dict[str, Any], *, scope: str) -> str:
    identity = {
        "scope": scope,
        "parent_trajectory_id": row["parent_trajectory_id"],
        "parent_checkpoint_label": row["parent_checkpoint_label"],
        "rollout_order_seed": int(row["rollout_order_seed"]),
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
    }
    return f"rollout_group_{canonical_sha256(identity)[:24]}"


def optimizer_row_identity(row: dict[str, Any], *, scope: str) -> str:
    return f"optimizer_{canonical_sha256({'scope': scope, **row, 'design_sha256': full.EXPECTED_DESIGN_SHA256})[:24]}"


def optimizer_group_identity(row: dict[str, Any], *, scope: str) -> str:
    identity = {
        "scope": scope,
        "parent_trajectory_id": row["parent_trajectory_id"],
        "parent_checkpoint_label": row["parent_checkpoint_label"],
        "parent_checkpoint_step": int(row["parent_checkpoint_step"]),
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
    }
    return f"optimizer_group_{canonical_sha256(identity)[:24]}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _component_cache_root(component: Any) -> str:
    cache_root = getattr(component, "cache_dir", None)
    if not cache_root:
        raise ValueError(f"Component lacks a cache_dir: {component}")
    return str(cache_root)


def _remote_file_sha256(uri: str) -> str:
    with fsspec.open(uri, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _remote_json(uri: str) -> dict[str, Any]:
    with fsspec.open(uri, "rb") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object at {uri}")
    return document


def _canary_configs() -> dict[str, Any]:
    canary_design = canary.load_canary_design()
    canary_steps = canary.build_training_steps(
        canary_design,
        marin_prefix=MARIN_PREFIX,
        tpu_type="v5p-8",
        tpu_region="us-central1",
        tpu_zone="us-central1-a",
        fork=False,
    )
    return {
        trajectory.trajectory_id: materialized_config(step, MARIN_PREFIX)
        for trajectory, step in zip(canary_design.trajectories, canary_steps, strict=True)
    }


def _full_configs() -> dict[str, Any]:
    trajectories, full_steps = full.build_training_steps(
        marin_prefix=MARIN_PREFIX,
        tpu_type="v5p-8",
        tpu_region="us-central1",
        tpu_zone="us-central1-a",
    )
    artifact_cache: dict[int, Any] = {}
    return {
        trajectory.trajectory_id: materialized_config(step, MARIN_PREFIX, artifact_cache=artifact_cache)
        for trajectory, step in zip(trajectories, full_steps, strict=True)
    }


def _materialized_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    return _canary_configs(), _full_configs()


def _config_identity(pod_config: Any) -> dict[str, Any]:
    train = pod_config.train_config
    tokenizer = train.data.the_tokenizer
    model = train.model
    configured_optimizer_horizon = train.optimizer_schedule_num_train_steps
    effective_optimizer_horizon = (
        train.trainer.num_train_steps if configured_optimizer_horizon is None else configured_optimizer_horizon
    )
    support_cap = train.data.max_train_batches
    support_start = train.data.max_train_batches_start
    if support_cap is None:
        if support_start is not None or train.data.max_train_batches_subset_seed is not None:
            raise ValueError("Full StarCoder support cannot carry a finite-support offset or seed")
        support_id = "full"
        support_batches = ""
        support_start_batches = ""
    else:
        if set(support_cap) != {"dolma/starcoder"}:
            raise ValueError(f"Unexpected finite-support components: {support_cap}")
        support_batches = int(support_cap["dolma/starcoder"])
        support_start_batches = 0 if support_start is None else int(support_start["dolma/starcoder"])
        if support_start_batches == 0:
            support_id = "m100a"
        elif support_start_batches == support_batches:
            support_id = "m100b"
        else:
            support_id = f"finite_start_{support_start_batches}"
    return {
        "output_path": pod_config.output_path,
        "training_seed": train.trainer.seed,
        "data_seed": train.data_seed,
        "total_steps": train.trainer.num_train_steps,
        "configured_optimizer_schedule_num_train_steps": (
            "" if configured_optimizer_horizon is None else configured_optimizer_horizon
        ),
        "effective_optimizer_schedule_num_train_steps": effective_optimizer_horizon,
        "train_batch_size": train.trainer.train_batch_size,
        "train_seq_len": train.train_seq_len,
        "model_type": type(model).__name__,
        "model_config_sha256": canonical_sha256(asdict(model)),
        "optimizer_type": type(train.optimizer).__name__,
        "optimizer_config_sha256": canonical_sha256(asdict(train.optimizer)),
        "tokenizer_name": str(train.data.tokenizer),
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_metadata_sha256": canonical_sha256(
            {
                "name_or_path": tokenizer.name_or_path,
                "vocab_size": tokenizer.vocab_size,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
        ),
        "train_weights_sha256": canonical_sha256(train.data.train_weights),
        "data_config_sha256": canonical_sha256(asdict(train.data)),
        "starcoder_support_id": support_id,
        "starcoder_support_batches": support_batches,
        "starcoder_support_start_batches": support_start_batches,
        "starcoder_support_pool_seed": (
            "" if train.data.max_train_batches_subset_seed is None else train.data.max_train_batches_subset_seed
        ),
        "starcoder_support_permutation_type": train.data.permutation_type,
        "full_train_config_sha256": canonical_sha256(asdict(train)),
    }


def _probe_rows(
    source_rows: list[dict[str, str]],
    *,
    scope: str,
    configs: dict[str, Any],
    target_sampling: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        trajectory_id = source["trajectory_id"]
        if trajectory_id not in configs:
            raise ValueError(f"Probe row has no materialized trajectory config: {trajectory_id}")
        row: dict[str, Any] = dict(source)
        design_blocks = int(source["replicate_blocks"])
        distribution = source["distribution_id"]
        if distribution in target_sampling:
            contract = target_sampling[distribution]
            expected_sequence_set_id = target_sequence_set_ids(int(configs[trajectory_id].train_config.trainer.seed))[
                distribution
            ]
            if source["probe_sequence_set_id"] != expected_sequence_set_id:
                raise ValueError(
                    f"Target sequence set drifted for {trajectory_id}/{distribution}: "
                    f"{source['probe_sequence_set_id']} != {expected_sequence_set_id}"
                )
            effective_blocks = min(design_blocks, int(contract["maximum_unique_full_blocks"]))
            if effective_blocks <= 0:
                raise ValueError(f"Target {distribution} cannot supply one complete probe block")
            row.update(
                {
                    "design_replicate_blocks": design_blocks,
                    "replicate_blocks": effective_blocks,
                    "target_available_sequence_count": contract["available_sequence_count"],
                    "target_unselected_sequence_count": (
                        int(contract["available_sequence_count"]) - effective_blocks * PROBE_SEQUENCES_PER_BLOCK
                    ),
                    "target_sampling_mode": "seeded_feistel_shuffle_without_replacement",
                }
            )
        else:
            row.update(
                {
                    "design_replicate_blocks": design_blocks,
                    "target_available_sequence_count": "",
                    "target_unselected_sequence_count": "",
                    "target_sampling_mode": "",
                }
            )
        row["expected_restored_state_step"] = expected_restored_state_step(int(source["checkpoint_step"]))
        row.update(
            {
                "scope": scope,
                "row_id": probe_row_identity(row, scope=scope),
                "group_id": probe_group_identity(row, scope=scope),
                "checkpoint_uri": checkpoint_uri(trajectory_id, int(source["checkpoint_step"]), scope=scope),
                "train_config_sha256": _config_identity(configs[trajectory_id])["full_train_config_sha256"],
                "scientific_inference_allowed": scope == "full",
                "endpoint_metrics_read": False,
            }
        )
        rows.append(row)
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate deterministic row identities in {scope} probe manifest")
    return rows


def _canary_rollout_rows(configs: dict[str, Any], target_sampling: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory_id in canary.EXPECTED_TRAJECTORY_IDS:
        support_id = _config_identity(configs[trajectory_id])["starcoder_support_id"]
        target_sequence_sets = target_sequence_set_ids(int(configs[trajectory_id].train_config.trainer.seed))
        for weight in CANARY_ROLLOUT_WEIGHTS:
            base_row: dict[str, Any] = {
                "parent_trajectory_id": trajectory_id,
                "parent_checkpoint_label": CANARY_ROLLOUT_CHECKPOINT_LABEL,
                "parent_checkpoint_step": CANARY_ROLLOUT_CHECKPOINT_STEP,
                "expected_restored_state_step": expected_restored_state_step(CANARY_ROLLOUT_CHECKPOINT_STEP),
                "starcoder_weight": weight,
                "nemotron_weight": 1.0 - weight,
                "source_support_id": support_id,
                "source_stream_rule": "continue_parent_support_with_frozen_per_source_order",
                "predicted_update_transform": "exact_optimizer_on_weighted_training_batch_gradient",
                "rollout_order_seed": 0,
                "updates": 512,
                "readout_steps": "|".join(map(str, CANARY_ROLLOUT_READOUTS)),
                "analysis_role": "pipeline_preflight_only",
                "target_block_counts_json": canonical_json(
                    {
                        distribution: min(PRIMARY_TARGET_BLOCKS, int(contract["maximum_unique_full_blocks"]))
                        for distribution, contract in sorted(target_sampling.items())
                    }
                ),
                "target_sequence_set_ids_json": canonical_json(target_sequence_sets),
            }
            rows.append(
                {
                    **base_row,
                    "scope": "canary",
                    "row_id": rollout_row_identity(base_row, scope="canary"),
                    "group_id": rollout_group_identity(base_row, scope="canary"),
                    "checkpoint_uri": checkpoint_uri(
                        trajectory_id,
                        CANARY_ROLLOUT_CHECKPOINT_STEP,
                        scope="canary",
                    ),
                    "train_config_sha256": _config_identity(configs[trajectory_id])["full_train_config_sha256"],
                    "scientific_inference_allowed": False,
                    "endpoint_metrics_read": False,
                }
            )
    return rows


def _full_rollout_rows(
    source_rows: list[dict[str, str]],
    configs: dict[str, Any],
    target_sampling: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    checkpoint_steps: dict[tuple[str, str], int] = {}
    for row in _read_csv(DESIGN_DIR / "checkpoint_manifest.csv"):
        checkpoint_steps[(row["trajectory_id"], row["checkpoint_label"])] = int(row["checkpoint_step"])

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        trajectory_id = source["parent_trajectory_id"]
        label = source["parent_checkpoint_label"]
        step = checkpoint_steps[(trajectory_id, label)]
        target_sequence_sets = target_sequence_set_ids(int(configs[trajectory_id].train_config.trainer.seed))
        base_row: dict[str, Any] = {
            **source,
            "parent_checkpoint_step": step,
            "expected_restored_state_step": expected_restored_state_step(step),
            "target_block_counts_json": canonical_json(
                {
                    distribution: min(PRIMARY_TARGET_BLOCKS, int(contract["maximum_unique_full_blocks"]))
                    for distribution, contract in sorted(target_sampling.items())
                }
            ),
            "target_sequence_set_ids_json": canonical_json(target_sequence_sets),
        }
        rows.append(
            {
                **base_row,
                "scope": "full",
                "row_id": rollout_row_identity(base_row, scope="full"),
                "group_id": rollout_group_identity(base_row, scope="full"),
                "checkpoint_uri": checkpoint_uri(trajectory_id, step, scope="full"),
                "train_config_sha256": _config_identity(configs[trajectory_id])["full_train_config_sha256"],
                "scientific_inference_allowed": True,
                "endpoint_metrics_read": False,
            }
        )
    return rows


def _optimizer_rows(
    source_rows: list[dict[str, str]],
    *,
    scope: str,
    configs: dict[str, Any],
    target_sampling: dict[str, dict[str, Any]],
    allowed_trajectories: set[str] | None = None,
) -> list[dict[str, Any]]:
    checkpoint_steps = {
        (row["trajectory_id"], row["checkpoint_label"]): int(row["checkpoint_step"])
        for row in _read_csv(DESIGN_DIR / "checkpoint_manifest.csv")
    }
    probe_source = (
        _read_csv(DESIGN_DIR / "probe_preflight_manifest.csv")
        if scope == "canary"
        else _read_csv(DESIGN_DIR / "gradient_probe_manifest.csv")
    )
    draw_counts = {
        (row["trajectory_id"], row["checkpoint_label"]): int(row["optimizer_update_draw_count"]) for row in probe_source
    }
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        trajectory_id = source["parent_trajectory_id"]
        if allowed_trajectories is not None and trajectory_id not in allowed_trajectories:
            continue
        label = source["parent_checkpoint_label"]
        step = checkpoint_steps[(trajectory_id, label)]
        target_sequence_sets = target_sequence_set_ids(int(configs[trajectory_id].train_config.trainer.seed))
        base_row: dict[str, Any] = {
            **source,
            "parent_checkpoint_step": step,
            "expected_restored_state_step": expected_restored_state_step(step),
            "optimizer_update_draw_count": draw_counts[(trajectory_id, label)],
            "target_block_counts_json": canonical_json(
                {
                    distribution: min(PRIMARY_TARGET_BLOCKS, int(contract["maximum_unique_full_blocks"]))
                    for distribution, contract in sorted(target_sampling.items())
                }
            ),
            "target_sequence_set_ids_json": canonical_json(target_sequence_sets),
        }
        rows.append(
            {
                **base_row,
                "scope": scope,
                "row_id": optimizer_row_identity(base_row, scope=scope),
                "group_id": optimizer_group_identity(base_row, scope=scope),
                "checkpoint_uri": checkpoint_uri(trajectory_id, step, scope=scope),
                "train_config_sha256": _config_identity(configs[trajectory_id])["full_train_config_sha256"],
                "scientific_inference_allowed": scope == "full",
                "endpoint_metrics_read": False,
            }
        )
    return rows


def _cache_provenance(canary_configs: dict[str, Any], full_configs: dict[str, Any]) -> list[dict[str, Any]]:
    representative_canary = next(iter(canary_configs.values())).train_config
    representative_full = next(iter(full_configs.values())).train_config
    train_length = representative_canary.train_seq_len or representative_canary.model.max_seq_len
    full_train_length = representative_full.train_seq_len or representative_full.model.max_seq_len
    if train_length != full_train_length:
        raise ValueError("Canary and full target-reference sequence lengths differ")
    Pos = Axis("position", train_length)
    canary_validation = representative_canary.data.validation_sets(Pos)
    full_validation = representative_full.data.validation_sets(Pos)
    names = (*TRAINING_COMPONENTS, *TARGET_COMPONENTS.values())
    rows: list[dict[str, Any]] = []
    for name in names:
        canary_component = representative_canary.data.components[name]
        full_component = representative_full.data.components[name]
        canary_root = _component_cache_root(canary_component)
        full_root = _component_cache_root(full_component)
        if canary_root != full_root:
            raise ValueError(f"Cache root differs between canary and full configs for {name}")
        is_training_source = name in TRAINING_COMPONENTS
        split = "train" if is_training_source else "validation"
        shard_ledger_uri = f"{canary_root}/{split}/shard_ledger.json"
        completion_uri = (
            f"{canary_root}/.executor_status" if is_training_source else f"{canary_root}/{split}/.stats.json"
        )
        completion = None if is_training_source else _remote_json(completion_uri)
        total_elements = "" if completion is None else int(completion["total_elements"])
        total_tokens = "" if completion is None else int(completion["total_tokens"])
        if is_training_source:
            materialized_sequence_count: int | str = ""
        else:
            canary_count = int(blocking_wait(canary_validation[name].async_len()))
            full_count = int(blocking_wait(full_validation[name].async_len()))
            if canary_count != full_count:
                raise ValueError(f"Canary and full target-reference lengths differ for {name}")
            materialized_sequence_count = canary_count
        rows.append(
            {
                "component_name": name,
                "role": "training_source" if is_training_source else "target_reference",
                "cache_root": canary_root,
                "split": split,
                "shard_ledger_uri": shard_ledger_uri,
                "shard_ledger_sha256": _remote_file_sha256(shard_ledger_uri),
                "completion_uri": completion_uri,
                "completion_sha256": _remote_file_sha256(completion_uri),
                "total_elements": total_elements,
                "total_tokens": total_tokens,
                "materialized_sequence_length": "" if completion is None else train_length,
                "materialized_sequence_count": materialized_sequence_count,
                "maximum_unique_full_probe_blocks": (
                    "" if completion is None else int(materialized_sequence_count) // PROBE_SEQUENCES_PER_BLOCK
                ),
                "target_distribution_id": next(
                    (distribution for distribution, component in TARGET_COMPONENTS.items() if component == name),
                    "",
                ),
            }
        )
    return rows


def _target_sampling_contract(cache_provenance: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    contract: dict[str, dict[str, Any]] = {}
    for row in cache_provenance:
        distribution = str(row["target_distribution_id"])
        if not distribution:
            continue
        available = int(row["materialized_sequence_count"])
        maximum_blocks = available // PROBE_SEQUENCES_PER_BLOCK
        if maximum_blocks <= 0:
            raise ValueError(f"Target {distribution} has only {available} sequences")
        contract[distribution] = {
            "component_name": row["component_name"],
            "available_sequence_count": available,
            "materialized_sequence_length": int(row["materialized_sequence_length"]),
            "sequences_per_block": PROBE_SEQUENCES_PER_BLOCK,
            "maximum_unique_full_blocks": maximum_blocks,
            "unused_tail_sequence_count": available - maximum_blocks * PROBE_SEQUENCES_PER_BLOCK,
            "sampling_mode": "seeded_feistel_shuffle_without_replacement",
        }
    if set(contract) != set(TARGET_COMPONENTS):
        raise ValueError(f"Target sampling contract is incomplete: {sorted(contract)}")
    return contract


def _config_provenance(scope: str, configs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"scope": scope, "trajectory_id": trajectory_id, **_config_identity(config)}
        for trajectory_id, config in sorted(configs.items())
    ]


def _manifest_summary(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": file_sha256(path),
        "row_count": len(rows),
    }
    if rows and "row_id" in rows[0]:
        summary["unique_row_count"] = len({row["row_id"] for row in rows})
    if rows and "group_id" in rows[0]:
        summary["unique_group_count"] = len({row["group_id"] for row in rows})
    return summary


def freeze() -> dict[str, Any]:
    """Materialize both releases and return the hash-pinned release document."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canary_configs, full_configs = _materialized_configs()
    cache_provenance = _cache_provenance(canary_configs, full_configs)
    target_sampling = _target_sampling_contract(cache_provenance)

    canary_probe = _probe_rows(
        _read_csv(DESIGN_DIR / "probe_preflight_manifest.csv"),
        scope="canary",
        configs=canary_configs,
        target_sampling=target_sampling,
    )
    full_probe = _probe_rows(
        _read_csv(DESIGN_DIR / "gradient_probe_manifest.csv"),
        scope="full",
        configs=full_configs,
        target_sampling=target_sampling,
    )
    canary_rollout = _canary_rollout_rows(canary_configs, target_sampling)
    full_rollout = _full_rollout_rows(_read_csv(DESIGN_DIR / "rollout_manifest.csv"), full_configs, target_sampling)
    optimizer_source = _read_csv(DESIGN_DIR / "optimizer_transform_manifest.csv")
    canary_optimizer = _optimizer_rows(
        optimizer_source,
        scope="canary",
        configs=canary_configs,
        target_sampling=target_sampling,
        allowed_trajectories=set(canary.EXPECTED_TRAJECTORY_IDS),
    )
    full_optimizer = _optimizer_rows(
        optimizer_source,
        scope="full",
        configs=full_configs,
        target_sampling=target_sampling,
    )
    config_provenance = [
        *_config_provenance("canary", canary_configs),
        *_config_provenance("full", full_configs),
    ]

    manifests = {
        "canary_probe": (OUTPUT_DIR / "canary_probe_manifest.csv", canary_probe),
        "canary_rollout": (OUTPUT_DIR / "canary_rollout_manifest.csv", canary_rollout),
        "canary_optimizer": (OUTPUT_DIR / "canary_optimizer_manifest.csv", canary_optimizer),
        "full_probe": (OUTPUT_DIR / "full_probe_manifest.csv", full_probe),
        "full_rollout": (OUTPUT_DIR / "full_rollout_manifest.csv", full_rollout),
        "full_optimizer": (OUTPUT_DIR / "full_optimizer_manifest.csv", full_optimizer),
        "cache_provenance": (OUTPUT_DIR / "cache_provenance.csv", cache_provenance),
        "config_provenance": (OUTPUT_DIR / "config_provenance.csv", config_provenance),
    }
    for path, rows in manifests.values():
        _write_csv(path, rows)

    source_design_manifest = json.loads((DESIGN_DIR / "design_manifest.json").read_text())
    canary_checkpoint_count = len(
        {
            (row["checkpoint_uri"], int(row.get("checkpoint_step", row.get("parent_checkpoint_step"))))
            for rows in (canary_probe, canary_optimizer, canary_rollout)
            for row in rows
        }
    )

    release = {
        "release_version": PROBE_RELEASE_VERSION,
        "release_sha256": "",
        "design_version": full.EXPECTED_DESIGN_VERSION,
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
        "design_manifest_sha256": full.EXPECTED_DESIGN_MANIFEST_SHA256,
        "training_release_sha256": full.EXPECTED_RELEASE_MANIFEST_SHA256,
        "result_root": RESULT_ROOT,
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": "gs://marin-us-central1",
        "canary_scientific_inference_allowed": False,
        "canary_provenance_limitation": (
            "The completed 2026-08-10 canaries predate review-v9's global training holdout. "
            "They validate restoration, numerics, optimizer-state use, rollout execution, and idempotence only."
        ),
        "full_scientific_inference_allowed": True,
        "endpoint_metrics_read": False,
        "artifact_triggered_async_readiness_implemented": False,
        "checkpoint_readiness_semaphore_limit": 64,
        "checkpoint_readiness_executor": "asyncio_default_thread_pool",
        "checkpoint_step_semantics": {
            "checkpoint_label_step": "zero-indexed update just completed, matching the step-N directory and metadata",
            "expected_restored_state_step": "next update index stored in TrainerState and optimizer counters",
            "continuation_rule": "expected_restored_state_step = checkpoint_label_step + 1",
        },
        "full_launch_authorized": False,
        "target_sampling_contract": target_sampling,
        "target_sampling_design_correction": (
            "The source design requested 64 independent 64-sequence target blocks. Frozen target caches are finite, "
            "so each target is capped using the materialized 2,048-token sequence population, not the raw document "
            "count. Complete blocks are sampled without replacement; the original request is retained as "
            "design_replicate_blocks, and training seed remains the scientific inferential unit."
        ),
        "target_reference_identity_rule": (
            "For a given training seed and target distribution, probe, optimizer-transform, and rollout rows use the "
            "same frozen sequence-set identity."
        ),
        "optimizer_update_statistic": (
            "Delta(g)-Delta(0) is an optimizer-aware finite difference under the restored optimizer state. "
            "Because MuonH is nonlinear, it is a mechanical diagnostic and not a causal attribution."
        ),
        "canary_checkpoint_count_reconciliation": {
            "source_design_expected_permanent_checkpoint_count": source_design_manifest[
                "canary_expected_permanent_checkpoint_count"
            ],
            "frozen_unique_permanent_checkpoint_count": canary_checkpoint_count,
            "explanation": (
                "The source design's count of 13 is stale. Two canary seeds each contribute seven distinct permanent "
                "checkpoint coordinates, so the execution manifests correctly require 14."
            ),
        },
        "manifests": {name: _manifest_summary(path, rows) for name, (path, rows) in manifests.items()},
        "source_design_files": {path.name: file_sha256(path) for path in sorted(DESIGN_DIR.iterdir()) if path.is_file()},
        "implementation_files": {
            str(Path(__file__).resolve().relative_to(REPO_ROOT)): file_sha256(Path(__file__).resolve()),
            str(PROBE_RUNTIME_PATH.relative_to(REPO_ROOT)): file_sha256(PROBE_RUNTIME_PATH),
        },
    }
    release["release_sha256"] = canonical_sha256({**release, "release_sha256": ""})
    release_path = OUTPUT_DIR / "release.json"
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    return release


def main() -> None:
    release = freeze()
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
