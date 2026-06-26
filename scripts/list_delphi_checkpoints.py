# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""List native Delphi Levanter checkpoints available on GCS.

This is an operator helper for true-midtraining / cooldown launches. It does
not choose a checkpoint, stage a checkpoint, or submit a job. It only lists the
native checkpoint steps that exist for a verified Delphi base and optionally
fails unless a user-selected exact step is available.

Examples:

    uv run python scripts/list_delphi_checkpoints.py --base 1e22
    uv run python scripts/list_delphi_checkpoints.py --base 1e22 --cooldown-ratio 0.2
    uv run python scripts/list_delphi_checkpoints.py --base 1e22 --expect-step 30000
    uv run python scripts/list_delphi_checkpoints.py --all --json

The command uses fsspec/gcsfs credentials, so local use requires GCP access
through the usual ADC / gcloud auth setup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import fsspec

from experiments.delphi_models import ALL_DELPHI_MODELS, DelphiModel, get_delphi_model

CHECKPOINT_PREFIX = "step-"
REQUIRED_CHECKPOINT_ARTIFACTS = ("manifest.ocdbt", "metadata.json", "d")


@dataclass(frozen=True)
class CheckpointRecord:
    base: str
    step: int
    progress_percent: float
    phase: str
    lr_fraction: float
    delta_to_decay_start: int
    remaining_steps: int
    path: str
    required_artifacts_ok: bool
    missing_artifacts: tuple[str, ...]
    metadata_step: int | None
    metadata_step_matches: bool | None


@dataclass(frozen=True)
class CheckpointDistance:
    step: int
    delta_steps: int
    progress_percent: float
    phase: str
    required_artifacts_ok: bool
    path: str


@dataclass(frozen=True)
class CooldownTarget:
    base: str
    cooldown_ratio: float
    train_prefix_ratio: float
    target_step: int
    target_progress_percent: float
    target_cooldown_steps: int
    closest_checkpoints: tuple[CheckpointDistance, ...]
    checkpoint_at_or_before: CheckpointDistance | None
    checkpoint_at_or_after: CheckpointDistance | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--base", help="Verified Delphi flops key, e.g. 3e20, 1e21, 1e22.")
    selector.add_argument("--all", action="store_true", help="List checkpoints for every verified Delphi base.")
    parser.add_argument(
        "--checkpoint-root",
        help=(
            "Override the checkpoint root to list. Use this for ad-hoc inspection; schedule metadata still comes "
            "from --base, so this flag is incompatible with --all."
        ),
    )
    parser.add_argument(
        "--expect-step",
        type=int,
        help=(
            "Require this exact step to exist with TensorStore artifacts. Exits nonzero if unavailable. "
            "Use this before setting CooldownResume.resume_step."
        ),
    )
    parser.add_argument(
        "--cooldown-ratio",
        type=float,
        help=(
            "Report the pretraining resume step implied by this cooldown ratio. For example, 0.2 means "
            "resume around 80%% of the original pretraining run and use the remaining 20%% for cooldown."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.checkpoint_root and args.all:
        raise ValueError("--checkpoint-root is only supported with --base, not --all")
    if args.cooldown_ratio is not None and not 0.0 < args.cooldown_ratio < 1.0:
        raise ValueError("--cooldown-ratio must be greater than 0 and less than 1")

    models = list(ALL_DELPHI_MODELS) if args.all else [get_delphi_model(args.base)]
    all_records: list[CheckpointRecord] = []
    cooldown_targets: list[CooldownTarget] = []
    for model in models:
        checkpoint_root = args.checkpoint_root or model.gcs_checkpoint_root
        records = list_checkpoints(model, checkpoint_root=checkpoint_root)
        all_records.extend(records)
        if args.cooldown_ratio is not None:
            cooldown_targets.append(cooldown_target(model, records, args.cooldown_ratio))

    if args.json:
        payload: Any
        if args.cooldown_ratio is None:
            payload = [asdict(record) for record in all_records]
        else:
            payload = {
                "checkpoints": [asdict(record) for record in all_records],
                "cooldown_targets": [asdict(target) for target in cooldown_targets],
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_table(all_records)
        if cooldown_targets:
            print_cooldown_targets(cooldown_targets)

    if args.expect_step is not None:
        return validate_expected_step(all_records, args.expect_step)
    if not all_records:
        return 1
    return 0


def list_checkpoints(model: DelphiModel, *, checkpoint_root: str | None = None) -> list[CheckpointRecord]:
    root = (checkpoint_root or model.gcs_checkpoint_root).rstrip("/")
    entries = list_checkpoint_entries(root)
    records = [
        checkpoint_record(model, step=step, path=f"{root}/{CHECKPOINT_PREFIX}{step}")
        for step in sorted(parse_checkpoint_steps(entries))
    ]
    return records


def list_checkpoint_entries(checkpoint_root: str) -> tuple[str, ...]:
    try:
        fs, path = fsspec.core.url_to_fs(checkpoint_root)
        return tuple(fs.ls(path, detail=False))
    except Exception as exc:
        raise RuntimeError(
            f"Could not list {checkpoint_root!r}. This helper requires local GCP access "
            "(for example `gcloud auth application-default login`) and permission to read the bucket."
        ) from exc


def parse_checkpoint_steps(entries: tuple[str, ...]) -> tuple[int, ...]:
    steps: list[int] = []
    for entry in entries:
        name = os.path.basename(entry.rstrip("/"))
        if not name.startswith(CHECKPOINT_PREFIX):
            continue
        step_s = name.removeprefix(CHECKPOINT_PREFIX)
        if step_s.isdigit():
            steps.append(int(step_s))
    return tuple(steps)


def checkpoint_record(model: DelphiModel, *, step: int, path: str) -> CheckpointRecord:
    missing = missing_checkpoint_artifacts(path)
    metadata_step = read_metadata_step(path)
    metadata_step_matches = None if metadata_step is None else metadata_step == step
    warmup_end = schedule_warmup_end(model)
    decay_start = schedule_decay_start(model)
    return CheckpointRecord(
        base=model.flops_key,
        step=step,
        progress_percent=100 * step / model.num_train_steps,
        phase=phase_for_step(step, model, warmup_end=warmup_end, decay_start=decay_start),
        lr_fraction=lr_fraction_for_step(step, model, warmup_end=warmup_end, decay_start=decay_start),
        delta_to_decay_start=step - decay_start,
        remaining_steps=max(model.num_train_steps - step, 0),
        path=path,
        required_artifacts_ok=not missing and metadata_step_matches is not False,
        missing_artifacts=missing,
        metadata_step=metadata_step,
        metadata_step_matches=metadata_step_matches,
    )


def cooldown_target(model: DelphiModel, records: list[CheckpointRecord], cooldown_ratio: float) -> CooldownTarget:
    prefix_ratio = 1.0 - cooldown_ratio
    target_step = fraction_or_steps_to_steps(prefix_ratio, model.num_train_steps)
    target_cooldown_steps = model.num_train_steps - target_step
    closest = tuple(
        checkpoint_distance(record, target_step)
        for record in sorted(records, key=lambda record: (abs(record.step - target_step), record.step))[:2]
    )
    before_records = [record for record in records if record.step <= target_step]
    after_records = [record for record in records if record.step >= target_step]
    checkpoint_at_or_before = (
        checkpoint_distance(max(before_records, key=lambda record: record.step), target_step) if before_records else None
    )
    checkpoint_at_or_after = (
        checkpoint_distance(min(after_records, key=lambda record: record.step), target_step) if after_records else None
    )
    return CooldownTarget(
        base=model.flops_key,
        cooldown_ratio=cooldown_ratio,
        train_prefix_ratio=prefix_ratio,
        target_step=target_step,
        target_progress_percent=100 * target_step / model.num_train_steps,
        target_cooldown_steps=target_cooldown_steps,
        closest_checkpoints=closest,
        checkpoint_at_or_before=checkpoint_at_or_before,
        checkpoint_at_or_after=checkpoint_at_or_after,
    )


def checkpoint_distance(record: CheckpointRecord, target_step: int) -> CheckpointDistance:
    return CheckpointDistance(
        step=record.step,
        delta_steps=record.step - target_step,
        progress_percent=record.progress_percent,
        phase=record.phase,
        required_artifacts_ok=record.required_artifacts_ok,
        path=record.path,
    )


def schedule_warmup_end(model: DelphiModel) -> int:
    return fraction_or_steps_to_steps(model.warmup_fraction, model.num_train_steps)


def schedule_decay_start(model: DelphiModel) -> int:
    decay_steps = fraction_or_steps_to_steps(model.decay_fraction, model.num_train_steps)
    return model.num_train_steps - decay_steps


def fraction_or_steps_to_steps(value: float, num_train_steps: int) -> int:
    """Match Levanter's fractional stage conversion semantics."""
    if value < 0.0 or (value > 1.0 and value % 1 != 0):
        raise ValueError(f"Invalid schedule stage value {value!r}")
    if value <= 1.0:
        return int(value * num_train_steps)
    return int(value)


def phase_for_step(step: int, model: DelphiModel, *, warmup_end: int, decay_start: int) -> str:
    if step < warmup_end:
        return "warmup"
    if step < decay_start:
        return "stable"
    if step < model.num_train_steps:
        return "decay"
    return "final"


def lr_fraction_for_step(step: int, model: DelphiModel, *, warmup_end: int, decay_start: int) -> float:
    if step < warmup_end:
        if warmup_end == 0:
            return 1.0
        return clamp(step / warmup_end)
    if step < decay_start:
        return 1.0
    decay_steps = max(model.num_train_steps - decay_start, 1)
    return clamp((model.num_train_steps - step) / decay_steps)


def clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def missing_checkpoint_artifacts(checkpoint_path: str) -> tuple[str, ...]:
    missing = []
    for artifact in REQUIRED_CHECKPOINT_ARTIFACTS:
        artifact_path = f"{checkpoint_path}/{artifact}"
        if not gcs_path_exists_or_has_children(artifact_path):
            missing.append(artifact)
    return tuple(missing)


def gcs_path_exists_or_has_children(uri: str) -> bool:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        return True
    try:
        return bool(fs.ls(path, detail=False))
    except (FileNotFoundError, OSError):
        return False


def read_metadata_step(checkpoint_path: str) -> int | None:
    metadata_path = f"{checkpoint_path}/metadata.json"
    try:
        with fsspec.open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return find_step_value(metadata)


def find_step_value(value: Any) -> int | None:
    if isinstance(value, dict):
        for key in ("step", "train_step", "global_step"):
            step = int_like(value.get(key))
            if step is not None:
                return step
        for child in value.values():
            step = find_step_value(child)
            if step is not None:
                return step
    if isinstance(value, list):
        for child in value:
            step = find_step_value(child)
            if step is not None:
                return step
    return None


def int_like(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def print_table(records: list[CheckpointRecord]) -> None:
    if not records:
        print("No checkpoints found.")
        return

    current_base = None
    for record in records:
        if record.base != current_base:
            current_base = record.base
            print_base_header(record)
        print_record(record)


def print_base_header(record: CheckpointRecord) -> None:
    print()
    print(f"base={record.base}")
    print(f"checkpoint_root={record.path.rsplit('/', 1)[0]}")
    print(
        f"{'step':>9}  {'progress':>8}  {'phase':<7}  {'lr':>6}  "
        f"{'delta_decay':>12}  {'remaining':>10}  {'artifacts':<18}  {'metadata':<12}  path"
    )


def print_record(record: CheckpointRecord) -> None:
    artifacts = "ok" if not record.missing_artifacts else "missing:" + ",".join(record.missing_artifacts)
    if record.metadata_step_matches is None:
        metadata = "unknown"
    elif record.metadata_step_matches:
        metadata = f"step={record.metadata_step}"
    else:
        metadata = f"mismatch:{record.metadata_step}"
    print(
        f"{record.step:9d}  "
        f"{record.progress_percent:7.2f}%  "
        f"{record.phase:<7}  "
        f"{record.lr_fraction:6.3f}  "
        f"{record.delta_to_decay_start:12d}  "
        f"{record.remaining_steps:10d}  "
        f"{artifacts:<18}  "
        f"{metadata:<12}  "
        f"{record.path}"
    )


def print_cooldown_targets(targets: list[CooldownTarget]) -> None:
    print()
    print("Cooldown targets")
    for target in targets:
        print()
        print(
            f"base={target.base} cooldown_ratio={target.cooldown_ratio:g} "
            f"train_prefix={target.train_prefix_ratio:.2%}"
        )
        print(
            f"target_step={target.target_step} "
            f"target_progress={target.target_progress_percent:.2f}% "
            f"target_cooldown_steps={target.target_cooldown_steps}"
        )
        if target.closest_checkpoints:
            print("closest checkpoints by absolute step distance:")
            for checkpoint in target.closest_checkpoints:
                print_checkpoint_distance("  ", checkpoint)
        else:
            print("closest checkpoints by absolute step distance: none")
        print("checkpoint bracket:")
        if target.checkpoint_at_or_before is None:
            print("  at_or_before: none")
        else:
            print_checkpoint_distance("  at_or_before: ", target.checkpoint_at_or_before)
        if target.checkpoint_at_or_after is None:
            print("  at_or_after: none")
        else:
            print_checkpoint_distance("  at_or_after:  ", target.checkpoint_at_or_after)


def print_checkpoint_distance(prefix: str, checkpoint: CheckpointDistance) -> None:
    status = "ok" if checkpoint.required_artifacts_ok else "bad"
    print(
        f"{prefix}step-{checkpoint.step} "
        f"delta={checkpoint.delta_steps:+d} "
        f"progress={checkpoint.progress_percent:.2f}% "
        f"phase={checkpoint.phase} "
        f"artifacts={status} "
        f"path={checkpoint.path}"
    )


def validate_expected_step(records: list[CheckpointRecord], expected_step: int) -> int:
    matches = [record for record in records if record.step == expected_step]
    if not matches:
        available = ", ".join(str(record.step) for record in records) or "(none)"
        print(f"\nERROR: expected step-{expected_step} was not found. Available steps: {available}", file=sys.stderr)
        return 2

    record = matches[0]
    errors = []
    if record.missing_artifacts:
        errors.append(f"missing artifacts: {', '.join(record.missing_artifacts)}")
    if record.metadata_step_matches is False:
        errors.append(f"metadata step is {record.metadata_step}, expected {expected_step}")
    if errors:
        print(f"\nERROR: step-{expected_step} is not a valid cooldown resume checkpoint: {'; '.join(errors)}")
        return 2

    print()
    print(f"OK: step-{expected_step} is available for base={record.base}.")
    print("Use this exact native checkpoint path in CooldownResume.pretrain_checkpoint_path:")
    print(f"  {record.path}")
    print(f"And set CooldownResume.resume_step={expected_step}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
