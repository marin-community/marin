# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""W&B configuration helpers for alternating RL roles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from levanter.tracker import Tracker
from levanter.tracker.wandb import WandbConfig
from levanter.utils.fsspec_utils import join_path

from marin.rl.alternating.config import AlternatingRLConfig
from marin.rl.alternating.state import AlternatingRunPaths

ALTERNATING_CONTROLLER_ROLE = "alternating-controller"
ALTERNATING_TRAIN_ROLE = "alternating-train"


def alternating_controller_run_id(run_id: str) -> str:
    return f"{run_id}-{ALTERNATING_CONTROLLER_ROLE}"


def alternating_controller_run_name(run_id: str) -> str:
    return f"{run_id}-{ALTERNATING_CONTROLLER_ROLE}"


def alternating_train_run_name(run_id: str) -> str:
    return f"{run_id}-{ALTERNATING_TRAIN_ROLE}"


def init_alternating_controller_tracker(
    config: AlternatingRLConfig,
    paths: AlternatingRunPaths,
) -> Tracker | None:
    base_wandb = _extract_wandb_config(getattr(config.trainer, "tracker", None))
    if base_wandb is None:
        return None

    controller_wandb = _alternating_wandb_config(
        base_wandb,
        run_id=config.run_id,
        role=ALTERNATING_CONTROLLER_ROLE,
        name=alternating_controller_run_name(config.run_id),
        replicate_path=join_path(
            join_path(paths.state_root, "wandb"),
            ALTERNATING_CONTROLLER_ROLE,
        ),
    )
    return controller_wandb.init(alternating_controller_run_id(config.run_id))


def alternating_training_tracker_config(tracker_config: Any, *, run_id: str) -> Any:
    if isinstance(tracker_config, WandbConfig):
        return _alternating_wandb_config(
            tracker_config,
            run_id=run_id,
            role=ALTERNATING_TRAIN_ROLE,
            name=alternating_train_run_name(run_id),
        )
    if isinstance(tracker_config, tuple):
        return tuple(alternating_training_tracker_config(item, run_id=run_id) for item in tracker_config)
    return tracker_config


def _extract_wandb_config(tracker_config: Any) -> WandbConfig | None:
    if isinstance(tracker_config, WandbConfig):
        return tracker_config
    if isinstance(tracker_config, Sequence):
        for item in tracker_config:
            if isinstance(item, WandbConfig):
                return item
    return None


def _alternating_wandb_config(
    config: WandbConfig,
    *,
    run_id: str,
    role: str,
    name: str,
    replicate_path: str | None = None,
) -> WandbConfig:
    role_tags = _append_unique_tags(config.tags, ["alternating", role])
    return replace(
        config,
        name=name,
        group=config.group or run_id,
        tags=role_tags,
        save_code=False,
        replicate_path=replicate_path or config.replicate_path,
    )


def _append_unique_tags(tags: list[str], extra_tags: list[str]) -> list[str]:
    combined: list[str] = []
    for tag in [*tags, *extra_tags]:
        if tag not in combined:
            combined.append(tag)
    return combined
