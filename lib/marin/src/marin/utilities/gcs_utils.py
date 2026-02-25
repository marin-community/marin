# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS-related utility helpers for Marin."""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import PurePath
from typing import Any

from iris.marin_fs import check_path_in_region, marin_region

logger = logging.getLogger(__name__)


def check_gcs_paths_same_region(
    obj: Any,
    *,
    local_ok: bool,
    region: str | None = None,
    skip_if_prefix_contains: Sequence[str] = ("train_urls", "validation_urls"),
    region_getter: Callable[[], str | None] = marin_region,
    path_checker: Callable[[str, str, str, bool], None] = check_path_in_region,
) -> None:
    """Validate that ``gs://`` paths in ``obj`` live in the current VM region."""
    if region is None:
        region = region_getter()
        if region is None:
            if local_ok:
                logger.warning("Could not determine the region of the VM. This is fine if you're running locally.")
                return
            raise ValueError("Could not determine the region of the VM. This is required for path checks.")

    _check_paths_recursively(
        obj,
        "",
        region=region,
        local_ok=local_ok,
        skip_if_prefix_contains=tuple(skip_if_prefix_contains),
        path_checker=path_checker,
    )


def _check_paths_recursively(
    obj: Any,
    path_prefix: str,
    *,
    region: str,
    local_ok: bool,
    skip_if_prefix_contains: tuple[str, ...],
    path_checker: Callable[[str, str, str, bool], None],
) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)
            _check_paths_recursively(
                value,
                new_prefix,
                region=region,
                local_ok=local_ok,
                skip_if_prefix_contains=skip_if_prefix_contains,
                path_checker=path_checker,
            )
        return

    if isinstance(obj, list | tuple):
        for index, item in enumerate(obj):
            new_prefix = f"{path_prefix}[{index}]"
            _check_paths_recursively(
                item,
                new_prefix,
                region=region,
                local_ok=local_ok,
                skip_if_prefix_contains=skip_if_prefix_contains,
                path_checker=path_checker,
            )
        return

    if isinstance(obj, str | os.PathLike):
        path_str = _normalize_path_like(obj)
        if path_str.startswith("gs://"):
            if any(skip_token in path_prefix for skip_token in skip_if_prefix_contains):
                return
            path_checker(path_prefix, path_str, region, local_ok)
        return

    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            new_prefix = f"{path_prefix}.{field.name}" if path_prefix else field.name
            _check_paths_recursively(
                getattr(obj, field.name),
                new_prefix,
                region=region,
                local_ok=local_ok,
                skip_if_prefix_contains=skip_if_prefix_contains,
                path_checker=path_checker,
            )
        return

    if not isinstance(obj, str | int | float | bool | type(None)):
        logger.warning(f"Found unexpected type {type(obj)} at {path_prefix}. Skipping.")


def _normalize_path_like(path: str | os.PathLike) -> str:
    if isinstance(path, os.PathLike):
        path_str = os.fspath(path)
        if isinstance(path, PurePath):
            parts = path.parts
            if parts and parts[0] == "gs:" and not path_str.startswith("gs://"):
                remainder = "/".join(parts[1:])
                return f"gs://{remainder}" if remainder else "gs://"
        return path_str
    return path
