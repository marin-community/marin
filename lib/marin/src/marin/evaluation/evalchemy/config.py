# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate portable Evalchemy configs in the pinned evaluator environment."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from marin.evaluation.isolated_driver import (
    ISOLATED_REQUEST_MODE,
    capture_driver,
    isolated_driver_environment,
)

_EVALCHEMY_ENV_CONFIG = ("config", "external", "evalchemy")
_PREFLIGHT_DRIVER = Path(__file__).with_name("preflight_driver.py")
RESERVED_ENDPOINT_MODEL_ARGS = frozenset({"model", "base_url", "tokenizer", "tokenizer_backend", "tokenized_requests"})


@dataclass(frozen=True)
class EvalchemyTaskOptions:
    """Normalized per-task behavior from Evalchemy's portable config."""

    num_fewshot: int | None = None
    task_alias: str | None = None
    generation: bool = False
    unsafe_code: bool = False
    completion_only: bool = False


@dataclass(frozen=True)
class ValidatedEvalchemyConfig:
    """One pinned-schema config plus the runtime extras selected from that package."""

    tasks: tuple[str, ...]
    task_options: Mapping[str, EvalchemyTaskOptions]
    apply_chat_template: bool
    limit: int | None
    num_fewshot: int | None
    batch_size: str
    seed: int | None
    gen_kwargs: Mapping[str, str]
    extra_model_args: Mapping[str, str | int | float | bool]
    max_length: int | None
    max_tokens: int | None
    runtime_extras: tuple[str, ...]


def _workspace_root() -> Path:
    for parent in Path(__file__).parents:
        if parent.joinpath(*_EVALCHEMY_ENV_CONFIG, "uv.lock").is_file():
            return parent
    raise RuntimeError("Evalchemy preflight requires a Marin workspace with a pinned external environment")


def _driver_command(request_path: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--isolated",
        "--frozen",
        "--project",
        str(_workspace_root().joinpath(*_EVALCHEMY_ENV_CONFIG)),
        "python",
        str(_PREFLIGHT_DRIVER),
        str(request_path),
    ]


def _parse_key_value_args(value: object, path: Path) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, str):
        raise ValueError(f"Evalchemy preflight returned invalid gen_kwargs metadata for {path}")
    parsed: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Evalchemy preflight returned invalid gen_kwargs item {item!r} for {path}")
        key, raw_value = item.split("=", 1)
        if not key.strip():
            raise ValueError(f"Evalchemy preflight returned an empty gen_kwargs key for {path}")
        parsed[key.strip()] = raw_value.strip()
    return parsed


def _validated_config(payload: object, path: Path) -> ValidatedEvalchemyConfig:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Evalchemy preflight returned a non-object result for {path}")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"Evalchemy preflight returned invalid config metadata for {path}")

    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(task, str) and task for task in tasks):
        raise ValueError(f"Evalchemy preflight returned invalid task metadata for {path}")
    options_payload = config.get("task_options", {})
    if not isinstance(options_payload, Mapping):
        raise ValueError(f"Evalchemy preflight returned invalid task_options metadata for {path}")
    task_options: dict[str, EvalchemyTaskOptions] = {}
    for task, options in options_payload.items():
        if not isinstance(task, str) or not isinstance(options, Mapping):
            raise ValueError(f"Evalchemy preflight returned invalid task_options metadata for {path}")
        task_options[task] = EvalchemyTaskOptions(
            num_fewshot=options.get("num_fewshot"),
            task_alias=options.get("task_alias"),
            generation=options.get("generation", False),
            unsafe_code=options.get("unsafe_code", False),
            completion_only=options.get("completion_only", False),
        )

    runtime_extras = payload.get("runtime_extras")
    if not isinstance(runtime_extras, list) or not all(isinstance(extra, str) for extra in runtime_extras):
        raise ValueError(f"Evalchemy preflight returned invalid runtime extras for {path}")
    extra_model_args = config.get("extra_model_args", {})
    if not isinstance(extra_model_args, Mapping):
        raise ValueError(f"Evalchemy preflight returned invalid extra_model_args metadata for {path}")

    return ValidatedEvalchemyConfig(
        tasks=tuple(tasks),
        task_options=task_options,
        apply_chat_template=config.get("apply_chat_template", False),
        limit=config.get("limit"),
        num_fewshot=config.get("num_fewshot"),
        batch_size=str(config.get("batch_size", 1)),
        seed=config.get("seed"),
        gen_kwargs=_parse_key_value_args(config.get("gen_kwargs"), path),
        extra_model_args=dict(extra_model_args),
        max_length=config.get("max_length"),
        max_tokens=config.get("max_tokens"),
        runtime_extras=tuple(runtime_extras),
    )


def preflight_evalchemy_configs(paths: Sequence[Path]) -> tuple[ValidatedEvalchemyConfig, ...]:
    """Normalize configs and resolve every task through the pinned evaluator catalogs."""
    if not paths:
        return ()
    with tempfile.TemporaryDirectory(prefix="marin-evalchemy-preflight-") as temp_dir:
        request_path = Path(temp_dir) / "requests.json"
        request_path.write_text(json.dumps([str(path.resolve()) for path in paths], separators=(",", ":")))
        request_path.chmod(ISOLATED_REQUEST_MODE)
        try:
            completed = capture_driver(_driver_command(request_path), isolated_driver_environment())
        except ValueError as exc:
            joined_paths = ", ".join(str(path) for path in paths)
            raise ValueError(f"invalid Evalchemy config in [{joined_paths}]: {exc}") from exc

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("Evalchemy preflight returned invalid JSON") from exc
    if not isinstance(response, list) or len(response) != len(paths):
        count = len(response) if isinstance(response, list) else "invalid"
        raise ValueError(f"Evalchemy preflight returned {count} result(s) for {len(paths)} config(s)")
    return tuple(_validated_config(payload, path) for payload, path in zip(response, paths, strict=True))
