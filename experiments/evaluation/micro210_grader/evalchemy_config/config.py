"""The portable, typed Evalchemy evaluation contract.

This package deliberately owns only evaluation intent.  Serving, credentials,
cluster placement, and endpoint lifecycle belong to the caller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .limits import MAX_OUTPUT_ALIASES, MODEL_LENGTH_ALIASES, format_key_value_args, parse_key_value_args, resolve_limit

_SCHEMA_VERSION = 1


class TaskOptions(BaseModel):
    """Per-task behavior that Evalchemy's one-task-at-a-time client must retain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_fewshot: int | None = None
    task_alias: str | None = None
    generation: bool = False
    unsafe_code: bool = False
    completion_only: bool = False


class EvaluationConfig(BaseModel):
    """Evaluation intent that is portable across Evalchemy launch environments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: list[str] = Field(default_factory=lambda: ["gsm8k"])
    task_options: dict[str, TaskOptions] = Field(default_factory=dict)
    apply_chat_template: bool = False
    limit: int | None = 200
    num_fewshot: int | None = None
    batch_size: int | str = 1
    seed: int | None = 1234
    gen_kwargs: str | None = None
    extra_model_args: dict[str, str | int | float | bool] = Field(default_factory=dict)
    max_length: int | None = None
    max_tokens: int | None = None

    @model_validator(mode="after")
    def task_options_match_tasks(self) -> "EvaluationConfig":
        unknown = sorted(set(self.task_options).difference(self.tasks))
        if unknown:
            raise ValueError(f"task_options refer to task(s) not selected in tasks: {unknown}")
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_limits(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        model_args = parse_key_value_args(data.get("extra_model_args"))
        generation_args = parse_key_value_args(data.get("gen_kwargs"))
        max_length = resolve_limit(
            "max_length",
            [("max_length", data.get("max_length"))]
            + [(f"extra_model_args.{key}", model_args.get(key)) for key in MODEL_LENGTH_ALIASES],
        )
        max_tokens = resolve_limit(
            "max_tokens",
            [("max_tokens", data.get("max_tokens"))]
            + [(f"gen_kwargs.{key}", generation_args.get(key)) for key in MAX_OUTPUT_ALIASES],
        )
        if max_length is not None:
            data["max_length"] = max_length
            model_args["max_length"] = max_length
            if "max_model_len" in model_args:
                model_args["max_model_len"] = max_length
            data["extra_model_args"] = model_args
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
            for key in MAX_OUTPUT_ALIASES:
                generation_args.pop(key, None)
            generation_args["max_gen_toks"] = max_tokens
            data["gen_kwargs"] = format_key_value_args(generation_args)
        return data


def _document_mapping(document: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(document, Mapping):
        return document
    if isinstance(document, Path):
        text = document.read_text()
    elif isinstance(document, str):
        candidate = Path(document)
        try:
            text = candidate.read_text() if candidate.exists() else document
        except OSError:
            text = document
    else:
        raise TypeError(f"unsupported evaluation config document: {type(document).__name__}")
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, Mapping):
        raise ValueError("evaluation config document must contain a mapping")
    return parsed


def load_evaluation_config(document: str | Path | Mapping[str, Any]) -> EvaluationConfig:
    """Load one YAML or mapping evaluation request."""
    return EvaluationConfig.model_validate(_document_mapping(document))


def apply_evaluation_patch(config: EvaluationConfig, patch: Mapping[str, Any]) -> EvaluationConfig:
    """Apply a typed replacement patch and revalidate the complete request."""
    data = config.model_dump(exclude_none=True)
    if "max_length" in patch and "extra_model_args" not in patch:
        model_args = parse_key_value_args(data.get("extra_model_args"))
        for key in MODEL_LENGTH_ALIASES:
            model_args.pop(key, None)
        data["extra_model_args"] = model_args
    if "max_tokens" in patch and "gen_kwargs" not in patch:
        generation_args = parse_key_value_args(data.get("gen_kwargs"))
        for key in MAX_OUTPUT_ALIASES:
            generation_args.pop(key, None)
        data["gen_kwargs"] = format_key_value_args(generation_args) or None
    return EvaluationConfig.model_validate({**data, **patch})


def canonical_json(config: EvaluationConfig) -> bytes:
    """Return stable serialized evaluation intent suitable for records and hashing."""
    return json.dumps(config.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":")).encode()


def fingerprint() -> str:
    """Return the stable schema fingerprint for artifact publication."""
    schema = {"version": _SCHEMA_VERSION, "schema": EvaluationConfig.model_json_schema()}
    return hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def materialize_eval_args(config: EvaluationConfig) -> list[str]:
    """Materialize Evalchemy CLI flags for the portable evaluation layer."""
    args = ["--tasks", ",".join(config.tasks)]
    if config.apply_chat_template:
        args.append("--apply_chat_template")
    if config.limit is not None:
        args.extend(["--limit", str(config.limit)])
    if config.num_fewshot is not None:
        args.extend(["--num_fewshot", str(config.num_fewshot)])
    if config.batch_size is not None:
        args.extend(["--batch_size", str(config.batch_size)])
    if config.seed is not None:
        args.extend(["--seed", str(config.seed)])
    if config.gen_kwargs:
        args.extend(["--gen_kwargs", config.gen_kwargs])
    if config.max_length is not None:
        args.extend(["--max_length", str(config.max_length)])
    if config.max_tokens is not None:
        args.extend(["--max_tokens", str(config.max_tokens)])
    return args
