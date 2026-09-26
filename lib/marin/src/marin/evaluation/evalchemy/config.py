# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load Evalchemy launch configuration without importing Evalchemy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from rigging.secrets import is_secret_reference

RESERVED_ENDPOINT_MODEL_ARGS = frozenset(
    {"model", "base_url", "tokenizer", "tokenizer_backend", "tokenized_requests", "chat_template_kwargs"}
)


class EvalchemyJudgeConfig(BaseModel):
    """External judge endpoint and credential references for a supported task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    model: str
    api_key: tuple[str, ...] = Field(min_length=1)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("judge.base_url must be an http(s) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("judge.base_url must not contain credentials, a query, or a fragment")
        return value

    @field_validator("model")
    @classmethod
    def nonempty_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("judge.model must not be empty")
        return value

    @field_validator("api_key")
    @classmethod
    def validate_api_key_references(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not is_secret_reference(value) for value in values):
            raise ValueError("judge.api_key accepts only env:, file:, or gcp-secret:// references")
        if any(value in {"env:", "file:", "gcp-secret://"} for value in values):
            raise ValueError("judge.api_key references must name a secret")
        return values


class EvalchemyTaskOptions(BaseModel):
    """Per-task behavior from an Evalchemy launch file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_fewshot: int | None = None
    task_alias: str | None = None
    generation: bool = False
    unsafe_code: bool = False
    completion_only: bool = False


class EvalchemyConfig(BaseModel):
    """Evalchemy launch settings consumed by Marin's endpoint runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: tuple[str, ...] = Field(min_length=1)
    task_options: Mapping[str, EvalchemyTaskOptions] = Field(default_factory=dict)
    apply_chat_template: bool | None = None
    limit: int | None = None
    num_fewshot: int | None = None
    batch_size: str | None = None
    seed: int | None = None
    gen_kwargs: Mapping[str, str] = Field(default_factory=dict)
    chat_template_kwargs: Mapping[str, bool] = Field(default_factory=dict)
    extra_model_args: Mapping[str, str | int | float | bool] = Field(default_factory=dict)
    max_length: int | None = None
    max_tokens: int | None = None
    runtime_extras: tuple[str, ...] = ()
    judge: EvalchemyJudgeConfig | None = None

    @model_validator(mode="after")
    def validate_judge_task(self) -> EvalchemyConfig:
        financebench = self.tasks == ("FinanceBench",)
        if self.judge is not None and not financebench:
            raise ValueError("judge is supported only for a single FinanceBench task")
        if financebench and self.judge is None:
            raise ValueError("FinanceBench requires an explicit judge configuration")
        return self

    @field_validator("tasks", "runtime_extras")
    @classmethod
    def nonempty_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values):
            raise ValueError("names must not be empty")
        return values

    @field_validator("batch_size", mode="before")
    @classmethod
    def normalize_batch_size(cls, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("batch_size must be a string, integer, or null")
        return str(value)

    @field_validator("gen_kwargs", mode="before")
    @classmethod
    def parse_gen_kwargs(cls, value: object) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, str):
            raise ValueError("gen_kwargs must be a comma-separated string")
        parsed: dict[str, str] = {}
        for item in value.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"invalid gen_kwargs item {item!r}")
            key, raw_value = item.split("=", 1)
            if not key.strip():
                raise ValueError("gen_kwargs keys must not be empty")
            parsed[key.strip()] = raw_value.strip()
        return parsed


def load_evalchemy_config(path: Path) -> EvalchemyConfig:
    """Load one Evalchemy launch file; task support is checked by the evaluator CLI at runtime."""
    try:
        document = yaml.safe_load(path.read_text())
        return EvalchemyConfig.model_validate(document)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"invalid Evalchemy config {path}: {exc}") from exc
