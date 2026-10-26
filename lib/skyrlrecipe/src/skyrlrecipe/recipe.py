# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sparse authored settings and complete recipes for Hydra composition."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Self

from pydantic import Field, PositiveInt, model_validator

from .budget import ContextBudget
from .documents import MISSING, get_path, merge_mappings, set_path
from .model import FrozenMap, OpenMap, freeze, thaw
from .ownership import OWNER_MESSAGES, REMOVED
from .rules import RLEntrypoint, validate_engine_init_kwargs, validate_tp_divides_heads
from .sections import ConfigGroups, RecipeSections
from .settings import parse_setting


def _reject_paths(document: Mapping[str, Any], table: Mapping[str, str]) -> None:
    problems = [f"{path}: {message}" for path, message in table.items() if get_path(document, path) is not MISSING]
    if problems:
        raise ValueError("; ".join(problems))


class RecipePatch(RecipeSections):
    """Sparse authored values with optional context."""

    entrypoint: RLEntrypoint = RLEntrypoint.STANDARD
    context_budget: ContextBudget | None = None
    config_groups: ConfigGroups = Field(default_factory=ConfigGroups)
    extra_env: OpenMap = Field(default_factory=FrozenMap)
    model_num_attention_heads: PositiveInt | None = None

    @model_validator(mode="after")
    def _author_rules(self) -> Self:
        document = self.to_skyrl()
        for table in (REMOVED, OWNER_MESSAGES):
            _reject_paths(document, table)
        validate_engine_init_kwargs(self.generator.engine_init_kwargs)
        return self

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        _reject_paths(document, REMOVED)
        _reject_paths(document, OWNER_MESSAGES)
        return cls.model_validate_json(json.dumps(thaw(freeze(document)), allow_nan=False))

    def merge(self, *patches: RecipePatch) -> Self:
        document = self.to_skyrl()
        for patch in patches:
            document = merge_mappings(document, patch.to_skyrl())
        return type(self).from_document(document)

    def with_settings(self, settings: Sequence[str]) -> Self:
        document = self.to_skyrl()
        for setting in settings:
            key, separator, raw = setting.partition("=")
            if not separator or not key:
                raise ValueError(f"a setting must have dotted.key=value form: {setting!r}")
            patch = {}
            set_path(patch, key, parse_setting(type(self), key, raw))
            document = merge_mappings(document, patch)
        return type(self).from_document(document)


class SkyRLRecipe(RecipePatch):
    """A complete sparse recipe with a required rollout context declaration."""

    context_budget: ContextBudget

    @model_validator(mode="after")
    def _complete_author_rules(self) -> Self:
        validate_tp_divides_heads(self.generator.inference_engine_tensor_parallel_size, self.model_num_attention_heads)
        return self
