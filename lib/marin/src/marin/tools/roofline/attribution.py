# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regex-based semantic attribution for profile rows."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class AttributionRule:
    semantic_op: str
    op_name_regex: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AttributionRule:
        match = payload.get("match", {})
        if not isinstance(match, dict):
            raise ValueError(f"Attribution rule for {payload.get('semantic_op')} has no match object")
        return cls(semantic_op=str(payload["semantic_op"]), op_name_regex=str(match["op_name_regex"]))

    def to_dict(self) -> dict[str, Any]:
        return {"semantic_op": self.semantic_op, "match": {"op_name_regex": self.op_name_regex}}


def load_attribution_rules() -> list[AttributionRule]:
    registry_text = resources.files(__package__).joinpath("default_attribution.json").read_text(encoding="utf-8")
    payload = json.loads(registry_text)
    return [AttributionRule.from_dict(rule) for rule in payload["rules"]]


def attribute_name(name: str, rules: list[AttributionRule]) -> str | None:
    for rule in rules:
        if re.search(rule.op_name_regex, name, flags=re.IGNORECASE):
            return rule.semantic_op
    return None


def suggested_regex(name: str) -> str:
    tokens = [token for token in re.split(r"[^A-Za-z0-9_]+", name) if len(token) >= 4]
    if not tokens:
        return re.escape(name[:64])
    return ".*".join(re.escape(token) for token in tokens[:3])
