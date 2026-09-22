# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load, validate, and render the structured agentic-lint catalog."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

CATALOG_SCHEMA_VERSION = 1
RULE_CODE = re.compile(r"^ml-[a-z0-9-]+$")
DEFAULT_CATALOG_DIR = Path(__file__).parent
CATALOG_FIELDS = frozenset({"schema_version", "shared_prompt", "lanes"})
LANE_FIELDS = frozenset({"name", "prompt", "include_complexity_leads", "min_diff_lines", "rules"})
RULE_FIELDS = frozenset({"schema_version", "code", "lane", "title", "minimum_confidence", "prompt"})


@dataclass(frozen=True)
class LintRule:
    """One rule exactly as stored and rendered into a lane prompt."""

    code: str
    lane: str
    title: str
    prompt: str
    minimum_confidence: float
    path: Path


@dataclass(frozen=True)
class LintLane:
    """One prompt lane and its ordered rules."""

    name: str
    prompt: str
    include_complexity_leads: bool
    min_diff_lines: int
    rules: tuple[LintRule, ...]


@dataclass(frozen=True)
class LintCatalog:
    """Validated shared policy and lane rules from one checkout."""

    root: Path
    shared_prompt: str
    lanes: tuple[LintLane, ...]

    @property
    def rules(self) -> tuple[LintRule, ...]:
        return tuple(rule for lane in self.lanes for rule in lane.rules)

    def lane(self, name: str) -> LintLane:
        for lane in self.lanes:
            if lane.name == name:
                return lane
        raise KeyError(f"unknown lint lane: {name}")

    def rule(self, code: str) -> LintRule:
        for rule in self.rules:
            if rule.code == code:
                return rule
        raise KeyError(f"unknown lint rule: {code}")


@dataclass(frozen=True)
class LoadedLane:
    lane: LintLane
    rule_paths: frozenset[Path]


def _mapping(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _reject_extra_fields(value: dict[str, object], allowed: frozenset[str], path: Path) -> None:
    if extra := value.keys() - allowed:
        raise ValueError(f"{path}: unknown fields: {sorted(extra)}")


def _text(value: object, *, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {field} must be non-empty text")
    return value.rstrip()


def _integer(value: object, *, path: Path, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path}: {field} must be an integer")
    return value


def _load_rule(rule_path: Path, lane_name: str, declared_code: str, config_path: Path) -> LintRule:
    if not rule_path.is_file():
        raise ValueError(f"{config_path}: declared rule does not exist: {rule_path}")
    value = _mapping(rule_path)
    _reject_extra_fields(value, RULE_FIELDS, rule_path)
    if value.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"{rule_path}: unsupported schema_version")
    code = _text(value.get("code"), path=rule_path, field="code")
    if RULE_CODE.fullmatch(code) is None:
        raise ValueError(f"{rule_path}: invalid rule code {code!r}")
    if code != declared_code:
        raise ValueError(f"{rule_path}: code {code!r} does not match catalog declaration {declared_code!r}")
    rule_lane = _text(value.get("lane"), path=rule_path, field="lane")
    if rule_lane != lane_name:
        raise ValueError(f"{rule_path}: lane {rule_lane!r} does not match directory {lane_name!r}")
    if rule_path.stem != code:
        raise ValueError(f"{rule_path}: filename must be {code}.yaml")
    confidence = value.get("minimum_confidence", 0.7)
    if not isinstance(confidence, int | float) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError(f"{rule_path}: minimum_confidence must be between 0 and 1")
    return LintRule(
        code=code,
        lane=rule_lane,
        title=_text(value.get("title"), path=rule_path, field="title"),
        prompt=_text(value.get("prompt"), path=rule_path, field="prompt"),
        minimum_confidence=float(confidence),
        path=rule_path,
    )


def _load_lane(root: Path, config_path: Path, lane_value: object) -> LoadedLane:
    if not isinstance(lane_value, dict):
        raise ValueError(f"{config_path}: each lane must be a mapping")
    _reject_extra_fields(lane_value, LANE_FIELDS, config_path)
    name = _text(lane_value.get("name"), path=config_path, field="lane.name")
    prompt = _text(lane_value.get("prompt"), path=config_path, field=f"lanes.{name}.prompt")
    include_leads = lane_value.get("include_complexity_leads", False)
    if not isinstance(include_leads, bool):
        raise ValueError(f"{config_path}: lanes.{name}.include_complexity_leads must be boolean")
    min_diff_lines = _integer(
        lane_value.get("min_diff_lines", 0), path=config_path, field=f"lanes.{name}.min_diff_lines"
    )
    if min_diff_lines < 0:
        raise ValueError(f"{config_path}: lanes.{name}.min_diff_lines must be non-negative")
    rule_codes = lane_value.get("rules")
    if not isinstance(rule_codes, list) or not rule_codes or not all(isinstance(code, str) for code in rule_codes):
        raise ValueError(f"{config_path}: lanes.{name}.rules must be a non-empty list of rule codes")
    if len(set(rule_codes)) != len(rule_codes):
        raise ValueError(f"{config_path}: lanes.{name}.rules contains duplicates")
    rule_paths = tuple(root / "rules" / name / f"{code}.yaml" for code in rule_codes)
    rules = tuple(_load_rule(path, name, code, config_path) for path, code in zip(rule_paths, rule_codes, strict=True))
    return LoadedLane(LintLane(name, prompt, include_leads, min_diff_lines, rules), frozenset(rule_paths))


def load_catalog(root: Path = DEFAULT_CATALOG_DIR) -> LintCatalog:
    """Load the catalog from ``root`` and reject ambiguous or stale files."""
    config_path = root / "catalog.yaml"
    config = _mapping(config_path)
    _reject_extra_fields(config, CATALOG_FIELDS, config_path)
    if config.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"{config_path}: unsupported schema_version")
    shared_prompt = _text(config.get("shared_prompt"), path=config_path, field="shared_prompt")
    lane_values = config.get("lanes")
    if not isinstance(lane_values, list) or not lane_values:
        raise ValueError(f"{config_path}: lanes must be a non-empty list")

    loaded = tuple(_load_lane(root, config_path, value) for value in lane_values)
    lanes = tuple(item.lane for item in loaded)
    lane_names = [lane.name for lane in lanes]
    if len(set(lane_names)) != len(lane_names):
        raise ValueError(f"{config_path}: duplicate lane names")
    rule_codes = [rule.code for lane in lanes for rule in lane.rules]
    if len(set(rule_codes)) != len(rule_codes):
        raise ValueError(f"{config_path}: duplicate rule codes")
    declared_rule_paths = {path.resolve() for item in loaded for path in item.rule_paths}
    actual_rule_paths = {path.resolve() for path in (root / "rules").glob("*/*.yaml")}
    if undeclared := actual_rule_paths - declared_rule_paths:
        raise ValueError(f"rules exist outside declared lanes: {sorted(str(path) for path in undeclared)}")
    return LintCatalog(root=root, shared_prompt=shared_prompt, lanes=lanes)


def render_lane(catalog: LintCatalog, lane_name: str) -> str:
    """Render one lane deterministically for the production reviewer prompt."""
    lane = catalog.lane(lane_name)
    rule_blocks = [f"### `{rule.code}` — {rule.title}\n\n{rule.prompt}" for rule in lane.rules]
    return "\n\n".join((lane.prompt, *rule_blocks)).rstrip() + "\n"


def catalog_sha(catalog: LintCatalog) -> str:
    """Return the SHA-256 identity of all structured catalog source files."""
    digest = hashlib.sha256()
    paths = [catalog.root / "catalog.yaml", *(rule.path for rule in catalog.rules)]
    for path in sorted(paths):
        digest.update(path.relative_to(catalog.root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
