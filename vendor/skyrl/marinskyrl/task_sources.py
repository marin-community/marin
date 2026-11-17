"""Typed data sources shared by the launch host and SkyRL runtime."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias


class DataSourceKind(StrEnum):
    DIRECTORY = "directory"
    TASKTROVE_PARQUET = "tasktrove_parquet"


class TaskTroveTagMatch(StrEnum):
    ALL = "all"
    ANY = "any"


def _normalized_values(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if any(not value.strip() for value in values):
        raise ValueError(f"TaskTrove {name} cannot contain blank values")
    if len(set(values)) != len(values):
        raise ValueError(f"TaskTrove {name} cannot contain duplicate values")
    return tuple(sorted(values))


@dataclass(frozen=True)
class TaskTroveSelection:
    """An exact metadata predicate over one TaskTrove Clean release."""

    sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    tag_match: TaskTroveTagMatch = TaskTroveTagMatch.ALL
    limit: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", _normalized_values("sources", self.sources))
        object.__setattr__(self, "tags", _normalized_values("tags", self.tags))
        object.__setattr__(self, "modes", _normalized_values("modes", self.modes))
        if not self.sources and not self.tags and not self.modes:
            raise ValueError("TaskTrove selection requires at least one source, tag, or mode")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("TaskTrove selection limit must be positive")


@dataclass(frozen=True)
class TaskTroveSelectionSnapshot:
    count: int
    digest: str
    distinct_environment_count: int


def _resolved_path(local_path: str, relative_path: str) -> str:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Data relative_path must stay below its source root: {relative_path!r}")
    return os.path.join(local_path, *relative.parts)


@dataclass(frozen=True)
class DirectoryDataSource:
    uri: str
    identity: str
    local_path: str
    relative_path: str
    kind: Literal[DataSourceKind.DIRECTORY] = DataSourceKind.DIRECTORY

    def resolved_path(self) -> str:
        return _resolved_path(self.local_path, self.relative_path)


@dataclass(frozen=True)
class TaskTroveParquetSource:
    uri: str
    identity: str
    local_path: str
    relative_path: str
    verifier_ref: str
    selection: TaskTroveSelection
    snapshot: TaskTroveSelectionSnapshot | None = None
    kind: Literal[DataSourceKind.TASKTROVE_PARQUET] = DataSourceKind.TASKTROVE_PARQUET

    def resolved_path(self) -> str:
        return _resolved_path(self.local_path, self.relative_path)


DataSource: TypeAlias = DirectoryDataSource | TaskTroveParquetSource


def data_source_dict(source: DataSource) -> dict[str, Any]:
    """Serialize a source with wire-format enum values."""
    value = asdict(source)
    value["kind"] = source.kind.value
    if isinstance(source, TaskTroveParquetSource):
        value["selection"]["tag_match"] = source.selection.tag_match.value
    return value


def data_source(value: Mapping[str, Any]) -> DataSource:
    """Parse one strict tagged data source from JSON-compatible values."""
    fields = dict(value)
    try:
        kind = DataSourceKind(fields.pop("kind"))
    except KeyError as error:
        raise ValueError("Data source requires a kind") from error
    if kind is DataSourceKind.DIRECTORY:
        return DirectoryDataSource(kind=kind, **fields)

    selection_value = fields.pop("selection")
    selection_fields = dict(selection_value)
    selection_fields["sources"] = tuple(selection_fields.get("sources", ()))
    selection_fields["tags"] = tuple(selection_fields.get("tags", ()))
    selection_fields["modes"] = tuple(selection_fields.get("modes", ()))
    selection_fields["tag_match"] = TaskTroveTagMatch(selection_fields.get("tag_match", "all"))
    snapshot_value = fields.pop("snapshot", None)
    snapshot = TaskTroveSelectionSnapshot(**snapshot_value) if snapshot_value is not None else None
    return TaskTroveParquetSource(
        kind=kind,
        selection=TaskTroveSelection(**selection_fields),
        snapshot=snapshot,
        **fields,
    )
