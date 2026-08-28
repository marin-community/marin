# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned table layout and operating policy declarations."""

from dataclasses import dataclass
from enum import StrEnum

from finelog.policy import StoragePolicy
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.schema import Schema, schema_to_proto


class L0Mode(StrEnum):
    """Durability boundary for newly sealed L0 segments."""

    LEGACY_LOCAL = "legacy_local"
    OBJECT_NATIVE = "object_native"


_L0_MODE_TO_PROTO = {
    L0Mode.LEGACY_LOCAL: stats_pb2.L0_MODE_LEGACY_LOCAL,
    L0Mode.OBJECT_NATIVE: stats_pb2.L0_MODE_OBJECT_NATIVE,
}


@dataclass(frozen=True)
class IdentityTransform:
    """Partition rows by their source-column value."""


@dataclass(frozen=True)
class BucketTransform:
    """Partition rows into a fixed number of stable hash buckets."""

    buckets: int

    def __post_init__(self) -> None:
        if self.buckets <= 0:
            raise ValueError("bucket count must be positive")


PartitionTransform = IdentityTransform | BucketTransform


@dataclass(frozen=True)
class PartitionField:
    source_column: str
    name: str
    transform: PartitionTransform

    def to_proto(self) -> stats_pb2.PartitionField:
        if isinstance(self.transform, IdentityTransform):
            return stats_pb2.PartitionField(
                source_column=self.source_column,
                name=self.name,
                identity=stats_pb2.IdentityTransform(),
            )
        return stats_pb2.PartitionField(
            source_column=self.source_column,
            name=self.name,
            bucket=stats_pb2.BucketTransform(buckets=self.transform.buckets),
        )


@dataclass(frozen=True)
class PartitionSpec:
    """Versioned physical partition declaration."""

    spec_id: int
    fields: tuple[PartitionField, ...] = ()

    def __post_init__(self) -> None:
        if self.spec_id <= 0:
            raise ValueError("partition spec_id must be positive")

    def to_proto(self) -> stats_pb2.PartitionSpec:
        return stats_pb2.PartitionSpec(spec_id=self.spec_id, fields=[field.to_proto() for field in self.fields])


@dataclass(frozen=True)
class SourceLayout:
    """Physical source-Parquet layout for a table version."""

    partition: PartitionSpec | None = None
    target_object_bytes: int = 0

    def to_proto(self, schema: Schema) -> stats_pb2.SourceLayout:
        layout = stats_pb2.SourceLayout(
            sort_columns=schema.sort_columns,
            max_row_group_rows=schema.max_row_group_rows,
            target_object_bytes=self.target_object_bytes,
        )
        if self.partition is not None:
            layout.partition.CopyFrom(self.partition.to_proto())
        return layout


@dataclass(frozen=True)
class OperatingPolicy:
    """Buffering, durability, cache, and query-lifetime policy."""

    l0_mode: L0Mode = L0Mode.OBJECT_NATIVE
    max_buffer_bytes: int = 0
    max_flush_age_ms: int = 0
    max_query_time_ms: int = 0

    def to_proto(self, local_cache: StoragePolicy) -> stats_pb2.OperatingPolicy:
        return stats_pb2.OperatingPolicy(
            l0_mode=_L0_MODE_TO_PROTO[self.l0_mode],
            local_cache=local_cache.to_proto(),
            remote_retention=stats_pb2.RemoteRetentionPolicy(retain_forever=True),
            max_buffer_bytes=self.max_buffer_bytes,
            max_flush_age_ms=self.max_flush_age_ms,
            max_query_time_ms=self.max_query_time_ms,
        )


@dataclass(frozen=True)
class TableSpec:
    """Complete versioned policy layered on a table's logical schema."""

    version: int
    source_layout: SourceLayout = SourceLayout()
    artifact_revision: int = 1
    operating_policy: OperatingPolicy = OperatingPolicy()

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("table spec version must be positive")
        if self.artifact_revision <= 0:
            raise ValueError("artifact revision must be positive")

    def to_proto(self, schema: Schema, local_cache: StoragePolicy) -> stats_pb2.TableSpec:
        """Encode the complete immutable policy for registration."""
        logical_schema = schema_to_proto(schema)
        indexes = [
            stats_pb2.ColumnArtifactPolicy(column=column.name, index=column.index)
            for column in logical_schema.columns
            if column.index.trigram or column.index.exact_values or column.index.value_counts
        ]
        return stats_pb2.TableSpec(
            version=self.version,
            logical_schema=logical_schema,
            source_layout=self.source_layout.to_proto(schema),
            artifact_policy=stats_pb2.ArtifactPolicy(
                revision=self.artifact_revision,
                indexes=indexes,
                projections=logical_schema.projections,
                grouped_extrema=logical_schema.grouped_extrema,
            ),
            operating_policy=self.operating_policy.to_proto(local_cache),
        )


@dataclass(frozen=True)
class TableStatus:
    """Active and desired version pointers for one registered table."""

    active_version: int
    desired_version: int | None
    migration_phase: str
    catalog_generation: int


def table_status_from_proto(response: stats_pb2.GetTableStatusResponse) -> TableStatus:
    active_version = response.active_table_spec.version if response.HasField("active_table_spec") else 0
    desired_version = response.desired_table_spec.version if response.HasField("desired_table_spec") else None
    migration_phase = (
        stats_pb2.MigrationPhase.Name(response.migration.phase)
        if response.HasField("migration")
        else stats_pb2.MigrationPhase.Name(stats_pb2.MIGRATION_PHASE_ACTIVATED)
    )
    return TableStatus(
        active_version=active_version,
        desired_version=desired_version,
        migration_phase=migration_phase,
        catalog_generation=response.catalog_generation,
    )
