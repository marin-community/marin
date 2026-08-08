# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable Node, capacity, and Slice projections for resource schema v2."""

from iris.cluster.controller.persistence.schema.base import metadata
from sqlalchemy import (
    REAL,
    CheckConstraint,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    Table,
    Text,
    UniqueConstraint,
    text,
)

rpc_nodes_table = Table(
    "rpc_nodes",
    metadata,
    Column("node_uid", Text, primary_key=True),
    Column("node_id", Text, nullable=False),
    Column("execution_cluster_id", Text, nullable=False),
    Column("backend_id", Text, nullable=False),
    Column("scaling_group_id", Text),
    Column("registered_at_ms", Integer, nullable=False),
    Column("last_seen_at_ms", Integer, nullable=False),
    Column("retired_at_ms", Integer),
    CheckConstraint("execution_cluster_id <> ''"),
    CheckConstraint("backend_id <> ''"),
)

Index(
    "current_rpc_node_logical_id",
    rpc_nodes_table.c.execution_cluster_id,
    rpc_nodes_table.c.backend_id,
    rpc_nodes_table.c.node_id,
    unique=True,
    sqlite_where=rpc_nodes_table.c.retired_at_ms.is_(None),
)
Index(
    "rpc_nodes_scaling_group",
    rpc_nodes_table.c.execution_cluster_id,
    rpc_nodes_table.c.backend_id,
    rpc_nodes_table.c.scaling_group_id,
)


rpc_node_details_table = Table(
    "rpc_node_details",
    metadata,
    Column("node_uid", Text, ForeignKey("rpc_nodes.node_uid", ondelete="CASCADE"), primary_key=True),
    Column("address", Text, nullable=False),
    Column("hostname", Text, nullable=False),
    Column("ip_address", Text, nullable=False),
    Column("provider_instance_id", Text, nullable=False),
    Column("provider_zone", Text, nullable=False),
    Column("provenance_json", Text, nullable=False),
    CheckConstraint("json_valid(provenance_json)"),
)


node_capacity_table = Table(
    "node_capacity",
    metadata,
    Column("node_uid", Text, ForeignKey("rpc_nodes.node_uid", ondelete="CASCADE"), primary_key=True),
    Column("cpu_millicores", Integer, nullable=False),
    Column("memory_bytes", Integer, nullable=False),
    Column("disk_bytes", Integer, nullable=False),
    Column("accelerator_kind", Text, nullable=False),
    Column("accelerator_variant", Text, nullable=False),
    Column("accelerator_count", Integer, nullable=False),
    CheckConstraint("cpu_millicores >= 0"),
    CheckConstraint("memory_bytes >= 0"),
    CheckConstraint("disk_bytes >= 0"),
    CheckConstraint("accelerator_count >= 0"),
)


node_attributes_table = Table(
    "node_attributes",
    metadata,
    Column("node_uid", Text, ForeignKey("rpc_nodes.node_uid", ondelete="CASCADE"), nullable=False),
    Column("key", Text, nullable=False),
    Column("value_type", Text, nullable=False),
    Column("str_value", Text),
    Column("int_value", Integer),
    Column("float_value", REAL),
    PrimaryKeyConstraint("node_uid", "key"),
    CheckConstraint("value_type IN ('str', 'int', 'float')"),
    CheckConstraint(
        "(value_type = 'str' AND str_value IS NOT NULL AND int_value IS NULL AND float_value IS NULL) OR "
        "(value_type = 'int' AND str_value IS NULL AND int_value IS NOT NULL AND float_value IS NULL) OR "
        "(value_type = 'float' AND str_value IS NULL AND int_value IS NULL AND float_value IS NOT NULL)"
    ),
)


scaling_groups_table = Table(
    "scaling_groups",
    metadata,
    Column("execution_cluster_id", Text, nullable=False),
    Column("backend_id", Text, nullable=False),
    Column("scaling_group_id", Text, nullable=False),
    Column("consecutive_failures", Integer, nullable=False, server_default=text("0")),
    Column("backoff_until_ms", Integer, nullable=False, server_default=text("0")),
    Column("last_scale_up_at_ms", Integer, nullable=False, server_default=text("0")),
    Column("last_scale_down_at_ms", Integer, nullable=False, server_default=text("0")),
    Column("quota_exceeded_until_ms", Integer, nullable=False, server_default=text("0")),
    Column("quota_reason", Text, nullable=False, server_default=""),
    Column("updated_at_ms", Integer, nullable=False),
    PrimaryKeyConstraint("execution_cluster_id", "backend_id", "scaling_group_id"),
)


slices_table = Table(
    "slices",
    metadata,
    Column("slice_uid", Text, primary_key=True),
    Column("slice_id", Text, nullable=False),
    Column("execution_cluster_id", Text, nullable=False),
    Column("backend_id", Text, nullable=False),
    Column("scaling_group_id", Text, nullable=False),
    Column("management_mode", Text, nullable=False),
    Column("lifecycle", Text, nullable=False),
    Column("membership_state", Text, nullable=False),
    Column("created_at_ms", Integer, nullable=False),
    Column("observed_at_ms", Integer),
    Column("error_message", Text, nullable=False, server_default=""),
    CheckConstraint("execution_cluster_id <> ''"),
    CheckConstraint("backend_id <> ''"),
    CheckConstraint("management_mode IN ('autoscaled', 'manual')"),
    CheckConstraint("lifecycle IN ('creating', 'ready', 'deleting', 'failed')"),
    CheckConstraint("membership_state IN ('unknown', 'observed')"),
    UniqueConstraint("execution_cluster_id", "backend_id", "slice_id"),
    ForeignKeyConstraint(
        ("execution_cluster_id", "backend_id", "scaling_group_id"),
        (
            "scaling_groups.execution_cluster_id",
            "scaling_groups.backend_id",
            "scaling_groups.scaling_group_id",
        ),
    ),
)

Index(
    "slices_scaling_group",
    slices_table.c.execution_cluster_id,
    slices_table.c.backend_id,
    slices_table.c.scaling_group_id,
)


slice_members_table = Table(
    "slice_members",
    metadata,
    Column("slice_uid", Text, ForeignKey("slices.slice_uid", ondelete="CASCADE"), nullable=False),
    Column("provider_node_id", Text, nullable=False),
    Column("observed_at_ms", Integer, nullable=False),
    PrimaryKeyConstraint("slice_uid", "provider_node_id"),
)
