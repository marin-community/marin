# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Final SQLAlchemy declarations for Iris resource schema v2."""

from iris.cluster.controller.persistence.schema.base import meta_table as meta_table
from iris.cluster.controller.persistence.schema.base import metadata as metadata
from iris.cluster.controller.persistence.schema.base import schema_migrations_table as schema_migrations_table
from iris.cluster.controller.persistence.schema.execution import node_attributes_table as node_attributes_table
from iris.cluster.controller.persistence.schema.execution import node_capacity_table as node_capacity_table
from iris.cluster.controller.persistence.schema.execution import rpc_node_details_table as rpc_node_details_table
from iris.cluster.controller.persistence.schema.execution import rpc_nodes_table as rpc_nodes_table
from iris.cluster.controller.persistence.schema.execution import scaling_groups_table as scaling_groups_table
from iris.cluster.controller.persistence.schema.execution import slice_members_table as slice_members_table
from iris.cluster.controller.persistence.schema.execution import slices_table as slices_table
from iris.cluster.controller.persistence.schema.federation import federated_jobs_table as federated_jobs_table
from iris.cluster.controller.persistence.schema.federation import federated_tasks_table as federated_tasks_table
from iris.cluster.controller.persistence.schema.federation import (
    federation_changelog_table as federation_changelog_table,
)
from iris.cluster.controller.persistence.schema.federation import (
    federation_sync_state_table as federation_sync_state_table,
)
from iris.cluster.controller.persistence.schema.operations import action_receipts_table as action_receipts_table
from iris.cluster.controller.persistence.schema.operations import user_budgets_table as user_budgets_table
from iris.cluster.controller.persistence.schema.version import RESOURCE_SCHEMA_EPOCH as RESOURCE_SCHEMA_EPOCH
from iris.cluster.controller.persistence.schema.version import RESOURCE_SCHEMA_NAME as RESOURCE_SCHEMA_NAME
from iris.cluster.controller.persistence.schema.workloads import (
    attempt_runtime_objects_table as attempt_runtime_objects_table,
)
from iris.cluster.controller.persistence.schema.workloads import attempts_table as attempts_table
from iris.cluster.controller.persistence.schema.workloads import endpoints_table as endpoints_table
from iris.cluster.controller.persistence.schema.workloads import job_specs_table as job_specs_table
from iris.cluster.controller.persistence.schema.workloads import job_workdir_files_table as job_workdir_files_table
from iris.cluster.controller.persistence.schema.workloads import jobs_table as jobs_table
from iris.cluster.controller.persistence.schema.workloads import tasks_table as tasks_table
