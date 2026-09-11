# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped command modules above the reconcile kernel.

Each submodule names an aggregate (job, worker, task) and exposes the
"open tx → load → kernel-or-direct-write → apply effects → commit" verbs
that controller RPC handlers and internal loops call.
"""

from iris.cluster.controller.ops import job, task, worker
