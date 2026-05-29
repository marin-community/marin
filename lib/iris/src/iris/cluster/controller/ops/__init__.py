# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped command modules above the reconcile kernel.

Each submodule names an aggregate (job, worker, task) and exposes the
"open tx → load → kernel-or-direct-write → apply effects → commit" verbs
that controller RPC handlers and internal loops call.

The submodule imports below make ``ops.job`` / ``ops.task`` / ``ops.worker``
resolvable for callers that do ``from iris.cluster.controller import ops`` and
then reach a verb via attribute access (the pattern used throughout the
controller and tests). Without them, attribute access would depend on some
other module having imported each submodule first.
"""

from iris.cluster.controller.ops import job, task, worker
