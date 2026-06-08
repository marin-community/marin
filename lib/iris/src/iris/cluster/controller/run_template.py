# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-job ``RunTaskRequest`` template cache.

A leaf module so both the dispatch path (``dispatch``) and the command
layer (``ops.job``) can name the cache type without pulling the dispatch logic
— and, transitively, the ``backend`` contract — into the ``ops`` aggregator's
import graph.
"""

from iris.cluster.controller.lru_cache import LRUCache
from iris.rpc import job_pb2

# Per-job RunTaskRequest templates are cached in RunTemplateCache.
# 4096 templates ~= worst-case concurrent job count we expect in a single
# controller process. Same-name replacement reuses the original ``job_id``,
# so ``submit_job`` evicts the cached entry before inserting the new row to
# prevent serving the prior submission's payload.
RUN_REQUEST_TEMPLATE_CACHE_SIZE = 4096

# LRU cache for per-job ``RunTaskRequest`` templates, keyed by wire job id.
# Templates carry the immutable per-job fields (entrypoint, environment,
# resources, constraints); per-attempt fields (``task_id``, ``attempt_id``)
# are stamped onto a copy at fan-out time.
RunTemplateCache = LRUCache[str, job_pb2.RunTaskRequest]


def new_run_template_cache() -> RunTemplateCache:
    return LRUCache(RUN_REQUEST_TEMPLATE_CACHE_SIZE)
