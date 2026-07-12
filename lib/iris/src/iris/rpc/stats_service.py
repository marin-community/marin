# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""StatsService implementation backed by :class:`RpcStatsCollector`.

Decoupled from any particular service so the controller, worker, or log
server can all mount their own instance. Currently wired in by
``ControllerDashboard``.
"""

from connectrpc.request import RequestContext

from iris.rpc import stats_pb2
from iris.rpc.stats import RpcStatsCollector


class RpcStatsService:
    """Serves :class:`iris.stats.StatsService`. Sync handler."""

    def __init__(self, collector: RpcStatsCollector):
        self._collector = collector

    def get_rpc_stats(
        self,
        request: stats_pb2.GetRpcStatsRequest,
        ctx: RequestContext,
    ) -> stats_pb2.GetRpcStatsResponse:
        del request, ctx
        return self._collector.snapshot_proto()
