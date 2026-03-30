# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Push-based log service for Iris.

The log service receives log entries from workers and the controller via
PushLogs RPC and serves them via QueryLogs. It is initially co-hosted on
the controller but designed for independent deployment.
"""

from iris.log_server.client import LogPusher, RemoteLogHandler
from iris.log_server.server import LogServiceImpl

__all__ = [
    "LogPusher",
    "LogServiceImpl",
    "RemoteLogHandler",
]
