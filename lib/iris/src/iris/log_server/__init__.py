# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Push-based log service for Iris.

The log service receives log entries via PushLogs and serves them via
FetchLogs. In production, it runs as a separate process started by the
controller's main() entry point. Tests use LogServiceImpl in-process.
"""

from iris.log_server.client import LogPusher, LogServiceProxy, RemoteLogHandler
from iris.log_server.server import LogServiceImpl

__all__ = [
    "LogPusher",
    "LogServiceImpl",
    "LogServiceProxy",
    "RemoteLogHandler",
]
