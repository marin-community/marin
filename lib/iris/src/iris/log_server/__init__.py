# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Push-based log service for Iris.

The log service receives log entries via PushLogs and serves them via
FetchLogs (same API as ControllerService.FetchLogs for client compat).
Co-hosted on the controller but designed for independent deployment.
"""

from iris.log_server.client import LogPusher, RemoteLogHandler
from iris.log_server.server import LogServiceImpl

__all__ = [
    "LogPusher",
    "LogServiceImpl",
    "RemoteLogHandler",
]
