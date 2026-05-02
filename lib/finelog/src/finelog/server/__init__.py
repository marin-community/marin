# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Push-based finelog log service.

Receives log entries via PushLogs and serves them via FetchLogs.
"""

from finelog.client import LogClient, RemoteLogHandler
from finelog.server.service import LogServiceImpl

__all__ = [
    "LogClient",
    "LogServiceImpl",
    "RemoteLogHandler",
]
