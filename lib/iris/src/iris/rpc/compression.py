# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared compression configuration for iris RPC servers and clients.

zstd is preferred and gzip is the fallback for older peers. Iris RPC traffic
is dominated by log payloads (FetchLogs responses, PushLogs requests); the
gzip path was the top allocator on prod (memray showed gzip.compress at ~66%
of allocated bytes in the finelog server alone). zstd cuts that meaningfully
without giving up interop with gzip-only clients.
"""

from __future__ import annotations

from connectrpc.compression.gzip import GzipCompression
from connectrpc.compression.zstd import ZstdCompression

# Order matters only on the client side (the negotiator walks the client's
# Accept-Encoding in order); we keep zstd first here for readability.
IRIS_RPC_COMPRESSIONS = (ZstdCompression(), GzipCompression())
