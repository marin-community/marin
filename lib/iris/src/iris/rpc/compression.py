# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared compression configuration for Iris RPC servers and clients.

Most Iris RPC traffic is response-dominated, so ordinary controller clients
only advertise ``Accept-Encoding`` via this list. Actor calls can carry large
request payloads as well; actor clients explicitly send with ``IRIS_RPC_ZSTD``.
Servers negotiate against ``IRIS_RPC_COMPRESSIONS``. zstd is listed first as
the preferred encoding; gzip is kept for interop with older peers.
"""

from connectrpc.compression.gzip import GzipCompression
from connectrpc.compression.zstd import ZstdCompression

# Importing this module installs the compact JSON codec; pulling it in alongside
# compression guarantees every iris RPC server/client gets the patched encoder
# without the entry points having to remember to import it themselves.
from iris.rpc import codecs as _codecs  # noqa: F401

# zstd level -1 ("fast") trades ratio for ~3-5x lower CPU at the encoder.
# Iris controller spent ~5% serving-thread CPU on zstd at the default level 3.
IRIS_RPC_ZSTD = ZstdCompression(level=-1)
IRIS_RPC_COMPRESSIONS = (IRIS_RPC_ZSTD, GzipCompression())
