# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared compression configuration for iris RPC servers and clients.

Iris RPC traffic is response-dominated (FetchLogs / list RPCs); requests are
small in practice, so clients pass ``send_compression=None`` and only
advertise ``Accept-Encoding`` via this list. Servers negotiate against it.
zstd is listed first as the preferred response encoding; gzip is kept for
interop with older peers.
"""

from __future__ import annotations

from connectrpc.compression.gzip import GzipCompression
from connectrpc.compression.zstd import ZstdCompression

# Importing this module installs the compact JSON codec; pulling it in alongside
# compression guarantees every iris RPC server/client gets the patched encoder
# without the entry points having to remember to import it themselves.
from iris.rpc import codecs as _codecs  # noqa: F401

IRIS_RPC_COMPRESSIONS = (ZstdCompression(), GzipCompression())
