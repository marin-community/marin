# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A finelog-backed sink for ``rigging.telltale`` metric forwarding.

``rigging.telltale`` owns the ``TelltaleMetric`` row and the forward loop; this
module is the finelog end of the handoff. Rows land in one namespace keyed on the
metric ``name``, so after compaction a metric's rows cluster into contiguous row
groups and ``WHERE name = ...`` prunes on parquet stats + bloom filters rather
than scanning every series. Leftover Prometheus labels ride in the native
``Map<Utf8,Utf8>`` ``labels`` column, readable with the ``json_get`` family of
UDFs.
"""

from collections.abc import Sequence

from rigging.telltale import TelltaleMetric

from finelog.client import LogClient

#: Namespace every process's telltale metrics land in.
TELLTALE_NAMESPACE = "telltale"


class FinelogMetricSink:
    """A ``rigging.telltale`` sink that appends rows to a finelog table.

    Owns the connection: construct it with the finelog endpoint, hand it to
    ``rigging.telltale.start_forwarding``, and the forwarder drives
    ``write``/``close``.
    """

    def __init__(self, endpoint: str, *, namespace: str = TELLTALE_NAMESPACE) -> None:
        self._client = LogClient.connect(endpoint)
        self._table = self._client.get_table(namespace, TelltaleMetric)

    def write(self, rows: Sequence[TelltaleMetric]) -> None:
        self._table.write(rows)

    def close(self) -> None:
        try:
            self._table.flush(timeout=5.0)
        finally:
            self._client.close()
