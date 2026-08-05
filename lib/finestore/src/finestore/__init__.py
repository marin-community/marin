# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""finestore — an append-only columnar archive of Parquet shards on object storage.

A :class:`DataStore` writes typed :class:`DataTable`s and opaque blobs under a URL prefix as
immutable Parquet shards; a :class:`CompositeReader` composes and deduplicates every writer's shards
on read, with column projection and predicate filtering. There is no manifest — shard membership is
a directory listing and schema comes from the Parquet footers. See :func:`compact` for merging
level-0 shards.
"""

from finestore.compaction import compact
from finestore.layout import build_uri, parse_uri
from finestore.reader import CompositeReader
from finestore.store import DataStore, DataTable
