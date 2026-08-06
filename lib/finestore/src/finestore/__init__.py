# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""finestore — an append-only columnar archive of Parquet shards on object storage.

A ``DataStore`` (``finestore.store``) writes typed ``DataTable``s and opaque blobs under a URL prefix
as immutable Parquet shards; a ``CompositeReader`` (``finestore.reader``) composes and deduplicates
every writer's shards on read, with column projection and predicate filtering. There is no manifest —
shard membership is a directory listing and schema comes from the Parquet footers. ``finestore.compaction``
merges a table's shards into one next-generation shard. Import from these submodules directly; this
package exports nothing.
"""
