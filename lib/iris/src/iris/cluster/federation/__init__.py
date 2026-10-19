# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federation: remote Iris clusters as peers.

A controller drives one local execution *backend* for its job DAG. A *peer* is a
full remote Iris cluster that owns its own DAG; whole jobs are handed off to it
and the parent caches what the peer reports. The two meet only at the submit-time
router.

This package is the self-contained home for the peer side: the per-peer
connection (:mod:`iris.cluster.federation.peer`), the capability heartbeat that
learns what each peer can currently schedule, the ``ListPeers`` view
(:mod:`iris.cluster.federation.manager`), and the submit-time peer router
(:mod:`iris.cluster.federation.router`). It never touches the DAG fold or the
backend seam.
"""
