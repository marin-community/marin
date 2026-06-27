# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote backend: the root side of the agent <-> root reconcile.

A :class:`~iris.cluster.backends.remote.backend.RemoteTaskBackend` runs in the
root controller as a CLUSTER_VIEW backend. It holds a
:class:`~iris.cluster.backends.remote.relay.BackendRelay` that the
:class:`~iris.cluster.backends.remote.server.RemoteAgentServer` reads/writes as
remote agents poll in.
"""
