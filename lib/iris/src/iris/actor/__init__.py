# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor system for distributed RPC.

Import concrete types from their defining modules: ``ActorServer``/``ActorId``
from ``iris.actor.server``, ``ActorClient`` from ``iris.actor.client``,
``ActorPool`` from ``iris.actor.pool``, and the resolver types from
``iris.actor.resolver``. Re-exporting the server here would pull it (and the
runtime telltale forwarder it starts, which imports ``iris.client``) into every
import of the package — and ``iris.client`` imports this package for the
resolver types, closing an import cycle.
"""
