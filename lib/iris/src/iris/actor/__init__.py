# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Actor system for distributed RPC.

Import concrete types from their defining modules: ``ActorServer``/``ActorId``
from ``iris.actor.server``, ``ActorClient`` from ``iris.actor.client``,
``ActorPool`` from ``iris.actor.pool``, and the resolver types from
``iris.actor.resolver``. Keeping concrete implementations in their defining
modules also avoids import cycles through ``iris.client``.
"""
