# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Projection package: in-memory materialized views over controller DB tables.

Each :class:`~iris.cluster.controller.projections.base.Projection` subclass
self-registers into ``db.caches`` at construction and is reached from any
cursor or the DB handle by concrete type — no module-global registry, no
threaded references.
"""
