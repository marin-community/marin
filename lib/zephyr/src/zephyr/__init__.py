# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr: Lightweight dataset library for distributed data processing.

Import from the defining submodule (``zephyr.dataset``, ``zephyr.execution``, ``zephyr.expr``,
``zephyr.readers``, ``zephyr.writers``, ``zephyr.plan``, ``zephyr.worker_context``, ``zephyr.counters``)
rather than the package: a re-export hub here ties every importer of any submodule to all of
them, which over-selects CI tests.
"""
