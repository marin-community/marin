# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter optimizers.

Import a config from its defining submodule (``levanter.optim.config`` for ``AdamConfig`` /
``OptimizerConfig``, ``levanter.optim.muon`` for ``MuonConfig``, ...) rather than the package.
The submodules register themselves with ``OptimizerConfig`` (a ``draccus.PluginRegistry``), which
discovers them lazily under ``levanter.optim`` on first parse -- so this module imports nothing
and a change to one optimizer no longer selects every test that touches ``levanter.optim``.
"""
