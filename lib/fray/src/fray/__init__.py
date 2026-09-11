# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fray: minimal job and actor scheduling interface.

Import from the defining submodule (``fray.client``, ``fray.actor``, ``fray.types``,
``fray.local_backend``, ``fray.current_client``) rather than the package: a re-export hub
here ties every importer of any submodule to all of them, which over-selects CI tests.
"""
