# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Text/LM dataset configs, formats, and tokenization caches.

Import from the defining submodule (``levanter.data.text.datasets``, ``levanter.data.text.formats``,
``levanter.data.text.preference``, ``levanter.data.text.examples``, ...) rather than the package.
``LmDatasetFormatBase`` is a ``draccus.PluginRegistry`` that discovers its format subclasses lazily
under ``levanter.data.text``, so this module imports nothing and a change to one submodule no longer
selects every test that touches ``levanter.data.text``.
"""
