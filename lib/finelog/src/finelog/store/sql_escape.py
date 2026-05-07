# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQL escape helpers shared by query and compaction paths.

DuckDB's ``CREATE VIEW`` does not accept prepared parameters, so namespace
identifiers and segment paths have to be inlined as SQL literals. The
namespace-name regex (``^[a-z][a-z0-9_.-]{0,63}$``) blocks the dangerous
characters in practice; these helpers provide defense-in-depth.
"""

from __future__ import annotations


def quote_ident(name: str) -> str:
    """Quote ``name`` as a double-quoted DuckDB identifier.

    Doubles any embedded ``"`` so identifiers containing one are still safe.
    """
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    """Quote ``value`` as a single-quoted DuckDB string literal.

    Doubles any embedded ``'``.
    """
    return "'" + value.replace("'", "''") + "'"
