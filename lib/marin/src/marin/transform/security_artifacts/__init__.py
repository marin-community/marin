# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Renderers and transforms for binary, network, and security artifact PPL slices.

The renderers in this package turn raw binary/tabular security data into the
textual surface form that a language model sees (hex dumps, Zeek TSV, etc.).
Transforms like :mod:`marin.transform.security_artifacts.zeek_to_dolma` wrap
those renderers and emit Dolma-format JSONL for use with
:func:`marin.evaluation.perplexity_gap.raw_text_dataset`.
"""

from marin.transform.security_artifacts.renderers import (
    DEFAULT_ZEEK_EMPTY_FIELD,
    DEFAULT_ZEEK_SET_SEPARATOR,
    DEFAULT_ZEEK_UNSET_FIELD,
    render_hex_dump,
    render_zeek_tsv_log,
    render_zeek_tsv_record,
    render_zeek_tsv_value,
)

__all__ = [
    "DEFAULT_ZEEK_EMPTY_FIELD",
    "DEFAULT_ZEEK_SET_SEPARATOR",
    "DEFAULT_ZEEK_UNSET_FIELD",
    "render_hex_dump",
    "render_zeek_tsv_log",
    "render_zeek_tsv_record",
    "render_zeek_tsv_value",
]
