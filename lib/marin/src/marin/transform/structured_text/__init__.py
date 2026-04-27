# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serializers and staging helpers for structured-data PPL eval slices.

Structured-data slices (CSV/TSV/JSON tables, time-series forecasting records,
geospatial records) need a different ingestion path from natural-language
corpora: the text the tokenizer ultimately sees must byte-preserve delimiters,
numeric literals, headers, and missing-value markers. Parsing a numeric field
and re-emitting it via ``repr(float(...))`` silently changes the digits the
tokenizer sees, which defeats the purpose of these slices.

See ``tabular`` for raw CSV/TSV staging, ``table_records`` for HF
parquet-based table datasets (ToTTo, WikiTableQuestions, GitTables), and
``web_data_commons`` for the WebTables sample archives.
"""
