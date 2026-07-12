# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample doubles as the finelog table schema, so it must satisfy finelog's
register_table contract: a declared key_column that names an existing column
(or an implicit `timestamp_ms` column). Without it the metrics table never
registers and FinelogTableSink drops every row."""

from finelog.client.log_client import schema_from_dataclass
from sample import Sample


def test_sample_schema_has_resolvable_finelog_key():
    schema = schema_from_dataclass(Sample)
    assert schema.key_column == "collected_at"
    assert any(column.name == schema.key_column for column in schema.columns)
