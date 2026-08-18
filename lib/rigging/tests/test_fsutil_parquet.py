# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fsutil's parquet previews, over local paths — which ``filesystem_for``
routes exactly as it routes an object store.

pyarrow is not a marin-rigging dependency, so these tests skip in an environment that
holds no parquet reader and run everywhere a parquet file could be browsed."""

import pytest
from click.testing import CliRunner
from rigging.fsutil import parquet
from rigging.fsutil.cli import cli
from rigging.fsutil.parquet import parquet_lines

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@pytest.fixture
def shard(tmp_path):
    """A two-row-group parquet file, so a preview reads the first group and no more."""
    path = tmp_path / "docs.parquet"
    table = pa.table({"step": list(range(6)), "text": [f"row {i}" for i in range(6)]})
    pq.write_table(table, path, row_group_size=3)
    return path


def test_parquet_preview_reports_schema_statistics_and_rows(shard):
    """Parquet's schema lives in a footer, so a preview must seek rather than read a head."""
    lines = parquet_lines(str(shard), rows=2)

    assert [line.split() for line in lines[1:4]] == [["column", "type"], ["------", "------"], ["step", "int64"]]
    assert "rows        6" in lines
    header = lines.index("step  text")
    assert lines[header + 2].split() == ["0", "row", "0"]
    assert lines[-1] == "[showing 2 of 6 rows]"


def test_parquet_preview_skips_rows_when_the_first_row_group_is_too_large(shard, monkeypatch):
    """One row is only readable by decoding the whole row group that holds it, so an
    oversized group is reported instead of pulled down."""
    monkeypatch.setattr(parquet, "MAX_PREVIEW_BYTES", 8)

    lines = parquet_lines(str(shard))

    assert lines[-1].startswith("[rows not read: row group 0 holds ")
    assert not any("row 0" in line for line in lines)


def test_cat_and_head_route_parquet_to_the_footer_reader(shard):
    """The commands dispatch on the name before they read bytes, and ``head -n`` bounds
    rows rather than lines, which the schema block would otherwise consume."""
    assert "row 0" in CliRunner().invoke(cli, ["cat", str(shard)]).output

    head = CliRunner().invoke(cli, ["head", "-n", "2", str(shard)])
    assert "row 0" in head.output
    assert head.output.splitlines()[-1] == "[showing 2 of 6 rows]"
