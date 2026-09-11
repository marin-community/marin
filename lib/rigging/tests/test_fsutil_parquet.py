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
from rigging.fsutil.parquet import ParquetViewSource, parquet_lines

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


def test_view_source_pages_rows_across_row_groups(shard):
    """The viewer scans forward one batch at a time: a fetch never decodes past its
    batch, row groups chain seamlessly, and the file closes with an end marker."""
    with ParquetViewSource(str(shard), batch_rows=2) as source:
        assert "rows        6" in source.head_lines()

        first = source.more_lines()
        assert first[0].split() == ["step", "text"]
        assert [line.split() for line in first[2:]] == [["0", "row", "0"], ["1", "row", "1"]]

        remaining = [source.more_lines() for _ in range(4)]
        rendered = [line for batch in remaining[:3] for line in batch]
        assert [line.split()[0] for line in rendered] == ["2", "3", "4", "5"]
        assert remaining[3] == ["[end of 6 rows]"]
        assert source.more_lines() == []


def test_view_source_keeps_first_batch_column_widths(shard):
    """Later batches pad to the widths of the first, so the table stays aligned as the
    viewer appends lines without re-rendering what is already on screen."""
    with ParquetViewSource(str(shard), batch_rows=3) as source:
        first = source.more_lines()
        second = source.more_lines()
    assert second[0].index("row 3") == first[2].index("row 0")


def test_view_source_skips_oversized_row_groups(shard, monkeypatch):
    """A row group above the preview limit is reported and stepped over, and the end
    marker admits the rows that were never shown."""
    monkeypatch.setattr(parquet, "MAX_PREVIEW_BYTES", 8)

    with ParquetViewSource(str(shard)) as source:
        batches = [source.more_lines() for _ in range(3)]

    assert batches[0][0].startswith("[row group 0 not read: ")
    assert batches[1][0].startswith("[row group 1 not read: ")
    assert batches[2] == ["[end: showed 0 of 6 rows]"]


def test_cat_and_head_route_parquet_to_the_footer_reader(shard):
    """The commands dispatch on the name before they read bytes, and ``head -n`` bounds
    rows rather than lines, which the schema block would otherwise consume."""
    assert "row 0" in CliRunner().invoke(cli, ["cat", str(shard)]).output

    head = CliRunner().invoke(cli, ["head", "-n", "2", str(shard)])
    assert "row 0" in head.output
    assert head.output.splitlines()[-1] == "[showing 2 of 6 rows]"
