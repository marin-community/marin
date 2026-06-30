# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from ducky.config import DuckyConfig
from ducky.runner import BucketNotAllowedError, QueryError, QueryRunner, disallowed_uris, duckdb_resource_settings
from iris.env_resources import TaskResources

_SMALL_HOST = TaskResources(memory_bytes=2 * 1024**3, cpu_cores=2, gpu_count=0, tpu_count=0)


def _make_config(scratch_bucket: str, **overrides) -> DuckyConfig:
    base = dict(
        region="us-east5",
        scratch_bucket=scratch_bucket,
        gcs_hmac_key_id="key",
        gcs_hmac_secret="secret",
        r2_endpoint="acct.r2.cloudflarestorage.com",
        r2_access_key="r2key",
        r2_secret_key="r2secret",
        cw_endpoint="cwobject.com",
        cw_access_key="cwkey",
        cw_secret_key="cwsecret",
    )
    base.update(overrides)
    return DuckyConfig(**base)


@pytest.fixture
def make_runner(tmp_path: Path) -> Callable[..., QueryRunner]:
    """Build runners that spill to a local scratch dir and close them on teardown."""
    (tmp_path / "ducky").mkdir()
    runners: list[QueryRunner] = []

    def factory(**config_overrides) -> QueryRunner:
        config = _make_config(str(tmp_path), **config_overrides)
        runner = QueryRunner(config, resources=_SMALL_HOST)
        runners.append(runner)
        return runner

    yield factory
    for runner in runners:
        runner.close()


def test_duckdb_resource_settings_scales_memory():
    settings = duckdb_resource_settings(_SMALL_HOST, memory_fraction=0.8)
    assert settings.threads == 2
    assert settings.memory_limit_bytes == int(2 * 1024**3 * 0.8)


def test_duckdb_resource_settings_unknown_memory_leaves_default():
    host = TaskResources(memory_bytes=0, cpu_cores=4, gpu_count=0, tpu_count=0)
    settings = duckdb_resource_settings(host, memory_fraction=0.8)
    assert settings.threads == 4
    assert settings.memory_limit_bytes == 0


def test_run_query_caps_preview_and_spills(make_runner):
    runner = make_runner(preview_row_cap=3)
    result = runner.run_query("SELECT * FROM range(5) t(x)", uuid.uuid4().hex)

    assert result.columns == ["x"]
    assert result.preview_rows == [[0], [1], [2]]
    assert result.total_rows == 5
    assert result.truncated is True
    assert result.result_path.endswith(".parquet")
    assert Path(result.result_path).exists()
    assert result.elapsed_ms >= 0
    assert result.result_bytes > 0  # the spilled parquet has content


def test_run_query_full_result_not_truncated(make_runner):
    runner = make_runner()
    result = runner.run_query("SELECT * FROM range(5) t(x)", uuid.uuid4().hex)

    assert result.total_rows == 5
    assert len(result.preview_rows) == 5
    assert result.truncated is False


def test_disallowed_uris_flags_only_unlisted_buckets():
    allowed = ("gs://marin-us-east5", "r2://")
    sql = (
        "SELECT * FROM read_parquet('gs://marin-us-central2/a.parquet') "
        "JOIN read_parquet('gs://marin-us-east5/b.parquet') USING (id) "
        "JOIN read_parquet('r2://anybucket/c.parquet') USING (id)"
    )
    assert disallowed_uris(sql, allowed) == ["gs://marin-us-central2/a.parquet"]


def test_disallowed_uris_prefix_semantics():
    # entries are prefixes: 'gs://marin-' allows every marin-* bucket
    allowed = ("gs://marin-", "s3://marin-na")
    assert disallowed_uris("read('gs://marin-us-east5/x') read('gs://marin-us-central2/y')", allowed) == []
    assert disallowed_uris("read('s3://marin-na/z')", allowed) == []
    assert disallowed_uris("read('gs://other-bucket/x')", allowed) == ["gs://other-bucket/x"]


def test_disallowed_uris_trailing_slash_bounds_to_bucket():
    # a bare prefix is loose; a trailing slash bounds the match to one bucket
    assert disallowed_uris("read('gs://marin-us-east5-evil/x')", ("gs://marin-us-east5",)) == []
    assert disallowed_uris("read('gs://marin-us-east5-evil/x')", ("gs://marin-us-east5/",))


def test_disallowed_uris_empty_allowlist_allows_all():
    assert disallowed_uris("read_parquet('gs://anywhere/x.parquet')", ()) == []


def test_run_query_refuses_bucket_outside_allowlist(make_runner):
    runner = make_runner(allowed_buckets=("gs://marin-us-east5",))
    with pytest.raises(BucketNotAllowedError, match="us-central2"):
        runner.run_query("SELECT * FROM read_parquet('gs://marin-us-central2/x.parquet')", uuid.uuid4().hex)


def test_run_query_allowlist_does_not_block_non_object_queries(make_runner):
    runner = make_runner(allowed_buckets=("gs://marin-us-east5",))
    result = runner.run_query("SELECT * FROM range(3) t(x)", uuid.uuid4().hex)
    assert result.total_rows == 3


def test_run_query_bad_sql_raises_query_error(make_runner):
    runner = make_runner()
    with pytest.raises(QueryError):
        runner.run_query("SELECT * FROM no_such_table", uuid.uuid4().hex)


def test_run_query_rejects_non_uuid_query_id(make_runner):
    runner = make_runner()
    with pytest.raises(ValueError):
        runner.run_query("SELECT 1", "../etc/passwd")


def test_run_query_coerces_non_scalar_cells_to_str(make_runner):
    runner = make_runner()
    result = runner.run_query("SELECT TIMESTAMP '2020-01-01 00:00:00' AS t", uuid.uuid4().hex)

    (cell,) = result.preview_rows[0]
    assert isinstance(cell, str)
    assert "2020-01-01" in cell


def test_run_query_strips_trailing_semicolon(make_runner):
    runner = make_runner()
    result = runner.run_query("SELECT 1 AS a;", uuid.uuid4().hex)
    assert result.columns == ["a"]
    assert result.preview_rows == [[1]]


def test_run_query_handles_trailing_line_comment(make_runner):
    runner = make_runner()
    result = runner.run_query("SELECT 1 AS a -- trailing comment", uuid.uuid4().hex)
    assert result.preview_rows == [[1]]


def test_run_query_coerces_non_finite_floats_to_str(make_runner):
    runner = make_runner()
    result = runner.run_query("SELECT 'nan'::DOUBLE AS x, 'inf'::DOUBLE AS y", uuid.uuid4().hex)
    nan_cell, inf_cell = result.preview_rows[0]
    assert isinstance(nan_cell, str) and isinstance(inf_cell, str)


def test_run_query_ignores_hive_partition_in_scratch_path(tmp_path):
    """A `ttl=Nd` segment in the scratch path must not leak a phantom partition column."""
    scratch = tmp_path / "tmp" / "ttl=7d"
    (scratch / "ducky").mkdir(parents=True)
    runner = QueryRunner(_make_config(str(scratch)), resources=_SMALL_HOST)
    try:
        result = runner.run_query("SELECT 1 AS a, 2 AS b", uuid.uuid4().hex)
    finally:
        runner.close()

    assert result.columns == ["a", "b"]  # no "ttl"
    assert result.preview_rows == [[1, 2]]
