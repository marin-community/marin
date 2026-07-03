# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pytest
from ducky.config import DuckyConfig
from ducky.runner import BucketNotAllowedError, QueryError, QueryRunner, disallowed_uris
from iris.env_resources import TaskResources

_SMALL_HOST = TaskResources(memory_bytes=2 * 1024**3, cpu_cores=2, gpu_count=0, tpu_count=0)


def _make_config(scratch_bucket: str, **overrides) -> DuckyConfig:
    base = dict(
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


def test_remote_scratch_blocks_local_filesystem_access(tmp_path):
    # results to object storage → user SQL must not read local files (e.g. /proc/self/environ)
    config = _make_config("gs://marin-us-east5/tmp/ttl=7d", spill_directory=str(tmp_path / "spill"))
    runner = QueryRunner(config, resources=_SMALL_HOST)
    try:
        with pytest.raises(duckdb.Error):
            runner._con.execute("SELECT * FROM read_text('/proc/self/environ')").fetchall()
    finally:
        runner.close()


@pytest.mark.parametrize(
    "sql, allowed, expected",
    [
        # only the unlisted bucket is flagged; listed gs:// prefix and all-of-r2 pass
        (
            "read('gs://marin-us-central2/a') read('gs://marin-us-east5/b') read('r2://any/c')",
            ("gs://marin-us-east5", "r2://"),
            ["gs://marin-us-central2/a"],
        ),
        # entries are prefixes: 'gs://marin-' allows every marin-* bucket
        ("read('gs://marin-us-east5/x') read('gs://marin-us-central2/y')", ("gs://marin-",), []),
        ("read('gs://other-bucket/x')", ("gs://marin-",), ["gs://other-bucket/x"]),
        # a bare prefix is loose; a trailing slash bounds the match to one bucket
        ("read('gs://marin-us-east5-evil/x')", ("gs://marin-us-east5",), []),
        ("read('gs://marin-us-east5-evil/x')", ("gs://marin-us-east5/",), ["gs://marin-us-east5-evil/x"]),
        # empty allowlist disables enforcement (allow all)
        ("read('gs://anywhere/x')", (), []),
    ],
)
def test_disallowed_uris(sql, allowed, expected):
    assert disallowed_uris(sql, allowed) == expected


def test_run_query_refuses_bucket_outside_allowlist(make_runner):
    runner = make_runner(allowed_buckets=("gs://marin-us-east5",))
    with pytest.raises(BucketNotAllowedError, match="us-central2"):
        runner.run_query("SELECT * FROM read_parquet('gs://marin-us-central2/x.parquet')", uuid.uuid4().hex)


def test_run_query_allowlist_does_not_block_non_object_queries(make_runner):
    runner = make_runner(allowed_buckets=("gs://marin-us-east5",))
    result = runner.run_query("SELECT * FROM range(3) t(x)", uuid.uuid4().hex)
    assert result.total_rows == 3


def test_run_query_spills_under_memory_pressure_and_survives(tmp_path):
    # tiny memory limit forces a big sort out-of-core; the query should still succeed
    # via the spill directory, and the runner must survive for the next query.
    (tmp_path / "ducky").mkdir()
    config = _make_config(str(tmp_path), spill_directory=str(tmp_path / "spill"))
    tiny = TaskResources(memory_bytes=200 * 1024 * 1024, cpu_cores=2, gpu_count=0, tpu_count=0)
    runner = QueryRunner(config, resources=tiny)
    try:
        spilled = runner.run_query("SELECT x FROM range(20000000) t(x) ORDER BY x DESC LIMIT 1", uuid.uuid4().hex)
        assert spilled.preview_rows == [[19999999]]
        assert runner.run_query("SELECT 1 AS a", uuid.uuid4().hex).preview_rows == [[1]]  # survives
    finally:
        runner.close()


def test_startup_wipes_orphaned_spill_files(tmp_path):
    # a killed process orphans temp files; a fresh runner must clear the spill dir on startup
    (tmp_path / "ducky").mkdir()
    spill = tmp_path / "spill"
    spill.mkdir()
    (spill / "orphan.tmp").write_text("leftover from a crashed process")
    runner = QueryRunner(_make_config(str(tmp_path), spill_directory=str(spill)), resources=_SMALL_HOST)
    try:
        assert not (spill / "orphan.tmp").exists()
    finally:
        runner.close()


def test_httpfs_retries_are_configured(make_runner):
    # transient object-store failures (incl. brief DNS/connection blips) should be retried.
    # queries run on a cursor, so the setting must be GLOBAL to be inherited — check via one.
    runner = make_runner()
    cursor = runner._con.cursor()
    assert cursor.execute("SELECT current_setting('http_retries')").fetchone()[0] == 10


def test_parquet_footer_cache_is_enabled(make_runner):
    # repeat queries over the same object-store files should reuse cached parquet footers;
    # GLOBAL so the per-query cursor inherits it.
    cursor = make_runner()._con.cursor()
    assert cursor.execute("SELECT current_setting('parquet_metadata_cache')").fetchone()[0] is True


def _write_parquet(con, path: Path, sql: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{path}'")


def test_catalog_views_are_queryable(make_runner, tmp_path):
    # a finelog-like local layout: <root>/<namespace>/seg_L*.parquet
    finelog_root = tmp_path / "finelog"
    con = duckdb.connect()
    _write_parquet(con, finelog_root / "log" / "seg_L0_0.parquet", "SELECT 1 AS level, 'hi' AS data")
    _write_parquet(con, finelog_root / "iris.task" / "seg_L0_0.parquet", "SELECT 'task-1' AS task_id")
    con.close()

    runner = make_runner(finelog_root=str(finelog_root))
    assert runner.run_query('SELECT data FROM finelog."log"', uuid.uuid4().hex).preview_rows == [["hi"]]
    # the dotted namespace resolves as a quoted view name
    assert runner.run_query('SELECT task_id FROM finelog."iris.task"', uuid.uuid4().hex).preview_rows == [["task-1"]]
    # created views are tracked (so the server advertises only what exists)
    assert {'finelog."log"', 'finelog."iris.task"'} <= runner.created_view_names


def test_configured_root_readable_despite_restrictive_allowlist(make_runner, tmp_path):
    # configuring finelog_root declares it readable, so its views are created and queryable
    # even when allowed_buckets doesn't cover it (the root joins effective_allowed_buckets).
    finelog_root = tmp_path / "finelog"
    con = duckdb.connect()
    _write_parquet(con, finelog_root / "log" / "seg_L0_0.parquet", "SELECT 1 AS level")
    con.close()

    runner = make_runner(finelog_root=str(finelog_root), allowed_buckets=("gs://marin-us-east5",))
    assert 'finelog."log"' in runner.created_view_names
    assert runner.run_query('SELECT * FROM finelog."log"', uuid.uuid4().hex).total_rows == 1


def test_configured_root_extends_literal_read_allowlist():
    # a literal read_parquet of the configured root prefix is allowed — consistent with the
    # view — while a different bucket in the same region stays blocked.
    config = DuckyConfig(
        scratch_bucket="/tmp/ducky",
        allowed_buckets=("gs://marin-us-east5",),
        finelog_root="gs://marin-us-central2/finelog/marin",
    )
    eff = config.effective_allowed_buckets
    assert disallowed_uris("read('gs://marin-us-central2/finelog/marin/log/s.parquet')", eff) == []
    assert disallowed_uris("read('gs://marin-us-central2/other/x.parquet')", eff) == [
        "gs://marin-us-central2/other/x.parquet"
    ]


def test_datakit_view_globs_hashed_dir(make_runner, tmp_path):
    datakit_root = tmp_path / "normalized"
    con = duckdb.connect()
    # <root>/<name>_<hash8>/outputs/main/part-*.parquet — the view must glob past the hash
    part = datakit_root / "finetranslations_abcd1234" / "outputs" / "main" / "part-00000-of-00001.parquet"
    _write_parquet(con, part, "SELECT 'doc-1' AS id, 'hello' AS text")
    con.close()

    runner = make_runner(datakit_root=str(datakit_root))
    assert runner.run_query('SELECT id FROM datakit."finetranslations"', uuid.uuid4().hex).preview_rows == [["doc-1"]]


def test_missing_catalog_dataset_is_skipped_not_fatal(make_runner, tmp_path):
    # a root with no matching parquet: view creation fails per-view, but the runner still
    # comes up and serves ordinary queries.
    runner = make_runner(finelog_root=str(tmp_path / "empty-finelog"))
    assert runner.run_query("SELECT 1 AS a", uuid.uuid4().hex).preview_rows == [[1]]
    with pytest.raises(QueryError):  # the un-created view is simply absent
        runner.run_query('SELECT * FROM finelog."log"', uuid.uuid4().hex)


def test_run_query_is_concurrency_safe(make_runner):
    runner = make_runner()  # one runner shared across threads; each query uses its own cursor

    def run(i: int) -> int:
        return runner.run_query(f"SELECT {i} AS v", uuid.uuid4().hex).preview_rows[0][0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = sorted(pool.map(run, range(8)))
    assert values == list(range(8))


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


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 AS a;",  # trailing semicolon
        "SELECT 1 AS a -- trailing comment",  # trailing line comment
        "SELECT 1 AS a; -- trailing comment",  # semicolon then comment
    ],
)
def test_run_query_accepts_trailing_semicolons_and_comments(make_runner, sql):
    result = make_runner().run_query(sql, uuid.uuid4().hex)
    assert result.columns == ["a"]
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
