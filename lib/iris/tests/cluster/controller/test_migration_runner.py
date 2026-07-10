# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``ControllerDB.apply_migrations`` runner.

A fresh DB is materialized from ``schema.py``'s metadata and is therefore already
at the current schema; its deltas are recorded as applied without running. A DB
that predates the baseline still runs every delta it has not recorded. These
tests pin both halves and pin that the two paths converge on the same schema.

Each test boots a real ``ControllerDB``, which migrates on construction. Canary
modules injected into the runner's module list create a table when they execute, so
whether a given delta ran is read back as persisted state.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from iris.cluster.controller import db as controller_db
from iris.cluster.controller.db import ControllerDB

MIGRATIONS_DIR = Path(controller_db.__file__).with_name("migrations")

# Every delta module on disk, enumerated independently of the runner's own glob.
REAL_DELTA_PATHS = sorted(path for path in MIGRATIONS_DIR.glob("*.py") if not path.name.startswith("__"))
DELTA_NAMES = {path.name for path in REAL_DELTA_PATHS}

CANARY_TEMPLATE = '''\
"""Records its own execution by creating a table no baseline declares."""


def migrate(raw_conn) -> None:
    raw_conn.execute("CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY)")
'''


def _ran_table(canary: Path) -> str:
    """The table a canary creates when it runs. SQLite identifiers cannot lead with a digit."""
    return f"ran_{canary.stem}"


def _write_canary(directory: Path, stem: str) -> Path:
    path = directory / f"{stem}.py"
    path.write_text(CANARY_TEMPLATE.format(table=_ran_table(path)))
    return path


@pytest.fixture
def canaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Append two canary deltas, which no baseline declares, after the real ones."""
    paths = [_write_canary(tmp_path, "9998_canary_a"), _write_canary(tmp_path, "9999_canary_b")]
    monkeypatch.setattr(ControllerDB, "_delta_migration_paths", staticmethod(lambda: [*REAL_DELTA_PATHS, *paths]))
    return paths


def _migrate(db_dir: Path) -> None:
    """Migrate ``db_dir`` the way a controller boot does — construction applies migrations."""
    ControllerDB(db_dir=db_dir).close()


def _connect(db_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_dir / ControllerDB.DB_FILENAME)
    conn.execute("ATTACH DATABASE ? AS auth", (str(db_dir / ControllerDB.AUTH_DB_FILENAME),))
    return conn


def _recorded_migrations(db_dir: Path) -> set[str]:
    with closing(_connect(db_dir)) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}


def _forget_migrations(db_dir: Path, names: set[str]) -> None:
    with closing(_connect(db_dir)) as conn:
        conn.executemany("DELETE FROM schema_migrations WHERE name = ?", [(name,) for name in names])
        conn.commit()


def _table_exists(db_dir: Path, table: str) -> bool:
    with closing(_connect(db_dir)) as conn:
        return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _column_names(db_dir: Path, table: str) -> set[str]:
    with closing(_connect(db_dir)) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _objects(conn: sqlite3.Connection, schema: str) -> list[tuple[str, str]]:
    return conn.execute(
        f"SELECT type, name FROM {schema}.sqlite_master "
        "WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()


def _schema(db_dir: Path) -> dict[str, object]:
    """Semantic shape of every table and index across both DB files.

    Tables map each column name to its type, nullability, default, and primary-key
    flag. Indexes map to their ``CREATE`` text, which carries the partial-index
    ``WHERE`` clause that no pragma exposes.
    """
    shape: dict[str, object] = {}
    with closing(_connect(db_dir)) as conn:
        for schema in ("main", "auth"):
            for kind, name in _objects(conn, schema):
                key = f"{schema}.{name}"
                if kind == "index":
                    shape[key] = conn.execute(
                        f"SELECT sql FROM {schema}.sqlite_master WHERE name = ?", (name,)
                    ).fetchone()[0]
                else:
                    # Column facts, not the table's CREATE text: ALTER TABLE ADD COLUMN
                    # appends its clause verbatim, so an upgraded DB spells a column
                    # "NOT NULL DEFAULT ''" where the baseline spells it "DEFAULT '' NOT NULL".
                    # table_info columns: (cid, name, type, notnull, dflt_value, pk)
                    shape[key] = {
                        row[1]: row[2:] for row in conn.execute(f"PRAGMA {schema}.table_info({name})").fetchall()
                    }
    return shape


def _column_order(db_dir: Path) -> dict[str, list[str]]:
    """Declared column order per table — stable only among DBs built from the baseline."""
    with closing(_connect(db_dir)) as conn:
        return {
            f"{schema}.{name}": [row[1] for row in conn.execute(f"PRAGMA {schema}.table_info({name})")]
            for schema in ("main", "auth")
            for kind, name in _objects(conn, schema)
            if kind == "table"
        }


def _make_legacy(db_dir: Path) -> None:
    """Regress a baseline DB into one that predates the baseline scheme.

    An empty migration ledger alongside surviving user tables is what
    ``apply_migrations`` reads as a pre-baseline DB. Dropping
    ``jobs.submitting_user`` gives one delta (``0041``) real forward work.
    """
    with closing(_connect(db_dir)) as conn:
        conn.execute("DELETE FROM schema_migrations")
        conn.execute("ALTER TABLE jobs DROP COLUMN submitting_user")
        conn.commit()


def test_fresh_db_marks_every_delta_applied_without_running_it(tmp_path: Path, canaries: list[Path]) -> None:
    db_dir = tmp_path / "fresh"
    _migrate(db_dir)

    expected = {ControllerDB.BASELINE_MIGRATION, *DELTA_NAMES, *(path.name for path in canaries)}
    assert _recorded_migrations(db_dir) == expected
    for canary in canaries:
        assert not _table_exists(db_dir, _ran_table(canary)), f"{canary.name} ran against a freshly created baseline"


def test_fresh_db_schema_matches_replaying_every_delta(tmp_path: Path) -> None:
    """The correctness claim: skipping the deltas loses no schema.

    Every delta carries an older DB toward a state the baseline declares outright,
    so replaying them over a fresh baseline yields the same tables and indexes. A
    delta that adds schema the baseline does not declare fails here.

    The comparison covers schema alone.
    """
    skipped = tmp_path / "skipped"
    _migrate(skipped)

    replayed = tmp_path / "replayed"
    _migrate(replayed)
    _forget_migrations(replayed, DELTA_NAMES)
    _migrate(replayed)

    assert _recorded_migrations(replayed) == _recorded_migrations(skipped)
    assert _schema(replayed) == _schema(skipped)
    # Both come from the baseline, so column order holds too.
    assert _column_order(replayed) == _column_order(skipped)


def test_legacy_db_runs_every_unrecorded_delta(tmp_path: Path) -> None:
    """A DB seeded before the baseline scheme existed still migrates forward.

    The comparison covers column facts, not column order: ``0041`` re-adds
    ``submitting_user`` with ``ALTER TABLE ... ADD COLUMN``, which appends it,
    while the baseline declares it third.
    """
    fresh = tmp_path / "fresh"
    _migrate(fresh)

    legacy = tmp_path / "legacy"
    _migrate(legacy)
    _make_legacy(legacy)
    assert "submitting_user" not in _column_names(legacy, "jobs")

    _migrate(legacy)

    assert "submitting_user" in _column_names(legacy, "jobs")
    assert _recorded_migrations(legacy) == {ControllerDB.BASELINE_MIGRATION, *DELTA_NAMES}
    assert _schema(legacy) == _schema(fresh)


def test_only_unrecorded_deltas_run(tmp_path: Path, canaries: list[Path]) -> None:
    """A DB behind by one delta runs that delta and no other."""
    first, last = canaries
    db_dir = tmp_path / "one_behind"
    _migrate(db_dir)

    _forget_migrations(db_dir, {last.name})
    _migrate(db_dir)

    assert _table_exists(db_dir, _ran_table(last))
    assert not _table_exists(db_dir, _ran_table(first)), "an already-recorded delta re-ran"


def test_reopening_a_migrated_db_changes_nothing(tmp_path: Path) -> None:
    """Every controller restart re-runs ``apply_migrations`` against its own DB."""
    db_dir = tmp_path / "twice"
    _migrate(db_dir)
    before = _schema(db_dir), _recorded_migrations(db_dir)

    _migrate(db_dir)

    assert (_schema(db_dir), _recorded_migrations(db_dir)) == before
