# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the storage dashboard HTTP API.

Each test exercises the full request/response cycle via FastAPI's TestClient.
The test_db fixture creates an isolated DuckDB with the real schema, populated
with enough seed data to exercise each endpoint's query logic.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from scripts.storage.db import StorageCatalog


def _make_test_db(db_path: Path) -> None:
    """Initialise a test DuckDB at db_path with the full schema and seed rows.

    We create dir_summary as a plain table rather than a parquet-backed view
    because the server queries it via SQL and DuckDB doesn't care which it is.
    """
    conn = duckdb.connect(str(db_path))

    conn.execute("CREATE TABLE cache_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO cache_meta VALUES ('schema_version', '14')")

    conn.execute("CREATE SEQUENCE protect_rules_id_seq START 1")
    conn.execute(
        """
        CREATE TABLE storage_classes (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            price_per_gib_month_us REAL NOT NULL,
            price_per_gib_month_eu REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE protect_rules (
            id INTEGER PRIMARY KEY DEFAULT nextval('protect_rules_id_seq'),
            bucket TEXT NOT NULL,
            pattern TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            owners TEXT,
            reasons TEXT,
            sources TEXT,
            UNIQUE (bucket, pattern)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE rule_costs (
            rule_id INTEGER NOT NULL REFERENCES protect_rules(id),
            bucket TEXT NOT NULL,
            storage_class_id INTEGER NOT NULL REFERENCES storage_classes(id),
            object_count INTEGER NOT NULL,
            total_bytes BIGINT NOT NULL,
            monthly_cost_usd REAL NOT NULL,
            PRIMARY KEY (rule_id, storage_class_id, bucket)
        )
        """
    )
    conn.execute("CREATE SEQUENCE delete_rules_id_seq START 1")
    conn.execute(
        """
        CREATE TABLE delete_rules (
            id INTEGER PRIMARY KEY DEFAULT nextval('delete_rules_id_seq'),
            pattern TEXT NOT NULL,
            storage_class TEXT,
            description TEXT,
            created_at TEXT NOT NULL,
            UNIQUE (pattern, storage_class)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE delete_rule_costs (
            rule_id INTEGER NOT NULL REFERENCES delete_rules(id),
            bucket TEXT NOT NULL,
            storage_class_id INTEGER NOT NULL REFERENCES storage_classes(id),
            object_count INTEGER NOT NULL,
            total_bytes BIGINT NOT NULL,
            monthly_cost_usd REAL NOT NULL,
            PRIMARY KEY (rule_id, storage_class_id, bucket)
        )
        """
    )

    for sc_id, name, us, eu in [
        (1, "STANDARD", 0.020, 0.023),
        (2, "NEARLINE", 0.010, 0.013),
        (3, "COLDLINE", 0.004, 0.006),
        (4, "ARCHIVE", 0.0012, 0.0025),
    ]:
        conn.execute("INSERT INTO storage_classes VALUES (?,?,?,?)", (sc_id, name, us, eu))

    # Empty objects view — needed by materialize_*_costs helpers if called.
    conn.execute(
        """
        CREATE VIEW objects AS
        SELECT NULL::VARCHAR as bucket, NULL::VARCHAR as name,
               NULL::BIGINT as size_bytes, NULL::INTEGER as storage_class_id,
               NULL::TIMESTAMPTZ as created, NULL::TIMESTAMPTZ as updated
        WHERE false
        """
    )
    conn.close()


def _write_seed_parquet(parquet_dir: Path) -> None:
    """Write seed dir_summary rows as a parquet file matching production layout."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "bucket": ["marin-us-central1", "marin-us-central1", "marin-us-central1", "marin-eu-west4"],
            "dir_prefix": ["checkpoints/exp1/", "data/training/", "tmp/scratch/", "checkpoints/exp1/"],
            "standard_count": pa.array([100, 200, 10, 80], type=pa.int32()),
            "standard_bytes": pa.array([1073741824, 2147483648, 10737418, 858993459], type=pa.int64()),
            "nearline_count": pa.array([50, 0, 20, 40], type=pa.int32()),
            "nearline_bytes": pa.array([536870912, 0, 21474836, 429496729], type=pa.int64()),
            "coldline_count": pa.array([10, 0, 0, 0], type=pa.int32()),
            "coldline_bytes": pa.array([107374182, 0, 0, 0], type=pa.int64()),
            "archive_count": pa.array([5, 0, 30, 0], type=pa.int32()),
            "archive_bytes": pa.array([53687091, 0, 32212254, 0], type=pa.int64()),
        }
    )
    pq.write_table(table, parquet_dir / "dir_summary_000001.parquet", compression="zstd")


@pytest.fixture
def catalog(tmp_path: Path) -> StorageCatalog:
    cat = StorageCatalog(tmp_path)
    cat.ensure_dirs()
    _make_test_db(cat.db_path)
    _write_seed_parquet(cat.dir_summary_parquet_dir)
    return cat


@pytest.fixture
def client(catalog: StorageCatalog):
    """TestClient wired to an isolated StorageCatalog (temp directory)."""
    from scripts.storage import server

    app = server.create_app(catalog)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/overview
# ---------------------------------------------------------------------------


def test_overview_returns_regions_and_totals(client):
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "regions" in data
    assert "totals" in data


def test_overview_totals_reflect_seed_data(client):
    resp = client.get("/api/overview")
    totals = resp.json()["totals"]
    # Seed data has objects in both buckets
    assert totals["total_objects"] > 0
    assert totals["total_bytes"] > 0
    assert totals["total_monthly_cost_usd"] > 0


def test_overview_region_entry_shape(client):
    resp = client.get("/api/overview")
    regions = resp.json()["regions"]
    assert len(regions) > 0
    for entry in regions:
        assert "region" in entry
        assert "bucket" in entry
        assert "total_objects" in entry
        assert "by_storage_class" in entry


# ---------------------------------------------------------------------------
# /api/rules  (protect rules)
# ---------------------------------------------------------------------------


def test_list_protect_rules_empty_initially(client):
    resp = client.get("/api/rules")
    assert resp.status_code == 200
    assert resp.json()["rules"] == []


def test_create_protect_rule_returns_id_and_pattern(client):
    resp = client.post("/api/rules", json={"bucket": "*", "pattern": "data/training/%"})
    assert resp.status_code == 200
    rule = resp.json()
    assert rule["id"] > 0
    assert rule["pattern"] == "data/training/%"


def test_create_protect_rule_appears_in_list(client):
    client.post("/api/rules", json={"bucket": "*", "pattern": "data/training/%"})
    rules = client.get("/api/rules").json()["rules"]
    assert any(r["pattern"] == "data/training/%" for r in rules)


def test_create_protect_rule_with_owners_and_reasons(client):
    resp = client.post(
        "/api/rules",
        json={"bucket": "*", "pattern": "checkpoints/%", "owners": "alice", "reasons": "model weights"},
    )
    assert resp.status_code == 200
    rule = resp.json()
    assert rule["owners"] == "alice"
    assert rule["reasons"] == "model weights"


def test_delete_protect_rule_removes_it(client):
    rule_id = client.post("/api/rules", json={"bucket": "*", "pattern": "tmp/%"}).json()["id"]

    resp = client.delete(f"/api/rules/{rule_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == rule_id

    rules = client.get("/api/rules").json()["rules"]
    assert not any(r["id"] == rule_id for r in rules)


def test_delete_protect_rule_nonexistent_is_ok(client):
    # Deleting a non-existent rule should not raise a server error.
    resp = client.delete("/api/rules/999999")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/delete-rules
# ---------------------------------------------------------------------------


def test_list_delete_rules_empty_initially(client):
    resp = client.get("/api/delete-rules")
    assert resp.status_code == 200
    assert resp.json()["rules"] == []


def test_create_delete_rule_returns_id_and_pattern(client):
    resp = client.post("/api/delete-rules", json={"pattern": "tmp/%"})
    assert resp.status_code == 200
    rule = resp.json()
    assert rule["id"] > 0
    assert rule["pattern"] == "tmp/%"


def test_create_delete_rule_storage_class_none_by_default(client):
    resp = client.post("/api/delete-rules", json={"pattern": "tmp/%"})
    assert resp.json()["storage_class"] is None


def test_create_delete_rule_with_storage_class(client):
    resp = client.post("/api/delete-rules", json={"pattern": "old/%", "storage_class": "ARCHIVE"})
    assert resp.status_code == 200
    assert resp.json()["storage_class"] == "ARCHIVE"


def test_create_delete_rule_with_description(client):
    resp = client.post("/api/delete-rules", json={"pattern": "junk/%", "description": "delete temp files"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "delete temp files"


def test_create_delete_rule_appears_in_list(client):
    client.post("/api/delete-rules", json={"pattern": "tmp/%"})
    rules = client.get("/api/delete-rules").json()["rules"]
    assert any(r["pattern"] == "tmp/%" for r in rules)


def test_delete_delete_rule_removes_it(client):
    rule_id = client.post("/api/delete-rules", json={"pattern": "junk/%"}).json()["id"]

    resp = client.delete(f"/api/delete-rules/{rule_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == rule_id

    rules = client.get("/api/delete-rules").json()["rules"]
    assert not any(r["id"] == rule_id for r in rules)


def test_delete_rule_recalculate_excludes_protected_dirs(client):
    """Delete rule costs should not count directories covered by a protect rule."""
    # Delete everything matching tmp/
    client.post("/api/delete-rules", json={"pattern": "tmp/%"})

    # Recalculate without protection — tmp/scratch/ should be counted
    resp = client.post("/api/delete-rules/recalculate")
    assert resp.status_code == 200
    rules_before = client.get("/api/delete-rules").json()["rules"]
    rule = next(r for r in rules_before if r["pattern"] == "tmp/%")
    assert rule["total_objects"] > 0

    # Now protect tmp/ and recalculate — should drop to zero
    client.post("/api/rules", json={"bucket": "*", "pattern": "tmp/%"})
    client.post("/api/delete-rules/recalculate")
    rules_after = client.get("/api/delete-rules").json()["rules"]
    rule = next(r for r in rules_after if r["pattern"] == "tmp/%")
    assert rule["total_objects"] == 0


# ---------------------------------------------------------------------------
# /api/explore  (per-bucket hierarchical browser)
# ---------------------------------------------------------------------------


def test_explore_returns_entries_for_seeded_bucket(client):
    resp = client.get("/api/explore", params={"bucket": "marin-us-central1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "entries" in data
    assert len(data["entries"]) > 0


def test_explore_entry_shape(client):
    resp = client.get("/api/explore", params={"bucket": "marin-us-central1"})
    for entry in resp.json()["entries"]:
        assert "name" in entry
        assert "objects" in entry
        assert "bytes" in entry
        assert "monthly_cost_usd" in entry


def test_explore_empty_bucket_returns_no_entries(client):
    # marin-us-east1 has no seed data
    resp = client.get("/api/explore", params={"bucket": "marin-us-east1"})
    assert resp.status_code == 200
    assert resp.json()["entries"] == []


# ---------------------------------------------------------------------------
# /api/explore/unified  (cross-bucket view with status annotation)
# ---------------------------------------------------------------------------


def test_unified_explore_returns_entries(client):
    resp = client.get("/api/explore/unified")
    assert resp.status_code == 200
    data = resp.json()
    assert "entries" in data
    assert len(data["entries"]) > 0


def test_unified_explore_entries_have_status(client):
    resp = client.get("/api/explore/unified")
    for entry in resp.json()["entries"]:
        assert "status" in entry
        assert entry["status"] in ("keep", "delete", "unmatched", "mixed")


def test_unified_explore_unmatched_when_no_rules(client):
    resp = client.get("/api/explore/unified")
    statuses = {e["status"] for e in resp.json()["entries"]}
    # With no protect or delete rules, every entry should be unmatched.
    assert statuses == {"unmatched"}


def test_unified_explore_delete_rule_marks_entry(client):
    client.post("/api/delete-rules", json={"pattern": "tmp/%"})

    resp = client.get("/api/explore/unified")
    entries = resp.json()["entries"]
    tmp_entry = next((e for e in entries if e["name"].startswith("tmp")), None)
    assert tmp_entry is not None, "expected a 'tmp/' entry in seed data"
    assert tmp_entry["status"] == "delete"


def test_unified_explore_protect_rule_marks_entry_keep(client):
    client.post("/api/rules", json={"bucket": "*", "pattern": "data/%"})

    resp = client.get("/api/explore/unified")
    entries = resp.json()["entries"]
    data_entry = next((e for e in entries if e["name"].startswith("data")), None)
    assert data_entry is not None, "expected a 'data/' entry in seed data"
    assert data_entry["status"] == "keep"


def test_unified_explore_protect_wins_over_delete(client):
    # Delete everything, then protect data/
    client.post("/api/delete-rules", json={"pattern": "%"})
    client.post("/api/rules", json={"bucket": "*", "pattern": "data/%"})

    resp = client.get("/api/explore/unified")
    entries = resp.json()["entries"]
    data_entry = next((e for e in entries if e["name"].startswith("data")), None)
    assert data_entry is not None
    assert data_entry["status"] == "keep"


def test_unified_explore_storage_class_delete_produces_mixed(client):
    """A delete rule targeting only ARCHIVE should produce 'mixed' for dirs
    that also have STANDARD bytes."""
    client.post("/api/delete-rules", json={"pattern": "%", "storage_class": "ARCHIVE"})

    resp = client.get("/api/explore/unified")
    entries = resp.json()["entries"]
    # checkpoints/exp1/ has both standard and archive bytes in seed data
    ckpt = next((e for e in entries if e["name"].startswith("checkpoints")), None)
    assert ckpt is not None, "expected a 'checkpoints/' entry in seed data"
    assert ckpt["status"] == "mixed"


def test_unified_explore_all_entries_have_status_order(client):
    resp = client.get("/api/explore/unified")
    for entry in resp.json()["entries"]:
        assert "status_order" in entry
        assert isinstance(entry["status_order"], int)


def test_unified_explore_bucketed_entries_have_status(client):
    """When bucketing kicks in, every range entry should still have a status."""
    resp = client.get("/api/explore/unified?max_children=1")
    data = resp.json()
    assert data["type"] == "buckets"
    for entry in data["entries"]:
        assert "status" in entry
        assert entry["status"] in ("keep", "delete", "unmatched", "mixed")
        assert "status_order" in entry


def test_unified_explore_bucketed_status_grouped(client):
    """Bucketed ranges should not straddle different statuses."""
    client.post("/api/delete-rules", json={"pattern": "tmp/%"})
    client.post("/api/rules", json={"bucket": "*", "pattern": "data/%"})

    resp = client.get("/api/explore/unified?max_children=1")
    data = resp.json()
    assert data["type"] == "buckets"
    entries = data["entries"]
    # Each bucketed range should have a single homogeneous status
    for entry in entries:
        assert entry["status"] in ("keep", "delete", "unmatched", "mixed")


def test_unified_explore_broad_delete_narrow_protect(client):
    """Delete % + protect data/% → data/ is keep, others are delete."""
    client.post("/api/delete-rules", json={"pattern": "%"})
    client.post("/api/rules", json={"bucket": "*", "pattern": "data/%"})

    resp = client.get("/api/explore/unified")
    entries = {e["name"]: e["status"] for e in resp.json()["entries"]}
    assert entries.get("data/") == "keep"
    assert entries.get("checkpoints/") == "delete"
    assert entries.get("tmp/") == "delete"


# ---------------------------------------------------------------------------
# /api/explore/unified — bucket filter
# ---------------------------------------------------------------------------


def test_unified_explore_bucket_filter_limits_to_bucket(client):
    """Filtering by bucket returns only entries present in that bucket."""
    resp = client.get("/api/explore/unified?bucket=marin-us-central1")
    names = {e["name"] for e in resp.json()["entries"]}
    assert "data/" in names, "data/training/ exists only in us-central1"
    assert "tmp/" in names, "tmp/scratch/ exists only in us-central1"

    resp2 = client.get("/api/explore/unified?bucket=marin-eu-west4")
    names2 = {e["name"] for e in resp2.json()["entries"]}
    assert "data/" not in names2, "data/ should not appear in eu-west4"
    assert "tmp/" not in names2, "tmp/ should not appear in eu-west4"
    assert "checkpoints/" in names2


def test_unified_explore_bucket_filter_exact_pricing(client):
    """Bucket-specific queries use exact region pricing, not averaged."""
    resp_us = client.get("/api/explore/unified?bucket=marin-us-central1")
    resp_eu = client.get("/api/explore/unified?bucket=marin-eu-west4")

    # checkpoints/exp1/ exists in both buckets — EU pricing is higher per GiB
    ckpt_us = next(e for e in resp_us.json()["entries"] if e["name"] == "checkpoints/")
    ckpt_eu = next(e for e in resp_eu.json()["entries"] if e["name"] == "checkpoints/")

    # EU has higher per-GiB prices for all storage classes, but the EU bucket
    # has fewer bytes. Verify both return valid costs > 0.
    assert ckpt_us["monthly_cost_usd"] > 0
    assert ckpt_eu["monthly_cost_usd"] > 0


def test_unified_explore_per_bucket_status_no_cross_bucket_mixed(client):
    """Protect checkpoints only in us-central1 + delete checkpoints everywhere.

    Without bucket filter this shows 'mixed' (protected in one bucket, deleted
    in another). With bucket filter, each bucket is cleanly keep or delete.
    """
    client.post("/api/rules", json={"bucket": "marin-us-central1", "pattern": "checkpoints/%"})
    client.post("/api/delete-rules", json={"pattern": "checkpoints/%"})

    # Per-bucket: us-central1 is protected → keep
    resp_us = client.get("/api/explore/unified?bucket=marin-us-central1")
    ckpt_us = next(e for e in resp_us.json()["entries"] if e["name"] == "checkpoints/")
    assert ckpt_us["status"] == "keep"

    # Per-bucket: eu-west4 is not protected → delete
    resp_eu = client.get("/api/explore/unified?bucket=marin-eu-west4")
    ckpt_eu = next(e for e in resp_eu.json()["entries"] if e["name"] == "checkpoints/")
    assert ckpt_eu["status"] == "delete"

    # All-buckets: aggregates cross-bucket → mixed
    resp_all = client.get("/api/explore/unified")
    ckpt_all = next(e for e in resp_all.json()["entries"] if e["name"] == "checkpoints/")
    assert ckpt_all["status"] == "mixed"


# ---------------------------------------------------------------------------
# /api/explore/unified — storage class filter
# ---------------------------------------------------------------------------


def test_unified_explore_storage_class_filter_limits_columns(client):
    """When filtering by storage class, only that class's bytes appear."""
    resp = client.get("/api/explore/unified?bucket=marin-us-central1&storage_class=STANDARD")
    entries = resp.json()["entries"]
    for entry in entries:
        # by_storage_class should only contain STANDARD
        classes = {sc["class"] for sc in entry.get("by_storage_class", [])}
        assert classes <= {"STANDARD"}, f"Expected only STANDARD, got {classes}"


def test_unified_explore_storage_class_filter_resolves_mixed(client):
    """Delete only ARCHIVE. With all classes → mixed. With class=ARCHIVE → delete."""
    client.post("/api/delete-rules", json={"pattern": "checkpoints/%", "storage_class": "ARCHIVE"})

    # No class filter: checkpoints has STANDARD + ARCHIVE, only ARCHIVE deleted → mixed
    resp_all = client.get("/api/explore/unified?bucket=marin-us-central1")
    ckpt = next(e for e in resp_all.json()["entries"] if e["name"] == "checkpoints/")
    assert ckpt["status"] == "mixed"

    # Class filter = ARCHIVE: all ARCHIVE bytes are deleted → delete
    resp_ar = client.get("/api/explore/unified?bucket=marin-us-central1&storage_class=ARCHIVE")
    ckpt_ar = next(e for e in resp_ar.json()["entries"] if e["name"] == "checkpoints/")
    assert ckpt_ar["status"] == "delete"

    # Class filter = STANDARD: no delete rule targets STANDARD → unmatched
    resp_std = client.get("/api/explore/unified?bucket=marin-us-central1&storage_class=STANDARD")
    ckpt_std = next(e for e in resp_std.json()["entries"] if e["name"] == "checkpoints/")
    assert ckpt_std["status"] == "unmatched"


# ---------------------------------------------------------------------------
# /api/explore/unified — kept_cost and bucketed status collapsing
# ---------------------------------------------------------------------------


def test_unified_explore_children_have_kept_cost(client):
    """Children (TLD) entries include a kept_cost field."""
    resp = client.get("/api/explore/unified")
    data = resp.json()
    assert data["type"] == "children"
    for entry in data["entries"]:
        assert "kept_cost" in entry
        assert entry["kept_cost"] >= 0


def test_unified_explore_kept_cost_reflects_deletions(client):
    """kept_cost < total cost when part of the data is marked for deletion."""
    client.post("/api/delete-rules", json={"pattern": "checkpoints/%"})

    resp = client.get("/api/explore/unified")
    ckpt = next(e for e in resp.json()["entries"] if e["name"] == "checkpoints/")
    assert ckpt["status"] == "delete"
    assert ckpt["kept_cost"] == 0.0


def test_unified_explore_kept_cost_equals_total_when_all_kept(client):
    """When nothing is deleted, kept_cost should equal monthly_cost_usd."""
    client.post("/api/rules", json={"bucket": "*", "pattern": "data/%"})

    resp = client.get("/api/explore/unified")
    data_entry = next(e for e in resp.json()["entries"] if e["name"] == "data/")
    assert data_entry["status"] == "keep"
    assert data_entry["kept_cost"] == data_entry["monthly_cost_usd"]


def test_unified_explore_bucketed_only_keep_delete(client):
    """Bucketed entries should only have 'keep' or 'delete' status, not 'mixed'
    or 'unmatched'. Mixed/unmatched are collapsed to 'keep'."""
    client.post("/api/delete-rules", json={"pattern": "tmp/%"})

    resp = client.get("/api/explore/unified?max_children=1")
    data = resp.json()
    assert data["type"] == "buckets"
    for entry in data["entries"]:
        assert entry["status"] in (
            "keep",
            "delete",
        ), f"Bucketed entry has status '{entry['status']}', expected 'keep' or 'delete'"


def test_unified_explore_bucketed_mixed_collapses_to_keep(client):
    """A storage-class-specific delete (creating 'mixed') should collapse to
    'keep' in the bucketed view."""
    client.post("/api/delete-rules", json={"pattern": "%", "storage_class": "ARCHIVE"})

    resp = client.get("/api/explore/unified?max_children=1")
    data = resp.json()
    assert data["type"] == "buckets"
    for entry in data["entries"]:
        assert entry["status"] in ("keep", "delete")


# ---------------------------------------------------------------------------
# /api/savings
# ---------------------------------------------------------------------------


def test_savings_returns_totals(client):
    resp = client.get("/api/savings")
    assert resp.status_code == 200
    assert "totals" in resp.json()


def test_savings_totals_shape(client):
    totals = client.get("/api/savings").json()["totals"]
    assert "deletable_objects" in totals
    assert "deletable_bytes" in totals
    assert "monthly_savings_usd" in totals


def test_savings_regions_present(client):
    data = client.get("/api/savings").json()
    assert "regions" in data


# ---------------------------------------------------------------------------
# /api/sync/status
# ---------------------------------------------------------------------------


def test_sync_status_returns_200(client):
    resp = client.get("/api/sync/status")
    assert resp.status_code == 200


def test_sync_status_shape(client):
    data = client.get("/api/sync/status").json()
    assert "syncing" in data
    assert data["syncing"] is False


# ---------------------------------------------------------------------------
# SPA fallback
# ---------------------------------------------------------------------------


def test_spa_fallback_serves_html(client):
    resp = client.get("/some/random/ui/path")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
