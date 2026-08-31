// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Catalog behavior tests, spanning every facet.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use buffa::MessageField;
use sha2::{Digest, Sha256};

use super::*;
use crate::partition_policy::SegmentPartition;
use crate::proto::finelog::stats::{
    ColumnType, MigrationPhase, OperatingPolicy, SourceLayout, TableSpec as ProtoTableSpec,
};
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_to_proto_owned, with_implicit_seq, Column, Schema};
use crate::store::table_spec::canonical_json_bytes;
use crate::store::types::{NamespaceStats, SegmentRow};

fn worker_stored() -> Schema {
    with_implicit_seq(Schema::new(
        vec![
            Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
            Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
        ],
        "",
    ))
}

fn table_spec(version: u64, target_object_bytes: u64) -> ProtoTableSpec {
    ProtoTableSpec {
        version: Some(version),
        logical_schema: MessageField::some(schema_to_proto_owned(&worker_stored())),
        source_layout: MessageField::some(SourceLayout {
            target_object_bytes: Some(target_object_bytes),
            ..Default::default()
        }),
        operating_policy: MessageField::some(OperatingPolicy::default()),
        ..Default::default()
    }
}

fn spec_hash(spec: &ProtoTableSpec) -> [u8; 32] {
    Sha256::digest(canonical_json_bytes(spec).unwrap()).into()
}

#[test]
fn table_spec_registration_is_monotonic_and_idempotent() {
    let catalog = Catalog::open(None).unwrap();
    let v1 = table_spec(1, 128);
    let status = catalog
        .register_table_spec("a", &v1, &spec_hash(&v1), false)
        .unwrap();
    assert_eq!(status.active_version(), 1);
    assert_eq!(status.catalog_generation, 1);

    let repeated = catalog
        .register_table_spec("a", &v1, &spec_hash(&v1), false)
        .unwrap();
    assert_eq!(repeated.catalog_generation, 1);

    let conflicting_v1 = table_spec(1, 256);
    assert!(matches!(
        catalog.register_table_spec("a", &conflicting_v1, &spec_hash(&conflicting_v1), false,),
        Err(StatsError::SchemaConflict(_))
    ));
    let v3 = table_spec(3, 128);
    assert!(matches!(
        catalog.register_table_spec("a", &v3, &spec_hash(&v3), false),
        Err(StatsError::SchemaConflict(_))
    ));
}

#[test]
fn source_layout_change_queues_activation_and_supports_abort() {
    let catalog = Catalog::open(None).unwrap();
    let v1 = table_spec(1, 128);
    catalog
        .register_table_spec("a", &v1, &spec_hash(&v1), false)
        .unwrap();
    let v2 = table_spec(2, 256);
    let pending = catalog
        .register_table_spec("a", &v2, &spec_hash(&v2), true)
        .unwrap();
    assert_eq!(pending.active_version(), 1);
    assert_eq!(pending.desired_version(), 2);
    assert_eq!(pending.phase, MigrationPhase::MIGRATION_PHASE_DUAL_WRITE);

    let activated = catalog.activate_desired_table_spec("a").unwrap();
    assert_eq!(activated.active_version(), 2);
    assert_eq!(activated.desired_version(), 0);
    assert!(activated
        .migration
        .as_ref()
        .and_then(|migration| migration.observation_deadline_ms)
        .is_some_and(|deadline| deadline > now_ms()));
    let aborted = catalog.abort_table_migration("a").unwrap();
    assert_eq!(aborted.active_version(), 1);
    assert!(aborted.migration.is_none());
    assert!(aborted.catalog_generation > activated.catalog_generation);
}

#[test]
fn table_spec_state_persists_across_catalog_reopen() {
    let dir = tempdir();
    let v1 = table_spec(1, 128);
    {
        let catalog = Catalog::open(Some(&dir)).unwrap();
        catalog
            .register_table_spec("a", &v1, &spec_hash(&v1), false)
            .unwrap();
    }
    let catalog = Catalog::open(Some(&dir)).unwrap();
    let status = catalog.table_spec_status("a").unwrap();
    assert_eq!(status.active.as_ref(), Some(&v1));
    assert_eq!(status.catalog_generation, 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn open_in_memory_and_register_fresh() {
    let cat = Catalog::open(None).unwrap();
    let (schema, policy) = cat
        .register_or_evolve("a", worker_stored(), StoragePolicy::default(), |_| {
            panic!("fresh register should not call merge")
        })
        .unwrap();
    assert_eq!(schema, worker_stored());
    assert!(policy.is_empty());
    assert!(cat.contains("a"));
}

#[test]
fn re_evolve_merges_existing() {
    let cat = Catalog::open(None).unwrap();
    cat.register_or_evolve(
        "a",
        worker_stored(),
        StoragePolicy::default(),
        |_| unreachable!(),
    )
    .unwrap();
    let (schema, _) = cat
        .register_or_evolve("a", worker_stored(), StoragePolicy::default(), |existing| {
            Ok(existing.clone())
        })
        .unwrap();
    assert_eq!(schema, worker_stored());
}

#[test]
fn upsert_schema_round_trips_through_json() {
    let cat = Catalog::open(None).unwrap();
    cat.upsert("a", &worker_stored()).unwrap();
    let all = cat.list_all().unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, "a");
    assert_eq!(all[0].1, worker_stored());
}

#[test]
fn upsert_preserves_registered_at_and_bumps_last_modified() {
    let cat = Catalog::open(None).unwrap();
    cat.upsert("a", &worker_stored()).unwrap();
    let inner = cat.inner.lock().unwrap();
    let (reg1, mod1): (i64, i64) = inner
        .conn
        .query_row(
            "SELECT registered_at_ms, last_modified_ms FROM namespaces WHERE namespace='a'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    drop(inner);
    std::thread::sleep(std::time::Duration::from_millis(2));
    cat.upsert("a", &worker_stored()).unwrap();
    let inner = cat.inner.lock().unwrap();
    let (reg2, mod2): (i64, i64) = inner
        .conn
        .query_row(
            "SELECT registered_at_ms, last_modified_ms FROM namespaces WHERE namespace='a'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(reg1, reg2, "registered_at preserved");
    assert!(mod2 >= mod1, "last_modified bumped");
}

#[test]
fn aggregate_stats_empty_when_no_segments() {
    let cat = Catalog::open(None).unwrap();
    cat.upsert("a", &worker_stored()).unwrap();
    assert_eq!(
        cat.aggregate_namespace_stats("a").unwrap(),
        NamespaceStats::empty()
    );
    assert!(cat.list_segments("a").unwrap().is_empty());
}

#[test]
fn begin_drop_fences_register() {
    let cat = Catalog::open(None).unwrap();
    cat.register_or_evolve(
        "a",
        worker_stored(),
        StoragePolicy::default(),
        |_| unreachable!(),
    )
    .unwrap();
    cat.begin_drop("a").unwrap();
    assert!(cat.is_dropping("a"));
    let err = cat.register_or_evolve(
        "a",
        worker_stored(),
        StoragePolicy::default(),
        |_| unreachable!(),
    );
    assert!(matches!(err, Err(StatsError::InvalidNamespace(_))));
    cat.finish_drop("a");
    assert!(!cat.is_dropping("a"));
}

#[test]
fn snapshot_live_returns_registration_order() {
    let cat = Catalog::open(None).unwrap();
    for name in ["zeta", "alpha", "mid"] {
        cat.register_or_evolve(
            name,
            worker_stored(),
            StoragePolicy::default(),
            |_| unreachable!(),
        )
        .unwrap();
    }
    let order: Vec<String> = cat.snapshot_live().into_iter().map(|ns| ns.name).collect();
    assert_eq!(order, vec!["zeta", "alpha", "mid"]);
}

#[test]
fn upsert_policy_empty_deletes_row() {
    let cat = Catalog::open(None).unwrap();
    cat.upsert("a", &worker_stored()).unwrap();
    cat.upsert_policy(
        "a",
        &StoragePolicy {
            max_segments: Some(7),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(cat.get_policy("a").unwrap().max_segments, Some(7));
    cat.upsert_policy("a", &StoragePolicy::default()).unwrap();
    assert!(cat.get_policy("a").unwrap().is_empty());
}

#[test]
fn a_failing_delete_leaves_every_namespace_row_in_place() {
    let cat = Catalog::open(None).unwrap();
    cat.upsert("a", &worker_stored()).unwrap();
    let v1 = table_spec(1, 128);
    cat.register_table_spec("a", &v1, &spec_hash(&v1), false)
        .unwrap();
    cat.upsert_policy(
        "a",
        &StoragePolicy {
            max_segments: Some(3),
            ..Default::default()
        },
    )
    .unwrap();
    cat.upsert_segment(&SegmentRow {
        namespace: "a".to_string(),
        path: "a.parquet".to_string(),
        level: 1,
        min_seq: 1,
        max_seq: 3,
        row_count: 3,
        byte_size: 100,
        created_at_ms: 1,
        min_key_value: None,
        max_key_value: None,
        partition: None,
        location: crate::store::types::SegmentLocation::Local,
    })
    .unwrap();
    cat.set_forward_cursor("hub", "a", 3).unwrap();

    // Fails the last of the delete's statement groups, after the groups that
    // clear the dependent tables have already run.
    cat.inner
        .lock()
        .unwrap()
        .conn
        .execute_batch(
            "CREATE TRIGGER pin_namespace BEFORE DELETE ON namespaces
             BEGIN SELECT RAISE(ABORT, 'namespace pinned'); END",
        )
        .unwrap();

    assert!(cat.delete("a").is_err());

    assert_eq!(cat.list_segments("a").unwrap().len(), 1);
    assert_eq!(cat.get_policy("a").unwrap().max_segments, Some(3));
    assert_eq!(cat.table_spec_status("a").unwrap().active_version(), 1);
    assert_eq!(cat.forward_cursor("hub", "a").unwrap(), Some(3));
    assert!(cat.list_all().unwrap().iter().any(|(name, _)| name == "a"));
}

#[test]
fn on_disk_catalog_persists_across_reopen() {
    let dir = tempdir();
    {
        let cat = Catalog::open(Some(&dir)).unwrap();
        cat.upsert("a", &worker_stored()).unwrap();
    }
    let cat = Catalog::open(Some(&dir)).unwrap();
    let all = cat.list_all().unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, "a");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn segment_partition_persists_across_catalog_reopen() {
    let dir = tempdir();
    let partition = SegmentPartition {
        spec_id: 1,
        values: BTreeMap::from([("name_bucket".to_string(), "6".to_string())]),
    };
    {
        let catalog = Catalog::open(Some(&dir)).unwrap();
        catalog.upsert("a", &worker_stored()).unwrap();
        catalog
            .upsert_segment(&SegmentRow {
                namespace: "a".to_string(),
                path: dir.join("a.parquet").to_string_lossy().into_owned(),
                level: 1,
                min_seq: 1,
                max_seq: 3,
                row_count: 3,
                byte_size: 100,
                created_at_ms: 1,
                min_key_value: None,
                max_key_value: None,
                partition: Some(partition.clone()),
                location: crate::store::types::SegmentLocation::Local,
            })
            .unwrap();
    }
    let catalog = Catalog::open(Some(&dir)).unwrap();
    assert_eq!(
        catalog.list_segments("a").unwrap()[0].partition,
        Some(partition)
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn opening_an_old_catalog_adds_partition_metadata_without_losing_segments() {
    let dir = tempdir();
    let path = dir.join(CATALOG_DB_FILENAME);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            r#"
            CREATE TABLE segments (
                namespace TEXT NOT NULL,
                path TEXT NOT NULL,
                level INTEGER NOT NULL,
                min_seq INTEGER NOT NULL,
                max_seq INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                byte_size INTEGER NOT NULL,
                created_at_ms INTEGER NOT NULL,
                min_key_value TEXT,
                max_key_value TEXT,
                location TEXT NOT NULL,
                PRIMARY KEY (namespace, path)
            );
            INSERT INTO segments VALUES
                ('a', '/old.parquet', 1, 1, 3, 3, 100, 1, NULL, NULL, 'LOCAL');
            "#,
        )
        .unwrap();
    drop(connection);

    let catalog = Catalog::open(Some(&dir)).unwrap();
    let old = catalog.list_segments("a").unwrap();
    assert_eq!(old.len(), 1);
    assert_eq!(old[0].path, "/old.parquet");
    assert_eq!(old[0].partition, None);

    let partition = SegmentPartition {
        spec_id: 1,
        values: BTreeMap::from([("name_bucket".to_string(), "6".to_string())]),
    };
    catalog
        .upsert_segment(&SegmentRow {
            namespace: "a".to_string(),
            path: "/new.parquet".to_string(),
            level: 1,
            min_seq: 4,
            max_seq: 4,
            row_count: 1,
            byte_size: 50,
            created_at_ms: 2,
            min_key_value: None,
            max_key_value: None,
            partition: Some(partition.clone()),
            location: crate::store::types::SegmentLocation::Local,
        })
        .unwrap();
    assert_eq!(
        catalog.list_segments("a").unwrap()[1].partition,
        Some(partition)
    );
    std::fs::remove_dir_all(&dir).ok();
}

fn tempdir() -> std::path::PathBuf {
    static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);
    let mut p = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let ordinal = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
    p.push(format!("finelog_catalog_test_{nanos}_{ordinal}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}
