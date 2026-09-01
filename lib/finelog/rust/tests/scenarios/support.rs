// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared journey fixture for whole-store scenario tests.
//!
//! A [`Cluster`] is one data directory and one object directory, reopened as
//! many times as a scenario needs; every store it opens carries a
//! [`FaultInjectingObjectStore`] seam in front of its object operations.
//! [`Invariants`] re-checks the durable contract after every step. The small
//! drivers below — [`write_row`], [`content_digest`], [`drive_to_phase`] —
//! keep each journey a short script over a real store.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use buffa::{Message, MessageView};

use finelog::proto::finelog::stats::{
    ColumnType, L0Mode, MigrationPhase, NamespaceCatalog, OperatingPolicy, RemoteRetentionPolicy,
    SourceLayout, TableSpec, TableSpecView,
};
use finelog::query::{make_ctx, run_query_over};
use finelog::store::object_store::{build_remote_object_store, ObjectStore};
use finelog::store::policy::StoragePolicy;
use finelog::store::schema::{schema_to_arrow, schema_to_proto_owned, Column, Schema};
use finelog::store::state_store::object::ObjectTableStateStore;
use finelog::store::store::{ServeMode, Store};
use finelog::store::table_spec::ValidatedTableSpec;
use finelog::store::TelemetryRootWriteMode;
use finelog::test_support::{unique_dir, FaultInjectingObjectStore, ObjectOp, ObjectPattern};

use buffa::MessageField;
use finelog::store::object_store::ObjectId;

pub const TABLE: &str = "iris.worker";
/// Long enough that a durability await never expires on a loaded machine, short
/// enough that a genuinely wedged flush still fails the test.
pub const PERSIST_BUDGET: Duration = Duration::from_secs(30);

pub fn worker_schema() -> Schema {
    Schema::new(
        vec![
            Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
            Column::new("mem_bytes", ColumnType::COLUMN_TYPE_INT64, false),
            Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
        ],
        "",
    )
}

pub fn object_backed_spec(version: u64, layout: SourceLayout) -> ValidatedTableSpec {
    let schema = worker_schema();
    let spec = TableSpec {
        version: Some(version),
        logical_schema: MessageField::some(schema_to_proto_owned(&schema)),
        source_layout: MessageField::some(layout),
        operating_policy: MessageField::some(OperatingPolicy {
            l0_mode: Some(L0Mode::L0_MODE_OBJECT_STORE.into()),
            remote_retention: MessageField::some(RemoteRetentionPolicy {
                retain_forever: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let encoded = spec.encode_to_vec();
    let view = TableSpecView::decode_view(&encoded).unwrap();
    ValidatedTableSpec::from_view(&view, &schema, &StoragePolicy::default()).unwrap()
}

/// The same logical schema at a different physical object size: a compatible
/// layout change, so registering it starts an automatic migration.
pub fn retargeted_spec(version: u64) -> ValidatedTableSpec {
    object_backed_spec(
        version,
        SourceLayout {
            target_object_bytes: Some(8 * 1024 * 1024),
            ..Default::default()
        },
    )
}

/// One data directory and one object directory, reopened as many times as a
/// scenario needs.
pub struct Cluster {
    pub data_dir: PathBuf,
    pub remote_dir: PathBuf,
}

impl Cluster {
    pub fn new(tag: &str) -> Self {
        Self {
            data_dir: unique_dir(&format!("{tag}_data")),
            remote_dir: unique_dir(&format!("{tag}_remote")),
        }
    }

    /// Open a store over `data_dir`, with a fault seam in front of every object
    /// operation it performs.
    pub fn open(&self) -> (Store, Arc<FaultInjectingObjectStore>) {
        self.open_from(&self.data_dir)
    }

    /// Open a store over its own data directory but the same object directory:
    /// a replacement writer, or a cold restart onto empty local state.
    pub fn open_from(&self, data_dir: &Path) -> (Store, Arc<FaultInjectingObjectStore>) {
        let seam: Arc<Mutex<Option<Arc<FaultInjectingObjectStore>>>> = Arc::new(Mutex::new(None));
        let captured = Arc::clone(&seam);
        let store = Store::open(
            Some(data_dir.to_path_buf()),
            self.remote_dir.to_string_lossy().into_owned(),
            finelog::indices::cache::DEFAULT_INDEX_CACHE_MB,
            // The scheduler owns a store's cadence but not its lifetime: it keeps
            // polling a dropped store's tables. Crash scenarios therefore never
            // start it and drive flush and maintenance directly, so a crashed
            // writer performs no work after the process that owned it is gone.
            ServeMode::Shadow,
            TelemetryRootWriteMode::SemanticOnly,
            None,
            Some(Arc::new(move |inner| {
                let faults = FaultInjectingObjectStore::new(inner);
                *captured.lock().unwrap() = Some(Arc::clone(&faults));
                faults as Arc<dyn ObjectStore>
            })),
        )
        .unwrap();
        let faults = seam.lock().unwrap().take().expect("object seam installed");
        (store, faults)
    }

    pub fn states(&self) -> ObjectTableStateStore {
        ObjectTableStateStore::new(Arc::new(
            build_remote_object_store(self.remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        ))
    }

    pub fn objects(&self) -> Arc<dyn ObjectStore> {
        Arc::new(
            build_remote_object_store(self.remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        )
    }

    pub fn cleanup(&self) {
        std::fs::remove_dir_all(&self.data_dir).ok();
        std::fs::remove_dir_all(&self.remote_dir).ok();
    }
}

/// Register the table as a version-0 legacy namespace, the state every
/// pre-object deployment starts from.
pub fn register_legacy(store: &Store) {
    store
        .register_table(TABLE, worker_schema(), StoragePolicy::default())
        .unwrap();
}

/// Register the table object-backed and publish its first revision.
pub async fn register_v1(store: &Store) {
    store
        .register_versioned_table(TABLE, object_backed_spec(1, SourceLayout::default()))
        .unwrap();
    store.publish_object_catalog(TABLE).await.unwrap();
}

/// Append one row and seal it, so the returned sequence number is acknowledged
/// durable before the caller proceeds.
pub async fn write_row(store: &Store, worker: &str, mem_bytes: i64) -> i64 {
    let batch_schema = schema_to_arrow(&worker_schema());
    let batch = RecordBatch::try_new(
        batch_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![worker])),
            Arc::new(Int64Array::from(vec![mem_bytes])),
            Arc::new(Int64Array::from(vec![mem_bytes])),
        ],
    )
    .unwrap();
    let ipc = finelog::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
    let (_, seq) = store.write_rows(TABLE, &ipc, None).unwrap();
    store.maintain_namespace(TABLE, false).await.unwrap();
    store
        .await_persisted(TABLE, seq, PERSIST_BUDGET)
        .await
        .unwrap();
    seq
}

pub async fn run_sql(store: &Store, sql: &str) -> Vec<RecordBatch> {
    try_run_sql(store, sql).await.unwrap()
}

pub async fn try_run_sql(store: &Store, sql: &str) -> Result<Vec<RecordBatch>, String> {
    let providers = store.query_providers().map_err(|error| error.to_string())?;
    run_query_over(&make_ctx(), providers, sql)
        .await
        .map(|result| result.batches)
        .map_err(|error| error.to_string())
}

pub fn seq_column(batches: &[RecordBatch]) -> Vec<i64> {
    let mut seqs: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            let column = batch
                .column_by_name("seq")
                .expect("every store-form row carries seq")
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..column.len())
                .map(|row| column.value(row))
                .collect::<Vec<_>>()
        })
        .collect();
    seqs.sort_unstable();
    seqs
}

/// An order-independent content digest of every row at `seq <= upto_seq`.
///
/// Aggregates only, so the value is identical whatever segment layout, level,
/// or storage backend serves the rows — the same equality the full-scale
/// rollout rehearsals check with per-column digest batteries.
pub async fn content_digest(store: &Store, upto_seq: i64) -> String {
    let sql = format!(
        "SELECT count(*) AS n, \
         sum(mem_bytes) AS mem_sum, min(mem_bytes) AS mem_min, max(mem_bytes) AS mem_max, \
         sum(length(worker_id)) AS id_len, min(worker_id) AS id_min, max(worker_id) AS id_max, \
         sum(seq) AS seq_sum \
         FROM \"{TABLE}\" WHERE seq <= {upto_seq}"
    );
    arrow::util::pretty::pretty_format_batches(&run_sql(store, &sql).await)
        .unwrap()
        .to_string()
}

/// Drive maintenance until the table's migration reaches `phase`, asserting it
/// gets there within `max_rounds` cycles.
pub async fn drive_to_phase(store: &Store, phase: MigrationPhase, max_rounds: usize) {
    for _ in 0..max_rounds {
        if store.spec_lifecycle(TABLE).unwrap().phase == phase {
            return;
        }
        store.maintain_namespace(TABLE, false).await.unwrap();
    }
    assert_eq!(
        store.spec_lifecycle(TABLE).unwrap().phase,
        phase,
        "the table never reached the expected migration phase"
    );
}

/// Every immutable object the given durable state references.
pub fn referenced_objects(state: &NamespaceCatalog) -> BTreeSet<String> {
    state
        .version_segments
        .iter()
        .flat_map(|version| {
            version
                .live_segments
                .iter()
                .chain(version.retired_segments.iter())
        })
        .chain(state.direct_query_segments.iter())
        .filter_map(|segment| {
            segment
                .source
                .as_option()
                .and_then(|source| source.object_id.clone())
        })
        .collect()
}

/// The objects the state's currently live segments name, whatever version they
/// belong to.
pub fn live_objects(state: &NamespaceCatalog) -> BTreeSet<String> {
    state
        .version_segments
        .iter()
        .flat_map(|version| version.live_segments.iter())
        .filter_map(|segment| {
            segment
                .source
                .as_option()
                .and_then(|source| source.object_id.clone())
        })
        .collect()
}

/// The compaction level of every live segment.
pub fn live_levels(state: &NamespaceCatalog) -> Vec<i32> {
    state
        .version_segments
        .iter()
        .flat_map(|version| version.live_segments.iter())
        .map(|segment| segment.level.unwrap_or(0))
        .collect()
}

/// The invariants every scenario step must leave intact.
///
/// The checker carries the history a single observation cannot see — the
/// highest revision published so far, and whether the table has been
/// tombstoned — so repeated calls catch a revision going backwards or a deleted
/// table coming back.
pub struct Invariants {
    remote_dir: PathBuf,
    highest_revision: u64,
    tombstoned: bool,
}

/// One consistent observation of a table, taken while the invariants held.
pub struct Observation {
    pub state: NamespaceCatalog,
    pub seqs: Vec<i64>,
}

impl Invariants {
    pub fn new(remote_dir: &Path) -> Self {
        Self {
            remote_dir: remote_dir.to_path_buf(),
            highest_revision: 0,
            tombstoned: false,
        }
    }

    /// Check every invariant against the object directory and `store`'s reads,
    /// and return what was observed so a scenario can assert its own specifics.
    pub async fn check(&mut self, store: &Store) -> Observation {
        let states = ObjectTableStateStore::new(Arc::new(
            build_remote_object_store(self.remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        ));
        // `load` is the contract for "HEAD names a state the controller
        // committed": it rejects a HEAD whose state document is missing, fails
        // its recorded SHA-256, or disagrees about revision, active version, or
        // tombstone.
        let selected = states
            .load(TABLE)
            .await
            .unwrap()
            .expect("a published table always has a loadable HEAD");
        let revision = selected.revision().get();
        assert!(
            revision >= self.highest_revision,
            "table revision moved backwards from {} to {revision}",
            self.highest_revision
        );
        self.highest_revision = revision;

        assert!(
            !self.tombstoned || selected.is_tombstoned(),
            "a tombstoned table came back to life at revision {revision}"
        );
        self.tombstoned = selected.is_tombstoned();

        let objects = Arc::new(
            build_remote_object_store(self.remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        for key in referenced_objects(&selected.catalog) {
            let id = ObjectId::parse(&key).unwrap();
            assert!(
                objects.read(&id).await.unwrap().is_some(),
                "state revision {revision} references missing object {key}"
            );
        }

        let seqs = seq_column(&run_sql(store, &format!("SELECT seq FROM \"{TABLE}\"")).await);
        let distinct: HashSet<i64> = seqs.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            seqs.len(),
            "duplicate sequence numbers in the readable row set: {seqs:?}"
        );
        if let (Some(first), Some(last)) = (seqs.first(), seqs.last()) {
            assert_eq!(
                *last - *first + 1,
                seqs.len() as i64,
                "acknowledged sequence coverage has a gap: {seqs:?}"
            );
        }
        Observation {
            state: selected.catalog,
            seqs,
        }
    }
}

/// A migration or compaction output object, matched so a fault can park the
/// writer that is producing it.
pub fn data_object_upload() -> (ObjectOp, ObjectPattern) {
    (
        ObjectOp::Write,
        ObjectPattern::Contains("/objects/".to_string()),
    )
}

/// The object IDs a phase resolved to local files, restricted to data objects.
pub fn localized_data_objects(faults: &FaultInjectingObjectStore) -> BTreeSet<String> {
    faults
        .keys_for(ObjectOp::LocalPath)
        .into_iter()
        .filter(|key| key.contains("/objects/"))
        .collect()
}

/// The data-object keys (`_finelog/.../objects/*.parquet`) the local cache
/// holds right now.
pub fn cached_data_objects(data_dir: &Path) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    let mut stack = vec![data_dir.join("_finelog")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(relative) = path.strip_prefix(data_dir) {
                let key = relative.to_string_lossy().into_owned();
                if key.contains("/objects/") {
                    keys.insert(key);
                }
            }
        }
    }
    keys
}

/// Wait for the background cache fill to settle on `expected`, returning
/// whatever the cache holds at the deadline.
pub async fn wait_for_cache_fill(data_dir: &Path, expected: &BTreeSet<String>) -> BTreeSet<String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let cached = cached_data_objects(data_dir);
        if &cached == expected || std::time::Instant::now() > deadline {
            return cached;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// Assert that a bootstrap resolved the table's state and nothing else: it read
/// HEAD, and it fetched no data object, index bundle, or projection.
pub fn assert_metadata_only_bootstrap(faults: &FaultInjectingObjectStore) {
    let fetched: Vec<String> = faults
        .calls()
        .into_iter()
        .filter(|call| matches!(call.op, ObjectOp::Read | ObjectOp::LocalPath))
        .map(|call| call.key)
        .filter(|key| {
            key.contains("/objects/") || key.contains("/indices/") || key.contains("/projections/")
        })
        .collect();
    assert!(
        fetched.is_empty(),
        "bootstrap must load metadata only, but fetched {fetched:?}"
    );
    assert!(
        faults
            .calls()
            .iter()
            .any(|call| call.key.ends_with("HEAD.json")),
        "the seam saw no HEAD read at all, so this assertion proved nothing"
    );
}
