//! Rebuild-from-disk catalog adoption.
//!
//! On a fresh boot over a `log_dir` that a prior server populated, there may be
//! no `_finelog_catalog.sqlite` sidecar — the catalog is empty, so the normal
//! `rehydrate_from_catalog` path builds no engines and adoption never runs.
//! This module reconstructs the catalog **purely by scanning the on-disk
//! parquet layout + footers**, never reading the sidecar.
//!
//! Layout adopted (per-namespace, the production layout):
//!
//! ```text
//! {data_dir}/<namespace>/seg_L<level>_<minseq:019>.parquet
//! {data_dir}/log/seg_L<level>_<minseq:019>.parquet   (the privileged log ns)
//! ```
//!
//! [`ensure_catalog_adopted`] is the boot orchestrator: a sentinel
//! (`{data_dir}/.finelog-rust-catalog`) fast-paths once adoption is `done`,
//! otherwise it footer-scans every namespace subdir and persists the recovered
//! namespace + segment rows into the sqlite catalog so the subsequent
//! `rehydrate_from_catalog` + `TableRuntime::open` (which re-discovers local
//! segments) materialize the live engines. It runs **before** the listener
//! binds, inside `Store::new`, between opening the catalog and rehydrating it.
//!
//! It is synchronous: footer reads are footer-only (no column scan), so a
//! moderate data dir is fast, and the done-sentinel fast-paths every later
//! boot. (Adoption runs before the reactor's request path, so the readiness
//! mitigation is the sentinel, not reactor offload.)
//!
//! ## Schema-recovery lossiness (documented)
//!
//! The sidecar stored `schema_json` per namespace; rebuild-from-disk can only
//! recover the Arrow schema from a parquet footer. Three facts are NOT
//! faithfully recoverable from parquet and are therefore lossy:
//!
//! 1. **`key_column`** — parquet carries no `key_column` metadata.
//!    [`recover_schema_from_segments`] resolves it with the `resolve_key_column`
//!    default rule: if the schema has an implicit `timestamp_ms` column, the key
//!    is left empty (the default resolves to `timestamp_ms`), matching a
//!    namespace registered with an empty `key_column`. A non-default key_column
//!    that differs from what the footer implies is unrecoverable from parquet
//!    alone. Two normalizations follow from this and are intentional (both
//!    re-established exactly by deploy's startup `RegisterTable`, so neither is
//!    observable past cutover): a namespace registered with an *explicit*
//!    `key_column = "timestamp_ms"` is recovered as the empty string (it
//!    resolves identically, but the wire `key_column` differs until
//!    RegisterTable runs); and when there is no `timestamp_ms` column the first
//!    non-seq column is taken as a best-effort key, which can differ from a
//!    genuine non-default registered key (a `warn!` is emitted in that branch so
//!    an operator can see which namespaces depend on deploy-time key
//!    re-establishment).
//! 2. **StoragePolicy** — never written to parquet. Adopted namespaces start
//!    with an EMPTY (inherit-all) policy, exactly like a catalog-less boot
//!    (`get_policy` returns the default when no row exists).
//! 3. **Column non-nullability of COMPACTED segments** — L0 segments preserve
//!    Arrow non-nullability in the footer (recovered faithfully), but a
//!    DuckDB-compacted L>=1 segment's parquet marks every column nullable
//!    (DuckDB's COPY does not carry Arrow non-nullability). So a column that was
//!    registered non-nullable but lives only in a compacted segment is adopted
//!    as nullable. This is benign: a nullable-superset never changes queryable
//!    contents (the cutover gate asserts Query Arrow + the data are identical),
//!    and a later RegisterTable with the original non-nullable schema is
//!    accepted as a no-op (`merge_schemas` treats a nullability difference on
//!    an existing column as compatible, keeping the adopted nullable form). The
//!    column stays nullable rather than narrowing back, which is harmless for
//!    every read and write path.
//!
//! Additive-merge history is likewise collapsed: the recovered schema is the
//! newest segment's columns (which already reflect every additive evolution
//! that has been flushed). Production cutover re-establishes the exact
//! registered schema + policy because deploy drives `RegisterTable` for every
//! known table at startup (an additive merge is a no-op when identical).

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Instant, UNIX_EPOCH};

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::errors::StatsError;
use crate::indices::local_sidecar_artifacts;
use crate::store::catalog::Catalog;
use crate::store::namespace_name::validate_namespace_name;
use crate::store::object_store::ObjectStore;
use crate::store::schema::{
    arrow_to_column_type, resolve_key_column, Column, Schema, IMPLICIT_KEY_COLUMN,
    IMPLICIT_SEQ_COLUMN,
};
use crate::store::segment::{discover_files, discover_segments, read_segment_footer};
use crate::store::table::object_segment_is_query_visible;
use crate::store::table::segment_view::{debug_assert_unique_paths, segment_artifacts};
use crate::store::types::{LocalSegment, SegmentLocation, SegmentRow};

/// Sentinel filename for the catalog-adoption state machine (the disk->catalog
/// rebuild).
pub const SENTINEL_FILENAME: &str = ".finelog-rust-catalog";

/// Sentinel schema version.
pub const SENTINEL_VERSION: u32 = 1;

/// The privileged log namespace name + dir (kept local to avoid a store->adopt
/// dependency cycle; the value is fixed by the proto contract).
const LOG_NAMESPACE_NAME: &str = "log";

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Sentinel state machine.
// ---------------------------------------------------------------------------

/// Adoption progress states. `in-progress` means a scan started (or crashed
/// mid-scan); `done` is the steady state that fast-paths every later boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdoptionState {
    #[serde(rename = "in-progress")]
    InProgress,
    #[serde(rename = "done")]
    Done,
}

/// The sentinel payload (single-line JSON, atomic tmp+rename).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdoptionSentinel {
    pub version: u32,
    pub state: AdoptionState,
    pub started_at: i64,
    pub finished_at: Option<i64>,
}

/// Read the sentinel `state`, or `None` if missing/malformed.
///
/// A malformed sentinel is treated as `None` (re-run adoption — the scan is
/// idempotent, so a re-run is safe and rewrites the sentinel on completion).
pub fn read_sentinel_state(data_dir: &Path) -> Option<AdoptionState> {
    let path = data_dir.join(SENTINEL_FILENAME);
    let raw = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<AdoptionSentinel>(&raw) {
        Ok(s) => Some(s.state),
        Err(_) => {
            tracing::warn!(
                path = %path.display(),
                "catalog adoption: malformed sentinel; treating as missing"
            );
            None
        }
    }
}

/// Atomically write the sentinel via a sibling `.tmp` + rename.
pub fn write_sentinel(
    data_dir: &Path,
    state: AdoptionState,
    started_at: i64,
    finished_at: Option<i64>,
) -> Result<(), StatsError> {
    let sentinel = AdoptionSentinel {
        version: SENTINEL_VERSION,
        state,
        started_at,
        finished_at,
    };
    let line = serde_json::to_string(&sentinel)
        .map_err(|e| StatsError::Internal(format!("sentinel json: {e}")))?;
    let final_path = data_dir.join(SENTINEL_FILENAME);
    let tmp_path = data_dir.join(format!("{SENTINEL_FILENAME}.tmp"));
    std::fs::write(&tmp_path, format!("{line}\n"))
        .map_err(|e| StatsError::Internal(format!("write sentinel tmp: {e}")))?;
    std::fs::rename(&tmp_path, &final_path)
        .map_err(|e| StatsError::Internal(format!("rename sentinel: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Directory scan -> in-memory catalog.
// ---------------------------------------------------------------------------

/// Recover the next seq to allocate from a set of adopted segment rows:
/// `max(max_seq) + 1`, floored at `1` for an empty set.
pub fn recover_next_seq(segments: &[SegmentRow]) -> i64 {
    segments
        .iter()
        .map(|s| s.max_seq + 1)
        .max()
        .unwrap_or(1)
        .max(1)
}

/// Footer-scan one namespace directory into `SegmentRow`s sorted by `min_seq`.
///
/// Each `seg_L*_*.parquet` file's footer gives `row_count` -> `max_seq`
/// (`min_seq` comes from the FILENAME); `byte_size` is the on-disk file length;
/// `created_at_ms` is the file mtime in ms; `location = LOCAL`. Key bounds are
/// the Int64 stats for the schema's resolved key column, stringified at the
/// catalog boundary. A corrupt or unparseable file is warn-and-skipped rather
/// than aborting the scan, so it is excluded from `segment_count`. This only
/// bites on already-unreadable data (the rows are lost regardless) and never
/// affects `next_seq` (re-derived from healthy footers at engine open) — a
/// genuinely empty *readable* 0-row segment is adopted normally.
pub fn adopt_namespace_from_disk(
    ns_dir: &Path,
    namespace: &str,
    schema: &Schema,
) -> Vec<SegmentRow> {
    let key_column = resolve_key_column(schema).ok();
    let mut rows: Vec<SegmentRow> = Vec::new();
    for path in discover_segments(ns_dir) {
        let Some(meta) = read_segment_footer(&path, key_column.as_deref()) else {
            tracing::warn!(path = %path.display(), "adopt: unreadable segment footer; skipping");
            continue;
        };
        let md = std::fs::metadata(&path);
        let byte_size = md.as_ref().map(|m| m.len() as i64).unwrap_or(0);
        let created_at_ms = md
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or_else(now_ms);
        rows.push(SegmentRow {
            namespace: namespace.to_string(),
            path: path.to_string_lossy().into_owned(),
            level: meta.level,
            min_seq: meta.min_seq,
            max_seq: meta.max_seq,
            row_count: meta.row_count,
            byte_size,
            created_at_ms,
            min_key_value: meta.min_key_value.map(|v| v.to_string()),
            max_key_value: meta.max_key_value.map(|v| v.to_string()),
            partition: meta.partition,
            location: SegmentLocation::Local,
        });
    }
    rows.sort_by_key(|r| r.min_seq);
    rows
}

// ---------------------------------------------------------------------------
// Schema recovery from the newest segment's parquet footer.
// ---------------------------------------------------------------------------

/// Reconstruct a namespace's store-form proto `Schema` from the parquet arrow
/// schema of its newest segment (highest `min_seq`).
///
/// The recovered columns are in on-disk order (the implicit `seq` column first,
/// re-marked as implicit Int64, then the registered columns). `key_column` is
/// recovered by the default rule: left EMPTY when the schema carries a
/// `timestamp_ms` column (so the default resolves to it), matching a namespace
/// registered with an empty key_column. See the module docstring for the lossy
/// edge.
///
/// Returns `None` when the directory has no readable segment (a namespace dir
/// with no parquet contributes nothing — the caller skips it).
pub fn recover_schema_from_segments(ns_dir: &Path) -> Option<Schema> {
    let newest = discover_segments(ns_dir)
        .into_iter()
        .filter_map(|path| {
            let (_, min_seq) =
                crate::store::types::parse_seg_filename(path.file_name()?.to_str()?)?;
            Some((min_seq, path))
        })
        .max_by_key(|(min_seq, _)| *min_seq)?
        .1;
    let file = std::fs::File::open(&newest).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    let arrow_schema = builder.schema();

    let mut columns: Vec<Column> = Vec::with_capacity(arrow_schema.fields().len());
    let mut has_timestamp_ms = false;
    for field in arrow_schema.fields() {
        let name = field.name();
        if name == IMPLICIT_SEQ_COLUMN {
            // Re-mark the implicit seq column exactly as `with_implicit_seq`
            // would (Int64, non-nullable). Its on-disk type is Int64 already.
            columns.push(Column::new(
                IMPLICIT_SEQ_COLUMN,
                crate::proto::finelog::stats::ColumnType::COLUMN_TYPE_INT64,
                false,
            ));
            continue;
        }
        let Ok(ctype) = arrow_to_column_type(field.data_type()) else {
            tracing::warn!(
                column = name,
                dtype = ?field.data_type(),
                "adopt: column type not representable as a proto ColumnType; skipping schema recovery"
            );
            return None;
        };
        if name == IMPLICIT_KEY_COLUMN {
            has_timestamp_ms = true;
        }
        columns.push(Column::new(name, ctype, field.is_nullable()));
    }

    // Leave key_column empty when the default (`timestamp_ms`) resolves; this
    // matches the common registration form (empty key_column) byte-for-byte on
    // the wire. Otherwise fall back to the implicit-key default name so
    // `resolve_key_column` still succeeds on the adopted schema.
    let key_column = if has_timestamp_ms {
        String::new()
    } else {
        // No timestamp_ms present: the original key_column is unrecoverable.
        // Use the first non-seq column as a best-effort key so the namespace is
        // still queryable; documented lossiness. Warn so an operator can see
        // which namespaces depend on deploy-time RegisterTable to re-establish
        // the genuine key (the parquet footer carries no key_column metadata).
        let fallback = columns
            .iter()
            .find(|c| c.name != IMPLICIT_SEQ_COLUMN)
            .map(|c| c.name.clone())
            .unwrap_or_default();
        tracing::warn!(
            dir = ?ns_dir,
            key_column = %fallback,
            "adopt: no timestamp_ms column; recovered key_column is a best-effort \
             first-column guess (re-established exactly by deploy RegisterTable)"
        );
        fallback
    };
    Some(Schema::new(columns, key_column))
}

// ---------------------------------------------------------------------------
// adopt_store_from_disk + ensure_catalog_adopted (the boot orchestrator).
// ---------------------------------------------------------------------------

/// Enumerate valid namespace subdirectories of `data_dir`. Internal roots such
/// as `_finelog` and migration scratch directories are excluded by the same
/// validator used at registration. The `log` dir is included (the privileged
/// namespace is adopted like any other, then
/// `ensure_log_namespace_registered` re-establishes its canonical schema).
fn enumerate_namespace_dirs(data_dir: &Path) -> Result<Vec<(String, PathBuf)>, StatsError> {
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    let entries = std::fs::read_dir(data_dir)
        .map_err(|e| StatsError::Internal(format!("read data_dir {}: {e}", data_dir.display())))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if validate_namespace_name(name, None).is_err() {
            continue;
        }
        out.push((name.to_string(), path));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Assert `data_dir` is in the per-namespace `seg_L` layout; fail loudly on a
/// flat (legacy `tmp_*`/`logs_*`) layout rather than migrating it.
///
/// A flat top-level parquet is an operational error, not something to silently
/// migrate.
fn assert_namespaced_layout(data_dir: &Path) -> Result<(), StatsError> {
    let entries = std::fs::read_dir(data_dir)
        .map_err(|e| StatsError::Internal(format!("read data_dir {}: {e}", data_dir.display())))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".parquet") {
                    return Err(StatsError::Internal(format!(
                        "data_dir {} holds a top-level parquet file {name:?}: this is the legacy \
                         flat layout, which the Rust server does not migrate. Run the Python \
                         layout migration first, or point --log-dir at a per-namespace seg_L tree.",
                        data_dir.display()
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Rebuild the catalog's `namespaces` + `segments` sqlite rows from the on-disk
/// parquet layout under `data_dir`.
///
/// For each namespace subdir: recover its schema (skip dirs with no readable
/// segment, except the privileged `log` ns which is materialized empty by
/// `ensure_log_namespace_registered`), persist the `namespaces` row, then
/// footer-scan its segments and upsert the `segments` rows. Footer reads are
/// footer-only (no column page scan), and adoption runs once before the
/// reactor's request path, so it is synchronous.
///
/// Does NOT touch the live registry — the caller's `rehydrate_from_catalog`
/// reads these persisted rows back and builds the engines.
pub fn adopt_store_from_disk(data_dir: &Path, catalog: &Catalog) -> Result<(), StatsError> {
    assert_namespaced_layout(data_dir)?;
    for (namespace, ns_dir) in enumerate_namespace_dirs(data_dir)? {
        adopt_one_namespace_dir(catalog, &namespace, &ns_dir)?;
    }
    Ok(())
}

/// Footer-scan one namespace dir and persist its `namespaces` + `segments` rows.
///
/// Returns `true` when the namespace was adopted (had a readable segment),
/// `false` when the dir had no readable segment (skipped). NOT the live
/// registry — the caller's `rehydrate_from_catalog` reads these rows back and
/// builds the engine. The store-form schema already carries `seq` (recovered
/// from the footer).
///
/// EXCEPTION: the privileged `log` namespace's schema is NOT recovered from
/// parquet — `Store::ensure_log_namespace_registered` re-establishes its
/// canonical fixed schema (key_column="key") on every boot. Skipping the `log`
/// schema row here means rehydrate builds no log engine, so
/// ensure_log_namespace_registered's register path runs with the canonical
/// schema. The log SEGMENTS are still persisted, and the log engine's
/// `adopt_local_segments` picks them up from the catalog.
fn adopt_one_namespace_dir(
    catalog: &Catalog,
    namespace: &str,
    ns_dir: &Path,
) -> Result<bool, StatsError> {
    let is_log = namespace == LOG_NAMESPACE_NAME;

    // Adoption is a one-time bootstrap input. Once a table's version-0 history
    // has been imported into immutable objects and activated, its liveness comes
    // from the table state alone, and the parquet files left in its directory
    // are no longer a load source.
    if catalog.filesystem_adoption_disabled(namespace)? {
        tracing::debug!(
            namespace,
            "adopt: table completed its version-0 import; skipping filesystem adoption"
        );
        return Ok(false);
    }

    let schema = match recover_schema_from_segments(ns_dir) {
        Some(s) => s,
        None => {
            // No readable segment. A non-log namespace dir with no parquet
            // contributes nothing: the dir's existence alone is not enough to
            // recover a schema, so we skip — the log ns is re-established by
            // ensure_log_namespace_registered regardless.
            if !is_log {
                tracing::info!(
                    namespace,
                    "adopt: namespace dir has no readable segment; skipping"
                );
            }
            return Ok(false);
        }
    };

    if !is_log {
        catalog.upsert(namespace, &schema)?;
    }

    // Footer-scan with the namespace's effective key column. For `log` the
    // canonical key is "key" (a STRING column carrying no Int64 stats, so key
    // bounds stay None), which the heuristic schema also resolves to.
    let rows = adopt_namespace_from_disk(ns_dir, namespace, &schema);
    catalog.upsert_segments(&rows)?;

    tracing::info!(
        namespace,
        segments = rows.len(),
        next_seq = recover_next_seq(&rows),
        "adopt: rebuilt namespace from disk"
    );
    Ok(true)
}

/// Reconcile namespace dirs present on disk but ABSENT from the persisted
/// catalog, WITHOUT a full rescan.
///
/// Runs on every `done`-sentinel boot. The sentinel fast-path otherwise blocks
/// the disk scan forever, so a namespace an OLDER binary skipped — e.g. a
/// microsecond-timestamp namespace that `recover_schema_from_segments` rejected
/// before the `arrow_to_column_type` fix — would stay permanently invisible
/// even after the binary is fixed. This re-discovers exactly those: it skips
/// `log` (re-established separately) and every already-cataloged namespace, so
/// the steady state is one `readdir` + footer reads of ONLY the not-yet-known
/// dirs (the large `log`/iris.* namespaces are never re-scanned).
fn adopt_missing_namespaces(data_dir: &Path, catalog: &Catalog) -> Result<(), StatsError> {
    assert_namespaced_layout(data_dir)?;
    let known: std::collections::HashSet<String> =
        catalog.list_all()?.into_iter().map(|(n, _)| n).collect();
    for (namespace, ns_dir) in enumerate_namespace_dirs(data_dir)? {
        if namespace == LOG_NAMESPACE_NAME || known.contains(&namespace) {
            continue;
        }
        if adopt_one_namespace_dir(catalog, &namespace, &ns_dir)? {
            tracing::warn!(
                namespace,
                "adopt: reconciled a namespace present on disk but missing from the catalog \
                 (an older binary likely skipped it before the done sentinel was stamped)"
            );
        }
    }
    Ok(())
}

/// Boot orchestrator: ensure the catalog has been adopted from disk, exactly
/// once, before the server binds.
///
/// Sentinel state machine (`{data_dir}/.finelog-rust-catalog`):
/// - `done` -> fast path, the sqlite sidecar is authoritative; skip the scan.
/// - missing / `in-progress` / malformed -> (re)run the scan. The scan is
///   idempotent (the (dir, footer) -> row mapping is a pure function of the
///   on-disk files), so a crash mid-scan re-converges on the next boot — the
///   directory IS the journal.
///
/// In-memory mode (`data_dir = None`) is a no-op (no disk to adopt).
pub fn ensure_catalog_adopted(
    data_dir: Option<&Path>,
    catalog: &Catalog,
) -> Result<(), StatsError> {
    let Some(data_dir) = data_dir else {
        return Ok(());
    };
    if read_sentinel_state(data_dir) == Some(AdoptionState::Done) {
        tracing::debug!("catalog adoption: done sentinel; skipping full disk scan");
        // The sidecar is authoritative for known namespaces, but still
        // reconcile any namespace dir on disk that the catalog doesn't know
        // about — otherwise a namespace an older binary skipped stays invisible
        // forever (the sentinel blocks the full scan). Cheap: skips `log` and
        // every cataloged namespace, footer-scanning only unknown dirs.
        adopt_missing_namespaces(data_dir, catalog)?;
        return Ok(());
    }
    // Cheap top-level pre-flight BEFORE stamping in-progress: a flat (legacy)
    // layout is a hard error, and we don't want to leave a dangling
    // `in-progress` sentinel on a dir we never actually adopt. `adopt_store_
    // from_disk` re-asserts this (defense in depth + its direct-call test).
    assert_namespaced_layout(data_dir)?;
    let started_at = now_ms();
    write_sentinel(data_dir, AdoptionState::InProgress, started_at, None)?;
    adopt_store_from_disk(data_dir, catalog)?;
    write_sentinel(data_dir, AdoptionState::Done, started_at, Some(now_ms()))?;
    tracing::info!("catalog adoption: rebuilt from disk and stamped done");
    Ok(())
}

/// Delete `*.parquet.tmp` left behind by a segment write or layout rewrite that
/// died before its rename. Nothing references them: the catalog only ever names
/// the final path, and `discover_segments` ignores the extension, so a survivor
/// is disk the table's own byte accounting cannot see.
fn discard_staging_files(dir: &Path, namespace: &str) {
    for path in discover_files(dir) {
        if path.extension().and_then(|extension| extension.to_str()) != Some("tmp") {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(namespace, file = %path.display(),
                "discarded an abandoned staging file"),
            Err(error) => {
                tracing::warn!(namespace, file = %path.display(), %error,
                "could not discard an abandoned staging file")
            }
        }
    }
}

/// Adopt a table's segments at boot, reconciling catalog rows against local
/// files.
///
/// Two-pass reconcile:
/// - **Pass 1** walks existing catalog rows. A catalog row with a local file
///   present enters the view (a `REMOTE` row whose file reappeared collapses to
///   `BOTH`). A `LOCAL` row whose file vanished is dropped (data lost). A `BOTH`
///   row whose file vanished collapses to `REMOTE` (durable archive survives).
///   A `REMOTE`-only row stays in the catalog but NEVER enters the view (queries
///   don't see archived data; stats exclude it).
/// - **Pass 2** walks local files not seen in pass 1 — files with no catalog row
///   — and adopts them as `LOCAL`, EXCEPT one whose seq range the catalog already
///   covers. A file with no catalog row is either genuinely-new flushed data
///   whose catalog upsert had not yet run (adopt it: crash recovery) or a
///   compaction input the catalog has already superseded — its row replaced by
///   the merge output — but whose unlink has not yet run. Adopting the latter
///   resurrects a phantom segment whose file is about to vanish: a dangling
///   reference that wedges compaction (#7361). Monotonic seq allocation
///   separates the two — a genuine flush orphan always sits strictly ABOVE the
///   cataloged high-water seq, a superseded input at or below it — so pass 2
///   skips any file whose `min_seq` is not past every catalog row's `max_seq`.
///
/// The result is sorted by `min_seq` so iteration matches the planner's
/// oldest-first expectation. Catalog REMOTE rows are left untouched.
pub fn adopt_local_segments(
    dir: &Path,
    key_column: Option<&str>,
    catalog: &Catalog,
    namespace: &str,
    objects: Option<&dyn ObjectStore>,
) -> Result<VecDeque<LocalSegment>, StatsError> {
    let started = Instant::now();
    let mut segments: Vec<LocalSegment> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    discard_staging_files(dir, namespace);

    let status = catalog.table_spec_status(namespace)?;
    let object_records: HashMap<_, _> = catalog
        .object_segments(namespace)?
        .into_iter()
        .map(|record| (record.path.clone(), record))
        .collect();

    let discover_started = Instant::now();
    let mut local_files: HashMap<String, PathBuf> = discover_segments(dir)
        .into_iter()
        .map(|path| (path.to_string_lossy().into_owned(), path))
        .collect();
    for record in object_records.values() {
        let path = PathBuf::from(&record.path);
        if path.exists() {
            local_files.insert(record.path.clone(), path);
        }
    }
    let discover_ms = discover_started.elapsed().as_millis() as u64;

    // Pass 1: catalog rows.
    let catalog_started = Instant::now();
    let catalog_rows = catalog.list_segments(namespace)?;
    let catalog_read_ms = catalog_started.elapsed().as_millis() as u64;
    let footer_reconcile_started = Instant::now();
    // Highest seq the catalog still ACCOUNTS FOR after reconciliation: a local
    // file past it is genuinely new (an uncataloged flush); a file at or below it
    // whose row is gone is a compaction input the catalog already superseded (the
    // pass-2 skip). A `LOCAL` row whose file vanished is dropped by pass 1 below —
    // its data is lost — so it must not extend the cutoff, or a lower-seq on-disk
    // file it does not actually cover would be misread as superseded and skipped.
    // Every other row's range stays covered: adopted to the view, or kept as a
    // `REMOTE` / `BOTH` durable archive.
    let max_catalog_seq = catalog_rows
        .iter()
        .filter(|row| row.location != SegmentLocation::Local || local_files.contains_key(&row.path))
        .map(|row| row.max_seq)
        .max();
    for row in &catalog_rows {
        seen.insert(row.path.clone());
        let Some(local_path) = local_files.get(&row.path) else {
            // Local file gone.
            match row.location {
                SegmentLocation::Local => {
                    // No durable copy — drop the row.
                    catalog.remove_segment(namespace, &row.path)?;
                }
                SegmentLocation::Both => {
                    // Bucket copy is durable; collapse to REMOTE.
                    catalog.set_location(namespace, &row.path, SegmentLocation::Remote)?;
                }
                SegmentLocation::Remote => {}
            }
            continue;
        };
        let Some(meta) = read_segment_footer(local_path, key_column) else {
            continue;
        };
        let location = if row.location == SegmentLocation::Remote {
            SegmentLocation::Both
        } else {
            row.location
        };
        let query_visible = match object_records.get(&row.path) {
            Some(record) => object_segment_is_query_visible(&status, record),
            None => status.active_version() == 0,
        };
        if !query_visible {
            continue;
        }
        let size = std::fs::metadata(local_path)
            .map(|metadata| metadata.len() as i64)
            .unwrap_or(0);
        segments.push(LocalSegment {
            path: row.path.clone(),
            size_bytes: size,
            level: meta.level,
            min_seq: meta.min_seq,
            max_seq: meta.max_seq,
            row_count: meta.row_count,
            created_at_ms: row.created_at_ms,
            min_key_value: meta.min_key_value,
            max_key_value: meta.max_key_value,
            partition: meta.partition,
            location,
            artifacts: segment_artifacts(objects, object_records.get(&row.path), local_path)?,
        });
    }

    // Pass 2: local files with no catalog row -> fresh LOCAL segments.
    for (path_str, path) in &local_files {
        if seen.contains(path_str) {
            continue;
        }
        if status.active_version() > 0 {
            continue;
        }
        let Some(meta) = read_segment_footer(path, key_column) else {
            continue;
        };
        // A file with no catalog row whose seq range the catalog already covers
        // is a compaction input whose row the merge output replaced but whose
        // unlink has not yet run (a concurrent adopt caught the post-splice /
        // pre-unlink window). Adopting it would resurrect a phantom segment whose
        // file is about to vanish, wedging compaction (#7361). A genuine
        // uncataloged flush always sits strictly above the cataloged high-water
        // seq, so this only skips the superseded case.
        if let Some(max_seq) = max_catalog_seq {
            if meta.min_seq <= max_seq {
                tracing::warn!(
                    namespace,
                    path = %path_str,
                    file_min_seq = meta.min_seq,
                    file_max_seq = meta.max_seq,
                    max_catalog_seq = max_seq,
                    "skipping orphan segment file already covered by the catalog (superseded compaction input mid-unlink)"
                );
                continue;
            }
        }
        let size = std::fs::metadata(path)
            .map(|metadata| metadata.len() as i64)
            .unwrap_or(0);
        let created_at_ms = std::fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
            .map(|elapsed| elapsed.as_millis() as i64)
            .unwrap_or_else(now_ms);
        segments.push(LocalSegment {
            path: path_str.clone(),
            size_bytes: size,
            level: meta.level,
            min_seq: meta.min_seq,
            max_seq: meta.max_seq,
            row_count: meta.row_count,
            created_at_ms,
            min_key_value: meta.min_key_value,
            max_key_value: meta.max_key_value,
            partition: meta.partition,
            location: SegmentLocation::Local,
            // A file with no catalog row is not object-backed, so its sidecars
            // come from the local layout.
            artifacts: local_sidecar_artifacts(path),
        });
    }

    segments.sort_by_key(|segment| segment.min_seq);
    let adopted: VecDeque<LocalSegment> = segments.into();
    debug_assert_unique_paths(&adopted);
    tracing::info!(
        namespace,
        local_files = local_files.len(),
        catalog_rows = catalog_rows.len(),
        adopted_segments = adopted.len(),
        discover_ms,
        catalog_read_ms,
        footer_reconcile_ms = footer_reconcile_started.elapsed().as_millis() as u64,
        total_ms = started.elapsed().as_millis() as u64,
        "finelog local segment adoption complete"
    );
    Ok(adopted)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch, StringArray};
    // `Arc` is used by `worker_batch` (arrow ArrowSchema wrapping).
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::ram_buffer::stamp_seq_and_build;
    use crate::store::schema::schema_to_arrow;
    use crate::store::segment::write_segment_to_dir;
    use crate::store::table::test_tables::{aligned, worker_schema};
    use crate::store::types::NamespaceStats;

    fn tempdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_adopt_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Build a seq-stamped worker batch: [seq, worker_id, mem_bytes, timestamp_ms].
    /// `timestamp_ms` is the key column (Int64), values = keys.
    fn worker_batch(first_seq: i64, keys: Vec<i64>) -> RecordBatch {
        let n = keys.len() as i64;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("mem_bytes", DataType::Int64, false),
            Field::new("timestamp_ms", DataType::Int64, false),
        ]));
        let seqs: Int64Array = (first_seq..first_seq + n).collect();
        let ids: Vec<String> = (0..n).map(|i| format!("w{i}")).collect();
        let mems: Int64Array = (0..n).map(|i| 100 + i).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(seqs),
                Arc::new(StringArray::from(ids)),
                Arc::new(mems),
                Arc::new(Int64Array::from(keys)),
            ],
        )
        .unwrap()
    }

    fn worker_store_schema() -> Schema {
        Schema::new(
            vec![
                Column::new("seq", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("mem_bytes", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        )
    }

    // ----- adopt:: -------------------------------------------------------

    #[test]
    fn adopt_namespace_from_disk_sorts_and_fills_fields() {
        let ns_dir = tempdir("ns");
        // Two segments out of filename order: seqs 4..5, then 1..3.
        write_segment_to_dir(&ns_dir, 0, 4, &worker_batch(4, vec![40, 50])).unwrap();
        write_segment_to_dir(&ns_dir, 0, 1, &worker_batch(1, vec![10, 20, 30])).unwrap();

        let rows = adopt_namespace_from_disk(&ns_dir, "iris.worker", &worker_store_schema());
        assert_eq!(rows.len(), 2);
        // Sorted by min_seq.
        assert_eq!(rows[0].min_seq, 1);
        assert_eq!(rows[0].max_seq, 3);
        assert_eq!(rows[0].row_count, 3);
        assert_eq!(rows[1].min_seq, 4);
        assert_eq!(rows[1].max_seq, 5);
        assert_eq!(rows[1].row_count, 2);
        for r in &rows {
            assert_eq!(r.location, SegmentLocation::Local);
            assert!(r.byte_size > 0);
            assert!(r.created_at_ms > 0);
        }
        // Key bounds from timestamp_ms stats (stringified).
        assert_eq!(rows[0].min_key_value.as_deref(), Some("10"));
        assert_eq!(rows[0].max_key_value.as_deref(), Some("30"));

        std::fs::remove_dir_all(&ns_dir).ok();
    }

    #[test]
    fn recover_next_seq_floor_and_max() {
        assert_eq!(recover_next_seq(&[]), 1);
        let rows = vec![
            SegmentRow {
                namespace: "n".into(),
                path: "p1".into(),
                level: 0,
                min_seq: 1,
                max_seq: 3,
                row_count: 3,
                byte_size: 1,
                created_at_ms: 1,
                min_key_value: None,
                max_key_value: None,
                partition: None,
                location: SegmentLocation::Local,
            },
            SegmentRow {
                namespace: "n".into(),
                path: "p2".into(),
                level: 0,
                min_seq: 4,
                max_seq: 5,
                row_count: 2,
                byte_size: 1,
                created_at_ms: 1,
                min_key_value: None,
                max_key_value: None,
                partition: None,
                location: SegmentLocation::Local,
            },
        ];
        assert_eq!(recover_next_seq(&rows), 6);
    }

    #[test]
    fn adopt_store_aggregates_match_aggregate_namespace_stats() {
        let data_dir = tempdir("store");
        let ns_dir = data_dir.join("iris.worker");
        std::fs::create_dir_all(&ns_dir).unwrap();
        write_segment_to_dir(&ns_dir, 0, 1, &worker_batch(1, vec![10, 20, 30])).unwrap();
        write_segment_to_dir(&ns_dir, 0, 4, &worker_batch(4, vec![40, 50])).unwrap();

        let catalog = Catalog::open(Some(&data_dir)).unwrap();
        adopt_store_from_disk(&data_dir, &catalog).unwrap();

        // Hand-computed aggregates vs the catalog's aggregate_namespace_stats.
        let stats = catalog.aggregate_namespace_stats("iris.worker").unwrap();
        assert_eq!(
            stats,
            NamespaceStats {
                row_count: 5,
                byte_size: stats.byte_size, // writer-dependent; checked > 0 below
                min_seq: 1,
                max_seq: 5,
                segment_count: 2,
            }
        );
        assert!(stats.byte_size > 0);

        // The namespaces row was persisted so rehydrate can pick it up.
        let all = catalog.list_all().unwrap();
        assert!(all.iter().any(|(n, _)| n == "iris.worker"));

        std::fs::remove_dir_all(&data_dir).ok();
    }

    // ----- adopt::schema -------------------------------------------------

    #[test]
    fn adopt_schema_recovers_columns_and_implicit_seq() {
        let ns_dir = tempdir("schema");
        write_segment_to_dir(&ns_dir, 0, 1, &worker_batch(1, vec![10, 20, 30])).unwrap();

        let recovered = recover_schema_from_segments(&ns_dir).unwrap();
        // seq is first and re-marked Int64 non-nullable.
        assert_eq!(recovered.columns[0].name, "seq");
        assert_eq!(recovered.columns[0].r#type, ColumnType::COLUMN_TYPE_INT64);
        assert!(!recovered.columns[0].nullable);
        let names: Vec<&str> = recovered.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["seq", "worker_id", "mem_bytes", "timestamp_ms"]);
        // timestamp_ms present -> default-resolvable -> key_column left empty.
        assert_eq!(recovered.key_column, "");
        // resolve_key_column resolves the adopted schema to the default key.
        assert_eq!(resolve_key_column(&recovered).unwrap(), "timestamp_ms");

        std::fs::remove_dir_all(&ns_dir).ok();
    }

    #[test]
    fn adopt_schema_none_for_empty_dir() {
        let ns_dir = tempdir("emptyschema");
        assert!(recover_schema_from_segments(&ns_dir).is_none());
        std::fs::remove_dir_all(&ns_dir).ok();
    }

    // ----- adopt::sentinel -----------------------------------------------

    #[test]
    fn ensure_catalog_adopted_writes_done_then_fast_paths() {
        let data_dir = tempdir("sentinel");
        let ns_dir = data_dir.join("iris.worker");
        std::fs::create_dir_all(&ns_dir).unwrap();
        write_segment_to_dir(&ns_dir, 0, 1, &worker_batch(1, vec![10, 20, 30])).unwrap();
        write_segment_to_dir(&ns_dir, 0, 4, &worker_batch(4, vec![40, 50])).unwrap();

        let catalog = Catalog::open(Some(&data_dir)).unwrap();
        // Cold start: sentinel missing.
        assert_eq!(read_sentinel_state(&data_dir), None);
        ensure_catalog_adopted(Some(&data_dir), &catalog).unwrap();
        assert_eq!(read_sentinel_state(&data_dir), Some(AdoptionState::Done));
        let after_first = catalog.aggregate_namespace_stats("iris.worker").unwrap();
        assert_eq!(after_first.segment_count, 2);

        // Second boot on the done sentinel: scan skipped (no new rows even if we
        // add a parquet, because the fast path returns before scanning).
        write_segment_to_dir(&ns_dir, 0, 6, &worker_batch(6, vec![60])).unwrap();
        let catalog2 = Catalog::open(Some(&data_dir)).unwrap();
        ensure_catalog_adopted(Some(&data_dir), &catalog2).unwrap();
        let after_second = catalog2.aggregate_namespace_stats("iris.worker").unwrap();
        // The sidecar persisted 2 segments; the fast path did not rescan, so the
        // new (untracked) segment is invisible until a real re-adoption.
        assert_eq!(after_second.segment_count, 2);

        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn ensure_catalog_adopted_reconciles_missing_namespace_on_done() {
        // Regression: a namespace dir present on disk but absent from the catalog
        // (e.g. one an older binary skipped) must be re-adopted on a done-sentinel
        // boot — otherwise the sentinel hides it forever.
        let data_dir = tempdir("reconcile");
        let worker_dir = data_dir.join("iris.worker");
        std::fs::create_dir_all(&worker_dir).unwrap();
        write_segment_to_dir(&worker_dir, 0, 1, &worker_batch(1, vec![10, 20, 30])).unwrap();

        // First boot: adopts iris.worker, stamps done.
        let catalog = Catalog::open(Some(&data_dir)).unwrap();
        ensure_catalog_adopted(Some(&data_dir), &catalog).unwrap();
        assert_eq!(read_sentinel_state(&data_dir), Some(AdoptionState::Done));

        // A namespace dir the catalog never learned about appears on disk.
        let probes_dir = data_dir.join("infra.canary.probes");
        std::fs::create_dir_all(&probes_dir).unwrap();
        write_segment_to_dir(&probes_dir, 0, 1, &worker_batch(1, vec![40, 50])).unwrap();

        // Second boot on the done sentinel: the missing namespace is reconciled
        // in (the full scan stays skipped).
        let catalog2 = Catalog::open(Some(&data_dir)).unwrap();
        ensure_catalog_adopted(Some(&data_dir), &catalog2).unwrap();
        let stats = catalog2
            .aggregate_namespace_stats("infra.canary.probes")
            .unwrap();
        assert_eq!(stats.segment_count, 1);
        assert_eq!(stats.row_count, 2);
        // The already-known namespace is untouched.
        assert!(catalog2
            .list_all()
            .unwrap()
            .iter()
            .any(|(n, _)| n == "iris.worker"));

        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn catalog_adoption_never_treats_object_cache_as_a_namespace() {
        let data_dir = tempdir("object_cache_reserved");
        let worker_dir = data_dir.join("iris.worker");
        std::fs::create_dir_all(&worker_dir).unwrap();
        write_segment_to_dir(&worker_dir, 0, 1, &worker_batch(1, vec![10])).unwrap();
        let object_cache = data_dir.join("_finelog/tables/iris.worker/objects/v1/backfill/source");
        std::fs::create_dir_all(&object_cache).unwrap();
        write_segment_to_dir(&object_cache, 0, 2, &worker_batch(2, vec![20])).unwrap();

        let catalog = Catalog::open(Some(&data_dir)).unwrap();
        ensure_catalog_adopted(Some(&data_dir), &catalog).unwrap();
        let namespaces = catalog
            .list_all()
            .unwrap()
            .into_iter()
            .map(|(namespace, _)| namespace)
            .collect::<Vec<_>>();
        assert!(namespaces
            .iter()
            .any(|namespace| namespace == "iris.worker"));
        assert!(!namespaces.iter().any(|namespace| namespace == "_finelog"));

        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn ensure_catalog_adopted_in_progress_reruns_idempotently() {
        let data_dir = tempdir("inprogress");
        let ns_dir = data_dir.join("iris.worker");
        std::fs::create_dir_all(&ns_dir).unwrap();
        write_segment_to_dir(&ns_dir, 0, 1, &worker_batch(1, vec![10, 20, 30])).unwrap();

        // Simulate a crash mid-scan: stamp in-progress, no catalog rows.
        write_sentinel(&data_dir, AdoptionState::InProgress, now_ms(), None).unwrap();
        let catalog = Catalog::open(Some(&data_dir)).unwrap();
        ensure_catalog_adopted(Some(&data_dir), &catalog).unwrap();
        assert_eq!(read_sentinel_state(&data_dir), Some(AdoptionState::Done));
        let stats = catalog.aggregate_namespace_stats("iris.worker").unwrap();
        assert_eq!(stats.segment_count, 1);
        assert_eq!(stats.row_count, 3);

        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn ensure_catalog_adopted_malformed_sentinel_treated_as_missing() {
        let data_dir = tempdir("malformed");
        let ns_dir = data_dir.join("iris.worker");
        std::fs::create_dir_all(&ns_dir).unwrap();
        write_segment_to_dir(&ns_dir, 0, 1, &worker_batch(1, vec![10])).unwrap();
        std::fs::write(data_dir.join(SENTINEL_FILENAME), b"{not json").unwrap();
        assert_eq!(read_sentinel_state(&data_dir), None);

        let catalog = Catalog::open(Some(&data_dir)).unwrap();
        ensure_catalog_adopted(Some(&data_dir), &catalog).unwrap();
        assert_eq!(read_sentinel_state(&data_dir), Some(AdoptionState::Done));

        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn adopt_store_flat_layout_is_a_hard_error() {
        let data_dir = tempdir("flat");
        // A top-level parquet => legacy flat layout.
        std::fs::write(
            data_dir.join("seg_L0_0000000000000000001.parquet"),
            b"not-real-parquet",
        )
        .unwrap();
        let catalog = Catalog::open(Some(&data_dir)).unwrap();
        let err = adopt_store_from_disk(&data_dir, &catalog).unwrap_err();
        match err {
            StatsError::Internal(msg) => assert!(msg.contains("flat layout"), "msg: {msg}"),
            other => panic!("expected Internal flat-layout error, got {other:?}"),
        }
        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn adopts_microsecond_timestamp_namespace() {
        use arrow::array::TimestampMicrosecondArray;
        use arrow::datatypes::TimeUnit;

        let data_dir = tempdir("us_ts");
        let ns_dir = data_dir.join("infra.canary.probes");
        std::fs::create_dir_all(&ns_dir).unwrap();

        // A segment whose timestamp column is microsecond — the legacy
        // duckdb-written form. The adopter must accept it (and report the logical
        // TIMESTAMP_MS type) rather than dropping the whole namespace.
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new(
                "started_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(Int64Array::from(vec![1_i64, 2])),
                Arc::new(StringArray::from(vec![
                    "controller-ping",
                    "iris-job-submit",
                ])),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    1_780_178_046_162_000_i64,
                    1_780_178_106_596_000,
                ])),
            ],
        )
        .unwrap();
        write_segment_to_dir(&ns_dir, 1, 1, &batch).unwrap();

        let recovered =
            recover_schema_from_segments(&ns_dir).expect("microsecond ts segment must adopt");
        assert_eq!(
            recovered.column("started_at").unwrap().r#type,
            ColumnType::COLUMN_TYPE_TIMESTAMP_MS
        );

        // End-to-end: the namespace is adopted with its segment, not skipped.
        let catalog = Catalog::open(Some(&data_dir)).unwrap();
        ensure_catalog_adopted(Some(&data_dir), &catalog).unwrap();
        let stats = catalog
            .aggregate_namespace_stats("infra.canary.probes")
            .unwrap();
        assert_eq!(stats.segment_count, 1);
        assert_eq!(stats.row_count, 2);

        std::fs::remove_dir_all(&data_dir).ok();
    }

    #[test]
    fn in_memory_mode_is_noop() {
        let catalog = Catalog::open(None).unwrap();
        ensure_catalog_adopted(None, &catalog).unwrap();
    }

    // ----- adopt_local_segments ------------------------------------------

    /// Write a `seg_L{level}_{first_seq}.parquet` of `n` worker rows to `dir` and
    /// return its path. Used to stage the on-disk state adoption reconciles.
    fn write_seg(dir: &Path, level: i32, first_seq: i64, n: i64) -> PathBuf {
        let arrow = schema_to_arrow(&worker_schema());
        let batch = stamp_seq_and_build(&aligned(n), first_seq, &arrow);
        write_segment_to_dir(dir, level, first_seq, &batch)
            .unwrap()
            .0
    }

    /// A `LOCAL` catalog `SegmentRow` for `path`. Key bounds are re-read from the
    /// file footer during adoption, so they are left `None` here.
    fn seg_row(path: &Path, level: i32, min_seq: i64, max_seq: i64) -> SegmentRow {
        SegmentRow {
            namespace: "iris.task".to_string(),
            path: path.to_string_lossy().into_owned(),
            level,
            min_seq,
            max_seq,
            row_count: max_seq - min_seq + 1,
            byte_size: 1,
            created_at_ms: 1,
            min_key_value: None,
            max_key_value: None,
            partition: None,
            location: SegmentLocation::Local,
        }
    }

    #[test]
    fn adopt_skips_superseded_compaction_input_still_on_disk() {
        // Regression for #7361. A compaction had committed its catalog splice —
        // `replace_segments` swapped the L0 input rows for the merged L1 output —
        // but had not yet unlinked the input files when adoption ran (a
        // re-register replacing the engine, or a crash, caught the post-splice /
        // pre-unlink window). Pass 2 must not resurrect those inputs as phantom
        // segments whose files are about to vanish, while still adopting a genuine
        // uncataloged flush orphan.
        let dir = tempdir("phantom");
        let table_dir = dir.join("iris.task");
        std::fs::create_dir_all(&table_dir).unwrap();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());

        // The committed merge output L1 [1..4]: on disk AND in the catalog.
        let l1 = write_seg(&table_dir, 1, 1, 4);
        catalog.upsert_segment(&seg_row(&l1, 1, 1, 4)).unwrap();

        // A superseded L0 input [1..2]: still on disk, its catalog row already
        // gone. Its seq range is covered by the L1 output — the phantom.
        let l0_input = write_seg(&table_dir, 0, 1, 2);

        // A genuine fresh flush orphan [5..6]: file written, catalog upsert not
        // yet run. It sits above the cataloged high-water seq (4) — adopt it.
        let l0_new = write_seg(&table_dir, 0, 5, 2);

        let deque = adopt_local_segments(
            &table_dir,
            Some("timestamp_ms"),
            &catalog,
            "iris.task",
            None,
        )
        .unwrap();
        let has = |p: &Path| deque.iter().any(|s| s.path == p.to_string_lossy());

        assert!(has(&l1), "the merge output is adopted");
        assert!(
            has(&l0_new),
            "a genuine flush orphan above the cataloged high-water seq is adopted"
        );
        assert!(
            !has(&l0_input),
            "the superseded compaction input must NOT be resurrected as a phantom"
        );
        assert_eq!(deque.len(), 2, "only the output and the genuine orphan");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn adopt_recovers_low_orphan_under_a_stale_high_catalog_row() {
        // The coverage cutoff must come only from rows whose data survives pass 1.
        // A stale LOCAL catalog row whose file is gone is dropped (its data lost),
        // so it must not extend the cutoff and mask a lower-seq on-disk file that
        // it never actually covered — that file is a recoverable orphan.
        let dir = tempdir("low_orphan");
        let table_dir = dir.join("iris.task");
        std::fs::create_dir_all(&table_dir).unwrap();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());

        // A stale LOCAL row [100..200] whose parquet was never written to disk.
        let gone = table_dir.join("seg_L0_00000000000000000100.parquet");
        catalog
            .upsert_segment(&seg_row(&gone, 0, 100, 200))
            .unwrap();

        // A real on-disk orphan [1..2], lower seq than the stale row, no catalog
        // row. It must be recovered, not skipped as covered.
        let orphan = write_seg(&table_dir, 0, 1, 2);

        let deque = adopt_local_segments(
            &table_dir,
            Some("timestamp_ms"),
            &catalog,
            "iris.task",
            None,
        )
        .unwrap();
        let has = |p: &Path| deque.iter().any(|s| s.path == p.to_string_lossy());
        assert!(
            has(&orphan),
            "a low-seq orphan is recovered even under a stale high catalog row"
        );
        assert_eq!(deque.len(), 1, "only the recovered orphan");
        std::fs::remove_dir_all(&dir).ok();
    }
}
