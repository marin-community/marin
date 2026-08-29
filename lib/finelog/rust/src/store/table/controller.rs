//! Private per-table durable-state controller.
//!
//! One controller owns everything that changes a table's durable state: the
//! writer fence, the selected [`StoredTableState`], the canonical object writes,
//! the local projection rebuild, and the published [`TableSnapshot`]. It is the
//! only publisher of a table's HEAD.
//!
//! Two entry points reach it, and both are serialized:
//!
//! - `mutation_gate` serializes the short synchronous transaction that allocates
//!   the next [`TableRevision`]. It is a plain lock because that transaction
//!   never awaits.
//! - the controller task is the only code that publishes state. Callers request
//!   publication over a mailbox and await the outcome, so concurrent flushes,
//!   compactions, migrations, and cursor advances never race for HEAD.
//!
//! Heavy work — Parquet encoding, index building, compaction merges — stays with
//! its caller. The controller serializes lease creation and the short commit
//! only. Appends never reach it: the RAM buffer's short lock is the ingest path.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use bytes::Bytes;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::sync::{mpsc, oneshot, watch};

use crate::errors::StatsError;
use crate::proto::finelog::stats::ObjectRef;
use crate::store::catalog::object_state_store::OBJECTS_PREFIX;
use crate::store::catalog::projection::namespace_catalog;
use crate::store::catalog::state_store::{StoredTableState, TableStateStore};
use crate::store::catalog::{Catalog, ObjectSegmentRecord, TableSpecStatus};
use crate::store::object_store::{
    ObjectByteStream, ObjectId, ObjectReference, ObjectStore, ObjectVersion,
};
use crate::store::table_state::{
    resolve_publication, ArtifactReferences, CommitError, CommitToken, Committed, LocalArtifacts,
    TableRevision, TableSnapshot, TableState, WriterFence,
};
use crate::store::types::{segment_relative_key, SegmentRow};

/// An immutable object this table just wrote, and the reference that names it.
pub struct WrittenObject {
    pub path: PathBuf,
    pub source: ObjectRef,
    pub byte_size: i64,
}

/// Object storage for one object-backed table.
pub struct ObjectPersistence {
    pub table_dir: PathBuf,
    pub store: Arc<dyn ObjectStore>,
    pub legacy_store: Arc<dyn ObjectStore>,
    pub state_store: Arc<dyn TableStateStore>,
}

/// Permission to run one compaction and commit its replacement.
///
/// The lease pins the definition version and the exact inputs the work reads.
/// It does not pin the table revision, so ordinary flushes and cursor advances
/// commit freely while the work runs; the commit is rebased onto whatever state
/// is current and rejected only when an input stopped being live, the definition
/// version moved, or this writer was fenced.
#[derive(Clone, Debug)]
pub struct MaintenanceLease {
    fence: WriterFence,
    definition_version: u64,
    inputs: Vec<String>,
}

impl MaintenanceLease {
    /// The exact immutable inputs this lease replaces.
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }
}

/// Work only the controller task performs.
enum ControllerCommand {
    Publish(oneshot::Sender<Result<Arc<TableSnapshot>, CommitError>>),
    PublishOwed(oneshot::Sender<Result<(), StatsError>>),
    Claim(oneshot::Sender<Result<(), StatsError>>),
    Tombstone(oneshot::Sender<Result<(), StatsError>>),
    GcStates {
        now_ms: i64,
        state_retention_ms: u64,
        orphan_grace_ms: u64,
        reply: oneshot::Sender<Result<usize, StatsError>>,
    },
}

const COMMAND_QUEUE_DEPTH: usize = 32;

/// Chunk size an artifact or compaction output streams to the object store in.
const UPLOAD_CHUNK_BYTES: usize = 8 * 1024 * 1024;

pub struct TableController {
    table: String,
    catalog: Arc<Catalog>,
    fence: WriterFence,
    /// Present exactly for object-backed tables. A legacy table commits to the
    /// local catalog and publishes no HEAD.
    objects: Option<ObjectPersistence>,
    /// Serializes the revision-allocating transaction. Held only across the
    /// synchronous mutation, never across I/O.
    mutation_gate: Mutex<()>,
    /// The state this writer last observed selected, and the token it presents
    /// on its next commit. Absent until this process claims the table.
    selected: Mutex<Option<StoredTableState>>,
    /// Set once this process owns durable writes for the table.
    claimed: AtomicBool,
    /// Cleared when another writer fences this one. A fenced table stops
    /// accepting writes; only a restart can re-claim it.
    writes_ready: AtomicBool,
    /// Set while a locally committed revision is not known to be published.
    publication_owed: AtomicBool,
    /// The latest committed state, republished after every transition.
    snapshot: watch::Sender<Option<Arc<TableSnapshot>>>,
    commands: Option<mpsc::Sender<ControllerCommand>>,
}

impl TableController {
    /// Build a controller and, for an object-backed table, start its task.
    ///
    /// A legacy table needs no task: it publishes no HEAD, so its only durable
    /// transition is the gated local mutation.
    pub fn start(
        table: String,
        catalog: Arc<Catalog>,
        objects: Option<ObjectPersistence>,
        fence: WriterFence,
    ) -> Arc<Self> {
        let (snapshot, _) = watch::channel(None);
        let object_backed = objects.is_some();
        let (sender, receiver) = mpsc::channel(COMMAND_QUEUE_DEPTH);
        let controller = Arc::new(Self {
            table,
            catalog,
            fence,
            objects,
            mutation_gate: Mutex::new(()),
            selected: Mutex::new(None),
            claimed: AtomicBool::new(false),
            writes_ready: AtomicBool::new(true),
            publication_owed: AtomicBool::new(false),
            snapshot,
            commands: object_backed.then(|| sender.clone()),
        });
        if object_backed {
            drop(sender);
            tokio::spawn(run_controller(Arc::downgrade(&controller), receiver));
        }
        controller
    }

    pub fn is_object_backed(&self) -> bool {
        self.objects.is_some()
    }

    /// Whether this writer still owns the table's durable state.
    pub fn writes_ready(&self) -> bool {
        self.writes_ready.load(Ordering::SeqCst)
    }

    /// Stop accepting writes for this table until a restart recovers it.
    pub fn mark_unready(&self, reason: &str) {
        self.writes_ready.store(false, Ordering::SeqCst);
        tracing::error!(
            table = %self.table,
            fence = %self.fence,
            reason,
            "table stops accepting writes"
        );
    }

    /// The latest published state, or `None` before the first transition.
    pub fn snapshot(&self) -> Option<Arc<TableSnapshot>> {
        self.snapshot.borrow().clone()
    }

    /// Follow this table's published state.
    pub fn watch_snapshot(&self) -> watch::Receiver<Option<Arc<TableSnapshot>>> {
        self.snapshot.subscribe()
    }

    /// The object store holding this table's immutable data and artifacts.
    pub fn object_store(&self) -> Option<&Arc<dyn ObjectStore>> {
        self.objects.as_ref().map(|objects| &objects.store)
    }

    pub fn legacy_store(&self) -> Option<Arc<dyn ObjectStore>> {
        self.objects
            .as_ref()
            .map(|objects| Arc::clone(&objects.legacy_store))
    }

    fn require_objects(&self) -> Result<&ObjectPersistence, StatsError> {
        self.objects.as_ref().ok_or_else(|| {
            StatsError::Internal(format!("table {:?} is not object-backed", self.table))
        })
    }

    fn require_commands(&self) -> Result<&mpsc::Sender<ControllerCommand>, StatsError> {
        self.commands.as_ref().ok_or_else(|| {
            StatsError::Internal(format!("table {:?} has no controller task", self.table))
        })
    }

    async fn dispatch<T>(
        &self,
        build: impl FnOnce(oneshot::Sender<T>) -> ControllerCommand,
    ) -> Result<T, StatsError> {
        let (reply, response) = oneshot::channel();
        self.require_commands()?
            .send(build(reply))
            .await
            .map_err(|_| {
                StatsError::Internal(format!("controller for {:?} stopped", self.table))
            })?;
        response.await.map_err(|_| {
            StatsError::Internal(format!("controller for {:?} dropped a reply", self.table))
        })
    }

    /// Record that a committed revision must still reach HEAD.
    pub fn mark_publication_owed(&self) {
        if self.is_object_backed() {
            self.publication_owed.store(true, Ordering::SeqCst);
        }
    }

    pub fn publication_owed(&self) -> bool {
        self.publication_owed.load(Ordering::SeqCst)
    }

    /// Take ownership of durable writes before the first commit.
    pub async fn claim_writer(&self) -> Result<(), StatsError> {
        if self.claimed.load(Ordering::SeqCst) || !self.is_object_backed() {
            return Ok(());
        }
        self.dispatch(ControllerCommand::Claim).await?
    }

    /// Adopt a state a bootstrap claim already selected.
    ///
    /// Recovery loads and claims each head before any table accepts writes; this
    /// seeds the controller with that outcome and publishes the table's initial
    /// snapshot without a second round trip to the state store.
    pub fn adopt_claimed(&self, claimed: StoredTableState) {
        let snapshot = TableSnapshot::from_stored(&claimed);
        *self.selected.lock().unwrap() = Some(claimed);
        self.claimed.store(true, Ordering::SeqCst);
        self.snapshot.send_replace(Some(Arc::new(snapshot)));
    }

    /// Apply one durable state transition and publish the resulting state.
    ///
    /// `mutation` runs the transaction that allocates the next revision under
    /// the mutation gate. For an object-backed table the committed state is then
    /// published by the controller task as the second half of one ordered
    /// operation, and an unresolved publication is settled against HEAD.
    pub async fn commit<T, F>(&self, mutation: F) -> Result<Committed<T>, CommitError>
    where
        F: FnOnce() -> Result<(TableRevision, T), StatsError>,
    {
        let (previous, revision, output) = self.apply(mutation)?;
        if !self.is_object_backed() {
            return Ok(Committed {
                token: CommitToken::local(revision, WriterFence::UNCLAIMED),
                output,
            });
        }
        if revision == previous && !self.publication_owed() {
            return Ok(Committed {
                token: CommitToken::local(revision, self.fence),
                output,
            });
        }
        let published = self.publish_state().await?;
        Ok(Committed {
            token: published.token(),
            output,
        })
    }

    /// Apply one durable state transition from a synchronous caller.
    ///
    /// The committed revision is owed to the table's maintenance loop, which
    /// publishes it — or a later revision containing it — through
    /// [`TableController::publish_owed`]. The locally durable state is published
    /// as a snapshot immediately, carrying a local commit token.
    pub fn commit_owing_publication<T, F>(&self, mutation: F) -> Result<Committed<T>, CommitError>
    where
        F: FnOnce() -> Result<(TableRevision, T), StatsError>,
    {
        let (previous, revision, output) = self.apply(mutation)?;
        let fence = if self.is_object_backed() {
            if revision > previous {
                self.publish_local_snapshot(revision);
            }
            self.fence
        } else {
            WriterFence::UNCLAIMED
        };
        Ok(Committed {
            token: CommitToken::local(revision, fence),
            output,
        })
    }

    /// Run the mutation under the gate and enforce revision monotonicity.
    ///
    /// A revision that advances is owed to HEAD from the moment it is durable
    /// locally, so a failure between here and publication republishes the same
    /// revision instead of undoing it.
    fn apply<T, F>(&self, mutation: F) -> Result<(TableRevision, TableRevision, T), CommitError>
    where
        F: FnOnce() -> Result<(TableRevision, T), StatsError>,
    {
        let _gate = self.mutation_gate.lock().unwrap();
        let previous = self.local_revision().map_err(CommitError::NotCommitted)?;
        let (revision, output) = mutation().map_err(CommitError::NotCommitted)?;
        assert!(
            revision >= previous,
            "table {:?} revision moved backwards from {previous} to {revision}",
            self.table
        );
        if revision > previous {
            self.mark_publication_owed();
        }
        Ok((previous, revision, output))
    }

    fn local_revision(&self) -> Result<TableRevision, StatsError> {
        Ok(TableRevision::new(
            self.catalog
                .table_spec_status(&self.table)?
                .catalog_generation,
        ))
    }

    /// Publish the locally durable state that a synchronous commit produced.
    ///
    /// The projection is skipped when it cannot be built yet — a first
    /// registration commits its specification before the state is complete.
    fn publish_local_snapshot(&self, revision: TableRevision) {
        let Some(objects) = &self.objects else {
            return;
        };
        let Ok(catalog) = namespace_catalog(&self.catalog, &self.table, &objects.table_dir) else {
            return;
        };
        let snapshot = TableSnapshot::new(
            TableState::new(catalog),
            CommitToken::local(revision, self.fence),
        );
        self.snapshot.send_replace(Some(Arc::new(snapshot)));
    }

    /// Publish the locally durable state as this table's first snapshot.
    ///
    /// A restart has local state before it has claimed HEAD, and readers need a
    /// pinned state to plan from. Recovery replaces this with the state the
    /// writer claim selects. A table that has already published keeps its
    /// snapshot.
    pub fn seed_local_snapshot(&self) {
        if !self.is_object_backed() || self.snapshot.borrow().is_some() {
            return;
        }
        let Ok(revision) = self.local_revision() else {
            return;
        };
        self.publish_local_snapshot(revision);
    }

    /// Publish the current local table state and settle the outcome.
    ///
    /// Returns the state HEAD selects, whose revision is at least the caller's
    /// committed revision.
    pub async fn publish_state(&self) -> Result<Arc<TableSnapshot>, CommitError> {
        self.mark_publication_owed();
        match self.dispatch(ControllerCommand::Publish).await {
            Ok(result) => result,
            Err(error) => Err(CommitError::PublicationDeferred(error)),
        }
    }

    /// Publish an owed revision. A table with nothing owed is already current.
    pub async fn publish_owed(&self) -> Result<(), StatsError> {
        if !self.publication_owed() {
            return Ok(());
        }
        self.dispatch(ControllerCommand::PublishOwed).await?
    }

    /// Publish a revision marking the table deleted.
    pub async fn tombstone(&self) -> Result<(), StatsError> {
        self.dispatch(ControllerCommand::Tombstone).await?
    }

    /// Remove superseded state documents and unreferenced objects.
    pub async fn gc_published(
        &self,
        now_ms: i64,
        state_retention_ms: u64,
        orphan_grace_ms: u64,
    ) -> Result<usize, StatsError> {
        self.dispatch(|reply| ControllerCommand::GcStates {
            now_ms,
            state_retention_ms,
            orphan_grace_ms,
            reply,
        })
        .await?
    }

    /// Take a lease over `inputs` for one compaction.
    ///
    /// The merge and encode run outside the controller; only the lease and the
    /// eventual commit are serialized here.
    pub fn begin_compaction(&self, inputs: Vec<String>) -> Result<MaintenanceLease, StatsError> {
        if !self.writes_ready() {
            return Err(StatsError::SchemaConflict(format!(
                "table {:?} is fenced by another writer",
                self.table
            )));
        }
        Ok(MaintenanceLease {
            fence: self.fence,
            definition_version: self
                .catalog
                .table_spec_status(&self.table)?
                .active_version(),
            inputs,
        })
    }

    /// Commit a compaction result, rebasing it onto the current state.
    ///
    /// The lease is rejected when this writer was fenced or the table's active
    /// definition version moved while the work ran. Input liveness is checked by
    /// `mutation`, which runs inside the same gated transaction that replaces
    /// them.
    pub async fn commit_maintenance<T, F>(
        &self,
        lease: &MaintenanceLease,
        mutation: F,
    ) -> Result<Committed<T>, CommitError>
    where
        F: FnOnce() -> Result<(TableRevision, T), StatsError>,
    {
        if lease.fence != self.fence || !self.writes_ready() {
            return Err(CommitError::Fenced(StatsError::SchemaConflict(format!(
                "table {:?} no longer accepts writes from writer {}",
                self.table, lease.fence
            ))));
        }
        let active = self
            .catalog
            .table_spec_status(&self.table)
            .map_err(CommitError::NotCommitted)?
            .active_version();
        if active != lease.definition_version {
            return Err(CommitError::NotCommitted(StatsError::SchemaConflict(
                format!(
                    "table {:?} moved from definition version {} to {active} while compaction ran",
                    self.table, lease.definition_version
                ),
            )));
        }
        self.commit(mutation).await
    }

    /// Write an immutable content-addressed Parquet object for this table.
    pub async fn write_parquet(&self, bytes: Bytes) -> Result<WrittenObject, StatsError> {
        let objects = self.require_objects()?;
        let sha256: [u8; 32] = Sha256::digest(&bytes).into();
        let id = ObjectId::table(
            &self.table,
            &format!("{OBJECTS_PREFIX}/{}.parquet", crate::hex::encode(&sha256)),
        )?;
        let version = objects.store.write(&id, bytes).await?;
        let reference = ObjectReference {
            id: id.clone(),
            version: version.clone(),
        };
        let path = objects.store.local_path(&reference).await?;
        Ok(WrittenObject {
            path,
            source: object_ref(&id, &version),
            byte_size: i64::try_from(version.byte_size).unwrap_or(i64::MAX),
        })
    }

    /// Upload a staged local file as an immutable content-addressed object.
    ///
    /// `kind` selects the object prefix (`objects`, `indices`, `projections`)
    /// and `extension` its suffix. The bytes stream from the file rather than
    /// being buffered whole, so a compaction output never sits in RAM twice.
    pub async fn write_staged_object(
        &self,
        kind: &str,
        extension: &str,
        staged: &Path,
    ) -> Result<WrittenObject, StatsError> {
        let objects = self.require_objects()?;
        let sha256 = file_sha256(staged)?;
        let id = ObjectId::table(
            &self.table,
            &format!("{kind}/{}.{extension}", crate::hex::encode(&sha256)),
        )?;
        let version = objects
            .store
            .write_stream(&id, file_byte_stream(staged).await?)
            .await?;
        let reference = ObjectReference {
            id: id.clone(),
            version: version.clone(),
        };
        let path = objects.store.local_path(&reference).await?;
        Ok(WrittenObject {
            path,
            source: object_ref(&id, &version),
            byte_size: i64::try_from(version.byte_size).unwrap_or(i64::MAX),
        })
    }

    /// Return the verified local file for one immutable object this table
    /// references. The reference, not adjacency to any other file, decides which
    /// bytes the caller reads.
    pub async fn localize(&self, reference: &ObjectRef) -> Result<PathBuf, StatsError> {
        let objects = self.require_objects()?;
        let reference = ObjectReference::try_from(reference)?;
        reference.id.table_relative(&self.table).ok_or_else(|| {
            StatsError::Internal(format!(
                "object {:?} belongs to another table",
                reference.id.as_str()
            ))
        })?;
        objects.store.local_path(&reference).await
    }

    /// The local file holding the rows a migration rewrites.
    ///
    /// An object-backed source resolves by exact reference. A version-0 source
    /// is the Parquet file the table's own directory holds; when only the legacy
    /// object layout still has it, its bytes are restored to that path first so
    /// the compaction executor reads it like any other input.
    pub async fn localize_source(
        &self,
        row: &SegmentRow,
        object_record: Option<&ObjectSegmentRecord>,
    ) -> Result<PathBuf, StatsError> {
        let objects = self.require_objects()?;
        if let Some(record) = object_record {
            return self.localize(&record.source).await;
        }
        let path = PathBuf::from(&row.path);
        if path.exists() {
            return Ok(path);
        }
        let key = segment_relative_key(&objects.table_dir, &row.path).ok_or_else(|| {
            StatsError::Internal(format!(
                "legacy migration source {} is outside {}",
                row.path,
                objects.table_dir.display()
            ))
        })?;
        let object = objects
            .legacy_store
            .read(&ObjectId::table(&self.table, &key)?)
            .await?
            .ok_or_else(|| {
                StatsError::Internal(format!(
                    "legacy migration source {key:?} is missing for {:?}",
                    self.table
                ))
            })?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                StatsError::Internal(format!(
                    "create legacy migration source directory {}: {error}",
                    parent.display()
                ))
            })?;
        }
        tokio::fs::write(&path, &object.bytes)
            .await
            .map_err(|error| {
                StatsError::Internal(format!(
                    "restore legacy migration source {}: {error}",
                    path.display()
                ))
            })?;
        Ok(path)
    }

    /// Collect whatever the object store's own retention allows.
    pub async fn gc_objects(&self) -> Result<(), StatsError> {
        self.require_objects()?.store.gc().await
    }

    // --- Bodies the controller task runs. ---

    async fn run_claim(&self) -> Result<(), StatsError> {
        if self.claimed.load(Ordering::SeqCst) {
            return Ok(());
        }
        let objects = self.require_objects()?;
        if let Some(published) = objects.state_store.load(&self.table).await? {
            let claimed = objects
                .state_store
                .claim_writer(&self.table, self.fence, &published)
                .await?;
            self.snapshot
                .send_replace(Some(Arc::new(TableSnapshot::from_stored(&claimed))));
            *self.selected.lock().unwrap() = Some(claimed);
        }
        self.claimed.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Build the next state, commit it under this writer's fence, and republish
    /// the resulting snapshot.
    ///
    /// Publication stays owed until HEAD is known to name that state, so the
    /// same revision is published again rather than undone. A fenced table stops
    /// owing publication: this writer must not overwrite the state another
    /// writer selected.
    async fn run_publish(&self) -> Result<Arc<TableSnapshot>, CommitError> {
        self.run_claim()
            .await
            .map_err(CommitError::PublicationDeferred)?;
        let objects = self
            .require_objects()
            .map_err(CommitError::PublicationDeferred)?;
        let state = TableState::new(
            namespace_catalog(&self.catalog, &self.table, &objects.table_dir)
                .map_err(CommitError::PublicationDeferred)?,
        );
        let expected = self.selected.lock().unwrap().clone();
        let outcome = objects
            .state_store
            .commit(
                &self.table,
                self.fence,
                expected.as_ref(),
                state.catalog().clone(),
            )
            .await;
        let published = match outcome {
            Ok(committed) => {
                let snapshot = TableSnapshot::from_stored(&committed);
                *self.selected.lock().unwrap() = Some(committed);
                snapshot
            }
            Err(error) => match self.resolve_lost_publication(&state, error).await {
                Ok(published) => published,
                Err(error) => {
                    if matches!(error, CommitError::Fenced(_)) {
                        self.publication_owed.store(false, Ordering::SeqCst);
                        self.mark_unready(&error.to_string());
                    }
                    return Err(error);
                }
            },
        };
        self.publication_owed.store(false, Ordering::SeqCst);
        let published = Arc::new(published);
        self.snapshot.send_replace(Some(Arc::clone(&published)));
        Ok(published)
    }

    /// Settle a commit whose outcome the state store did not report against the
    /// state HEAD now selects.
    async fn resolve_lost_publication(
        &self,
        attempted: &TableState,
        error: StatsError,
    ) -> Result<TableSnapshot, CommitError> {
        let objects = self
            .require_objects()
            .map_err(CommitError::PublicationDeferred)?;
        let published = objects
            .state_store
            .load(&self.table)
            .await
            .map_err(|head_error| {
                CommitError::PublicationDeferred(StatsError::AmbiguousCommit(format!(
                    "committing {:?} failed with {error}; reading HEAD to resolve it failed with {head_error}",
                    self.table
                )))
            })?;
        let snapshot = published.as_ref().map(TableSnapshot::from_stored);
        let resolved =
            resolve_publication(&self.table, attempted, self.fence, snapshot.as_ref(), error)?;
        *self.selected.lock().unwrap() = published;
        Ok(resolved)
    }

    async fn run_tombstone(&self) -> Result<(), StatsError> {
        let objects = self.require_objects()?;
        let Some(selected) = objects.state_store.load(&self.table).await? else {
            return Ok(());
        };
        let tombstoned = objects
            .state_store
            .tombstone(&self.table, self.fence, &selected)
            .await?;
        self.publication_owed.store(false, Ordering::SeqCst);
        self.snapshot
            .send_replace(Some(Arc::new(TableSnapshot::from_stored(&tombstoned))));
        Ok(())
    }

    async fn run_gc_states(
        &self,
        now_ms: i64,
        state_retention_ms: u64,
        orphan_grace_ms: u64,
    ) -> Result<usize, StatsError> {
        let objects = self.require_objects()?;
        objects
            .state_store
            .gc_obsolete_states(
                &self.table,
                now_ms,
                state_retention_ms,
                orphan_grace_ms,
                self.fence,
            )
            .await
    }
}

/// The controller task: the only publisher of one table's durable state.
///
/// It holds a weak reference so dropping the controller ends the task, and it
/// exits when the last command sender goes away.
async fn run_controller(
    controller: Weak<TableController>,
    mut commands: mpsc::Receiver<ControllerCommand>,
) {
    while let Some(command) = commands.recv().await {
        let Some(controller) = controller.upgrade() else {
            return;
        };
        match command {
            ControllerCommand::Publish(reply) => {
                let _ = reply.send(controller.run_publish().await);
            }
            ControllerCommand::PublishOwed(reply) => {
                let result = if controller.publication_owed() {
                    controller
                        .run_publish()
                        .await
                        .map(|_| ())
                        .map_err(StatsError::from)
                } else {
                    Ok(())
                };
                let _ = reply.send(result);
            }
            ControllerCommand::Claim(reply) => {
                let _ = reply.send(controller.run_claim().await);
            }
            ControllerCommand::Tombstone(reply) => {
                let _ = reply.send(controller.run_tombstone().await);
            }
            ControllerCommand::GcStates {
                now_ms,
                state_retention_ms,
                orphan_grace_ms,
                reply,
            } => {
                let _ = reply.send(
                    controller
                        .run_gc_states(now_ms, state_retention_ms, orphan_grace_ms)
                        .await,
                );
            }
        }
    }
}

/// Whether a committed object segment belongs to the version a query reads.
///
/// A segment is visible under the active version, under a desired version whose
/// rows were written after the migration fence, and under an in-flight migration
/// aliasing the source version onto the target.
pub fn object_segment_is_query_visible(
    status: &TableSpecStatus,
    record: &ObjectSegmentRecord,
) -> bool {
    record.table_spec_version == status.active_version()
        || (status.desired_version() == record.table_spec_version && !record.migration_backfill)
        || (status.migration.as_ref().is_some_and(|migration| {
            migration.from_version == Some(status.active_version())
                && migration.to_version == Some(record.table_spec_version)
                && !record.migration_backfill
        }))
}

/// Resolve the local files one object-backed segment's artifact references
/// name.
///
/// Each path comes from the artifact object's own identity, so an empty cache
/// resolves the same filenames a warm one does without consulting the local
/// directory.
pub fn local_artifacts(
    store: &dyn ObjectStore,
    references: &ArtifactReferences,
) -> Result<LocalArtifacts, StatsError> {
    let mut local = LocalArtifacts {
        binding: references.binding.clone(),
        ..Default::default()
    };
    if let Some(bundle) = references.bundle.as_ref() {
        local.bundle = Some(planned_path(store, bundle)?);
    }
    for (name, object) in &references.projections {
        local
            .projections
            .insert(name.clone(), planned_path(store, object)?);
    }
    Ok(local)
}

fn planned_path(store: &dyn ObjectStore, reference: &ObjectRef) -> Result<PathBuf, StatsError> {
    let id =
        ObjectId::parse(reference.object_id.as_deref().ok_or_else(|| {
            StatsError::Internal("artifact reference has no object ID".to_string())
        })?)?;
    store.planned_local_path(&id)
}

fn object_ref(id: &ObjectId, version: &ObjectVersion) -> ObjectRef {
    ObjectRef {
        object_id: Some(id.as_str().to_string()),
        provider_version: version.provider_version.clone(),
        etag: version.e_tag.clone(),
        byte_size: Some(version.byte_size),
        sha256: Some(version.content_sha256.to_vec()),
        ..Default::default()
    }
}

/// Stream a staged file in bounded chunks, so an upload never holds the whole
/// object in RAM.
async fn file_byte_stream(path: &Path) -> Result<ObjectByteStream, StatsError> {
    let file = tokio::fs::File::open(path).await.map_err(|error| {
        StatsError::Internal(format!("open staged object {}: {error}", path.display()))
    })?;
    Ok(futures::stream::try_unfold(file, |mut file| async move {
        let mut chunk = vec![0_u8; UPLOAD_CHUNK_BYTES];
        let mut filled = 0;
        while filled < chunk.len() {
            let read = file
                .read(&mut chunk[filled..])
                .await
                .map_err(|error| StatsError::Internal(format!("read staged object: {error}")))?;
            if read == 0 {
                break;
            }
            filled += read;
        }
        if filled == 0 {
            return Ok(None);
        }
        chunk.truncate(filled);
        Ok(Some((Bytes::from(chunk), file)))
    })
    .boxed())
}

/// Content SHA-256 of a staged file, read in bounded chunks.
pub fn file_sha256(path: &Path) -> Result<[u8; 32], StatsError> {
    let mut file = std::fs::File::open(path).map_err(|error| {
        StatsError::Internal(format!("open {} for hashing: {error}", path.display()))
    })?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .map_err(|error| StatsError::Internal(format!("hash {}: {error}", path.display())))?;
    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    use buffa::MessageField;

    use crate::proto::finelog::stats::{
        ColumnType, OperatingPolicy, SourceLayout, TableSpec as ProtoTableSpec,
    };
    use crate::store::catalog::object_state_store::ObjectTableStateStore;
    use crate::store::object_store::build_remote_object_store;
    use crate::store::schema::{schema_to_proto_owned, with_implicit_seq, Column, Schema};
    use crate::store::table_spec::canonical_json_bytes;
    use crate::test_support::{
        lost_head_response, FaultAction, FaultInjectingObjectStore, ObjectFault, ObjectOp,
        ObjectPattern,
    };

    const TABLE: &str = "iris.worker";

    /// The HEAD pointer every table-state commit swaps.
    fn head_swap() -> (ObjectOp, ObjectPattern) {
        (
            ObjectOp::CompareAndSwap,
            ObjectPattern::EndsWith("HEAD.json".to_string()),
        )
    }

    fn registered_catalog() -> Arc<Catalog> {
        let catalog = Catalog::open(None).unwrap();
        let schema = with_implicit_seq(Schema::new(
            vec![Column::new(
                "timestamp_ms",
                ColumnType::COLUMN_TYPE_INT64,
                false,
            )],
            "",
        ));
        let spec = ProtoTableSpec {
            version: Some(1),
            logical_schema: MessageField::some(schema_to_proto_owned(&schema)),
            source_layout: MessageField::some(SourceLayout::default()),
            operating_policy: MessageField::some(OperatingPolicy::default()),
            ..Default::default()
        };
        let hash: [u8; 32] = Sha256::digest(canonical_json_bytes(&spec).unwrap()).into();
        catalog
            .register_table_spec(TABLE, &spec, &hash, false)
            .unwrap();
        Arc::new(catalog)
    }

    fn object_controller(
        table_dir: PathBuf,
        catalog: Arc<Catalog>,
        store: Arc<dyn ObjectStore>,
        state_store: Arc<dyn TableStateStore>,
        fence: u64,
    ) -> Arc<TableController> {
        TableController::start(
            TABLE.to_string(),
            catalog,
            Some(ObjectPersistence {
                table_dir,
                store: Arc::clone(&store),
                legacy_store: store,
                state_store,
            }),
            WriterFence::new(fence),
        )
    }

    /// A controller whose state store writes through a fault seam, plus the
    /// seam and a clean view of the same objects for direct inspection.
    fn faulted_controller(
        tag: &str,
        fence: u64,
    ) -> (
        Arc<TableController>,
        ObjectTableStateStore,
        Arc<FaultInjectingObjectStore>,
    ) {
        let remote_dir = crate::test_support::unique_dir(tag);
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = ObjectTableStateStore::new(remote.clone());
        let faults = FaultInjectingObjectStore::new(remote.clone());
        let controller = object_controller(
            remote_dir,
            registered_catalog(),
            remote,
            Arc::new(ObjectTableStateStore::new(
                Arc::clone(&faults) as Arc<dyn ObjectStore>
            )),
            fence,
        );
        (controller, states, faults)
    }

    #[tokio::test]
    async fn a_commit_that_lost_its_response_is_durable_when_head_names_it() {
        let (controller, _states, faults) = faulted_controller("controller_lost_response", 11);
        let (op, pattern) = head_swap();
        faults.arm(
            ObjectFault::new(
                op,
                pattern,
                FaultAction::LoseResponse {
                    error: lost_head_response(),
                    gate: None,
                },
            )
            .forever(),
        );

        let published = controller.publish_state().await.unwrap();

        assert_eq!(published.revision().get(), 1);
        assert_eq!(published.fence(), WriterFence::new(11));
        assert!(!controller.publication_owed());
        assert!(controller.writes_ready());
    }

    #[tokio::test]
    async fn a_commit_the_store_never_applied_stays_owed_at_the_same_revision() {
        let (controller, _states, faults) = faulted_controller("controller_unapplied", 11);
        let (op, pattern) = head_swap();
        faults.arm(
            ObjectFault::new(
                op,
                pattern,
                FaultAction::Fail(StatsError::SchemaConflict(
                    "object pointer changed concurrently".to_string(),
                )),
            )
            .forever(),
        );

        let error = controller.publish_state().await.unwrap_err();

        assert!(matches!(error, CommitError::PublicationDeferred(_)));
        assert!(error.is_committed());
        assert!(controller.publication_owed());
        assert!(controller.writes_ready());
        assert_eq!(
            controller
                .catalog
                .table_spec_status(TABLE)
                .unwrap()
                .catalog_generation,
            1
        );
    }

    /// Claiming a table another writer published takes ownership of exactly the
    /// state HEAD selects.
    #[tokio::test]
    async fn claiming_an_existing_head_retains_the_selected_state() {
        let (controller, states, _faults) = faulted_controller("controller_claim", 11);
        let objects = controller.objects.as_ref().unwrap();
        let state = namespace_catalog(&controller.catalog, TABLE, &objects.table_dir).unwrap();
        states
            .commit(TABLE, WriterFence::new(12), None, state.clone())
            .await
            .unwrap();

        controller.claim_writer().await.unwrap();

        let selected = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(selected.fence(), WriterFence::new(11));
        assert_eq!(selected.catalog, state);
    }

    /// A second process claims a table this writer already published, then this
    /// writer commits again from a state it loaded after the claim.
    #[tokio::test]
    async fn a_replacement_claim_fences_every_later_commit_from_the_stale_writer() {
        let remote_dir = crate::test_support::unique_dir("controller_replacement_claim");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = Arc::new(ObjectTableStateStore::new(remote.clone()));
        let catalog = registered_catalog();
        let stale = object_controller(
            remote_dir.clone(),
            Arc::clone(&catalog),
            remote.clone(),
            states.clone(),
            11,
        );
        stale.publish_state().await.unwrap();

        let replacement = object_controller(
            remote_dir,
            Arc::clone(&catalog),
            remote.clone(),
            states.clone(),
            12,
        );
        replacement.claim_writer().await.unwrap();

        // The stale writer advances its local revision and republishes.
        catalog.set_forward_cursor("hub", TABLE, 7).unwrap();
        let error = stale.publish_state().await.unwrap_err();

        assert!(matches!(error, CommitError::Fenced(_)));
        assert!(!stale.writes_ready());
        let selected = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(selected.fence(), WriterFence::new(12));
        assert_eq!(selected.revision().get(), 1);
    }

    /// A fenced controller refuses to lease maintenance work.
    #[tokio::test]
    async fn a_fenced_controller_refuses_maintenance_and_reports_unready() {
        let remote_dir = crate::test_support::unique_dir("controller_fenced_maintenance");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = Arc::new(ObjectTableStateStore::new(remote.clone()));
        let catalog = registered_catalog();
        let stale = object_controller(
            remote_dir.clone(),
            Arc::clone(&catalog),
            remote.clone(),
            states.clone(),
            11,
        );
        stale.publish_state().await.unwrap();
        let lease = stale
            .begin_compaction(vec!["a.parquet".to_string()])
            .unwrap();

        let replacement = object_controller(
            remote_dir,
            Arc::clone(&catalog),
            remote.clone(),
            states.clone(),
            12,
        );
        replacement.claim_writer().await.unwrap();
        catalog.set_forward_cursor("hub", TABLE, 7).unwrap();
        stale.publish_state().await.unwrap_err();

        assert!(!stale.writes_ready());
        let rejected = stale
            .commit_maintenance(&lease, || Ok((TableRevision::new(99), ())))
            .await
            .map(|committed| committed.token.revision());
        assert!(matches!(rejected, Err(CommitError::Fenced(_))));
        assert!(stale.begin_compaction(Vec::new()).is_err());
    }

    /// The local projection is rebuildable; losing it never disturbs the state
    /// HEAD selects.
    #[tokio::test]
    async fn a_lost_local_projection_leaves_the_committed_state_selected() {
        let remote_dir = crate::test_support::unique_dir("controller_lost_projection");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = Arc::new(ObjectTableStateStore::new(remote.clone()));
        let catalog = registered_catalog();
        let controller =
            object_controller(remote_dir, Arc::clone(&catalog), remote, states.clone(), 11);
        let committed = controller.publish_state().await.unwrap();
        assert_eq!(committed.revision().get(), 1);

        catalog.delete(TABLE).unwrap();

        // The projection is gone, so no next state can be built, and the
        // committed revision remains the selected one.
        let error = controller.publish_state().await.unwrap_err();
        assert!(matches!(error, CommitError::PublicationDeferred(_)));
        let selected = states.load(TABLE).await.unwrap().unwrap();
        assert_eq!(selected.revision().get(), 1);
        assert_eq!(selected.fence(), WriterFence::new(11));
    }

    /// Every committed transition republishes the table's read snapshot.
    #[tokio::test]
    async fn each_committed_transition_publishes_a_new_snapshot() {
        let remote_dir = crate::test_support::unique_dir("controller_snapshot_watch");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = Arc::new(ObjectTableStateStore::new(remote.clone()));
        let catalog = registered_catalog();
        let controller = object_controller(remote_dir, Arc::clone(&catalog), remote, states, 11);
        let mut watcher = controller.watch_snapshot();
        assert!(watcher.borrow_and_update().is_none());

        let first = controller
            .commit(|| {
                let revision = catalog.set_forward_cursor("hub", TABLE, 5)?;
                Ok((revision, ()))
            })
            .await
            .unwrap();

        assert!(watcher.has_changed().unwrap());
        let published = watcher.borrow_and_update().clone().unwrap();
        assert_eq!(published.revision(), first.token.revision());
        assert_eq!(
            published.state().catalog().forward_cursors[0].cursor,
            Some(5)
        );

        controller
            .commit(|| {
                let revision = catalog.set_forward_cursor("hub", TABLE, 9)?;
                Ok((revision, ()))
            })
            .await
            .unwrap();

        assert!(watcher.has_changed().unwrap());
        let second = watcher.borrow_and_update().clone().unwrap();
        assert!(second.revision() > published.revision());
        assert_eq!(second.state().catalog().forward_cursors[0].cursor, Some(9));
    }
}
