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
//! - the controller task is the only code that publishes state. Durability
//!   callers request publication over a mailbox and await the outcome, while
//!   replayable maintenance leaves its revision owed. Concurrent flushes,
//!   compactions, migrations, and cursor advances therefore never race for
//!   HEAD.
//!
//! Heavy work — Parquet encoding, index building, compaction merges — stays with
//! its caller. The controller serializes lease creation and the short commit
//! only. Appends never reach it: the RAM buffer's short lock is the ingest path.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use bytes::Bytes;
use tokio::sync::{mpsc, oneshot, watch};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{MigrationPhase, NamespaceCatalog, ObjectRef};
use crate::store::catalog::projection::namespace_catalog;
use crate::store::catalog::{Catalog, ObjectSegmentRecord, SpecLifecycle};
use crate::store::object_store::OBJECTS_PREFIX;
use crate::store::object_store::{
    ObjectId, ObjectPrefix, ObjectReference, ObjectStore, ObjectVersion,
};
use crate::store::state_store::object::{ObjectTableStateStore, StateGcPolicy};
use crate::store::state_store::StoredTableState;
use crate::store::table_spec::TablePolicy;
use crate::store::table_state::{
    resolve_publication, CommitError, CommitToken, Committed, TableRevision, TableSnapshot,
    TableState, WriterFence,
};
use crate::store::types::{segment_relative_key, SegmentRow};

/// An immutable object this table just wrote, and the reference that names it.
pub struct WrittenObject {
    pub path: PathBuf,
    pub source: ObjectRef,
    pub byte_size: i64,
}

/// The object storage this server can persist a table through.
///
/// Present whenever the server is configured with a remote log directory, for
/// every table it serves. Holding it says nothing about whether a particular
/// table keeps its data in objects — see
/// [`TableController::is_object_backed`].
pub struct ObjectPersistence {
    pub table_dir: PathBuf,
    pub store: Arc<dyn ObjectStore>,
    pub legacy_store: Arc<dyn ObjectStore>,
    pub state_store: Arc<ObjectTableStateStore>,
}

/// Permission to run one compaction and commit its replacement.
///
/// The lease pins the writer fence and the definition version. It does not pin
/// the table revision, so ordinary flushes and cursor advances commit freely
/// while the work runs; the commit is rebased onto whatever state is current
/// and rejected only when the definition version moved or this writer was
/// fenced. Input liveness is checked by the commit mutation itself.
#[derive(Clone, Debug)]
pub struct MaintenanceLease {
    fence: WriterFence,
    lifecycle: LeaseLifecycle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LeaseLifecycle {
    active_version: u64,
    desired_version: u64,
    phase: MigrationPhase,
}

struct AppliedMutation<T> {
    previous: TableRevision,
    revision: TableRevision,
    output: T,
}

impl LeaseLifecycle {
    fn from_status(status: &SpecLifecycle) -> Self {
        Self {
            active_version: status.active_version(),
            desired_version: status.desired_version(),
            phase: status.phase,
        }
    }
}

/// Work only the controller task performs.
enum ControllerCommand {
    Publish {
        required: TableRevision,
        reply: oneshot::Sender<Result<Arc<TableSnapshot>, CommitError>>,
    },
    PublishOwed(oneshot::Sender<Result<(), StatsError>>),
    Claim(oneshot::Sender<Result<(), StatsError>>),
    Tombstone(oneshot::Sender<Result<(), StatsError>>),
}

const COMMAND_QUEUE_DEPTH: usize = 32;

/// Floor between owed publications of one table. Every flush and compaction
/// commit marks HEAD owed, and each publication PUTs a full catalog snapshot
/// and rewrites the same `HEAD.json` object — which object stores rate-limit
/// per object. Local-disk acknowledgement can defer publication to batch staged
/// uploads and snapshot churn. Object-store acknowledgement uses an explicit
/// publication command and bypasses this throttle.
const MIN_PUBLISH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

pub struct TableController {
    table: String,
    catalog: Arc<Catalog>,
    fence: WriterFence,
    /// Present whenever this server has object persistence configured. Server
    /// configuration, shared by every table it serves.
    objects: Option<ObjectPersistence>,
    /// Whether the table's own specification puts its data in immutable
    /// objects. Refreshed from the catalog by every durable transition that
    /// advances the table's revision, which is how a specification lands.
    spec_selects_objects: AtomicBool,
    /// Set once this table is known to have a published HEAD, whether this
    /// writer wrote it or claimed one another writer left. It outlives the
    /// specification that introduced it: a transition rolled back to the
    /// version-0 layout still owns its objects and its HEAD.
    head_published: AtomicBool,
    /// Serializes revision-allocating transactions with the synchronous
    /// projection of one revision into a publishable state. Never held across
    /// object-store I/O.
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
    /// When the last successful publication finished, for the owed-publication
    /// throttle. Explicit `Publish` commands bypass it.
    last_published: Mutex<Option<tokio::time::Instant>>,
    /// Objects staged locally whose upload has not completed, by object id.
    /// Publication uploads them before it swaps HEAD; a crash repopulates the
    /// missing entries through the once-per-boot reconcile below.
    staged: Mutex<std::collections::HashMap<String, ObjectReference>>,
    /// Set after the first publication this process runs has verified — by one
    /// remote listing — that every referenced data object is remotely durable,
    /// re-uploading any the local cache still holds.
    boot_reconciled: AtomicBool,
    /// Set while the table's durable state needs an operator — e.g. its remote
    /// root holds catalog history but no HEAD. The table keeps serving reads
    /// and accumulating local writes; publication stays deferred.
    degraded: Mutex<Option<String>>,
    /// The latest committed state, republished after every transition.
    snapshot: watch::Sender<Option<Arc<TableSnapshot>>>,
    /// Highest sequence selected by a remotely committed HEAD.
    published_high_water: watch::Sender<i64>,
    commands: Option<mpsc::Sender<ControllerCommand>>,
}

impl TableController {
    /// Build a controller and, where objects can be persisted, start its task.
    ///
    /// The task publishes HEAD, so it exists whenever this server could publish
    /// one. A legacy table simply never sends it a command: it publishes no
    /// HEAD, so its only durable transition is the gated local mutation.
    pub fn start(
        table: String,
        catalog: Arc<Catalog>,
        objects: Option<ObjectPersistence>,
        fence: WriterFence,
    ) -> Arc<Self> {
        let (snapshot, _) = watch::channel(None);
        let (published_high_water, _) = watch::channel(0);
        let object_persistence = objects.is_some();
        let (sender, receiver) = mpsc::channel(COMMAND_QUEUE_DEPTH);
        let controller = Arc::new(Self {
            table,
            catalog,
            fence,
            objects,
            spec_selects_objects: AtomicBool::new(false),
            head_published: AtomicBool::new(false),
            mutation_gate: Mutex::new(()),
            selected: Mutex::new(None),
            claimed: AtomicBool::new(false),
            writes_ready: AtomicBool::new(true),
            publication_owed: AtomicBool::new(false),
            last_published: Mutex::new(None),
            staged: Mutex::new(std::collections::HashMap::new()),
            boot_reconciled: AtomicBool::new(false),
            degraded: Mutex::new(None),
            snapshot,
            published_high_water,
            commands: object_persistence.then(|| sender.clone()),
        });
        if let Err(error) = controller.refresh_object_backing() {
            // The next durable transition re-reads it. Serving a table as
            // legacy is the safe reading: it commits locally and publishes no
            // HEAD.
            tracing::error!(
                table = %controller.table,
                %error,
                "reading a table specification to classify its storage failed"
            );
        }
        if object_persistence {
            drop(sender);
            tokio::spawn(run_controller(Arc::downgrade(&controller), receiver));
        }
        controller
    }

    /// Whether this server can persist objects for this table at all: a remote
    /// object store, its legacy view, and the durable state authority are all
    /// configured.
    ///
    /// This is a property of the server, not of the table. Every table on a
    /// server with a remote log directory has object persistence available, and
    /// the legacy ones among them are still not object-backed.
    pub fn object_persistence_configured(&self) -> bool {
        self.objects.is_some()
    }

    /// Whether this table's own durable state keeps its data in immutable
    /// objects under a published HEAD.
    ///
    /// The table's specification decides it — the desired version while a
    /// transition onto object storage runs, the active version otherwise — and
    /// a claimed HEAD is equally conclusive. A legacy table on a
    /// remote-configured server is not object-backed: it commits to the local
    /// catalog, records artifacts beside its Parquet, and publishes no state.
    pub fn is_object_backed(&self) -> bool {
        self.object_persistence_configured()
            && (self.spec_selects_objects.load(Ordering::SeqCst)
                || self.head_published.load(Ordering::SeqCst))
    }

    /// The sequence high-water mark of the claimed durable state, or 0 when no
    /// state is claimed. It can exceed the max seq in the published segments (a
    /// legacy import excludes archive-only rows; retirement deletes legacy
    /// rows), so the allocator must seed past it.
    pub fn claimed_high_water(&self) -> i64 {
        self.selected
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|selected| selected.catalog.persisted_high_water)
            .unwrap_or(0)
    }

    /// Re-read the table's specification and cache whether it selects
    /// object-store L0, returning the status the read produced.
    ///
    /// [`TablePolicy`] is the one place a specification resolves to operating
    /// knobs, so the flush path and this classification cannot drift apart.
    fn refresh_object_backing(&self) -> Result<SpecLifecycle, StatsError> {
        let status = self.catalog.spec_lifecycle(&self.table)?;
        self.spec_selects_objects.store(
            TablePolicy::resolve(status.operative()).object_backed(),
            Ordering::SeqCst,
        );
        Ok(status)
    }

    /// Whether this writer still owns the table's durable state.
    pub fn writes_ready(&self) -> bool {
        self.writes_ready.load(Ordering::SeqCst)
    }

    /// Flag durable state that needs an operator, without stopping writes:
    /// rows keep accumulating locally while publication stays deferred.
    pub fn mark_degraded(&self, reason: &str) {
        let mut degraded = self.degraded.lock().unwrap();
        if degraded.as_deref() != Some(reason) {
            tracing::error!(table = %self.table, fence = %self.fence, reason, "table is degraded");
        }
        *degraded = Some(reason.to_string());
    }

    pub fn degraded_reason(&self) -> Option<String> {
        self.degraded.lock().unwrap().clone()
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

    /// Highest sequence recoverable from the object state HEAD.
    pub fn published_high_water(&self) -> i64 {
        *self.published_high_water.borrow()
    }

    /// Wait for object state HEAD to cover `target`.
    pub async fn await_published_high_water(
        &self,
        target: i64,
        timeout: std::time::Duration,
    ) -> Result<(), StatsError> {
        if target < 0 {
            return Ok(());
        }
        let mut receiver = self.published_high_water.subscribe();
        if *receiver.borrow() >= target {
            return Ok(());
        }
        let wait = async {
            while *receiver.borrow() < target {
                if receiver.changed().await.is_err() {
                    return;
                }
            }
        };
        match tokio::time::timeout(timeout, wait).await {
            Ok(()) if *self.published_high_water.borrow() >= target => Ok(()),
            Ok(()) => Err(StatsError::Internal(format!(
                "table {:?} published-high-water channel closed before seq>={target}",
                self.table
            ))),
            Err(_) => Err(StatsError::DeadlineExceeded(format!(
                "timed out waiting for table {:?} object state to publish seq>={target}",
                self.table
            ))),
        }
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
            StatsError::Internal(format!(
                "table {:?} has no object persistence configured",
                self.table
            ))
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
    pub fn adopt_claimed(&self, claimed: StoredTableState) -> Result<(), StatsError> {
        // Recovery has already reconciled the local projection with this HEAD.
        // When HEAD covers the local revision, every referenced object was
        // necessarily durable before HEAD selected it, so the first new write
        // need only upload objects staged by this process. A projection ahead
        // of HEAD may contain a pre-crash local commit whose object upload did
        // not finish, and therefore retains the once-per-boot remote scan.
        let selected_covers_local = claimed.revision() >= self.local_revision()?;
        let snapshot = TableSnapshot::from_stored(&claimed);
        self.record_published_high_water(claimed.catalog.persisted_high_water.unwrap_or(0));
        *self.selected.lock().unwrap() = Some(claimed);
        self.claimed.store(true, Ordering::SeqCst);
        self.head_published.store(true, Ordering::SeqCst);
        self.boot_reconciled
            .store(selected_covers_local, Ordering::SeqCst);
        self.snapshot.send_replace(Some(Arc::new(snapshot)));
        Ok(())
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
        let AppliedMutation {
            previous,
            revision,
            output,
        } = self.apply(mutation)?;
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
        let AppliedMutation {
            previous,
            revision,
            output,
        } = self.apply(mutation)?;
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
    fn apply<T, F>(&self, mutation: F) -> Result<AppliedMutation<T>, CommitError>
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
            // The mutation may have registered or activated a specification, so
            // the storage classification is re-read before this commit decides
            // whether it owes a publication.
            self.refresh_object_backing()
                .map_err(CommitError::NotCommitted)?;
            self.mark_publication_owed();
        }
        Ok(AppliedMutation {
            previous,
            revision,
            output,
        })
    }

    fn local_revision(&self) -> Result<TableRevision, StatsError> {
        Ok(TableRevision::new(
            self.refresh_object_backing()?.catalog_generation,
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
        let required = self
            .local_revision()
            .map_err(CommitError::PublicationDeferred)?;
        match self
            .dispatch(|reply| ControllerCommand::Publish { required, reply })
            .await
        {
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
    ///
    /// Collection runs outside the publication mailbox: state pin retention
    /// protects a newly written document until long after its HEAD swap, and
    /// orphan grace similarly protects newly written data objects. The state
    /// store also verifies this writer's fence before deleting anything. A
    /// slow listing or deletion therefore cannot delay a durability commit.
    pub(crate) async fn gc_published(
        &self,
        now_ms: i64,
        policy: StateGcPolicy,
    ) -> Result<usize, StatsError> {
        self.require_objects()?
            .state_store
            .gc_obsolete_states(&self.table, now_ms, policy, self.fence)
            .await
    }

    /// Take a lease for one compaction.
    ///
    /// The merge and encode run outside the controller; only the lease and the
    /// eventual commit are serialized here.
    pub fn begin_compaction(&self) -> Result<MaintenanceLease, StatsError> {
        if !self.writes_ready() {
            return Err(StatsError::SchemaConflict(format!(
                "table {:?} is fenced by another writer",
                self.table
            )));
        }
        let lifecycle = self.catalog.spec_lifecycle(&self.table)?;
        Ok(MaintenanceLease {
            fence: self.fence,
            lifecycle: LeaseLifecycle::from_status(&lifecycle),
        })
    }

    /// Take a lease only if the lifecycle used to select the compaction inputs
    /// is still current. This closes the window between planning and leasing.
    pub fn begin_compaction_for(
        &self,
        expected: &SpecLifecycle,
    ) -> Result<MaintenanceLease, StatsError> {
        let lease = self.begin_compaction()?;
        let expected = LeaseLifecycle::from_status(expected);
        if lease.lifecycle != expected {
            return Err(StatsError::SchemaConflict(format!(
                "table {:?} changed lifecycle while compaction was planned",
                self.table
            )));
        }
        Ok(lease)
    }

    /// Commit replayable maintenance locally, rebasing it onto the current state.
    ///
    /// The lease is rejected when this writer was fenced or the table's active
    /// definition version moved while the work ran. Input liveness is checked by
    /// `mutation`, which runs inside the same gated transaction that replaces
    /// them.
    ///
    /// Maintenance does not wait for HEAD publication. Its immutable outputs
    /// are already remotely durable, and the previously published inputs stay
    /// live until a later client write or maintenance tick publishes the new
    /// revision. A crash can therefore lose maintenance progress but cannot
    /// lose acknowledged rows.
    pub fn commit_maintenance<T, F>(
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
        let current = self
            .catalog
            .spec_lifecycle(&self.table)
            .map_err(CommitError::NotCommitted)?;
        let current = LeaseLifecycle::from_status(&current);
        if current != lease.lifecycle {
            return Err(CommitError::NotCommitted(StatsError::SchemaConflict(
                format!(
                    "table {:?} changed lifecycle from {:?} to {:?} while compaction ran",
                    self.table, lease.lifecycle, current
                ),
            )));
        }
        self.commit_owing_publication(mutation)
    }

    /// Stage an immutable Parquet object locally.
    ///
    /// The bytes are durable on local disk when this returns; publication
    /// uploads them before HEAD names the revision that references them.
    pub async fn stage_parquet(&self, bytes: Bytes) -> Result<WrittenObject, StatsError> {
        let objects = self.require_objects()?;
        let key = format!("{OBJECTS_PREFIX}/{}.parquet", uuid::Uuid::new_v4());
        let id = ObjectId::table(&self.table, &key)?;
        let version = objects.store.stage(&id, bytes).await?;
        let reference = ObjectReference {
            id: id.clone(),
            version: version.clone(),
        };
        let path = objects.store.planned_local_path(&id)?;
        self.staged
            .lock()
            .unwrap()
            .insert(id.as_str().to_string(), reference);
        Ok(WrittenObject {
            path,
            source: object_ref(&id, &version),
            byte_size: i64::try_from(version.byte_size).unwrap_or(i64::MAX),
        })
    }

    /// Upload a staged local file as an immutable object.
    ///
    /// `kind` selects the object prefix (`objects`, `indices`, `projections`)
    /// and `extension` its suffix.
    pub async fn write_staged_object(
        &self,
        kind: &str,
        extension: &str,
        staged: &Path,
    ) -> Result<WrittenObject, StatsError> {
        let bytes = Bytes::from(tokio::fs::read(staged).await.map_err(|error| {
            StatsError::Internal(format!("read staged object {}: {error}", staged.display()))
        })?);
        let key = format!("{kind}/{}.{extension}", uuid::Uuid::new_v4());
        self.write_immutable(&key, bytes).await
    }

    async fn write_immutable(
        &self,
        relative_key: &str,
        bytes: Bytes,
    ) -> Result<WrittenObject, StatsError> {
        let objects = self.require_objects()?;
        let id = ObjectId::table(&self.table, relative_key)?;
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

    /// Return the local file for one immutable object this table
    /// references. The reference, not adjacency to any other file, decides which
    /// bytes the caller reads.
    pub async fn localize(&self, reference: &ObjectRef) -> Result<PathBuf, StatsError> {
        let objects = self.require_objects()?;
        let reference = ObjectReference::try_from(reference)?;
        objects.store.local_path(&reference).await
    }

    /// The local file holding the rows a migration rewrites, or `None` when
    /// nothing holds them.
    ///
    /// An object-backed source resolves by exact reference. A version-0 source
    /// is the Parquet file the table's own directory holds. Callers select
    /// sources the catalog reports as locally present, so the archive read below
    /// repairs a torn eviction — the file was unlinked but the catalog row still
    /// claims it, and the rows it holds are still query-visible and owed a
    /// rewrite. A segment the catalog itself reports as archived is never a
    /// migration source and never reaches this path. `None` means the row's
    /// bytes exist neither locally nor in the archive: no reader can serve them
    /// and no source can supply them.
    pub async fn localize_source(
        &self,
        row: &SegmentRow,
        object_record: Option<&ObjectSegmentRecord>,
    ) -> Result<Option<PathBuf>, StatsError> {
        let objects = self.require_objects()?;
        if let Some(record) = object_record {
            return self.localize(&record.source).await.map(Some);
        }
        let path = PathBuf::from(&row.path);
        if path.exists() {
            return Ok(Some(path));
        }
        let key = segment_relative_key(&objects.table_dir, &row.path).ok_or_else(|| {
            StatsError::Internal(format!(
                "legacy migration source {} is outside {}",
                row.path,
                objects.table_dir.display()
            ))
        })?;
        let Some(object) = objects
            .legacy_store
            .read(&ObjectId::table(&self.table, &key)?)
            .await?
        else {
            return Ok(None);
        };
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
        Ok(Some(path))
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
            self.record_published_high_water(claimed.catalog.persisted_high_water.unwrap_or(0));
            *self.selected.lock().unwrap() = Some(claimed);
            self.head_published.store(true, Ordering::SeqCst);
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
        // `namespace_catalog` reads several catalog relations. Keep the
        // revision gate until that projection is complete so a concurrent
        // cursor or segment commit cannot produce a hybrid value carrying the
        // old revision with fields from the new one.
        let expected = self.selected.lock().unwrap().clone();
        let mut catalog = {
            let _gate = self.mutation_gate.lock().unwrap();
            let mut projected = namespace_catalog(&self.catalog, &self.table, &objects.table_dir)
                .map_err(CommitError::PublicationDeferred)?;
            if let Some(previous) = &expected {
                let floor = previous.catalog.persisted_high_water.unwrap_or(0);
                if projected.persisted_high_water.unwrap_or(0) < floor {
                    projected.persisted_high_water = Some(floor);
                }
                if projected.catalog_generation == previous.catalog.catalog_generation
                    && projected != previous.catalog
                {
                    let revision = self
                        .catalog
                        .advance_object_state_revision(&self.table)
                        .map_err(CommitError::PublicationDeferred)?;
                    tracing::warn!(
                        table = %self.table,
                        revision = revision.get(),
                        "advanced a divergent recovered projection to a fresh revision"
                    );
                    projected = namespace_catalog(&self.catalog, &self.table, &objects.table_dir)
                        .map_err(CommitError::PublicationDeferred)?;
                    if projected.persisted_high_water.unwrap_or(0) < floor {
                        projected.persisted_high_water = Some(floor);
                    }
                }
            }
            projected
        };
        // A root that holds catalog history without a HEAD is an external
        // anomaly (the software never deletes HEAD). Creating a fresh HEAD
        // would start a second history over the first, so publication defers
        // until an operator restores HEAD from the newest catalog document.
        if expected.is_none()
            && objects
                .state_store
                .catalog_history_exists(&self.table)
                .await
                .map_err(CommitError::PublicationDeferred)?
        {
            let reason = format!(
                "table {:?} remote root holds catalog history but no HEAD; refusing to start a \
                 new history — restore HEAD.json from the newest catalog document and restart",
                self.table
            );
            self.mark_degraded(&reason);
            return Err(CommitError::PublicationDeferred(StatsError::Internal(
                reason,
            )));
        }
        // HEAD must never name a state whose data objects are not remotely
        // durable, so the upload happens here before the swap.
        self.sync_referenced_objects(&catalog)
            .await
            .map_err(CommitError::PublicationDeferred)?;
        // The high-water mark is monotone across revisions: the local aggregate
        // it is computed from shrinks when retirement deletes legacy rows, and
        // a shrunken mark would let a later recovery reissue sequence numbers.
        if let Some(previous) = &expected {
            let floor = previous.catalog.persisted_high_water.unwrap_or(0);
            if catalog.persisted_high_water.unwrap_or(0) < floor {
                catalog.persisted_high_water = Some(floor);
            }
        }
        let state = TableState::new(catalog);
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
        // HEAD now names a state for this table, and keeps doing so however
        // its specification later moves.
        self.head_published.store(true, Ordering::SeqCst);
        *self.last_published.lock().unwrap() = Some(tokio::time::Instant::now());
        *self.degraded.lock().unwrap() = None;
        let published = Arc::new(published);
        // A local mutation may commit while the remote swap is in flight. Make
        // the owed decision under the same short gate as mutation allocation,
        // so an older publication cannot clear a newer revision's obligation.
        {
            let _gate = self.mutation_gate.lock().unwrap();
            let still_owed = match self.local_revision() {
                Ok(local) => local > published.revision(),
                Err(error) => {
                    tracing::warn!(
                        table = %self.table,
                        %error,
                        "published table state but could not compare the local revision"
                    );
                    true
                }
            };
            self.publication_owed.store(still_owed, Ordering::SeqCst);
        }
        self.record_published_high_water(
            published
                .state()
                .catalog()
                .persisted_high_water
                .unwrap_or(0),
        );
        self.snapshot.send_replace(Some(Arc::clone(&published)));
        Ok(published)
    }

    fn record_published_high_water(&self, high_water: i64) {
        let current = *self.published_high_water.borrow();
        if high_water > current {
            self.published_high_water.send_replace(high_water);
        }
    }

    /// Whether an owed publication is past the per-table throttle.
    fn owed_publication_due(&self) -> bool {
        self.last_published
            .lock()
            .unwrap()
            .is_none_or(|at| at.elapsed() >= MIN_PUBLISH_INTERVAL)
    }

    /// Make every data object the state references remotely durable before
    /// HEAD names it.
    ///
    /// The fast path uploads the objects this process staged — no remote round
    /// trip beyond the uploads themselves. The first publication after a boot
    /// additionally reconciles against one remote listing: a crash between a
    /// local commit and its upload leaves objects the staged map cannot know
    /// about, and their cache files hold the only bytes. A live reference
    /// whose bytes exist neither remotely nor locally is refused — the local
    /// catalog and the remote root describe different histories. A retired
    /// reference in that state is rollback bookkeeping and is logged instead.
    async fn sync_referenced_objects(&self, catalog: &NamespaceCatalog) -> Result<(), StatsError> {
        let objects = self.require_objects()?;
        let staged: Vec<ObjectReference> = self.staged.lock().unwrap().values().cloned().collect();
        for reference in staged {
            objects.store.upload_staged(&reference).await?;
            self.staged.lock().unwrap().remove(reference.id.as_str());
        }
        if self.boot_reconciled.load(Ordering::SeqCst) {
            return Ok(());
        }
        let data_object_prefix = format!("{OBJECTS_PREFIX}/");
        let mut live = Vec::new();
        let mut retired = Vec::new();
        for version in &catalog.version_segments {
            let version_number = version.table_spec_version.unwrap_or(0);
            let segments = version
                .live_segments
                .iter()
                .map(|segment| (segment, true))
                .chain(
                    version
                        .retired_segments
                        .iter()
                        .map(|segment| (segment, false)),
                );
            for (segment, is_live) in segments {
                let table_spec_version = segment.table_spec_version.unwrap_or(version_number);
                if table_spec_version == 0 {
                    continue;
                }
                let Some(source) = segment.source.as_option() else {
                    return Err(StatsError::Internal(format!(
                        "table {:?} version {table_spec_version} segment {:?} has no source object",
                        self.table, segment.segment_id
                    )));
                };
                let reference = ObjectReference::try_from(source).map_err(|error| {
                    StatsError::Internal(format!(
                        "table {:?} version {table_spec_version} segment {:?} has an invalid source object: {error}",
                        self.table, segment.segment_id
                    ))
                })?;
                let relative = reference.id.table_relative(&self.table).ok_or_else(|| {
                    StatsError::Internal(format!(
                        "table {:?} version {table_spec_version} segment {:?} references object {:?} from another table",
                        self.table,
                        segment.segment_id,
                        reference.id.as_str()
                    ))
                })?;
                if !relative.starts_with(&data_object_prefix) {
                    return Err(StatsError::Internal(format!(
                        "table {:?} version {table_spec_version} segment {:?} source {:?} is outside the data-object prefix",
                        self.table,
                        segment.segment_id,
                        reference.id.as_str()
                    )));
                }
                if is_live {
                    live.push(reference);
                } else {
                    retired.push(reference);
                }
            }
        }
        if live.is_empty() && retired.is_empty() {
            self.boot_reconciled.store(true, Ordering::SeqCst);
            return Ok(());
        }
        let present: HashSet<String> = objects
            .store
            .list(&ObjectPrefix::table(&self.table, OBJECTS_PREFIX)?)
            .await?
            .into_iter()
            .map(|metadata| metadata.id.as_str().to_string())
            .collect();
        let mut unservable = Vec::new();
        for reference in live {
            if present.contains(reference.id.as_str()) {
                continue;
            }
            if let Err(error) = objects.store.upload_staged(&reference).await {
                unservable.push((reference, error));
            }
        }
        if let Some((reference, error)) = unservable.first() {
            return Err(StatsError::Internal(format!(
                "table {:?} publication references {} data objects that are neither remotely \
                 durable nor locally staged (first: {:?}: {error}); the local catalog and the \
                 remote log dir describe different histories",
                self.table,
                unservable.len(),
                reference.id.as_str(),
            )));
        }
        for reference in retired {
            if present.contains(reference.id.as_str()) {
                continue;
            }
            if let Err(error) = objects.store.upload_staged(&reference).await {
                tracing::warn!(
                    table = %self.table,
                    object = %reference.id.as_str(),
                    %error,
                    "retired object is not remotely durable and cannot be re-uploaded"
                );
            }
        }
        self.boot_reconciled.store(true, Ordering::SeqCst);
        Ok(())
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
            ControllerCommand::Publish { required, reply } => {
                // One publication reads the latest local catalog, so it may
                // cover several requests queued while its remote swap was in
                // flight. Avoid another HEAD read for every already-covered
                // request; a later caller re-arms `publication_owed` before it
                // enters the queue and therefore still validates its state.
                let covered = (!controller.publication_owed()).then(|| {
                    controller
                        .selected
                        .lock()
                        .unwrap()
                        .as_ref()
                        .filter(|selected| selected.revision() >= required)
                        .map(TableSnapshot::from_stored)
                        .map(Arc::new)
                });
                let result = if let Some(snapshot) = covered.flatten() {
                    Ok(snapshot)
                } else {
                    controller.run_publish().await
                };
                let _ = reply.send(result);
            }
            ControllerCommand::PublishOwed(reply) => {
                // Within the throttle the revision simply stays owed; a later
                // cycle publishes it. Explicit Publish commands are not
                // throttled — activation and shutdown want HEAD current.
                let result = if controller.publication_owed() && controller.owed_publication_due() {
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
        }
    }
}

fn object_ref(id: &ObjectId, version: &ObjectVersion) -> ObjectRef {
    ObjectRef {
        object_id: Some(id.as_str().to_string()),
        provider_version: version.provider_version.clone(),
        etag: version.e_tag.clone(),
        byte_size: Some(version.byte_size),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use buffa::MessageField;
    use sha2::{Digest, Sha256};

    use crate::proto::finelog::stats::{
        CatalogSegment, ColumnType, ForwardCursor, L0Mode, OperatingPolicy, SourceLayout,
        TableSpec as ProtoTableSpec, TableVersionSegments,
    };
    use crate::store::object_store::build_remote_object_store;
    use crate::store::schema::{schema_to_proto_owned, with_implicit_seq, Column, Schema};
    use crate::store::state_store::object::ObjectTableStateStore;
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

    /// A catalog holding one table whose specification puts its data in
    /// objects, which is what makes its controller publish a HEAD.
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
            operating_policy: MessageField::some(OperatingPolicy {
                l0_mode: Some(L0Mode::L0_MODE_OBJECT_STORE.into()),
                ..Default::default()
            }),
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
        state_store: Arc<ObjectTableStateStore>,
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
    async fn publication_rejects_a_malformed_live_object_reference() {
        let (controller, _states, _faults) = faulted_controller("controller_malformed_ref", 11);
        let state = NamespaceCatalog {
            version_segments: vec![TableVersionSegments {
                table_spec_version: Some(1),
                live_segments: vec![CatalogSegment {
                    table_spec_version: Some(1),
                    source: MessageField::some(ObjectRef {
                        object_id: Some("not/a/canonical/object".to_string()),
                        byte_size: Some(1),
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };

        assert!(matches!(
            controller.sync_referenced_objects(&state).await,
            Err(StatsError::Internal(_))
        ));
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
        assert!(controller.publication_owed());
        assert!(controller.writes_ready());
        assert_eq!(
            controller
                .catalog
                .spec_lifecycle(TABLE)
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

    #[tokio::test]
    async fn adopting_a_claimed_head_only_reconciles_an_ahead_local_projection() {
        let remote_dir = crate::test_support::unique_dir("controller_adopt_reconcile");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = Arc::new(ObjectTableStateStore::new(remote.clone()));
        let published_catalog = registered_catalog();
        let publisher = object_controller(
            remote_dir.clone(),
            Arc::clone(&published_catalog),
            remote.clone(),
            states.clone(),
            11,
        );
        publisher.publish_state().await.unwrap();
        let selected = states.load(TABLE).await.unwrap().unwrap();

        let matching = object_controller(
            remote_dir.clone(),
            registered_catalog(),
            remote.clone(),
            states.clone(),
            11,
        );
        matching.adopt_claimed(selected.clone()).unwrap();
        assert!(matching.boot_reconciled.load(Ordering::SeqCst));

        let ahead_catalog = registered_catalog();
        ahead_catalog.set_forward_cursor("hub", TABLE, 5).unwrap();
        let ahead = object_controller(remote_dir, ahead_catalog, remote, states, 11);
        ahead.adopt_claimed(selected).unwrap();
        assert!(!ahead.boot_reconciled.load(Ordering::SeqCst));
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
        let lease = stale.begin_compaction().unwrap();

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
            .map(|committed| committed.token.revision());
        assert!(matches!(rejected, Err(CommitError::Fenced(_))));
        assert!(stale.begin_compaction().is_err());
    }

    #[tokio::test]
    async fn maintenance_lease_rejects_migration_lifecycle_changes() {
        let catalog = registered_catalog();
        let active = catalog.spec_lifecycle(TABLE).unwrap().active.unwrap();
        let mut next = active;
        next.version = Some(2);
        next.source_layout
            .get_or_insert_default()
            .target_object_bytes = Some(8 * 1024 * 1024);
        let hash: [u8; 32] = Sha256::digest(canonical_json_bytes(&next).unwrap()).into();
        let dual_write = catalog
            .register_table_spec(TABLE, &next, &hash, true)
            .unwrap();
        assert_eq!(dual_write.phase, MigrationPhase::MIGRATION_PHASE_DUAL_WRITE);

        let remote_dir = crate::test_support::unique_dir("controller_lifecycle_lease");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let controller = object_controller(
            remote_dir,
            Arc::clone(&catalog),
            remote.clone(),
            Arc::new(ObjectTableStateStore::new(remote)),
            11,
        );

        catalog
            .update_migration_phase(
                TABLE,
                MigrationPhase::MIGRATION_PHASE_DUAL_WRITE,
                MigrationPhase::MIGRATION_PHASE_BACKFILL,
            )
            .unwrap();
        assert!(controller.begin_compaction_for(&dual_write).is_err());

        let backfill = catalog.spec_lifecycle(TABLE).unwrap();
        let lease = controller.begin_compaction_for(&backfill).unwrap();
        catalog
            .update_migration_phase(
                TABLE,
                MigrationPhase::MIGRATION_PHASE_BACKFILL,
                MigrationPhase::MIGRATION_PHASE_VERIFY,
            )
            .unwrap();
        let mut invoked = false;
        let rejected = controller.commit_maintenance(&lease, || {
            invoked = true;
            Ok((TableRevision::new(99), ()))
        });
        assert!(matches!(
            rejected,
            Err(CommitError::NotCommitted(StatsError::SchemaConflict(_)))
        ));
        assert!(!invoked);
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

    #[tokio::test]
    async fn queued_publications_skip_revisions_an_earlier_request_already_published() {
        let (controller, states, faults) = faulted_controller("controller_publish_coalesce", 11);
        let gate = crate::test_support::FaultGate::new();
        let (op, pattern) = head_swap();
        faults.arm(ObjectFault::new(
            op,
            pattern,
            FaultAction::Park(gate.clone()),
        ));

        let first = {
            let controller = Arc::clone(&controller);
            tokio::spawn(async move { controller.publish_state().await.unwrap() })
        };
        gate.entered().await;

        controller
            .catalog
            .set_forward_cursor("hub", TABLE, 5)
            .unwrap();
        let second = {
            let controller = Arc::clone(&controller);
            tokio::spawn(async move { controller.publish_state().await.unwrap() })
        };
        tokio::task::yield_now().await;
        controller
            .catalog
            .set_forward_cursor("hub", TABLE, 9)
            .unwrap();
        let third = {
            let controller = Arc::clone(&controller);
            tokio::spawn(async move { controller.publish_state().await.unwrap() })
        };
        tokio::task::yield_now().await;

        gate.release();
        let first = first.await.unwrap();
        let second = second.await.unwrap();
        let third = third.await.unwrap();

        assert_eq!(first.revision().get(), 1);
        assert_eq!(second.revision().get(), 3);
        assert_eq!(third.revision().get(), 3);
        let head_reads = faults
            .keys_for(ObjectOp::Read)
            .into_iter()
            .filter(|key| key.ends_with("HEAD.json"))
            .count();
        assert_eq!(head_reads, 3);
        assert_eq!(faults.keys_for(ObjectOp::CompareAndSwap).len(), 2);
        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().revision().get(),
            3
        );
        assert!(!controller.publication_owed());
    }

    #[tokio::test]
    async fn a_commit_during_publication_stays_owed_after_the_older_revision_lands() {
        let (controller, states, faults) = faulted_controller("controller_publish_race", 11);
        let gate = crate::test_support::FaultGate::new();
        let (op, pattern) = head_swap();
        faults.arm(ObjectFault::new(
            op,
            pattern,
            FaultAction::Park(gate.clone()),
        ));

        let first = {
            let controller = Arc::clone(&controller);
            tokio::spawn(async move { controller.publish_state().await.unwrap() })
        };
        gate.entered().await;
        controller
            .commit_owing_publication(|| {
                let revision = controller.catalog.set_forward_cursor("hub", TABLE, 5)?;
                Ok((revision, ()))
            })
            .unwrap();
        gate.release();

        assert_eq!(first.await.unwrap().revision().get(), 1);
        assert!(controller.publication_owed());
        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().revision().get(),
            1
        );

        let second = controller.publish_state().await.unwrap();
        assert_eq!(second.revision().get(), 2);
        assert!(!controller.publication_owed());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publication_projects_after_an_in_flight_local_mutation() {
        let (controller, states, _faults) = faulted_controller("controller_projection_race", 11);
        controller.publish_state().await.unwrap();
        let (entered_send, entered_receive) = std::sync::mpsc::channel();
        let (release_send, release_receive) = std::sync::mpsc::channel();
        let mutation = {
            let controller = Arc::clone(&controller);
            std::thread::spawn(move || {
                controller
                    .commit_owing_publication(|| {
                        entered_send.send(()).unwrap();
                        release_receive.recv().unwrap();
                        let revision = controller.catalog.set_forward_cursor("hub", TABLE, 5)?;
                        Ok((revision, ()))
                    })
                    .unwrap();
            })
        };
        entered_receive.recv().unwrap();

        let mut publication = {
            let controller = Arc::clone(&controller);
            tokio::spawn(async move { controller.publish_state().await.unwrap() })
        };
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(250), &mut publication)
                .await
                .is_err(),
            "publication must wait for the revision-owning mutation"
        );

        release_send.send(()).unwrap();
        mutation.join().unwrap();
        let published = publication.await.unwrap();
        assert_eq!(published.revision().get(), 2);
        assert_eq!(
            published.state().catalog().forward_cursors[0].cursor,
            Some(5)
        );
        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().revision().get(),
            2
        );
    }

    #[tokio::test]
    async fn divergent_recovered_projection_gets_a_fresh_revision() {
        let (controller, states, _faults) =
            faulted_controller("controller_recovered_projection_revision", 11);
        let first = controller.publish_state().await.unwrap();
        assert_eq!(first.revision().get(), 1);

        let mut divergent = first.state().catalog().clone();
        divergent.forward_cursors.push(ForwardCursor {
            target: Some("hub".to_string()),
            cursor: Some(5),
            ..Default::default()
        });
        let schema = with_implicit_seq(Schema::new(
            vec![Column::new(
                "timestamp_ms",
                ColumnType::COLUMN_TYPE_INT64,
                false,
            )],
            "",
        ));
        controller
            .catalog
            .replace_with_published_snapshot(
                TABLE,
                schema,
                crate::store::policy::StoragePolicy::default(),
                &divergent,
                &[],
            )
            .unwrap();

        let repaired = controller.publish_state().await.unwrap();
        assert_eq!(repaired.revision().get(), 2);
        assert_eq!(
            repaired.state().catalog().forward_cursors[0].cursor,
            Some(5)
        );
        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().revision().get(),
            2
        );
        assert!(controller.writes_ready());
    }

    #[tokio::test]
    async fn maintenance_commits_while_an_older_publication_is_in_flight() {
        let (controller, states, faults) = faulted_controller("controller_maintenance_publish", 11);
        let gate = crate::test_support::FaultGate::new();
        let (op, pattern) = head_swap();
        faults.arm(ObjectFault::new(
            op,
            pattern,
            FaultAction::Park(gate.clone()),
        ));
        let lease = controller.begin_compaction().unwrap();

        let first = {
            let controller = Arc::clone(&controller);
            tokio::spawn(async move { controller.publish_state().await.unwrap() })
        };
        gate.entered().await;

        let committed = controller
            .commit_maintenance(&lease, || {
                let revision = controller.catalog.set_forward_cursor("hub", TABLE, 5)?;
                Ok((revision, ()))
            })
            .unwrap();
        assert_eq!(committed.token.revision().get(), 2);
        assert!(controller.publication_owed());
        assert!(states.load(TABLE).await.unwrap().is_none());

        gate.release();
        assert_eq!(first.await.unwrap().revision().get(), 1);
        assert!(controller.publication_owed());

        let second = controller.publish_state().await.unwrap();
        assert_eq!(second.revision().get(), 2);
        assert!(!controller.publication_owed());
    }

    #[tokio::test]
    async fn object_collection_does_not_block_a_durability_publication() {
        let (controller, _states, faults) = faulted_controller("controller_gc_publish", 11);
        controller.publish_state().await.unwrap();
        let gate = crate::test_support::FaultGate::new();
        faults.arm(ObjectFault::new(
            ObjectOp::List,
            ObjectPattern::Contains("/catalogs".to_string()),
            FaultAction::Park(gate.clone()),
        ));
        let collection = {
            let controller = Arc::clone(&controller);
            tokio::spawn(async move {
                controller
                    .gc_published(
                        crate::store::table::now_ms(),
                        StateGcPolicy {
                            pin_retention_ms: 600_000,
                            state_retention_ms: 600_000,
                            orphan_grace_ms: 86_400_000,
                            sweep_orphans: false,
                        },
                    )
                    .await
                    .unwrap()
            })
        };
        gate.entered().await;

        controller
            .catalog
            .set_forward_cursor("hub", TABLE, 5)
            .unwrap();
        let published = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            controller.publish_state(),
        )
        .await
        .expect("publication must not wait for object collection")
        .unwrap();
        assert_eq!(published.revision().get(), 2);

        gate.release();
        collection.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn owed_publications_coalesce_until_the_interval_elapses() {
        let remote_dir = crate::test_support::unique_dir("controller_publish_throttle");
        let remote = Arc::new(
            build_remote_object_store(remote_dir.to_str().unwrap())
                .unwrap()
                .unwrap(),
        );
        let states = Arc::new(ObjectTableStateStore::new(remote.clone()));
        let catalog = registered_catalog();
        let controller =
            object_controller(remote_dir, Arc::clone(&catalog), remote, states.clone(), 11);
        controller.publish_state().await.unwrap();

        controller
            .commit_owing_publication(|| {
                let revision = catalog.set_forward_cursor("hub", TABLE, 9)?;
                Ok((revision, ()))
            })
            .unwrap();
        controller.publish_owed().await.unwrap();

        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().revision().get(),
            1
        );
        assert!(controller.publication_owed());

        tokio::time::advance(MIN_PUBLISH_INTERVAL).await;
        controller.publish_owed().await.unwrap();

        assert_eq!(
            states.load(TABLE).await.unwrap().unwrap().revision().get(),
            2
        );
        assert!(!controller.publication_owed());
    }
}
