//! The durable state values one table's commits move between.
//!
//! One table has one monotonic [`TableRevision`], allocated by the transaction
//! that mutates its rows and mirrored into the published object catalog. A
//! [`CommitToken`] proves a revision is durable, and a [`TableSnapshot`] is the
//! immutable read view of one committed state. The commit path that produces
//! them is
//! [`TableController`](crate::store::table::TableController).
//!
//! A revision never decreases. A publication whose outcome is unknown is
//! settled against HEAD by [`resolve_publication`]; a revision that is
//! committed locally but not published is repaired by publishing that same
//! revision again, never by undoing the local commit.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{NamespaceCatalog, ObjectRef};
use crate::store::catalog::state_store::{BackendToken, StoredTableState};
use crate::store::object_store::ObjectVersion;
use crate::store::types::SegmentRow;

/// Monotonic durable revision of one table's state.
///
/// Wraps `table_heads.catalog_generation`. Revision zero means the table has
/// no durable state yet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TableRevision(u64);

impl TableRevision {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for TableRevision {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Identity of the process that owns durable writes for a table.
///
/// Wraps the writer epoch recorded in a published HEAD. Zero means no writer
/// has claimed the table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriterFence(u64);

impl WriterFence {
    pub const UNCLAIMED: WriterFence = WriterFence(0);

    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for WriterFence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Proof that one table revision is durable.
#[derive(Clone, Debug)]
pub struct CommitToken {
    revision: TableRevision,
    fence: WriterFence,
    /// CAS token of the published HEAD naming this revision. Absent while the
    /// revision lives only in the local catalog.
    head_version: Option<ObjectVersion>,
}

impl CommitToken {
    /// A revision that is durable locally and not published.
    pub fn local(revision: TableRevision, fence: WriterFence) -> Self {
        Self {
            revision,
            fence,
            head_version: None,
        }
    }

    pub fn published(
        revision: TableRevision,
        fence: WriterFence,
        head_version: ObjectVersion,
    ) -> Self {
        Self {
            revision,
            fence,
            head_version: Some(head_version),
        }
    }

    pub fn revision(&self) -> TableRevision {
        self.revision
    }

    pub fn fence(&self) -> WriterFence {
        self.fence
    }

    /// The backend CAS token a fenced commit presents as its expected version.
    pub fn head_version(&self) -> Option<&ObjectVersion> {
        self.head_version.as_ref()
    }
}

/// The complete durable state of one table at a revision.
#[derive(Clone, Debug)]
pub struct TableState {
    catalog: NamespaceCatalog,
}

impl TableState {
    pub fn new(catalog: NamespaceCatalog) -> Self {
        Self { catalog }
    }

    pub fn revision(&self) -> TableRevision {
        TableRevision::new(self.catalog.catalog_generation.unwrap_or(0))
    }

    pub fn catalog(&self) -> &NamespaceCatalog {
        &self.catalog
    }
}

/// Immutable read view of one published table state and its commit token.
#[derive(Clone, Debug)]
pub struct TableSnapshot {
    state: TableState,
    token: CommitToken,
}

impl TableSnapshot {
    pub fn new(state: TableState, token: CommitToken) -> Self {
        Self { state, token }
    }

    /// Adapt the state a [`TableStateStore`](crate::store::catalog::state_store::TableStateStore)
    /// selected.
    pub fn from_stored(stored: &StoredTableState) -> Self {
        let state = TableState::new(stored.catalog.clone());
        let token = match stored.token() {
            BackendToken::Head(version) => {
                CommitToken::published(stored.revision(), stored.fence(), version.clone())
            }
            BackendToken::Local(revision) => CommitToken::local(*revision, stored.fence()),
        };
        Self { state, token }
    }

    pub fn revision(&self) -> TableRevision {
        self.token.revision()
    }

    pub fn fence(&self) -> WriterFence {
        self.token.fence()
    }

    pub fn state(&self) -> &TableState {
        &self.state
    }

    pub fn token(&self) -> CommitToken {
        self.token.clone()
    }

    /// Whether this snapshot selects exactly `state`, comparing the revision
    /// and the complete state value HEAD's catalog reference is hashed over.
    pub fn selects(&self, state: &TableState) -> bool {
        self.revision() == state.revision() && self.state.catalog == state.catalog
    }
}

/// One immutable data object, the catalog row that references it, and the
/// derived artifacts it advertises.
#[derive(Clone, Debug)]
pub struct SegmentDescriptor {
    pub row: SegmentRow,
    pub source: ObjectRef,
    pub artifacts: ArtifactReferences,
}

/// Exact references to one segment's derived index artifacts.
///
/// Membership is by reference. An artifact belongs to a segment because the
/// segment names it, never because a file sits beside the segment's Parquet.
/// `binding` carries the identity the FIDX container itself records — the
/// Parquet footer UUID and row count — so a reader rejects an artifact bound to
/// a different physical segment even when both resolve to the same filename.
///
/// This is the durable form: object identities only. The local files those
/// objects resolve to are [`LocalArtifacts`], derived at read time from the
/// object store.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ArtifactReferences {
    pub binding: SourceBinding,
    /// The FIDX bundle object, absent when the segment has none.
    pub bundle: Option<ObjectRef>,
    /// External covering-projection Parquet objects the bundle describes, keyed
    /// by projection name.
    pub projections: BTreeMap<String, ObjectRef>,
}

impl ArtifactReferences {
    pub fn is_empty(&self) -> bool {
        self.bundle.is_none() && self.projections.is_empty()
    }
}

/// The physical segment a set of artifacts is bound to.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceBinding {
    /// Parquet footer UUID, rendered as text. Absent for a segment whose footer
    /// carries none.
    pub segment_uuid: Option<String>,
    pub row_count: i64,
}

/// The local files one segment's artifacts resolve to.
///
/// A query holds these, never a rule for deriving them. An object-backed
/// segment resolves each path from its object identity; a legacy table's
/// sidecars are recorded once when the segment is adopted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LocalArtifacts {
    pub binding: SourceBinding,
    pub bundle: Option<PathBuf>,
    pub projections: BTreeMap<String, PathBuf>,
}

impl LocalArtifacts {
    pub fn is_empty(&self) -> bool {
        self.bundle.is_none() && self.projections.is_empty()
    }
}

/// Why a durable state transition did not complete.
#[derive(Debug)]
pub enum CommitError {
    /// Nothing was committed. The caller may retry the whole operation.
    NotCommitted(StatsError),
    /// The revision is durable in the local catalog but HEAD does not name it.
    /// Only republishing that same revision repairs it; retrying the operation
    /// would duplicate committed state.
    PublicationDeferred(StatsError),
    /// Another writer owns this table's published state. This process must
    /// stop mutating the table and replan from the selected state.
    Fenced(StatsError),
}

impl CommitError {
    /// Whether durable local state advanced despite the failure.
    pub fn is_committed(&self) -> bool {
        !matches!(self, CommitError::NotCommitted(_))
    }
}

impl From<CommitError> for StatsError {
    fn from(error: CommitError) -> StatsError {
        match error {
            CommitError::NotCommitted(error)
            | CommitError::PublicationDeferred(error)
            | CommitError::Fenced(error) => error,
        }
    }
}

impl std::fmt::Display for CommitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommitError::NotCommitted(error)
            | CommitError::PublicationDeferred(error)
            | CommitError::Fenced(error) => write!(formatter, "{error}"),
        }
    }
}

/// A durable table revision and the mutation's own result.
pub struct Committed<T> {
    pub token: CommitToken,
    pub output: T,
}

/// Settle a publication whose outcome the object store did not report.
///
/// `published` is the state HEAD selects after the failure. HEAD recording
/// another writer's fence means this writer no longer owns the table, whatever
/// revision HEAD holds. Otherwise the attempted state is durable when HEAD
/// names it, or when this writer has already published a later revision that
/// contains it; HEAD behind the attempted revision means the publication did
/// not apply and must be retried at the same revision.
pub fn resolve_publication(
    table: &str,
    attempted: &TableState,
    fence: WriterFence,
    published: Option<&TableSnapshot>,
    error: StatsError,
) -> Result<TableSnapshot, CommitError> {
    let Some(published) = published else {
        return Err(CommitError::PublicationDeferred(error));
    };
    if published.fence() != fence {
        return Err(CommitError::Fenced(StatsError::SchemaConflict(format!(
            "table {table:?} is published at revision {} by writer {}, fencing writer {fence}",
            published.revision(),
            published.fence()
        ))));
    }
    if published.revision() > attempted.revision() {
        return Ok(published.clone());
    }
    if published.revision() == attempted.revision() {
        if published.selects(attempted) {
            return Ok(published.clone());
        }
        return Err(CommitError::Fenced(StatsError::SchemaConflict(format!(
            "table {table:?} publishes a different state at revision {}",
            attempted.revision()
        ))));
    }
    Err(CommitError::PublicationDeferred(error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::catalog::object_state_store::TABLE_STATE_FORMAT_VERSION;

    fn state(revision: u64, active_version: u64) -> TableState {
        TableState::new(NamespaceCatalog {
            format_version: Some(TABLE_STATE_FORMAT_VERSION),
            namespace: Some("iris.worker".to_string()),
            catalog_generation: Some(revision),
            active_table_spec_version: Some(active_version),
            ..Default::default()
        })
    }

    fn snapshot(state: TableState, fence: u64) -> TableSnapshot {
        let revision = state.revision();
        TableSnapshot::new(state, CommitToken::local(revision, WriterFence::new(fence)))
    }

    fn lost_response() -> StatsError {
        StatsError::AmbiguousCommit("HEAD swap response was lost".to_string())
    }

    #[test]
    fn head_naming_the_attempted_state_makes_the_commit_durable() {
        let attempted = state(7, 2);
        let published = resolve_publication(
            "iris.worker",
            &attempted,
            WriterFence::new(11),
            Some(&snapshot(state(7, 2), 11)),
            lost_response(),
        )
        .unwrap();
        assert_eq!(published.revision(), TableRevision::new(7));
    }

    #[test]
    fn a_later_revision_from_the_same_writer_contains_the_attempted_state() {
        let attempted = state(7, 2);
        let published = resolve_publication(
            "iris.worker",
            &attempted,
            WriterFence::new(11),
            Some(&snapshot(state(9, 2), 11)),
            lost_response(),
        )
        .unwrap();
        assert_eq!(published.revision(), TableRevision::new(9));
    }

    #[test]
    fn head_behind_the_attempted_revision_defers_publication() {
        let attempted = state(7, 2);
        let error = resolve_publication(
            "iris.worker",
            &attempted,
            WriterFence::new(11),
            Some(&snapshot(state(6, 2), 11)),
            lost_response(),
        )
        .unwrap_err();
        assert!(matches!(error, CommitError::PublicationDeferred(_)));
        assert!(error.is_committed());
    }

    #[test]
    fn a_missing_head_defers_publication() {
        let attempted = state(1, 1);
        let error = resolve_publication(
            "iris.worker",
            &attempted,
            WriterFence::new(11),
            None,
            lost_response(),
        )
        .unwrap_err();
        assert!(matches!(error, CommitError::PublicationDeferred(_)));
    }

    #[test]
    fn a_writer_that_owns_head_behind_the_attempted_revision_fences_this_one() {
        let attempted = state(7, 2);
        let error = resolve_publication(
            "iris.worker",
            &attempted,
            WriterFence::new(11),
            Some(&snapshot(state(6, 2), 12)),
            lost_response(),
        )
        .unwrap_err();
        assert!(matches!(error, CommitError::Fenced(_)));
    }

    #[test]
    fn another_writer_at_the_attempted_revision_fences_this_one() {
        let attempted = state(7, 2);
        let error = resolve_publication(
            "iris.worker",
            &attempted,
            WriterFence::new(11),
            Some(&snapshot(state(7, 2), 12)),
            lost_response(),
        )
        .unwrap_err();
        assert!(matches!(error, CommitError::Fenced(_)));
        assert!(error.is_committed());
    }

    #[test]
    fn a_different_state_at_the_attempted_revision_fences_this_writer() {
        let attempted = state(7, 2);
        let error = resolve_publication(
            "iris.worker",
            &attempted,
            WriterFence::new(11),
            Some(&snapshot(state(7, 3), 11)),
            lost_response(),
        )
        .unwrap_err();
        assert!(matches!(error, CommitError::Fenced(_)));
    }
}
