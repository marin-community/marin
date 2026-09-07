//! Cursor-gated retirement for object-native forwarding relays.
//!
//! The forwarding cursor is the downstream settlement boundary. A relay may
//! remove a whole immutable segment from its current table state only when the
//! segment is older than the local visibility grace and its maximum sequence is
//! at or below that boundary. Older retained states continue to reference the
//! object until normal snapshot, rollback, and orphan collection windows pass.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use crate::errors::StatsError;
use crate::store::table::maintenance::WorkOutcome;
use crate::store::table::runtime::TableRuntime;
use crate::store::table_state::CommitError;

const RETIREMENT_GRACE: Duration = Duration::from_secs(15 * 60);
const SEGMENTS_PER_TICK: usize = 64;

/// Retire one bounded batch and report whether another immediate cycle is due.
pub async fn maintain(
    runtime: &Arc<TableRuntime>,
    target: &str,
) -> Result<WorkOutcome, StatsError> {
    let Some(cursor) = runtime.catalog.forward_cursor(target, runtime.name())? else {
        return Ok(WorkOutcome::Complete);
    };
    let cutoff_ms = crate::store::table::now_ms()
        .saturating_sub(i64::try_from(RETIREMENT_GRACE.as_millis()).unwrap_or(i64::MAX));
    let object_paths: HashSet<String> = runtime
        .catalog
        .object_segments(runtime.name())?
        .into_iter()
        .map(|record| record.path)
        .collect();
    let mut eligible = runtime
        .catalog
        .list_segments(runtime.name())?
        .into_iter()
        .filter(|segment| {
            object_paths.contains(&segment.path)
                && segment.max_seq <= cursor
                && segment.created_at_ms <= cutoff_ms
        })
        .collect::<Vec<_>>();
    eligible.sort_by(|left, right| {
        (left.max_seq, left.created_at_ms, &left.path).cmp(&(
            right.max_seq,
            right.created_at_ms,
            &right.path,
        ))
    });
    let pending = eligible.len() > SEGMENTS_PER_TICK;
    eligible.truncate(SEGMENTS_PER_TICK);
    if eligible.is_empty() {
        return Ok(WorkOutcome::Complete);
    }

    let paths = eligible
        .iter()
        .map(|segment| segment.path.clone())
        .collect::<Vec<_>>();
    let rows = eligible
        .iter()
        .map(|segment| segment.row_count)
        .sum::<i64>();
    let bytes = eligible
        .iter()
        .map(|segment| segment.byte_size)
        .sum::<i64>();
    let lifecycle = runtime.catalog.spec_lifecycle(runtime.name())?;
    let lease = runtime.controller.begin_compaction_for(&lifecycle)?;
    let committed = runtime
        .controller
        .commit_maintenance(&lease, || {
            let revision = runtime
                .catalog
                .retire_object_segments(runtime.name(), &paths)?;
            Ok((revision, ()))
        })
        .await;

    match committed {
        Ok(_) | Err(CommitError::PublicationDeferred(_)) => {
            runtime.segments.replace(&paths, Vec::new());
            tracing::info!(
                namespace = %runtime.name(),
                target,
                cursor,
                segments = paths.len(),
                rows,
                bytes,
                "retired downstream-settled relay segments"
            );
            Ok(WorkOutcome::from_pending(pending))
        }
        Err(CommitError::NotCommitted(StatsError::SchemaConflict(error))) => {
            tracing::info!(
                namespace = %runtime.name(),
                target,
                %error,
                "relay retirement lost a concurrent table-state change"
            );
            Ok(WorkOutcome::MoreWork)
        }
        Err(error) => Err(error.into()),
    }
}
