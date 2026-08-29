//! The operating knobs a table's active specification resolves to.
//!
//! A table's durable specification is versioned and lives in the catalog. The
//! runtime reads the few values it needs on every flush, compaction, and
//! collection decision from this snapshot instead of re-reading the
//! specification each time.

use std::time::Duration;

use crate::proto::finelog::stats::{L0Mode, SourceLayout};
use crate::store::catalog::TableSpecStatus;

/// Buffered-byte size at which an append forces an early flush, short-circuiting
/// the flush-rate cooldown so a write burst can't buffer unboundedly (and bounds
/// a single L0's size).
pub const SEGMENT_TARGET_BYTES: i64 = 100 * 1024 * 1024;

/// Maximum idle gap before a buffer is flushed on age alone. With steady writes
/// the per-append nudge drives flushes; this is the ceiling for a quiet table.
pub const DEFAULT_FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// The operating policy one table runs under.
#[derive(Debug, Clone)]
pub struct TableRuntimePolicy {
    pub l0_mode: L0Mode,
    pub table_spec_version: u64,
    pub max_buffer_bytes: i64,
    pub max_flush_age: Duration,
    pub max_query_time_ms: u64,
    pub rollback_window_ms: u64,
    pub target_object_bytes: i64,
    pub source_layout: Option<SourceLayout>,
}

impl Default for TableRuntimePolicy {
    fn default() -> Self {
        Self {
            l0_mode: L0Mode::L0_MODE_LEGACY_LOCAL,
            table_spec_version: 0,
            max_buffer_bytes: SEGMENT_TARGET_BYTES,
            max_flush_age: DEFAULT_FLUSH_INTERVAL,
            max_query_time_ms: crate::store::table_spec::DEFAULT_MAX_QUERY_TIME_MS,
            rollback_window_ms: crate::store::table_spec::DEFAULT_ROLLBACK_WINDOW_MS,
            target_object_bytes: crate::store::table_spec::DEFAULT_TARGET_OBJECT_BYTES as i64,
            source_layout: None,
        }
    }
}

impl TableRuntimePolicy {
    /// Resolve the desired specification's policy, falling back to the active
    /// one and then to the defaults a table without a specification runs under.
    pub fn from_status(status: &TableSpecStatus) -> Self {
        let Some(spec) = status.desired.as_ref().or(status.active.as_ref()) else {
            return Self::default();
        };
        let Some(operating) = spec.operating_policy.as_option() else {
            return Self::default();
        };
        let l0_mode = operating
            .l0_mode
            .and_then(|mode| mode.as_known())
            .filter(|mode| *mode != L0Mode::L0_MODE_UNSPECIFIED)
            .unwrap_or(L0Mode::L0_MODE_LEGACY_LOCAL);
        Self {
            l0_mode,
            table_spec_version: spec.version.unwrap_or(0),
            max_buffer_bytes: i64::try_from(
                operating
                    .max_buffer_bytes
                    .unwrap_or(SEGMENT_TARGET_BYTES as u64),
            )
            .unwrap_or(i64::MAX),
            max_flush_age: Duration::from_millis(
                operating
                    .max_flush_age_ms
                    .unwrap_or(DEFAULT_FLUSH_INTERVAL.as_millis() as u64),
            ),
            max_query_time_ms: crate::store::table_spec::max_query_time_ms(spec),
            rollback_window_ms: crate::store::table_spec::rollback_window_ms(spec),
            target_object_bytes: spec
                .source_layout
                .as_option()
                .and_then(|layout| layout.target_object_bytes)
                .and_then(|bytes| i64::try_from(bytes).ok())
                .unwrap_or(crate::store::table_spec::DEFAULT_TARGET_OBJECT_BYTES as i64),
            source_layout: spec.source_layout.as_option().cloned(),
        }
    }

    /// Whether this table's L0 is written as immutable objects rather than local
    /// files.
    pub fn object_backed(&self) -> bool {
        self.l0_mode == L0Mode::L0_MODE_OBJECT_STORE
    }
}
