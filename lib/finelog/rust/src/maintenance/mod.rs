//! Background maintenance cadence and its process-wide resource limits.
//!
//! [`MaintenanceScheduler`](scheduler::MaintenanceScheduler) is the only cadence
//! owner in the process. Tables run no timer loops of their own: the scheduler
//! polls the registry, decides which table owes flush or maintenance work, and
//! dispatches it under the limits held in [`MaintenanceLimits`]. A table's
//! controller still decides whether the work it is asked for is legal; the
//! scheduler never edits durable state or files.

pub mod scheduler;

use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Rate cap on sealing L0 segments. A table that keeps receiving appends seals
/// at most one segment per interval, so a write burst coalesces instead of
/// producing one tiny segment per nudge. A buffer that already holds a whole
/// segment bypasses the cap.
pub const MIN_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// How often a table with a pending physical-layout migration is polled. The
/// streaming rebuild continues at this cadence instead of waiting out the
/// ordinary compaction check interval.
pub const LAYOUT_MIGRATION_RETRY_INTERVAL: Duration = Duration::from_millis(100);

/// Longest the scheduler parks when no table has nearer work. Bounds how long a
/// table registered after the current round waits for its first poll.
pub const MAX_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Shortest the scheduler parks between rounds, so a table whose work is
/// perpetually due cannot turn the loop into a spin.
pub const MIN_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How often an object-backed table collects superseded state documents.
pub const OBJECT_GC_INTERVAL: Duration = Duration::from_secs(5 * 60);

/// How long an object must have been unreferenced by every retained state
/// before object GC deletes it. A sweep deletes an object only once it is both
/// unreferenced and older than this, so a reader that opened a stale state, or
/// a writer whose commit raced the sweep, still finds its bytes.
pub const OBJECT_ORPHAN_GRACE: Duration = Duration::from_secs(24 * 60 * 60);

/// Wall-clock budget one maintenance cycle spends re-encoding segments whose
/// physical layout predates the current writer policy.
///
/// Budgeted by time rather than by count because segment sizes span three orders
/// of magnitude — on the marin hub a small L1 re-encodes in milliseconds where a
/// 290 MiB terminal segment takes about 11 s — so a fixed count leaves the
/// cycle's duration unpredictable and can starve compaction, sync, and eviction
/// queued behind it. A single over-budget segment can still overrun, because a
/// rewrite already in flight is never abandoned.
///
/// The work is a storage and footer-size win rather than a correctness fix, so
/// it stays a minority of the cycle.
pub const REWRITE_LAYOUT_BUDGET: Duration = Duration::from_secs(3);

/// Process-wide concurrency limits every table's maintenance shares.
///
/// One instance exists per store. Holding these here rather than per table is
/// what keeps a store with many tables from saturating the process with
/// concurrent scans, compression, and Parquet rewrites.
pub struct MaintenanceLimits {
    /// At most one historical projection/index backfill runs at a time, so a
    /// cold store cannot spend every core re-serializing old segments.
    index_backfill: tokio::sync::Mutex<()>,
    /// At most one two-worker physical-layout migration wave runs at a time.
    layout_migration: Mutex<()>,
    /// How many tables may run a maintenance cycle concurrently.
    maintenance_cycles: tokio::sync::Semaphore,
    /// A table whose spec migration is mid-flight cycles in this dedicated
    /// slot instead of a shared maintenance slot. Its heavy CPU is already
    /// paced by the nice(19) rewrite thread, and queuing a many-hundred-batch
    /// backfill behind other tables' multi-minute backlog drains stretches it
    /// from hours into days.
    spec_migration: tokio::sync::Mutex<()>,
    /// How many tables may flush concurrently. Flushes are short and are the
    /// durability path, so this is looser than the maintenance limit.
    flushes: tokio::sync::Semaphore,
}

/// Concurrent maintenance cycles across all tables.
const MAX_CONCURRENT_MAINTENANCE_CYCLES: usize = 2;

/// Concurrent flushes across all tables.
const MAX_CONCURRENT_FLUSHES: usize = 4;

impl MaintenanceLimits {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            index_backfill: tokio::sync::Mutex::new(()),
            layout_migration: Mutex::new(()),
            maintenance_cycles: tokio::sync::Semaphore::new(MAX_CONCURRENT_MAINTENANCE_CYCLES),
            spec_migration: tokio::sync::Mutex::new(()),
            flushes: tokio::sync::Semaphore::new(MAX_CONCURRENT_FLUSHES),
        })
    }

    pub fn index_backfill(&self) -> &tokio::sync::Mutex<()> {
        &self.index_backfill
    }

    pub fn layout_migration(&self) -> &Mutex<()> {
        &self.layout_migration
    }

    pub fn maintenance_cycles(&self) -> &tokio::sync::Semaphore {
        &self.maintenance_cycles
    }

    pub fn spec_migration(&self) -> &tokio::sync::Mutex<()> {
        &self.spec_migration
    }

    pub fn flushes(&self) -> &tokio::sync::Semaphore {
        &self.flushes
    }
}
