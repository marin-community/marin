//! The process's single maintenance cadence.
//!
//! One task polls the table registry. For each live table it decides whether a
//! flush or a maintenance cycle is due, then dispatches the work under the
//! process-wide limits in [`MaintenanceLimits`]. Dispatched work is registered
//! with the table it belongs to, so a table shutting down (a re-register
//! replacement, a drop, or process shutdown) still joins or aborts it inside the
//! same bounded window a per-table task used to.
//!
//! The scheduler decides *when* work runs. Whether the work is legal — the
//! writer fence, the definition version, input liveness — remains the table
//! controller's decision, and the scheduler never touches durable state or
//! files itself.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::Notify;
use tokio::time::Instant;

use crate::maintenance::{
    MaintenanceLimits, LAYOUT_MIGRATION_RETRY_INTERVAL, MAX_POLL_INTERVAL, MIN_FLUSH_INTERVAL,
    MIN_POLL_INTERVAL,
};
use crate::store::store::ServeMode;
use crate::store::table::{TableManager, TableRuntime, TableWork};

/// What one table's scheduling decisions are based on between rounds.
struct TableCadence {
    last_flush: Instant,
    last_maintenance: Instant,
    flush_running: bool,
    maintenance_running: bool,
    /// Set from the previous cycle's outcome. A table rebuilding its physical
    /// layout is polled at [`LAYOUT_MIGRATION_RETRY_INTERVAL`] instead of its
    /// ordinary compaction check interval.
    migration_pending: bool,
    /// Cleared at the end of each round; an entry that stays false belongs to a
    /// table that has left the registry.
    live: bool,
}

impl TableCadence {
    /// A table joins the registry able to flush at once — its first append must
    /// not wait out a coalescing window it never used — and owing its first
    /// maintenance cycle one interval later.
    fn new(now: Instant) -> Self {
        Self {
            last_flush: now.checked_sub(MIN_FLUSH_INTERVAL).unwrap_or(now),
            last_maintenance: now,
            flush_running: false,
            maintenance_running: false,
            migration_pending: false,
            live: true,
        }
    }
}

pub struct MaintenanceScheduler {
    tables: Arc<TableManager>,
    limits: Arc<MaintenanceLimits>,
    mode: ServeMode,
    /// Signalled by an append that wants a flush and by dispatched work as it
    /// finishes, so neither waits out the poll interval.
    wake: Arc<Notify>,
    stop: Arc<Notify>,
    stopped: AtomicBool,
    cadence: Arc<Mutex<HashMap<String, TableCadence>>>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl MaintenanceScheduler {
    pub fn new(tables: Arc<TableManager>) -> Arc<Self> {
        let wake = Arc::clone(tables.maintenance_wake());
        let limits = Arc::clone(tables.maintenance_limits());
        let mode = tables.mode();
        Arc::new(Self {
            tables,
            limits,
            mode,
            wake,
            stop: Arc::new(Notify::new()),
            stopped: AtomicBool::new(false),
            cadence: Arc::new(Mutex::new(HashMap::new())),
            task: Mutex::new(None),
        })
    }

    /// Start the polling loop. Called once after bootstrap, before serving.
    ///
    /// A shadow store polls for flushes — a rehearsal still accepts writes and
    /// owes them durability — but never dispatches a maintenance cycle, so it
    /// cannot compact, evict, or rewrite the copy it was handed.
    pub fn start(self: &Arc<Self>) {
        let scheduler = Arc::clone(self);
        let handle = tokio::spawn(async move { scheduler.run().await });
        *self.task.lock().unwrap() = Some(handle);
    }

    /// Stop polling and join the loop. Dispatched work is owned by the tables it
    /// was registered with, which join or abort it during their own shutdown.
    pub async fn shutdown(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.stop.notify_waiters();
        let handle = self.task.lock().unwrap().take();
        if let Some(handle) = handle {
            let _ = handle.await;
        }
    }

    async fn run(self: Arc<Self>) {
        loop {
            if self.stopped.load(Ordering::SeqCst) {
                return;
            }
            let delay = self.poll_round();
            let stopped = self.stop.notified();
            let woken = self.wake.notified();
            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                _ = woken => {}
                _ = stopped => return,
            }
        }
    }

    /// Dispatch everything due across the registry and return how long to park
    /// before the next round.
    pub(crate) fn poll_round(&self) -> Duration {
        let now = Instant::now();
        let runtimes = self.tables.runtimes();
        let mut next = MAX_POLL_INTERVAL;
        let mut cadence = self.cadence.lock().unwrap();
        for entry in cadence.values_mut() {
            entry.live = false;
        }
        for runtime in runtimes {
            if runtime.is_stopped() || !runtime.is_disk_backed() {
                continue;
            }
            let entry = cadence
                .entry(runtime.name().to_string())
                .or_insert_with(|| TableCadence::new(now));
            entry.live = true;
            next = next.min(self.schedule_flush(&runtime, entry, now));
            next = next.min(self.schedule_maintenance(&runtime, entry, now));
        }
        cadence.retain(|_, entry| entry.live);
        next.clamp(MIN_POLL_INTERVAL, MAX_POLL_INTERVAL)
    }

    /// Flush the table if its buffer is due, and return the delay until its next
    /// flush decision.
    ///
    /// A buffer that already holds a whole segment flushes immediately; an
    /// ordinary nudge waits out [`MIN_FLUSH_INTERVAL`] so the appends in that
    /// window coalesce into one segment; a table that lost its nudge still
    /// flushes once its buffer reaches the definition's maximum flush age.
    fn schedule_flush(
        &self,
        runtime: &Arc<TableRuntime>,
        entry: &mut TableCadence,
        now: Instant,
    ) -> Duration {
        let demand = runtime.flush_demand();
        if entry.flush_running {
            return MAX_POLL_INTERVAL;
        }
        let since = now.saturating_duration_since(entry.last_flush);
        let wait = if demand.forced {
            Duration::ZERO
        } else if demand.requested {
            MIN_FLUSH_INTERVAL.saturating_sub(since)
        } else {
            demand.max_flush_age.saturating_sub(since)
        };
        if !wait.is_zero() {
            return wait;
        }
        entry.last_flush = now;
        entry.flush_running = true;
        // Clear before the flush runs so an append that lands during it re-arms
        // the demand rather than being swallowed by this cycle.
        runtime.clear_flush_demand();
        let table = runtime.name().to_string();
        let limits = Arc::clone(&self.limits);
        let wake = Arc::clone(&self.wake);
        let cadence = Arc::clone(&self.cadence);
        let tables = Arc::clone(&self.tables);
        let dispatched = runtime.clone().spawn_tracked({
            let runtime = Arc::clone(runtime);
            async move {
                let _permit = limits.flushes().acquire().await;
                if let Err(error) = tables.run_work(&runtime, TableWork::Flush).await {
                    tracing::warn!(namespace = %runtime.name(), %error, "scheduled flush failed");
                }
                if let Some(entry) = cadence.lock().unwrap().get_mut(&table) {
                    entry.flush_running = false;
                }
                wake.notify_one();
            }
        });
        if !dispatched {
            entry.flush_running = false;
        }
        MIN_FLUSH_INTERVAL
    }

    /// Run one maintenance cycle if the table is due, and return the delay until
    /// its next maintenance decision.
    fn schedule_maintenance(
        &self,
        runtime: &Arc<TableRuntime>,
        entry: &mut TableCadence,
        now: Instant,
    ) -> Duration {
        if self.mode == ServeMode::Shadow || entry.maintenance_running {
            return MAX_POLL_INTERVAL;
        }
        let interval = if entry.migration_pending {
            LAYOUT_MIGRATION_RETRY_INTERVAL
        } else {
            runtime.maintenance_interval()
        };
        let since = now.saturating_duration_since(entry.last_maintenance);
        let wait = interval.saturating_sub(since);
        if !wait.is_zero() {
            return wait;
        }
        entry.last_maintenance = now;
        entry.maintenance_running = true;
        let table = runtime.name().to_string();
        let limits = Arc::clone(&self.limits);
        let wake = Arc::clone(&self.wake);
        let cadence = Arc::clone(&self.cadence);
        let tables = Arc::clone(&self.tables);
        let dispatched = runtime.clone().spawn_tracked({
            let runtime = Arc::clone(runtime);
            async move {
                let _permit = limits.maintenance_cycles().acquire().await;
                let cycle = TableWork::Cycle {
                    force_compact_l0: false,
                };
                let migration_pending = match tables.run_work(&runtime, cycle).await {
                    Ok(outcome) => outcome.pending,
                    Err(error) => {
                        // Fall back to the ordinary interval after an error. A
                        // bad input must not turn this into a hot loop.
                        tracing::warn!(namespace = %runtime.name(), %error, "scheduled maintenance failed");
                        false
                    }
                };
                if let Some(entry) = cadence.lock().unwrap().get_mut(&table) {
                    entry.maintenance_running = false;
                    entry.migration_pending = migration_pending;
                }
                wake.notify_one();
            }
        });
        if !dispatched {
            entry.maintenance_running = false;
        }
        interval
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::catalog::Catalog;
    use crate::store::policy::StoragePolicy;
    use crate::store::schema::{with_implicit_seq, Column, Schema};
    use crate::store::table_state::WriterFence;

    const TABLE: &str = "iris.worker";

    fn worker_schema() -> Schema {
        with_implicit_seq(Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "timestamp_ms",
        ))
    }

    fn manager_with_one_table(dir: PathBuf, mode: ServeMode) -> Arc<TableManager> {
        std::fs::create_dir_all(dir.join(TABLE)).unwrap();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let manager = TableManager::new(
            Some(dir),
            mode,
            catalog,
            None,
            None,
            None,
            WriterFence::UNCLAIMED,
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
        );
        manager
            .register(TABLE, worker_schema(), StoragePolicy::default())
            .unwrap();
        manager
    }

    /// A rehearsal still owes its writers durability, so it flushes — but it
    /// must never compact, evict, or rewrite the copy it was handed. Gating only
    /// the boot path would leave a namespace registered at runtime unguarded.
    #[tokio::test(start_paused = true)]
    async fn a_shadow_store_schedules_flushes_but_no_maintenance_cycle() {
        let mut dispatched = Vec::new();
        for mode in [ServeMode::Live, ServeMode::Shadow] {
            let dir = crate::test_support::unique_dir("scheduler_mode");
            let manager = manager_with_one_table(dir.clone(), mode);
            let scheduler = MaintenanceScheduler::new(Arc::clone(&manager));
            // The first round only registers the table's cadence. Advancing past
            // both the flush age and the compaction check interval then makes
            // every kind of work this table can owe due.
            scheduler.poll_round();
            tokio::time::advance(Duration::from_secs(120)).await;
            scheduler.poll_round();
            dispatched.push(manager.require(TABLE).unwrap().background_task_count());
            std::fs::remove_dir_all(&dir).ok();
        }
        let (live, shadow) = (dispatched[0], dispatched[1]);
        assert_eq!(shadow, 1, "a shadow store dispatches only the flush");
        assert_eq!(
            live, 2,
            "a live store also dispatches its maintenance cycle"
        );
    }

    /// The coalescing window is the only thing between a write burst and one
    /// tiny L0 per append, so an ordinary nudge waits it out while a buffer that
    /// already holds a whole segment does not.
    #[tokio::test(start_paused = true)]
    async fn a_full_buffer_bypasses_the_flush_coalescing_window() {
        let dir = crate::test_support::unique_dir("scheduler_flush_window");
        let manager = manager_with_one_table(dir.clone(), ServeMode::Live);
        let scheduler = MaintenanceScheduler::new(Arc::clone(&manager));
        let table = manager.require(TABLE).unwrap();

        table.request_flush(false);
        scheduler.poll_round();
        assert!(
            !table.flush_demand().requested,
            "a table's first nudge flushes without waiting"
        );
        // Let the dispatched flush finish, still well inside the window.
        tokio::time::sleep(Duration::from_millis(100)).await;

        table.request_flush(false);
        scheduler.poll_round();
        assert!(
            table.flush_demand().requested,
            "a nudge inside the coalescing window is deferred, not dropped"
        );

        table.request_flush(true);
        scheduler.poll_round();
        assert!(
            !table.flush_demand().requested,
            "a full buffer flushes without waiting out the window"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
