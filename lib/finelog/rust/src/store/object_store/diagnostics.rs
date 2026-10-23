use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::admission::RequestHistory;

const CACHE_LOG_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Clone, Copy)]
pub(super) enum FillReason {
    Query,
    Required,
    Repair,
}

#[derive(Default)]
pub(super) struct FillCounters {
    objects: AtomicU64,
    bytes: AtomicU64,
}

impl FillCounters {
    pub fn record(&self, bytes: u64) {
        self.objects.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

pub(super) struct CacheDiagnostics {
    last_log: Mutex<Instant>,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub query_admitted: AtomicU64,
    pub query_cold: AtomicU64,
    pub query_duplicate: AtomicU64,
    pub query_object_limit: AtomicU64,
    pub query_byte_limit: AtomicU64,
    pub warm_busy: AtomicU64,
    pub warm_errors: AtomicU64,
    pub cleanup_batches: AtomicU64,
    pub cleanup_lock_wait_us: AtomicU64,
    pub cleanup_lock_hold_us: AtomicU64,
    pub cleanup_max_lock_hold_us: AtomicU64,
    query: FillCounters,
    required: FillCounters,
    repair: FillCounters,
    pub writes: FillCounters,
}

impl Default for CacheDiagnostics {
    fn default() -> Self {
        Self {
            last_log: Mutex::new(Instant::now()),
            hits: AtomicU64::default(),
            misses: AtomicU64::default(),
            query_admitted: AtomicU64::default(),
            query_cold: AtomicU64::default(),
            query_duplicate: AtomicU64::default(),
            query_object_limit: AtomicU64::default(),
            query_byte_limit: AtomicU64::default(),
            warm_busy: AtomicU64::default(),
            warm_errors: AtomicU64::default(),
            cleanup_batches: AtomicU64::default(),
            cleanup_lock_wait_us: AtomicU64::default(),
            cleanup_lock_hold_us: AtomicU64::default(),
            cleanup_max_lock_hold_us: AtomicU64::default(),
            query: FillCounters::default(),
            required: FillCounters::default(),
            repair: FillCounters::default(),
            writes: FillCounters::default(),
        }
    }
}

impl CacheDiagnostics {
    pub fn fills(&self, reason: FillReason) -> &FillCounters {
        match reason {
            FillReason::Query => &self.query,
            FillReason::Required => &self.required,
            FillReason::Repair => &self.repair,
        }
    }

    /// Counters describe the interval; gauges describe the instant of emission.
    /// Lookups include maintenance and download rechecks, not just SQL scans.
    pub fn log_if_due(
        &self,
        history: &Mutex<RequestHistory>,
        active_fetches: usize,
        staged_objects: usize,
    ) {
        let mut last = self.last_log.lock().unwrap();
        let elapsed = last.elapsed();
        if elapsed < CACHE_LOG_INTERVAL {
            return;
        }
        *last = Instant::now();
        let history = history.lock().unwrap().take_diagnostics();
        tracing::info!(
            interval_ms = elapsed.as_millis() as u64,
            lookup_hits = take(&self.hits),
            lookup_misses = take(&self.misses),
            query_admitted = take(&self.query_admitted),
            query_cold = take(&self.query_cold),
            query_duplicate = take(&self.query_duplicate),
            query_object_limit = take(&self.query_object_limit),
            query_byte_limit = take(&self.query_byte_limit),
            warm_busy = take(&self.warm_busy),
            warm_errors = take(&self.warm_errors),
            query_fill_objects = take(&self.query.objects),
            query_fill_bytes = take(&self.query.bytes),
            required_fill_objects = take(&self.required.objects),
            required_fill_bytes = take(&self.required.bytes),
            repair_fill_objects = take(&self.repair.objects),
            repair_fill_bytes = take(&self.repair.bytes),
            write_objects = take(&self.writes.objects),
            write_bytes = take(&self.writes.bytes),
            history_entries = history.entries,
            history_prunes = history.prunes,
            history_pruned_entries = history.pruned_entries,
            active_fetches,
            staged_objects,
            cleanup_batches = take(&self.cleanup_batches),
            cleanup_lock_wait_us = take(&self.cleanup_lock_wait_us),
            cleanup_lock_hold_us = take(&self.cleanup_lock_hold_us),
            cleanup_max_lock_hold_us = take(&self.cleanup_max_lock_hold_us),
            "object cache activity"
        );
    }
}

fn take(counter: &AtomicU64) -> u64 {
    counter.swap(0, Ordering::Relaxed)
}
