//! One request's absolute read deadline, shared by every table and scan.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

pub(crate) struct ReadScope {
    pub index_reads: Arc<crate::indices::input::IndexReadStats>,
    deadline: Mutex<Option<tokio::time::Instant>>,
    pub candidates: AtomicU64,
    pub prepared: AtomicU64,
    pub pruned: AtomicU64,
    pub source_opens: AtomicU64,
    pub range_reads: AtomicU64,
    pub range_bytes: AtomicU64,
    active: AtomicU64,
    peak_active: AtomicU64,
    first_batch: AtomicU64,
    started: tokio::time::Instant,
    stages: Mutex<Stages>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum ReadPhase {
    #[default]
    Setup,
    LogicalPlan,
    PhysicalPlan,
    Execution,
    Normalize,
    Complete,
}

struct Stages {
    phase: ReadPhase,
    since: tokio::time::Instant,
    logical: u64,
    physical: u64,
    execution: u64,
}

pub(crate) struct ReadReport {
    pub phase: ReadPhase,
    pub logical_ms: u64,
    pub physical_ms: u64,
    pub execution_ms: u64,
    pub candidates: u64,
    pub prepared: u64,
    pub pruned: u64,
    pub source_opens: u64,
    pub range_reads: u64,
    pub range_bytes: u64,
    pub peak_active: u64,
    pub first_batch_ms: Option<u64>,
}

pub(crate) struct FilePreparation(Arc<ReadScope>);

#[derive(Debug)]
enum ReadOutcome {
    Cancelled,
    Succeeded,
    Failed,
}

pub(crate) struct ReadTrace {
    scope: Arc<ReadScope>,
    sql: String,
    kind: &'static str,
    outcome: ReadOutcome,
}

impl ReadTrace {
    pub fn new(scope: Arc<ReadScope>, sql: &str, kind: &'static str) -> Self {
        Self {
            scope,
            sql: super::truncate_sql_for_log(sql),
            kind,
            outcome: ReadOutcome::Cancelled,
        }
    }

    pub fn finish<T>(&mut self, result: &datafusion::error::Result<T>) {
        self.outcome = if result.is_ok() {
            ReadOutcome::Succeeded
        } else {
            ReadOutcome::Failed
        };
    }
}

impl Drop for ReadTrace {
    fn drop(&mut self) {
        let elapsed = self.scope.started.elapsed().as_millis();
        if elapsed < super::slow_query_log_ms() && matches!(self.outcome, ReadOutcome::Succeeded) {
            return;
        }
        let report = self.scope.report();
        tracing::warn!(
            kind = self.kind,
            outcome = ?self.outcome,
            phase = ?report.phase,
            elapsed_ms = elapsed as u64,
            logical_plan_ms = report.logical_ms,
            physical_plan_ms = report.physical_ms,
            execution_ms = report.execution_ms,
            first_batch_ms = report.first_batch_ms,
            candidate_files = report.candidates,
            prepared_files = report.prepared,
            pruned_files = report.pruned,
            source_opens = report.source_opens,
            source_range_reads = report.range_reads,
            source_range_bytes = report.range_bytes,
            peak_file_preparations = report.peak_active,
            index_range_reads = self.scope.index_reads.range_reads.load(Ordering::Relaxed),
            index_range_bytes = self.scope.index_reads.range_bytes.load(Ordering::Relaxed),
            index_cache_hits = self.scope.index_reads.cache_hits.load(Ordering::Relaxed),
            index_unavailable = self.scope.index_reads.unavailable.load(Ordering::Relaxed),
            index_timeouts = self.scope.index_reads.timed_out.load(Ordering::Relaxed),
            index_corrupt = self.scope.index_reads.corrupt.load(Ordering::Relaxed),
            index_stale = self.scope.index_reads.stale.load(Ordering::Relaxed),
            index_missing_bundles = self.scope.index_reads.missing_bundle.load(Ordering::Relaxed),
            sql = self.sql,
            "read execution report",
        );
    }
}

impl Drop for FilePreparation {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Default for ReadScope {
    fn default() -> Self {
        let started = tokio::time::Instant::now();
        Self {
            index_reads: Arc::new(crate::indices::input::IndexReadStats::default()),
            deadline: Mutex::new(None),
            candidates: AtomicU64::new(0),
            prepared: AtomicU64::new(0),
            pruned: AtomicU64::new(0),
            source_opens: AtomicU64::new(0),
            range_reads: AtomicU64::new(0),
            range_bytes: AtomicU64::new(0),
            active: AtomicU64::new(0),
            peak_active: AtomicU64::new(0),
            first_batch: AtomicU64::new(u64::MAX),
            started,
            stages: Mutex::new(Stages {
                phase: ReadPhase::Setup,
                since: started,
                logical: 0,
                physical: 0,
                execution: 0,
            }),
        }
    }
}

impl ReadScope {
    pub fn constrain(&self, deadline: Option<tokio::time::Instant>) {
        let Some(deadline) = deadline else {
            return;
        };
        let mut current = self.deadline.lock().unwrap();
        *current = Some(current.map_or(deadline, |current| current.min(deadline)));
    }

    pub fn deadline(&self) -> Option<tokio::time::Instant> {
        *self.deadline.lock().unwrap()
    }

    pub fn phase(&self, phase: ReadPhase) {
        let mut stages = self.stages.lock().unwrap();
        stages.finish_interval();
        stages.phase = phase;
    }

    pub fn prepare(self: &Arc<Self>) -> FilePreparation {
        self.prepared.fetch_add(1, Ordering::Relaxed);
        let active = self.active.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_active.fetch_max(active, Ordering::Relaxed);
        FilePreparation(Arc::clone(self))
    }

    pub fn first_batch(&self) {
        let _ = self.first_batch.compare_exchange(
            u64::MAX,
            self.started.elapsed().as_millis() as u64,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    pub fn report(&self) -> ReadReport {
        let mut stages = self.stages.lock().unwrap();
        stages.finish_interval();
        let first_batch = self.first_batch.load(Ordering::Relaxed);
        ReadReport {
            phase: stages.phase,
            logical_ms: stages.logical,
            physical_ms: stages.physical,
            execution_ms: stages.execution,
            candidates: self.candidates.load(Ordering::Relaxed),
            prepared: self.prepared.load(Ordering::Relaxed),
            pruned: self.pruned.load(Ordering::Relaxed),
            source_opens: self.source_opens.load(Ordering::Relaxed),
            range_reads: self.range_reads.load(Ordering::Relaxed),
            range_bytes: self.range_bytes.load(Ordering::Relaxed),
            peak_active: self.peak_active.load(Ordering::Relaxed),
            first_batch_ms: (first_batch != u64::MAX).then_some(first_batch),
        }
    }
}

impl Stages {
    fn finish_interval(&mut self) {
        let now = tokio::time::Instant::now();
        let elapsed = (now - self.since).as_millis() as u64;
        match self.phase {
            ReadPhase::LogicalPlan => self.logical += elapsed,
            ReadPhase::PhysicalPlan => self.physical += elapsed,
            ReadPhase::Execution => self.execution += elapsed,
            _ => {}
        }
        self.since = now;
    }
}
