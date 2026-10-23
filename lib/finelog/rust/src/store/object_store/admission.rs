//! Admission for optional query cache fills. Required local reads and writes
//! bypass this policy because their local files are part of the storage contract.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{ObjectId, ObjectReference};

const QUERY_WARM_BYTES: u64 = 256 * 1024 * 1024;
const QUERY_WARM_OBJECTS: usize = 64;
const MIN_QUERY_REQUESTS: u8 = 2;
const REQUEST_WINDOW: Duration = Duration::from_secs(10 * 60);
const REQUESTS_PER_PRUNE: usize = 32_768;

pub(super) enum AdmissionDecision {
    Admitted,
    Duplicate,
    Cold,
    ObjectLimit,
    ByteLimit,
}

/// Shared by every table and scan in one query's SessionConfig.
pub struct QueryCacheAdmission {
    state: Mutex<QueryBudget>,
}

struct QueryBudget {
    seen: HashSet<ObjectId>,
    remaining_bytes: u64,
    remaining_objects: usize,
}

impl Default for QueryCacheAdmission {
    fn default() -> Self {
        Self::new(QUERY_WARM_BYTES, QUERY_WARM_OBJECTS)
    }
}

impl QueryCacheAdmission {
    pub(crate) fn new(bytes: u64, objects: usize) -> Self {
        Self {
            state: Mutex::new(QueryBudget {
                seen: HashSet::new(),
                remaining_bytes: bytes,
                remaining_objects: objects,
            }),
        }
    }

    pub(super) fn admit(
        &self,
        reference: &ObjectReference,
        history: &Mutex<RequestHistory>,
    ) -> AdmissionDecision {
        let mut budget = self.state.lock().unwrap();
        // Replanning and self-joins must not turn one request into reuse.
        if !budget.seen.insert(reference.id.clone()) {
            return AdmissionDecision::Duplicate;
        }
        let repeated = history
            .lock()
            .unwrap()
            .observe(&reference.id, Instant::now());
        if !repeated {
            return AdmissionDecision::Cold;
        }
        if budget.remaining_objects == 0 {
            return AdmissionDecision::ObjectLimit;
        }
        if reference.version.byte_size > budget.remaining_bytes {
            return AdmissionDecision::ByteLimit;
        }
        budget.remaining_objects -= 1;
        budget.remaining_bytes -= reference.version.byte_size;
        AdmissionDecision::Admitted
    }
}

/// Approximate recent demand without maintaining an eviction order. Each request
/// adds at most one to the total count, and each pruning halves it. The total
/// count, and therefore the entry count, stays below twice REQUESTS_PER_PRUNE.
#[derive(Default)]
pub(super) struct RequestHistory {
    objects: HashMap<ObjectId, RequestStats>,
    requests_since_prune: usize,
    prunes: u64,
    pruned_entries: usize,
}

struct RequestStats {
    requests: u8,
    last_seen: Instant,
}

impl RequestHistory {
    /// Record a request and return whether its decaying count demonstrates reuse.
    fn observe(&mut self, id: &ObjectId, now: Instant) -> bool {
        let stats = self.objects.entry(id.clone()).or_insert(RequestStats {
            requests: 0,
            last_seen: now,
        });
        if now.duration_since(stats.last_seen) >= REQUEST_WINDOW {
            stats.requests = 0;
        }
        stats.requests = stats.requests.saturating_add(1);
        stats.last_seen = now;
        let repeated = stats.requests >= MIN_QUERY_REQUESTS;

        self.requests_since_prune += 1;
        if self.requests_since_prune == REQUESTS_PER_PRUNE {
            let before = self.objects.len();
            self.objects.retain(|_, stats| {
                stats.requests /= 2;
                stats.requests > 0 && now.duration_since(stats.last_seen) < REQUEST_WINDOW
            });
            self.prunes += 1;
            self.pruned_entries += before - self.objects.len();
            self.requests_since_prune = 0;
        }
        repeated
    }

    pub(super) fn take_diagnostics(&mut self) -> HistoryDiagnostics {
        HistoryDiagnostics {
            entries: self.objects.len(),
            prunes: std::mem::take(&mut self.prunes),
            pruned_entries: std::mem::take(&mut self.pruned_entries),
        }
    }
}

pub(super) struct HistoryDiagnostics {
    pub entries: usize,
    pub prunes: u64,
    pub pruned_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_requests_refresh_the_idle_window() {
        let mut history = RequestHistory::default();
        let now = Instant::now();
        let id = ObjectId::table("log", "objects/old.parquet").unwrap();
        assert!(!history.observe(&id, now));
        assert!(history.observe(&id, now + REQUEST_WINDOW / 2));
        assert!(history.observe(&id, now + REQUEST_WINDOW));
        assert!(!history.observe(&id, now + REQUEST_WINDOW * 2));
    }

    #[test]
    fn periodic_pruning_preserves_reuse_and_forgets_one_off_scans() {
        let mut history = RequestHistory::default();
        let now = Instant::now();
        let hot = ObjectId::table("log", "objects/hot.parquet").unwrap();
        assert!(!history.observe(&hot, now));
        assert!(history.observe(&hot, now));
        let cold: Vec<_> = (0..REQUESTS_PER_PRUNE - 2)
            .map(|index| ObjectId::table("log", &format!("objects/{index}.parquet")).unwrap())
            .collect();
        for id in &cold {
            assert!(!history.observe(id, now));
        }
        // The oldest object survives because it was requested twice; the
        // one-off scan is forgotten in the same amortized pruning pass.
        assert!(history.observe(&hot, now));
        assert!(!history.observe(&cold[0], now));
    }
}
