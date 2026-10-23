//! Admission for optional query cache fills. Required local reads and writes
//! bypass this policy because their local files are part of the storage contract.

use std::collections::{HashSet, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::{ObjectId, ObjectReference};

const QUERY_WARM_BYTES: u64 = 256 * 1024 * 1024;
const QUERY_WARM_OBJECTS: usize = 64;
const REQUEST_WINDOW: Duration = Duration::from_secs(10 * 60);
const REQUEST_HISTORY_OBJECTS: usize = 65_536;

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
    ) -> bool {
        let mut budget = self.state.lock().unwrap();
        // Replanning and self-joins must not turn one request into reuse.
        if !budget.seen.insert(reference.id.clone()) {
            return false;
        }
        let repeated = history
            .lock()
            .unwrap()
            .observe(&reference.id, Instant::now());
        if !repeated
            || budget.remaining_objects == 0
            || reference.version.byte_size > budget.remaining_bytes
        {
            return false;
        }
        budget.remaining_objects -= 1;
        budget.remaining_bytes -= reference.version.byte_size;
        true
    }
}

/// A bounded FIFO of first observations. An object qualifies on another query
/// within the window; old scans cannot accumulate popularity indefinitely.
#[derive(Default)]
pub(super) struct RequestHistory {
    first_seen: HashSet<ObjectId>,
    order: VecDeque<(ObjectId, Instant)>,
}

impl RequestHistory {
    fn observe(&mut self, id: &ObjectId, now: Instant) -> bool {
        while self
            .order
            .front()
            .is_some_and(|(_, first)| now.duration_since(*first) >= REQUEST_WINDOW)
        {
            let (expired, _) = self.order.pop_front().unwrap();
            self.first_seen.remove(&expired);
        }
        if self.first_seen.contains(id) {
            return true;
        }
        if self.order.len() == REQUEST_HISTORY_OBJECTS {
            let (oldest, _) = self.order.pop_front().unwrap();
            self.first_seen.remove(&oldest);
        }
        self.first_seen.insert(id.clone());
        self.order.push_back((id.clone(), now));
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_expires_and_history_evicts_old_observations() {
        let mut history = RequestHistory::default();
        let now = Instant::now();
        let id = ObjectId::table("log", "objects/old.parquet").unwrap();
        assert!(!history.observe(&id, now));
        assert!(history.observe(&id, now + REQUEST_WINDOW / 2));
        assert!(!history.observe(&id, now + REQUEST_WINDOW));
        for index in 0..REQUEST_HISTORY_OBJECTS {
            let other = ObjectId::table("log", &format!("objects/{index}.parquet")).unwrap();
            assert!(!history.observe(&other, now + REQUEST_WINDOW));
        }
        assert!(!history.observe(&id, now + REQUEST_WINDOW));
    }
}
