// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Fixtures shared by tests across the crate's modules. Module-specific ones (a served
//! store, a Connect client) live beside the module they exercise.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::errors::StatsError;
use crate::store::object_store::{
    ObjectId, ObjectMetadata, ObjectPrefix, ObjectReference, ObjectStore, ObjectVersion,
    StoredObject,
};

/// A fresh directory under the system temp dir, unique per call.
pub fn unique_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "finelog_{tag}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One operation of the [`ObjectStore`] contract, as a fault matches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectOp {
    Write,
    Read,
    LocalPath,
    CompareAndSwap,
}

/// Which object IDs a fault applies to.
#[derive(Clone, Debug)]
pub enum ObjectPattern {
    /// Object IDs whose key ends with this text, e.g. `"HEAD.json"`.
    EndsWith(String),
    /// Object IDs whose key contains this text, e.g. `"/objects/"`.
    Contains(String),
}

impl ObjectPattern {
    fn matches(&self, key: &str) -> bool {
        match self {
            ObjectPattern::EndsWith(suffix) => key.ends_with(suffix.as_str()),
            ObjectPattern::Contains(text) => key.contains(text.as_str()),
        }
    }
}

/// A rendezvous between a parked object operation and the test that released it.
///
/// [`FaultGate::entered`] resolves once the operation has parked, so a test
/// never guesses when a writer reached the fault; [`FaultGate::release`] lets it
/// continue.
pub struct FaultGate {
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}

impl FaultGate {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            entered: tokio::sync::Semaphore::new(0),
            release: tokio::sync::Semaphore::new(0),
        })
    }

    /// Wait until a faulted operation parks on this gate.
    pub async fn entered(&self) {
        self.entered.acquire().await.unwrap().forget();
    }

    /// Let one parked operation continue.
    pub fn release(&self) {
        self.release.add_permits(1);
    }

    async fn park(&self) {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
    }
}

/// What a matched operation does instead of running normally.
#[derive(Clone)]
pub enum FaultAction {
    /// Fail without performing the operation.
    Fail(StatsError),
    /// Perform the operation, then report `error`: the response was lost after
    /// the write applied. A `gate` holds the caller inside the operation until
    /// it is released, so another writer can act on the state that did apply.
    LoseResponse {
        error: StatsError,
        gate: Option<Arc<FaultGate>>,
    },
    /// Park until the gate is released, then perform the operation normally.
    Park(Arc<FaultGate>),
}

/// One armed fault: which operation it matches, how many matches it lets
/// through first, how many times it fires, and what it does.
#[derive(Clone)]
pub struct ObjectFault {
    op: ObjectOp,
    pattern: ObjectPattern,
    skip: usize,
    times: usize,
    action: FaultAction,
}

impl ObjectFault {
    pub fn new(op: ObjectOp, pattern: ObjectPattern, action: FaultAction) -> Self {
        Self {
            op,
            pattern,
            skip: 0,
            times: 1,
            action,
        }
    }

    /// Let the first `matches` matching calls through untouched.
    pub fn after(mut self, matches: usize) -> Self {
        self.skip = matches;
        self
    }

    /// Fire on every match rather than once.
    pub fn forever(mut self) -> Self {
        self.times = usize::MAX;
        self
    }
}

/// One recorded object operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectCall {
    pub op: ObjectOp,
    pub key: String,
}

/// An [`ObjectStore`] decorator that records every operation and applies armed
/// faults to the ones a test selected.
///
/// Faults are an explicit ordered queue, never sampled: the first armed fault
/// matching an operation's kind and object ID owns that call, so a scenario
/// reproduces byte for byte on every run.
pub struct FaultInjectingObjectStore {
    inner: Arc<dyn ObjectStore>,
    faults: Mutex<Vec<ObjectFault>>,
    calls: Mutex<Vec<ObjectCall>>,
}

impl FaultInjectingObjectStore {
    pub fn new(inner: Arc<dyn ObjectStore>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            faults: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
        })
    }

    pub fn arm(&self, fault: ObjectFault) {
        self.faults.lock().unwrap().push(fault);
    }

    /// Every operation performed so far, in order.
    pub fn calls(&self) -> Vec<ObjectCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Forget the recorded history, so the next assertion covers one phase.
    pub fn clear_calls(&self) {
        self.calls.lock().unwrap().clear();
    }

    /// Disarm every armed fault — the outage ends.
    pub fn clear_faults(&self) {
        self.faults.lock().unwrap().clear();
    }

    /// The object IDs `op` was called with, in order.
    pub fn keys_for(&self, op: ObjectOp) -> Vec<String> {
        self.calls()
            .into_iter()
            .filter(|call| call.op == op)
            .map(|call| call.key)
            .collect()
    }

    /// Take the action the armed queue selects for this call, if any.
    fn selected(&self, op: ObjectOp, key: &str) -> Option<FaultAction> {
        let mut faults = self.faults.lock().unwrap();
        let index = faults
            .iter()
            .position(|fault| fault.op == op && fault.pattern.matches(key))?;
        let fault = &mut faults[index];
        if fault.skip > 0 {
            fault.skip -= 1;
            return None;
        }
        let action = fault.action.clone();
        if fault.times != usize::MAX {
            fault.times -= 1;
            if fault.times == 0 {
                faults.remove(index);
            }
        }
        Some(action)
    }

    fn record(&self, op: ObjectOp, key: &str) -> Option<FaultAction> {
        self.calls.lock().unwrap().push(ObjectCall {
            op,
            key: key.to_string(),
        });
        self.selected(op, key)
    }
}

/// How a faulted call proceeds once its action is known.
enum Proceed {
    /// Run the operation and return its own result.
    Run,
    /// Do not run the operation; return this error.
    Reject(StatsError),
    /// Run the operation, then park on any gate and report this error.
    RunThenFail {
        error: StatsError,
        gate: Option<Arc<FaultGate>>,
    },
}

/// Apply everything an action decides before the operation runs.
async fn resolve(action: Option<FaultAction>) -> Proceed {
    match action {
        None => Proceed::Run,
        Some(FaultAction::Fail(error)) => Proceed::Reject(error),
        Some(FaultAction::Park(gate)) => {
            gate.park().await;
            Proceed::Run
        }
        Some(FaultAction::LoseResponse { error, gate }) => Proceed::RunThenFail { error, gate },
    }
}

/// Hold the caller inside a lost-response fault until its gate opens, so
/// another writer observes the applied state while this one is still waiting.
async fn lost(error: StatsError, gate: Option<Arc<FaultGate>>) -> StatsError {
    if let Some(gate) = gate {
        gate.park().await;
    }
    error
}

#[async_trait]
impl ObjectStore for FaultInjectingObjectStore {
    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        match resolve(self.record(ObjectOp::Write, id.as_str())).await {
            Proceed::Run => self.inner.write(id, bytes).await,
            Proceed::Reject(error) => Err(error),
            Proceed::RunThenFail { error, gate } => {
                self.inner.write(id, bytes).await?;
                Err(lost(error, gate).await)
            }
        }
    }

    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError> {
        match resolve(self.record(ObjectOp::Read, id.as_str())).await {
            Proceed::Run => self.inner.read(id).await,
            Proceed::Reject(error) => Err(error),
            Proceed::RunThenFail { error, gate } => {
                self.inner.read(id).await?;
                Err(lost(error, gate).await)
            }
        }
    }

    /// Staging is a local write; faults model the remote, so it always runs.
    async fn stage(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        self.inner.stage(id, bytes).await
    }

    /// The upload half of a staged write is a remote write: `Write` faults
    /// apply, so an outage scenario fails uploads while staging succeeds.
    async fn upload_staged(&self, reference: &ObjectReference) -> Result<(), StatsError> {
        match resolve(self.record(ObjectOp::Write, reference.id.as_str())).await {
            Proceed::Run => self.inner.upload_staged(reference).await,
            Proceed::Reject(error) => Err(error),
            Proceed::RunThenFail { error, gate } => {
                self.inner.upload_staged(reference).await?;
                Err(lost(error, gate).await)
            }
        }
    }

    async fn local_path(&self, reference: &ObjectReference) -> Result<PathBuf, StatsError> {
        match resolve(self.record(ObjectOp::LocalPath, reference.id.as_str())).await {
            Proceed::Run => self.inner.local_path(reference).await,
            Proceed::Reject(error) => Err(error),
            Proceed::RunThenFail { error, gate } => {
                self.inner.local_path(reference).await?;
                Err(lost(error, gate).await)
            }
        }
    }

    fn planned_local_path(&self, id: &ObjectId) -> Result<PathBuf, StatsError> {
        self.inner.planned_local_path(id)
    }

    fn remote_scan_url(&self, id: &ObjectId) -> Option<String> {
        self.inner.remote_scan_url(id)
    }

    async fn cached_path(
        &self,
        reference: &ObjectReference,
    ) -> Result<Option<PathBuf>, StatsError> {
        self.inner.cached_path(reference).await
    }

    fn warm(&self, reference: &ObjectReference) {
        self.inner.warm(reference)
    }

    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        match resolve(self.record(ObjectOp::CompareAndSwap, id.as_str())).await {
            Proceed::Run => self.inner.compare_and_swap(id, expected, bytes).await,
            Proceed::Reject(error) => Err(error),
            Proceed::RunThenFail { error, gate } => {
                self.inner.compare_and_swap(id, expected, bytes).await?;
                Err(lost(error, gate).await)
            }
        }
    }

    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError> {
        self.inner.delete(id).await
    }

    async fn list(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError> {
        self.inner.list(prefix).await
    }

    async fn list_tables(&self) -> Result<Vec<String>, StatsError> {
        self.inner.list_tables().await
    }

    async fn gc(&self) -> Result<(), StatsError> {
        self.inner.gc().await
    }
}

/// The error a state store reports when its HEAD swap response is lost.
pub fn lost_head_response() -> StatsError {
    StatsError::AmbiguousCommit("HEAD swap response was lost".to_string())
}
