//! Bound synchronous read work through completion, including after cancellation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::errors::StatsError;

pub(crate) struct ReadWorkers {
    permits: Arc<Semaphore>,
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

impl ReadWorkers {
    pub fn new(limit: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(limit)),
        }
    }

    pub async fn run<T: Send + 'static>(
        &self,
        work: impl FnOnce() -> Result<T, StatsError> + Send + 'static,
    ) -> Result<T, StatsError> {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|error| StatsError::Internal(format!("read workers closed: {error}")))?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let _cancel = CancelOnDrop(Arc::clone(&cancelled));
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            if cancelled.load(Ordering::Acquire) {
                return Err(StatsError::DeadlineExceeded(
                    "read cancelled before blocking work started".into(),
                ));
            }
            work()
        })
        .await
        .map_err(|error| StatsError::Internal(format!("blocking read worker: {error}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_queued_work_never_performs_its_filesystem_operation() {
        let root = crate::test_support::unique_dir("cancelled_read_worker");
        let output = root.join("unexpected");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let (entered, entered_rx) = tokio::sync::oneshot::channel();
            let (release, release_rx) = std::sync::mpsc::channel();
            let occupied = tokio::task::spawn_blocking(move || {
                entered.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            entered_rx.await.unwrap();
            let workers = ReadWorkers::new(1);
            let path = output.clone();
            {
                let operation = workers.run(move || {
                    std::fs::write(path, "must not run")
                        .map_err(|error| StatsError::Internal(error.to_string()))
                });
                futures::pin_mut!(operation);
                assert!(futures::poll!(operation.as_mut()).is_pending());
                // The only blocking thread is occupied, so dropping the future
                // cancels an operation that is definitely still queued.
            }
            release.send(()).unwrap();
            occupied.await.unwrap();
            tokio::task::spawn_blocking(|| ()).await.unwrap();
            assert!(!output.exists());
            assert_eq!(workers.run(|| Ok(7)).await.unwrap(), 7);
        });
        std::fs::remove_dir_all(root).unwrap();
    }
}
