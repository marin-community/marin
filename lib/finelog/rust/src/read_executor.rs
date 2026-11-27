// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Give public reads separate workers with lower Linux scheduling priority.

use std::future::Future;
use std::io;

use tokio::runtime::{Builder, Runtime};
use tokio::task::{JoinError, JoinSet};

#[cfg(target_os = "linux")]
const READ_NICE_INCREMENT: i32 = 10;

/// Public-read workers owned by one RPC server.
pub struct ReadExecutor {
    runtime: Option<Runtime>,
    workers: usize,
}

impl ReadExecutor {
    /// Size the pool to the process CPU budget; idle ingestion capacity is
    /// available to reads without changing ingestion's worker pool or priority.
    pub fn for_process() -> io::Result<Self> {
        Self::with_workers(std::thread::available_parallelism()?.get())
    }

    fn with_workers(workers: usize) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        let nice = (thread_nice(std::process::id())? + READ_NICE_INCREMENT).min(19);
        let runtime = Builder::new_multi_thread()
            .worker_threads(workers)
            // Nested cache I/O needs Tokio's separate blocking capacity. Both
            // kinds of worker yield CPU to ingestion at the same priority.
            .thread_name("finelog-read")
            .on_thread_start(move || {
                #[cfg(target_os = "linux")]
                if unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice) } != 0 {
                    panic!(
                        "configure read worker priority: {}",
                        io::Error::last_os_error()
                    );
                }
            })
            .enable_all()
            .build()?;
        tracing::info!(workers, "read executor configured");
        Ok(Self {
            runtime: Some(runtime),
            workers,
        })
    }

    /// Async read workers, independent of ingestion's runtime workers.
    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Dropping the caller aborts its task instead of detaching an expired read.
    pub(crate) async fn run<T: Send + 'static>(
        &self,
        work: impl Future<Output = T> + Send + 'static,
    ) -> Result<T, JoinError> {
        let mut tasks = JoinSet::new();
        tasks.spawn_on(
            work,
            self.runtime.as_ref().expect("live read executor").handle(),
        );
        tasks.join_next().await.expect("one read task was spawned")
    }
}

impl Drop for ReadExecutor {
    fn drop(&mut self) {
        // Routers can release their last owner on a Tokio worker, where
        // Runtime's blocking Drop is illegal.
        self.runtime
            .take()
            .expect("live read executor")
            .shutdown_background();
    }
}

#[cfg(target_os = "linux")]
fn thread_nice(thread: libc::id_t) -> io::Result<i32> {
    // -1 is also a valid nice value, so errno distinguishes failure.
    unsafe { *libc::__errno_location() = 0 };
    let nice = unsafe { libc::getpriority(libc::PRIO_PROCESS, thread) };
    if nice == -1 && io::Error::last_os_error().raw_os_error() != Some(0) {
        return Err(io::Error::last_os_error());
    }
    Ok(nice)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn reads_borrow_all_cpus_without_lowering_ingestion_priority() {
        fn affinity() -> Vec<usize> {
            let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
            assert_eq!(
                unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&mask), &mut mask) },
                0
            );
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
                .collect()
        }
        let ingress_cpus = affinity();
        let ingress_nice = thread_nice(0).unwrap();
        let executor = ReadExecutor::for_process().unwrap();
        let (async_cpus, async_nice, blocking_cpus, blocking_nice) = executor
            .run(async {
                let async_cpus = affinity();
                let async_nice = thread_nice(0).unwrap();
                let (blocking_cpus, blocking_nice) =
                    tokio::task::spawn_blocking(|| (affinity(), thread_nice(0).unwrap()))
                        .await
                        .unwrap();
                (async_cpus, async_nice, blocking_cpus, blocking_nice)
            })
            .await
            .unwrap();
        assert_eq!(async_cpus, ingress_cpus);
        assert_eq!(blocking_cpus, ingress_cpus);
        assert_eq!(affinity(), ingress_cpus);
        assert_eq!(thread_nice(0).unwrap(), ingress_nice);
        assert!(async_nice > ingress_nice);
        // Blocking threads can be created by a read worker. Adding nice again
        // would push them below async reads, starving query cache operations.
        assert_eq!(async_nice, blocking_nice);
    }

    #[tokio::test]
    async fn blocked_query_dispatch_does_not_block_durable_writes() {
        let store = crate::server::test_support::disk_store("read_cpu_ingestion");
        let executor = Arc::new(ReadExecutor::with_workers(1).unwrap());
        let arrived = Arc::new(tokio::sync::Notify::new());
        let observed = Arc::clone(&arrived);
        let app = crate::server::build_app_with_config(
            Arc::clone(&store),
            crate::server::ServerConfig::default(),
            Arc::clone(&executor),
        )
        .layer(axum::middleware::from_fn(
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                let observed = Arc::clone(&observed);
                async move {
                    if request.uri().path().ends_with("/Query") {
                        observed.notify_one();
                    }
                    next.run(request).await
                }
            },
        ));
        let (address, server) = crate::server::test_support::serve_app(app).await;
        let (release, blocked) =
            crate::test_support::block_worker(executor.runtime.as_ref().unwrap().handle()).await;
        let mut query = tokio::spawn(async move {
            crate::server::test_support::stats_client(address)
                .query(crate::proto::finelog::stats::QueryRequest::default().with_sql("SELECT 1"))
                .await
        });
        arrived.notified().await;
        // The request has reached the server. It must remain queued for the
        // occupied executor rather than execute on the ingestion runtime.
        let queued = tokio::time::timeout(std::time::Duration::from_millis(200), &mut query).await;
        let client = crate::server::test_support::client(address);
        let write = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.push_logs(
                crate::proto::finelog::logging::PushLogsRequest {
                    entries: vec![crate::proto::finelog::logging::LogEntry::default()
                        .with_data("ingestion keeps progressing")],
                    ..Default::default()
                }
                .with_key("/test/ingestion"),
            ),
        )
        .await;
        drop(release);
        blocked.await.unwrap();
        assert!(queued.is_err(), "Query bypassed the occupied read executor");
        write
            .expect("writes must not wait for a read worker")
            .unwrap();
        assert_eq!(store.namespace_persisted_seq("log").unwrap(), 1);
        assert_eq!(query.await.unwrap().unwrap().view().row_count, Some(1));
        server.abort();
        store.shutdown(std::time::Duration::from_secs(5)).await;
    }

    #[tokio::test]
    async fn cancelling_a_read_drops_its_work() {
        let (started, started_rx) = tokio::sync::oneshot::channel();
        let (dropped, dropped_rx) = tokio::sync::oneshot::channel::<()>();
        let executor = Arc::new(ReadExecutor::for_process().unwrap());
        let task_executor = Arc::clone(&executor);
        let task = tokio::spawn(async move {
            task_executor
                .run(async move {
                    let _dropped = dropped;
                    started.send(()).unwrap();
                    std::future::pending::<()>().await;
                })
                .await
        });
        started_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(5), dropped_rx)
                .await
                .expect("cancelled read remained attached to the live executor")
                .is_err()
        );
        drop(executor);
    }
}
