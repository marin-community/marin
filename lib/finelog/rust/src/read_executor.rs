// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Isolate public reads, reserving CPUs on multicore Linux hosts.

use std::future::Future;
use std::io;

use tokio::runtime::{Builder, Runtime};
use tokio::task::{JoinError, JoinSet};

const READ_NICE: i32 = if cfg!(target_os = "linux") { 5 } else { 0 };

/// CPU-bounded workers owned by one public RPC server.
pub struct ReadExecutor {
    runtime: Option<Runtime>,
    cpus: Vec<usize>,
}

impl ReadExecutor {
    /// Reserve half the process CPU budget from public read execution.
    pub fn for_process() -> io::Result<Self> {
        let (available, parallelism) = process_capacity()?;
        let parallelism = parallelism.min(available.len());
        let count = (parallelism / 2).max(1);
        let cpus = available[available.len() - count..].to_vec();
        Self::on_cpus(cpus)
    }

    fn on_cpus(cpus: Vec<usize>) -> io::Result<Self> {
        let count = cpus.len();
        let worker_cpus = cpus.clone();
        let runtime = Builder::new_multi_thread()
            .worker_threads(count)
            // Keep Tokio's blocking capacity for nested cache I/O; affinity
            // bounds CPU use on both async and blocking workers.
            .thread_name("finelog-read")
            .on_thread_start(move || {
                restrict_read_thread(&worker_cpus).expect("configure read worker CPU affinity");
            })
            .enable_all()
            .build()?;
        tracing::info!(
            ?cpus,
            workers = count,
            nice = READ_NICE,
            "read executor configured"
        );
        Ok(Self {
            runtime: Some(runtime),
            cpus,
        })
    }

    /// CPU IDs selected for this executor's workers.
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
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
fn cpu_affinity(thread: libc::pid_t) -> io::Result<Vec<usize>> {
    let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    if unsafe { libc::sched_getaffinity(thread, std::mem::size_of_val(&mask), &mut mask) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .collect())
}

#[cfg(target_os = "linux")]
fn set_cpu_affinity(cpus: &[usize]) -> io::Result<()> {
    let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    for &cpu in cpus {
        unsafe { libc::CPU_SET(cpu, &mut mask) };
    }
    if unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&mask), &mask) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn process_capacity() -> io::Result<(Vec<usize>, usize)> {
    #[cfg(target_os = "linux")]
    {
        // A library caller may already be pinned. Measure the cgroup-aware CPU
        // budget with the process leader's mask without changing that caller.
        std::thread::Builder::new()
            .name("finelog-cpus".into())
            .spawn(|| {
                let cpus = cpu_affinity(std::process::id() as libc::pid_t)?;
                set_cpu_affinity(&cpus)?;
                Ok((cpus, std::thread::available_parallelism()?.get()))
            })?
            .join()
            .expect("CPU capacity probe panicked")
    }
    #[cfg(not(target_os = "linux"))]
    {
        let count = std::thread::available_parallelism()?.get();
        Ok(((0..count).collect(), count))
    }
}

fn restrict_read_thread(_cpus: &[usize]) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    set_cpu_affinity(_cpus)?;
    #[cfg(target_os = "linux")]
    if unsafe { libc::nice(READ_NICE) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn async_and_blocking_reads_leave_ingestion_cpus_available() {
        let ingress_cpus = cpu_affinity(0).unwrap();
        let executor = ReadExecutor::for_process().unwrap();
        let (async_cpus, blocking_cpus) = executor
            .run(async {
                let async_cpus = cpu_affinity(0).unwrap();
                let blocking_cpus = tokio::task::spawn_blocking(|| cpu_affinity(0))
                    .await
                    .unwrap()
                    .unwrap();
                (async_cpus, blocking_cpus)
            })
            .await
            .unwrap();
        assert_eq!(async_cpus, executor.cpus());
        assert_eq!(blocking_cpus, executor.cpus());
        assert_eq!(cpu_affinity(0).unwrap(), ingress_cpus);
        if std::thread::available_parallelism().unwrap().get() > 1 {
            assert!(async_cpus.len() < ingress_cpus.len());
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_pinned_initializer_preserves_the_process_cpu_budget() {
        let expected = process_capacity().unwrap();
        let first_cpu = expected.0[0];
        let measured = std::thread::spawn(move || {
            set_cpu_affinity(&[first_cpu]).unwrap();
            process_capacity().unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(measured, expected);
    }

    #[tokio::test]
    async fn blocked_query_dispatch_does_not_block_durable_writes() {
        let store = crate::server::test_support::disk_store("read_cpu_ingestion");
        let cpu = *process_capacity().unwrap().0.last().unwrap();
        let executor = Arc::new(ReadExecutor::on_cpus(vec![cpu]).unwrap());
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
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
            .unwrap()
        });
        let (started, started_rx) = tokio::sync::oneshot::channel();
        let (release, release_rx) = std::sync::mpsc::channel::<()>();
        let worker = Arc::clone(&executor);
        let blocked = tokio::spawn(async move {
            worker
                .run(async move {
                    started.send(()).unwrap();
                    let _ = release_rx.recv();
                })
                .await
                .unwrap()
        });
        started_rx.await.unwrap();
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
        let executor = ReadExecutor::for_process().unwrap();
        let task = tokio::spawn(async move {
            executor
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
        assert!(dropped_rx.await.is_err());
    }
}
