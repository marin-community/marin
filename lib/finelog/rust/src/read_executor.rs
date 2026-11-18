// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Keep public reads and maintenance off the CPUs reserved for ingestion.

use std::future::Future;
use std::io;
use std::sync::OnceLock;

use tokio::runtime::{Builder, Runtime};
use tokio::task::{JoinError, JoinSet};

const READ_NICE: i32 = 5;
// Maintenance readers can await nested blocking cache reads. CPU affinity,
// rather than a tiny blocking pool, limits their CPU use without deadlocking I/O.
const MAX_BLOCKING_THREADS: usize = 32;

pub(crate) struct ReadExecutor {
    runtime: Runtime,
    cpus: Vec<usize>,
}

impl ReadExecutor {
    fn new() -> io::Result<Self> {
        let available = available_cpus()?;
        let parallelism = std::thread::available_parallelism()?
            .get()
            .min(available.len());
        let count = (parallelism / 2).max(1);
        let cpus = available[available.len() - count..].to_vec();
        let worker_cpus = cpus.clone();
        let runtime = Builder::new_multi_thread()
            .worker_threads(count)
            .max_blocking_threads(MAX_BLOCKING_THREADS)
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
        Ok(Self { runtime, cpus })
    }

    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }

    /// Dropping the caller aborts its task instead of detaching an expired read.
    pub async fn run<T: Send + 'static>(
        &self,
        work: impl Future<Output = T> + Send + 'static,
    ) -> Result<T, JoinError> {
        let mut tasks = JoinSet::new();
        tasks.spawn_on(work, self.runtime.handle());
        tasks.join_next().await.expect("one read task was spawned")
    }
}

pub(crate) fn read_executor() -> &'static ReadExecutor {
    static EXECUTOR: OnceLock<ReadExecutor> = OnceLock::new();
    EXECUTOR.get_or_init(|| ReadExecutor::new().expect("create read executor"))
}

#[cfg(target_os = "linux")]
fn available_cpus() -> io::Result<Vec<usize>> {
    let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    if unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&mask), &mut mask) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .collect())
}

#[cfg(not(target_os = "linux"))]
fn available_cpus() -> io::Result<Vec<usize>> {
    Ok((0..std::thread::available_parallelism()?.get()).collect())
}

fn restrict_read_thread(cpus: &[usize]) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        for &cpu in cpus {
            unsafe { libc::CPU_SET(cpu, &mut mask) };
        }
        if unsafe { libc::sched_setaffinity(0, std::mem::size_of_val(&mask), &mask) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(unix)]
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
        let ingress_cpus = available_cpus().unwrap();
        let executor = read_executor();
        let (async_cpus, blocking_cpus) = executor
            .run(async {
                let async_cpus = available_cpus().unwrap();
                let blocking_cpus = tokio::task::spawn_blocking(available_cpus)
                    .await
                    .unwrap()
                    .unwrap();
                (async_cpus, blocking_cpus)
            })
            .await
            .unwrap();
        assert_eq!(async_cpus, executor.cpus());
        assert_eq!(blocking_cpus, executor.cpus());
        assert_eq!(available_cpus().unwrap(), ingress_cpus);
        if std::thread::available_parallelism().unwrap().get() > 1 {
            assert!(async_cpus.len() < ingress_cpus.len());
        }
    }

    #[tokio::test]
    async fn writes_become_durable_while_all_read_workers_are_occupied() {
        let store = crate::server::test_support::disk_store("read_cpu_ingestion");
        let (address, _) = crate::server::test_support::serve(
            Arc::clone(&store),
            crate::server::AuthPolicy::allow_localhost(),
        )
        .await;
        let executor = read_executor();
        let mut releases = Vec::new();
        let mut reads = Vec::new();
        for _ in executor.cpus() {
            let (started, started_rx) = tokio::sync::oneshot::channel();
            let (release, release_rx) = std::sync::mpsc::channel::<()>();
            releases.push(release);
            reads.push(tokio::spawn(executor.run(async move {
                started.send(()).unwrap();
                // Hold every async read worker to reproduce CPU-bound reads
                // that cannot yield to a durability flush on a shared pool.
                let _ = release_rx.recv();
            })));
            started_rx.await.unwrap();
        }
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
        drop(releases);
        for read in reads {
            read.await.unwrap().unwrap();
        }
        write
            .expect("writes must not wait for a read worker")
            .unwrap();
        assert_eq!(store.namespace_persisted_seq("log").unwrap(), 1);
        store.shutdown(std::time::Duration::from_secs(5)).await;
    }

    #[tokio::test]
    async fn cancelling_a_read_drops_its_work() {
        let (started, started_rx) = tokio::sync::oneshot::channel();
        let (dropped, dropped_rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(read_executor().run(async move {
            let _dropped = dropped;
            started.send(()).unwrap();
            std::future::pending::<()>().await;
        }));
        started_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(dropped_rx.await.is_err());
    }
}
