//! Periodic pool/RSS diagnostics line.
//!
//! Every [`POOL_DIAGNOSTICS_INTERVAL`] a tokio task emits one `tracing::info!`
//! line carrying the process RSS + VmSize (parsed from `/proc/self/status`) and
//! the store memory summary (`namespaces` / `ram_bytes` / `chunks`).
//!
//! The line is a diagnostic aid, not a contract, so no test asserts on its
//! text. The task selects against a shutdown signal and exits when it fires.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;

use crate::store::Store;

/// Diagnostics emit cadence.
pub const POOL_DIAGNOSTICS_INTERVAL: Duration = Duration::from_secs(60);

/// Read a `/proc/self/status` field value in KiB (Linux-only). Returns 0 when
/// the field is missing or the file is unreadable (non-Linux); 0 means
/// "unknown".
pub fn read_proc_self_status_kb(field: &str) -> u64 {
    match std::fs::read_to_string("/proc/self/status") {
        Ok(contents) => parse_proc_self_status_kb(&contents, field),
        Err(_) => 0,
    }
}

/// Parse a KiB field out of a `/proc/self/status` blob. Pure (testable) half of
/// [`read_proc_self_status_kb`]. The line shape is `VmRSS:\t  12345 kB`.
fn parse_proc_self_status_kb(contents: &str, field: &str) -> u64 {
    let prefix = format!("{field}:");
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix(&prefix) {
            // The value is the first whitespace-separated token after the colon.
            if let Some(tok) = rest.split_whitespace().next() {
                return tok.parse().unwrap_or(0);
            }
        }
    }
    0
}

/// Emit one diagnostics line.
fn emit(store: &Store) {
    let summary = store.memory_summary();
    let rss_kb = read_proc_self_status_kb("VmRSS");
    let vmsize_kb = read_proc_self_status_kb("VmSize");
    tracing::info!(
        rss_kb,
        vmsize_kb,
        namespaces = summary.namespaces,
        ram_bytes = summary.ram_bytes,
        chunks = summary.chunks,
        "pool_diag rss_kb={rss_kb} vmsize_kb={vmsize_kb} namespaces={} ram_bytes={} chunks={}",
        summary.namespaces,
        summary.ram_bytes,
        summary.chunks,
    );
}

/// Spawn the diagnostics task. It emits once per [`POOL_DIAGNOSTICS_INTERVAL`]
/// and exits when `stop` is latched (set the `AtomicBool` THEN `notify_waiters`
/// the `Notify`).
///
/// The latch is load-bearing: `Notify::notify_waiters` stores no permit, so a
/// notify that fires while the task is between the prior `select!` and
/// re-registering `notified()` would be lost and the task would sleep up to a
/// full interval before noticing shutdown. Checking the latch at loop-top and
/// after the select closes that race (the same pattern the per-namespace
/// flush/maintenance tasks use), so shutdown is prompt and cannot stall the
/// process drain.
pub fn spawn_pool_diagnostics(
    store: Arc<Store>,
    stop: Arc<AtomicBool>,
    shutdown: Arc<tokio::sync::Notify>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            if stop.load(Ordering::SeqCst) {
                return;
            }
            let stopped = shutdown.notified();
            tokio::select! {
                _ = tokio::time::sleep(POOL_DIAGNOSTICS_INTERVAL) => {}
                _ = stopped => return,
            }
            if stop.load(Ordering::SeqCst) {
                return;
            }
            emit(&store);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_proc_self_status_parses_vmrss() {
        let fixture = "Name:\tfinelog\n\
VmPeak:\t  123456 kB\n\
VmSize:\t  120000 kB\n\
VmRSS:\t   45678 kB\n\
Threads:\t8\n";
        assert_eq!(parse_proc_self_status_kb(fixture, "VmRSS"), 45678);
        assert_eq!(parse_proc_self_status_kb(fixture, "VmSize"), 120000);
        // A missing field returns 0 (the "unknown" sentinel).
        assert_eq!(parse_proc_self_status_kb(fixture, "VmHWM"), 0);
    }

    #[test]
    fn read_proc_self_status_handles_garbage() {
        // A line whose value isn't a number falls back to 0, not a panic.
        assert_eq!(
            parse_proc_self_status_kb("VmRSS:\tnotanumber kB\n", "VmRSS"),
            0
        );
        assert_eq!(parse_proc_self_status_kb("", "VmRSS"), 0);
    }

    #[tokio::test]
    async fn diagnostics_task_stops_promptly_on_shutdown() {
        // The task parks in a 60s select; latch + notify must drain it FAR sooner
        // than POOL_DIAGNOSTICS_INTERVAL, proving the shutdown path cannot stall
        // the process for an interval (the lost-wakeup-race guard).
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(tokio::sync::Notify::new());
        let handle =
            spawn_pool_diagnostics(Arc::clone(&store), Arc::clone(&stop), Arc::clone(&notify));
        stop.store(true, Ordering::SeqCst);
        notify.notify_waiters();
        let joined = tokio::time::timeout(Duration::from_secs(2), handle).await;
        assert!(
            joined.is_ok(),
            "diagnostics task did not stop within 2s of the shutdown latch"
        );
    }
}
