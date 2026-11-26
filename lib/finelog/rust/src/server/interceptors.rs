//! Connect/RPC interceptors for the finelog server.
//!
//! Two interceptors:
//!
//! - [`SlowRpcInterceptor`] — times `next.run` and emits a single
//!   `tracing::warn!` when a call exceeds a per-method ms threshold (default
//!   7000ms, `<= 0` disables for that method).
//!   Registered FIRST (outermost) so it times the WHOLE chain, including the
//!   concurrency wait.
//!
//! - [`ConcurrencyInterceptor`] — independent SQL and log-fetch semaphores;
//!   acquires permits BEFORE
//!   `next.run` and holds it across the handler (bounding the parquet working
//!   set). One deadline bounds admission and the whole handler; expiry drops
//!   queued requests and running reads with `deadline_exceeded`.
//!   Per-method caps bound the parquet working set; separate lower-priority
//!   read workers preserve ingestion's runtime. Writes bypass admission.
//!   Registered AFTER SlowRpc.
//!
//! Both identify the dispatched RPC through `ctx.spec()`. The
//! spec is populated ONLY when the services are registered through the generated
//! `register()` (which `server::app::build_app` uses); a `None` spec means the
//! interceptor cannot identify the method and passes through untouched, so the
//! whole chain would silently no-op. `build_app` therefore MUST use `register()`
//! (never `into_axum_service`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use connectrpc::interceptor::{UnaryRequest, UnaryResponse};
use connectrpc::spec::Spec;
use connectrpc::{async_trait, ConnectError, Interceptor, Next, RequestContext};
use tokio::sync::Semaphore;

use crate::proto::finelog::logging::LOG_SERVICE_FETCH_LOGS_SPEC;
use crate::proto::finelog::stats::STATS_SERVICE_QUERY_SPEC;
use crate::query::{query_timeout, run_within_query_timeout};

/// Default per-method slow-RPC threshold.
/// Picked well above the durable-write floor (one bg flush cycle) so a normal
/// WriteRows that waits for its L0 segment doesn't spam warnings.
pub const DEFAULT_SLOW_RPC_THRESHOLD_MS: i64 = 7000;

pub const MAX_CONCURRENT_FETCH_LOGS: usize = 4;
pub const MAX_CONCURRENT_QUERY: usize = 2;

#[derive(Clone, Copy)]
pub(crate) enum ReadProcedure {
    FetchLogs,
    Query,
}

const READ_PROCEDURES: [(Spec, ReadProcedure); 2] = [
    (LOG_SERVICE_FETCH_LOGS_SPEC, ReadProcedure::FetchLogs),
    (STATS_SERVICE_QUERY_SPEC, ReadProcedure::Query),
];

pub(crate) fn read_procedure(path: &str) -> Option<ReadProcedure> {
    READ_PROCEDURES
        .iter()
        .find(|(spec, _)| spec.procedure == path)
        .map(|(_, method)| *method)
}

/// The dispatched method's short name (`"FetchLogs"`, `"Query"`, ...), or `None`
/// when the spec is absent (manual `route_*` registration without `with_spec`).
/// `register()` always populates it; see the module doc.
fn method_name(ctx: &RequestContext) -> Option<&'static str> {
    ctx.spec().map(|s| s.method())
}

/// Logs a WARNING when a unary RPC exceeds a per-method ms threshold.
///
/// Times `next.run` and, if the elapsed wall time is `>= threshold` for the
/// dispatched method, emits one `tracing::warn!`. A threshold `<= 0` disables
/// the warning for that method (the timing branch is skipped). Methods absent
/// from `thresholds` use `default_threshold_ms`.
pub struct SlowRpcInterceptor {
    default_threshold_ms: i64,
    thresholds: HashMap<String, i64>,
}

impl SlowRpcInterceptor {
    /// Build with the given default threshold and no per-method overrides.
    pub fn new(default_threshold_ms: i64) -> Self {
        Self {
            default_threshold_ms,
            thresholds: HashMap::new(),
        }
    }

    /// Set a per-method threshold override (`0`/negative disables that method).
    pub fn with_method_threshold(mut self, method: impl Into<String>, threshold_ms: i64) -> Self {
        self.thresholds.insert(method.into(), threshold_ms);
        self
    }

    fn threshold(&self, method: &str) -> i64 {
        self.thresholds
            .get(method)
            .copied()
            .unwrap_or(self.default_threshold_ms)
    }
}

#[async_trait]
impl Interceptor for SlowRpcInterceptor {
    async fn intercept_unary(
        &self,
        req: UnaryRequest,
        next: Next<'_>,
    ) -> Result<UnaryResponse, ConnectError> {
        // Resolve the method + threshold before `req` is moved into `next.run`.
        let method = method_name(&req.ctx).unwrap_or("");
        let threshold = self.threshold(method);
        if threshold <= 0 {
            return next.run(req).await;
        }
        let method = method.to_owned();
        let started = Instant::now();
        let resp = next.run(req).await;
        let elapsed_ms = started.elapsed().as_millis() as i64;
        if elapsed_ms >= threshold {
            // The warning text is a diagnostic aid, not a contract — no test
            // asserts on it.
            tracing::warn!(
                method = %method,
                elapsed_ms,
                threshold_ms = threshold,
                "Slow RPC {method}: {elapsed_ms}ms (threshold: {threshold}ms)",
            );
        }
        resp
    }
}

/// Bound read concurrency and the entire read RPC, including admission wait.
pub struct ConcurrencyInterceptor {
    fetch: Arc<Semaphore>,
    query: Arc<Semaphore>,
}

impl ConcurrencyInterceptor {
    /// Bound each method's working set without SQL consuming log-fetch permits.
    pub fn new(fetch_limit: usize, query_limit: usize) -> Self {
        Self {
            fetch: Arc::new(Semaphore::new(fetch_limit)),
            query: Arc::new(Semaphore::new(query_limit)),
        }
    }

    fn semaphore_for(&self, procedure: ReadProcedure) -> &Semaphore {
        match procedure {
            ReadProcedure::FetchLogs => &self.fetch,
            ReadProcedure::Query => &self.query,
        }
    }
}

/// True if the request's Connect deadline has already elapsed. `time_remaining`
/// saturates at zero on expiry, so `Some(ZERO)` is the expired signal (`None`
/// means no deadline was asserted).
fn deadline_expired(ctx: &RequestContext) -> bool {
    ctx.time_remaining().is_some_and(|d| d.is_zero())
}

fn deadline_error(method: &str) -> ConnectError {
    ConnectError::deadline_exceeded(format!("RPC {method}: deadline exceeded"))
}

#[async_trait]
impl Interceptor for ConcurrencyInterceptor {
    async fn intercept_unary(
        &self,
        mut req: UnaryRequest,
        next: Next<'_>,
    ) -> Result<UnaryResponse, ConnectError> {
        let method = method_name(&req.ctx).unwrap_or("");
        let Some(procedure) = req
            .ctx
            .spec()
            .and_then(|spec| read_procedure(spec.procedure))
        else {
            return next.run(req).await;
        };
        let semaphore = self.semaphore_for(procedure);
        // Pre-acquire shed: a request that arrived already past its deadline is
        // not worth queuing.
        if deadline_expired(&req.ctx) {
            return Err(deadline_error(method));
        }
        let timeout = query_timeout(req.ctx.time_remaining(), None);
        if let Some(timeout) = timeout {
            req.ctx = req.ctx.with_deadline(Some(Instant::now() + timeout));
        }
        // Admission, snapshot construction, visibility waits, execution, and
        // encoding all spend the same request budget. Dropping the handler also
        // releases its permits and cancels its optional cache downloads.
        run_within_query_timeout(
            timeout,
            async {
                let _read_permit = semaphore
                    .acquire()
                    .await
                    .map_err(|_| ConnectError::internal("read semaphore closed"))?;
                next.run(req).await
            },
            |_| deadline_error(method),
            |error| error,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use axum::http::HeaderMap;
    use bytes::Bytes;
    use connectrpc::codec::CodecFormat;
    use connectrpc::interceptor::run_chain;
    use connectrpc::response::{EncodedBody, EncodedResponse};
    use connectrpc::spec::{Spec, StreamType};
    use tokio::sync::Notify;
    use tracing::subscriber;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::Layer;

    use super::*;

    /// A `UnaryRequest` whose ctx carries the spec for `procedure` and an
    /// optional absolute deadline.
    fn request(procedure: &'static str, deadline: Option<Instant>) -> UnaryRequest {
        let ctx = RequestContext::new(HeaderMap::new())
            .with_spec(Some(Spec::server(procedure, StreamType::Unary)))
            .with_path(procedure)
            .with_deadline(deadline);
        UnaryRequest::new(ctx, Bytes::new(), CodecFormat::Proto)
    }

    fn ok_response() -> UnaryResponse {
        UnaryResponse::from_encoded(
            EncodedResponse::new(EncodedBody::Contiguous(Bytes::new())),
            CodecFormat::Proto,
        )
    }

    #[tokio::test(start_paused = true)]
    async fn queued_read_expires_without_entering_the_handler() {
        let chain: Vec<Arc<dyn Interceptor>> = vec![Arc::new(ConcurrencyInterceptor::new(0, 0))];
        let result = run_chain(
            &chain,
            request(
                STATS_SERVICE_QUERY_SPEC.procedure,
                Some(Instant::now() + Duration::from_millis(25)),
            ),
            |_| async {
                panic!("expired queued query reached the handler");
            },
        )
        .await;
        assert_eq!(
            result.unwrap_err().code,
            connectrpc::ErrorCode::DeadlineExceeded
        );
    }

    #[tokio::test(start_paused = true)]
    async fn sql_saturation_keeps_log_and_write_admission_available() {
        let chain: Vec<Arc<dyn Interceptor>> = vec![Arc::new(ConcurrencyInterceptor::new(1, 1))];
        let release = Notify::new();
        let query = run_chain(
            &chain,
            request(STATS_SERVICE_QUERY_SPEC.procedure, None),
            |_| async {
                release.notified().await;
                Ok(ok_response())
            },
        );
        futures::pin_mut!(query);
        assert!(futures::poll!(query.as_mut()).is_pending());
        let second_query = run_chain(
            &chain,
            request(STATS_SERVICE_QUERY_SPEC.procedure, None),
            |_| async { Ok(ok_response()) },
        );
        futures::pin_mut!(second_query);
        assert!(futures::poll!(second_query.as_mut()).is_pending());
        run_chain(
            &chain,
            request(LOG_SERVICE_FETCH_LOGS_SPEC.procedure, None),
            |_| async { Ok(ok_response()) },
        )
        .await
        .unwrap();
        let fetch = run_chain(
            &chain,
            request(LOG_SERVICE_FETCH_LOGS_SPEC.procedure, None),
            |_| async {
                release.notified().await;
                Ok(ok_response())
            },
        );
        futures::pin_mut!(fetch);
        assert!(futures::poll!(fetch.as_mut()).is_pending());
        for method in [
            crate::proto::finelog::stats::STATS_SERVICE_WRITE_ROWS_SPEC.procedure,
            crate::proto::finelog::logging::LOG_SERVICE_PUSH_LOGS_SPEC.procedure,
        ] {
            run_chain(&chain, request(method, None), |_| async {
                Ok(ok_response())
            })
            .await
            .unwrap();
        }
        release.notify_waiters();
        query.await.unwrap();
        fetch.await.unwrap();
        second_query.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn read_deadline_cancels_handler_and_releases_admission() {
        let chain: Vec<Arc<dyn Interceptor>> = vec![Arc::new(ConcurrencyInterceptor::new(1, 1))];
        for procedure in [
            STATS_SERVICE_QUERY_SPEC.procedure,
            LOG_SERVICE_FETCH_LOGS_SPEC.procedure,
        ] {
            let held = Semaphore::new(1);
            let result = run_chain(&chain, request(procedure, None), |_| async {
                let _active = held.acquire().await.unwrap();
                std::future::pending::<Result<UnaryResponse, ConnectError>>().await
            })
            .await;
            assert_eq!(
                result.unwrap_err().code,
                connectrpc::ErrorCode::DeadlineExceeded
            );
            assert_eq!(held.available_permits(), 1);
            run_chain(&chain, request(procedure, None), |_| async {
                Ok(ok_response())
            })
            .await
            .unwrap();
        }
    }

    /// A tracing layer that counts events at WARN level.
    #[derive(Clone, Default)]
    struct WarnCounter {
        count: Arc<AtomicUsize>,
    }

    impl<S: tracing::Subscriber> Layer<S> for WarnCounter {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if *event.metadata().level() == tracing::Level::WARN {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// Run a single request through the slow interceptor with a terminal that
    /// sleeps `terminal_delay`, returning the number of WARN events emitted.
    async fn slow_warn_count(
        interceptor: SlowRpcInterceptor,
        procedure: &'static str,
        terminal_delay: Duration,
    ) -> usize {
        let counter = WarnCounter::default();
        let count = Arc::clone(&counter.count);
        let chain: Vec<Arc<dyn Interceptor>> = vec![Arc::new(interceptor)];
        let sub = tracing_subscriber::registry().with(counter);
        subscriber::with_default(sub, || {
            futures::executor::block_on(async {
                run_chain(&chain, request(procedure, None), |_req| async move {
                    tokio::time::sleep(terminal_delay).await;
                    Ok(ok_response())
                })
                .await
                .unwrap();
            });
        });
        count.load(Ordering::SeqCst)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_rpc_warns_over_threshold() {
        // A 1ms threshold against a 20ms terminal -> exactly one WARN.
        let n = slow_warn_count(
            SlowRpcInterceptor::new(1),
            LOG_SERVICE_FETCH_LOGS_SPEC.procedure,
            Duration::from_millis(20),
        )
        .await;
        assert_eq!(n, 1, "slow call over threshold emits exactly one WARN");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_rpc_silent_under_threshold() {
        // A huge threshold against a fast terminal -> no WARN.
        let n = slow_warn_count(
            SlowRpcInterceptor::new(100_000),
            STATS_SERVICE_QUERY_SPEC.procedure,
            Duration::from_millis(1),
        )
        .await;
        assert_eq!(n, 0, "fast call under threshold is silent");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_rpc_zero_disables() {
        // Threshold 0 for the method disables timing even on a slow terminal.
        let interceptor = SlowRpcInterceptor::new(DEFAULT_SLOW_RPC_THRESHOLD_MS)
            .with_method_threshold("FetchLogs", 0);
        let n = slow_warn_count(
            interceptor,
            LOG_SERVICE_FETCH_LOGS_SPEC.procedure,
            Duration::from_millis(20),
        )
        .await;
        assert_eq!(n, 0, "threshold<=0 disables the warning for that method");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn slow_rpc_default_for_unlisted() {
        // An unlisted method falls back to the (tiny) default -> WARN fires.
        let interceptor = SlowRpcInterceptor::new(1).with_method_threshold("Query", 0);
        let n = slow_warn_count(
            interceptor,
            LOG_SERVICE_FETCH_LOGS_SPEC.procedure,
            Duration::from_millis(20),
        )
        .await;
        assert_eq!(n, 1, "unlisted method uses the default threshold");
    }

    /// Drive `count` concurrent FetchLogs requests through the concurrency
    /// interceptor against a terminal that parks on `gate` (no sleep). Returns
    /// the observed peak number of terminals in flight at once.
    ///
    /// Each terminal increments a counter on entry, records the peak, then waits
    /// on the shared `Notify`; the test releases all permits at once after
    /// asserting the peak. This is a deterministic exact-cap proof (run_chain +
    /// parked terminal, no wall-clock sleep for the assertion).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrency_caps_in_flight() {
        let limit = 2usize;
        let callers = limit + 3;
        let interceptor: Arc<dyn Interceptor> =
            Arc::new(ConcurrencyInterceptor::new(limit, MAX_CONCURRENT_QUERY));
        let chain: Arc<Vec<Arc<dyn Interceptor>>> = Arc::new(vec![interceptor]);

        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(Mutex::new(0usize));
        let gate = Arc::new(Notify::new());
        // Counts terminals that have entered (acquired a permit + started).
        let entered = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..callers {
            let chain = Arc::clone(&chain);
            let in_flight = Arc::clone(&in_flight);
            let peak = Arc::clone(&peak);
            let gate = Arc::clone(&gate);
            let entered = Arc::clone(&entered);
            handles.push(tokio::spawn(async move {
                run_chain(
                    &chain,
                    request(LOG_SERVICE_FETCH_LOGS_SPEC.procedure, None),
                    move |_req| {
                        let in_flight = Arc::clone(&in_flight);
                        let peak = Arc::clone(&peak);
                        let gate = Arc::clone(&gate);
                        let entered = Arc::clone(&entered);
                        async move {
                            let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                            {
                                let mut p = peak.lock().unwrap();
                                *p = (*p).max(now);
                            }
                            entered.fetch_add(1, Ordering::SeqCst);
                            gate.notified().await;
                            in_flight.fetch_sub(1, Ordering::SeqCst);
                            Ok(ok_response())
                        }
                    },
                )
                .await
                .unwrap();
            }));
        }

        // Wait until exactly `limit` terminals are parked and no more can enter.
        // Poll the entered-count via yields (no wall-clock sleep): once `limit`
        // have entered, the remaining callers are blocked on the semaphore.
        loop {
            if entered.load(Ordering::SeqCst) >= limit {
                break;
            }
            tokio::task::yield_now().await;
        }
        // Give any erroneously-admitted extra caller a chance to enter, then
        // assert the cap held. Yield a bounded number of times so a regression
        // (limit+1 admitted) is observed without a timer.
        for _ in 0..1000 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            in_flight.load(Ordering::SeqCst),
            limit,
            "exactly `limit` terminals are in flight; the cap is enforced"
        );

        // Release all parked terminals; every caller completes.
        gate.notify_waiters();
        // notify_waiters only wakes current waiters; release in a loop until all
        // callers drained (later-admitted callers re-park as permits free up).
        for h in handles {
            // Keep nudging the gate so callers admitted after the first release
            // also wake.
            loop {
                gate.notify_waiters();
                tokio::task::yield_now().await;
                if h.is_finished() {
                    break;
                }
            }
            h.await.unwrap();
        }
        assert_eq!(*peak.lock().unwrap(), limit, "peak in-flight == cap");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrency_passthrough_unmatched() {
        // RegisterTable is uncapped: it always reaches the terminal even with a
        // cap of zero on the matched methods.
        let interceptor: Arc<dyn Interceptor> = Arc::new(ConcurrencyInterceptor::new(0, 0));
        let chain: Vec<Arc<dyn Interceptor>> = vec![interceptor];
        let ran = Arc::new(AtomicUsize::new(0));
        let ran2 = Arc::clone(&ran);
        run_chain(
            &chain,
            request("/finelog.stats.StatsService/RegisterTable", None),
            move |_req| {
                let ran2 = Arc::clone(&ran2);
                async move {
                    ran2.fetch_add(1, Ordering::SeqCst);
                    Ok(ok_response())
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "uncapped method passes through"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrency_sheds_expired_deadline() {
        // A request whose deadline is already in the past is shed with
        // DEADLINE_EXCEEDED before the terminal runs.
        let interceptor: Arc<dyn Interceptor> = Arc::new(ConcurrencyInterceptor::new(4, 4));
        let chain: Vec<Arc<dyn Interceptor>> = vec![interceptor];
        let ran = Arc::new(AtomicUsize::new(0));
        let ran2 = Arc::clone(&ran);
        // An expired deadline: an Instant in the past saturates time_remaining
        // to Some(ZERO).
        let past = Instant::now() - Duration::from_secs(1);
        let result = run_chain(
            &chain,
            request(LOG_SERVICE_FETCH_LOGS_SPEC.procedure, Some(past)),
            move |_req| {
                let ran2 = Arc::clone(&ran2);
                async move {
                    ran2.fetch_add(1, Ordering::SeqCst);
                    Ok(ok_response())
                }
            },
        )
        .await;
        assert!(result.is_err(), "expired deadline is shed");
        assert_eq!(
            result.unwrap_err().code,
            connectrpc::ErrorCode::DeadlineExceeded
        );
        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "the doomed terminal never ran"
        );
    }
}
