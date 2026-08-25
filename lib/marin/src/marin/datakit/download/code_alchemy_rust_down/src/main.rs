//! Software Heritage blob downloader -> per-shard uncompressed parquet.
//!
//! For each input file in INPUT_DIR (one blob_id per line; plain `.txt` or
//! zstd-compressed `.txt.zst`) we atomically produce one parquet file in
//! OUTPUT_DIR with `blob_id: Utf8` and `content: LargeBinary` (the raw,
//! already-gzipped object bytes). Non-success IDs are written to a sibling
//! TSV with an explicit not-found, retryable-failure, or permanent-failure
//! class. Aggregate accounting is atomically written to METRICS_PATH.
//! ## Connection architecture (the part that matters at 100k+ req/s)
//!
//! The bucket is PUBLIC: anonymous GETs, no AWS SDK, no signing. We do NOT use
//! a pooled client. hyper-util's legacy pool races "wait for an idle conn"
//! against "open a brand-new connection" for every request, so at high
//! concurrency nearly every request starts a TCP+TLS connect even though it
//! ends up served by a pooled conn (measured: ~1 connect per 3 requests, 4M
//! client RSTs). That handshake storm fills the Oracle NAT gateway's
//! per-destination connection table and the whole box blackholes for ~90s at
//! a time.
//!
//! Instead: a fixed FLEET of workers, each owning one long-lived hand-rolled
//! HTTP/1.1 connection, pulling blob ids from one shared MPMC queue. S3
//! honors request PIPELINING (verified) and serves exactly 100 responses per
//! connection, marking the 100th `Connection: close` and silently dropping
//! anything pipelined behind it -- so each worker keeps up to PIPELINE
//! requests on the wire, never sends more than REQS_PER_CONN (=100) per
//! connection, and lets *S3* close the socket (TIME_WAIT accrues on Amazon's
//! side, not ours). Batched writes + multi-response reads cut syscalls per
//! object by ~an order of magnitude vs request-per-wakeup clients. Workers
//! rotate reconnects across many S3 IPs collected by a background DNS
//! harvester (8 A records per 5s-TTL query; we accumulate them) so the NAT's
//! per-(dst ip,port) limits never concentrate. SCHEME=http by default: the
//! data is public and immutable, and skipping TLS makes the forced reconnect
//! every 100 requests a single RTT.
//!
//! Parquet: UNCOMPRESSED, PLAIN, no dictionary, no statistics (content is
//! high-entropy gzip); row groups flushed by accumulated bytes (ROW_GROUP_MB).
//! Each download holds its queue slot until the bytes reach the writer, so
//! disk backpressure reaches the network.
//!
//! Env knobs:
//!   INPUT_DIR (~/nt_swh_chunks)  OUTPUT_DIR (/mnt/temp/nt_swh_files)
//!   METRICS_PATH (OUTPUT_DIR/download-metrics.json)
//!   BUCKET (softwareheritage)    SCHEME (http)
//!   FLEET (1024, falls back to GLOBAL_CONCURRENCY)  PIPELINE (16)
//!   ACTIVE_FILES (16)            WRITER_CHAN_CAP (1024)  ROW_GROUP_MB (192)
//!   MAX_ATTEMPTS (6)             ATTEMPT_TIMEOUT_S (15)  CONNECT_TIMEOUT_S (5)
//!   REQS_PER_CONN (100)          RCVBUF (131072)  SNDBUF (16384)
//!   DNS_REFRESH_S (2)            MAX_IPS (512)
//!   TOKIO_WORKERS (2; 0 = current_thread runtime)
//!   SHARD_COUNT / SHARD_INDEX    (scale out across processes)

use std::env;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_compression::tokio::bufread::ZstdDecoder;
use bytes::{Buf, Bytes, BytesMut};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpSocket;
use tokio::sync::{Semaphore, mpsc};
use tokio::task::JoinSet;
use tokio::time::timeout;

use arrow::array::{ArrayRef, LargeBinaryBuilder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::{EnabledStatistics, WriterProperties};

// ----------------------------------------------------------------------------

#[derive(Default)]
struct Counters {
    ok: AtomicU64,
    not_found: AtomicU64,
    retryable_failure: AtomicU64,
    permanent_failure: AtomicU64,
    retry_events: AtomicU64,
    throttled: AtomicU64,
    bytes: AtomicU64,
    conn_opens: AtomicU64,
    conn_errors: AtomicU64,
    err_samples: AtomicU64,
}

#[derive(Clone, Copy)]
enum FailureKind {
    NotFound,
    Retryable,
    Permanent,
}

impl FailureKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::Retryable => "retryable_failure",
            Self::Permanent => "permanent_failure",
        }
    }
}

enum Record {
    Ok(String, Bytes),
    Failed(String, FailureKind),
}

/// One blob to fetch. Carries its shard's writer handle so the shared worker
/// fleet can serve many shards at once; the writer channel closes (and the
/// parquet file finalizes) when the reader and every outstanding Job for that
/// shard are done.
struct Job {
    id: String,
    tx: mpsc::Sender<Record>,
    attempts: u8,
}

struct Cfg {
    /// Request tail bytes appended after the blob id:
    /// `" HTTP/1.1\r\nhost: <host>\r\n\r\n"`.
    req_tail: Vec<u8>,
    tls: Option<(
        tokio_rustls::TlsConnector,
        rustls::pki_types::ServerName<'static>,
    )>,
    reqs_per_conn: u32,
    pipeline: usize,
    max_attempts: u8,
    attempt_timeout: Duration,
    connect_timeout: Duration,
    rcvbuf: usize,
    sndbuf: usize,
    /// Source address to bind (BIND_IP env). With secondary VNICs each
    /// process can egress from its own IP -> its own NAT/public identity,
    /// multiplying any per-source-IP ceiling (S3 or NAT gateway side).
    bind_ip: Option<std::net::IpAddr>,
}

// ----------------------------------------------------------------------------
// DNS harvester + latency-scored IP set. Each query returns 8 A records with
// a 5s TTL from a much larger rotating pool; spreading the fleet across many
// destination IPs keeps per-destination NAT tables shallow. Not all fronts
// are equally fast from here (measured ~20% spread), so workers report a
// per-connection service-time EWMA and selection uses power-of-two-choices
// (with a 1/128 random exploration pick so stale scores get refreshed).

struct IpEntry {
    addr: SocketAddr,
    ewma_us: AtomicU64, // microseconds per request on this front; 0 = unsampled
}

struct IpSet {
    entries: RwLock<Vec<IpEntry>>,
}

impl IpSet {
    fn pick(&self, rng: &mut u64) -> (usize, SocketAddr) {
        let g = self.entries.read().unwrap();
        let n = g.len();
        debug_assert!(n > 0);
        let r = xorshift(rng);
        let i = (r as usize) % n;
        if n == 1 || (r >> 32) % 128 == 0 {
            return (i, g[i].addr); // exploration pick
        }
        let j = (xorshift(rng) as usize) % n;
        let (ei, ej) = (
            g[i].ewma_us.load(Ordering::Relaxed),
            g[j].ewma_us.load(Ordering::Relaxed),
        );
        // Prefer unsampled fronts so new addresses get scored quickly.
        let k = match (ei, ej) {
            (0, _) => i,
            (_, 0) => j,
            _ if ei <= ej => i,
            _ => j,
        };
        (k, g[k].addr)
    }

    fn report(&self, idx: usize, sample_us: u64) {
        let g = self.entries.read().unwrap();
        if let Some(e) = g.get(idx) {
            let old = e.ewma_us.load(Ordering::Relaxed);
            let new = if old == 0 {
                sample_us
            } else {
                (old * 7 + sample_us) / 8
            };
            e.ewma_us.store(new.max(1), Ordering::Relaxed);
        }
    }

    fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }
}

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

async fn harvest_ips(ips: Arc<IpSet>, host: String, port: u16, every: Duration, cap: usize) {
    loop {
        if let Ok(found) = tokio::net::lookup_host((host.as_str(), port)).await {
            let mut g = ips.entries.write().unwrap();
            for a in found {
                if a.is_ipv4() && g.len() < cap && !g.iter().any(|e| e.addr == a) {
                    g.push(IpEntry {
                        addr: a,
                        ewma_us: AtomicU64::new(0),
                    });
                }
            }
        }
        tokio::time::sleep(every).await;
    }
}

// ----------------------------------------------------------------------------
// Fleet worker: one long-lived PIPELINED connection per worker. S3 honors
// HTTP/1.1 pipelining and serves exactly 100 responses per connection (the
// 100th carries `Connection: close`; later pipelined requests are silently
// discarded). So we write GETs in batches of PIPELINE while never sending
// more than REQS_PER_CONN (=100) per connection, parse responses in arrival
// order, and let *S3* close the socket -- the TIME_WAIT lands on Amazon's
// side, and we requeue nothing.

trait Conn: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Conn for T {}

async fn connect_stream(cfg: &Cfg, addr: SocketAddr) -> Result<Box<dyn Conn>> {
    let socket = TcpSocket::new_v4()?;
    if let Some(local) = cfg.bind_ip {
        socket.bind(SocketAddr::new(local, 0))?;
    }
    if cfg.rcvbuf > 0 {
        socket.set_recv_buffer_size(cfg.rcvbuf as u32)?;
    }
    if cfg.sndbuf > 0 {
        socket.set_send_buffer_size(cfg.sndbuf as u32)?;
    }
    let stream = timeout(cfg.connect_timeout, socket.connect(addr))
        .await
        .map_err(|_| anyhow::anyhow!("connect timeout to {addr}"))??;
    stream.set_nodelay(true)?;

    match &cfg.tls {
        None => Ok(Box::new(stream)),
        Some((connector, name)) => {
            let tls = timeout(cfg.connect_timeout, connector.connect(name.clone(), stream))
                .await
                .map_err(|_| anyhow::anyhow!("tls timeout to {addr}"))??;
            Ok(Box::new(tls))
        }
    }
}

/// One parsed response: status, server-asked-close, body bytes.
struct Resp {
    status: u16,
    close: bool,
    body: Bytes,
}

/// Try to parse one complete HTTP/1.1 response from the front of `buf`,
/// consuming it. Ok(None) = need more bytes. Errors are connection-fatal.
fn try_parse_response(buf: &mut BytesMut) -> Result<Option<Resp>> {
    let Some(hdr_len) = find_crlfcrlf(&buf[..]) else {
        anyhow::ensure!(buf.len() <= 64 * 1024, "header block > 64KB");
        return Ok(None);
    };
    let hdr_end = hdr_len + 4;

    anyhow::ensure!(
        buf.len() >= 12 && buf.starts_with(b"HTTP/1.1 "),
        "bad status line"
    );
    let d = &buf[9..12];
    anyhow::ensure!(
        d[0].is_ascii_digit() && d[1].is_ascii_digit() && d[2].is_ascii_digit(),
        "bad status code"
    );
    let status = (d[0] - b'0') as u16 * 100 + (d[1] - b'0') as u16 * 10 + (d[2] - b'0') as u16;

    let mut content_len: Option<usize> = None;
    let mut chunked = false;
    let mut close = false;
    for line in buf[..hdr_len].split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Some(colon) = line.iter().position(|&b| b == b':') else {
            continue;
        };
        let (name, rest) = line.split_at(colon);
        let val = rest[1..].trim_ascii();
        if name.eq_ignore_ascii_case(b"content-length") {
            let s = std::str::from_utf8(val).context("content-length utf8")?;
            content_len = Some(s.parse().context("content-length parse")?);
        } else if name.eq_ignore_ascii_case(b"connection") {
            close = val.eq_ignore_ascii_case(b"close");
        } else if name.eq_ignore_ascii_case(b"transfer-encoding") {
            // S3 returns chunked bodies on some error responses (e.g. 503
            // SlowDown XML). Decode them in-band instead of killing the
            // connection -- otherwise one throttle event requeues the whole
            // pipeline. Object GET 200s always carry content-length.
            chunked = val
                .split(|&b| b == b',')
                .any(|t| t.trim_ascii().eq_ignore_ascii_case(b"chunked"));
            anyhow::ensure!(chunked, "unsupported transfer-encoding");
        }
    }

    if chunked {
        let mut pos = hdr_end;
        let mut body: Vec<u8> = Vec::new();
        loop {
            let Some(rel) = find_crlf(&buf[pos..]) else {
                anyhow::ensure!(buf.len() - pos <= 64 * 1024, "chunk header > 64KB");
                return Ok(None);
            };
            let size_field = buf[pos..pos + rel].split(|&b| b == b';').next().unwrap();
            let hex = std::str::from_utf8(size_field.trim_ascii()).context("chunk size utf8")?;
            let size = usize::from_str_radix(hex, 16).context("chunk size parse")?;
            if size == 0 {
                // Final chunk: consume through the trailer-terminating CRLFCRLF.
                let Some(end) = find_crlfcrlf(&buf[pos..]) else {
                    return Ok(None);
                };
                buf.advance(pos + end + 4);
                return Ok(Some(Resp {
                    status,
                    close,
                    body: Bytes::from(body),
                }));
            }
            let data = pos + rel + 2;
            if buf.len() < data + size + 2 {
                return Ok(None);
            }
            body.extend_from_slice(&buf[data..data + size]);
            pos = data + size + 2; // skip chunk data + trailing CRLF
        }
    }

    let cl = content_len.context("response without content-length")?;
    if buf.len() < hdr_end + cl {
        return Ok(None);
    }
    let mut resp = buf.split_to(hdr_end + cl);
    let body = resp.split_off(hdr_end).freeze();
    Ok(Some(Resp {
        status,
        close,
        body,
    }))
}

fn find_crlfcrlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn find_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\r\n")
}

fn append_get(wbuf: &mut BytesMut, cfg: &Cfg, id: &str) {
    wbuf.extend_from_slice(b"GET /content/");
    wbuf.extend_from_slice(id.as_bytes());
    wbuf.extend_from_slice(&cfg.req_tail);
}

/// Deliver a parsed response for `job`. Returns the job back if it must be
/// retried (throttle / 5xx), along with whether we were throttled.
async fn deliver(mut job: Job, resp: Resp, cfg: &Cfg, counters: &Counters) -> (Option<Job>, bool) {
    match resp.status {
        200 => {
            counters.ok.fetch_add(1, Ordering::Relaxed);
            counters
                .bytes
                .fetch_add(resp.body.len() as u64, Ordering::Relaxed);
            let _ = job.tx.send(Record::Ok(job.id, resp.body)).await;
            (None, false)
        }
        404 => {
            counters.not_found.fetch_add(1, Ordering::Relaxed);
            let _ = job
                .tx
                .send(Record::Failed(job.id, FailureKind::NotFound))
                .await;
            (None, false)
        }
        429 | 500..=599 => {
            let throttled = resp.status == 503 || resp.status == 429;
            counters.retry_events.fetch_add(1, Ordering::Relaxed);
            if throttled {
                counters.throttled.fetch_add(1, Ordering::Relaxed);
            }
            job.attempts += 1;
            if job.attempts >= cfg.max_attempts {
                sample_err(counters, &job.id, &format!("HTTP {}", resp.status));
                counters.retryable_failure.fetch_add(1, Ordering::Relaxed);
                let _ = job
                    .tx
                    .send(Record::Failed(job.id, FailureKind::Retryable))
                    .await;
                (None, throttled)
            } else {
                (Some(job), throttled)
            }
        }
        other => {
            sample_err(counters, &job.id, &format!("HTTP {other}"));
            counters.permanent_failure.fetch_add(1, Ordering::Relaxed);
            let _ = job
                .tx
                .send(Record::Failed(job.id, FailureKind::Permanent))
                .await;
            (None, false)
        }
    }
}

async fn worker(
    wid: usize,
    queue: async_channel::Receiver<Job>,
    ips: Arc<IpSet>,
    cfg: Arc<Cfg>,
    counters: Arc<Counters>,
) {
    let mut retry: Vec<Job> = Vec::new();
    let mut pending: Option<Job> = None;
    let mut rng: u64 = (wid as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let mut consec_conn_fails: u32 = 0;
    let mut wbuf = BytesMut::with_capacity(4096);
    let mut rbuf = BytesMut::with_capacity(64 * 1024);
    // (ip index, established-at, responses served) of the finished connection,
    // reported to the scoreboard at the top of the next iteration.
    let mut last_conn: Option<(usize, Instant, u32)> = None;

    'outer: loop {
        if let Some((idx, started, served)) = last_conn.take() {
            if served >= 16 {
                let us = started.elapsed().as_micros() as u64 / served as u64;
                ips.report(idx, us);
            }
        }

        // Acquire the first job before connecting (don't hold idle conns).
        if pending.is_none() {
            pending = match retry.pop() {
                Some(j) => Some(j),
                None => match queue.recv().await {
                    Ok(j) => Some(j),
                    Err(_) => break 'outer, // queue closed and drained
                },
            };
        }

        let (ip_idx, addr) = ips.pick(&mut rng);
        let mut stream = match connect_stream(&cfg, addr).await {
            Ok(s) => {
                consec_conn_fails = 0;
                counters.conn_opens.fetch_add(1, Ordering::Relaxed);
                s
            }
            Err(e) => {
                counters.conn_errors.fetch_add(1, Ordering::Relaxed);
                consec_conn_fails += 1;
                if consec_conn_fails >= 10 {
                    // Network looks dead from here. Classify the exhausted
                    // request as retryable so a later run can recover it.
                    if let Some(job) = pending.take() {
                        sample_err(&counters, &job.id, &format!("connect: {e:#}"));
                        counters.retry_events.fetch_add(1, Ordering::Relaxed);
                        counters.retryable_failure.fetch_add(1, Ordering::Relaxed);
                        let _ = job
                            .tx
                            .send(Record::Failed(job.id, FailureKind::Retryable))
                            .await;
                    }
                    consec_conn_fails = 0;
                }
                let backoff = 50u64 << consec_conn_fails.min(6);
                tokio::time::sleep(Duration::from_millis(backoff + (wid as u64 % 47))).await;
                continue 'outer;
            }
        };

        last_conn = Some((ip_idx, Instant::now(), 0));
        let mut inflight: std::collections::VecDeque<Job> =
            std::collections::VecDeque::with_capacity(cfg.pipeline);
        let mut sent: u32 = 0;
        rbuf.clear();

        'conn: loop {
            // FILL: top up the pipeline without exceeding the per-conn budget.
            wbuf.clear();
            while inflight.len() < cfg.pipeline && sent < cfg.reqs_per_conn {
                let job = if let Some(j) = pending.take() {
                    j
                } else if let Some(j) = retry.pop() {
                    j
                } else {
                    match queue.try_recv() {
                        Ok(j) => j,
                        Err(_) => break,
                    }
                };
                append_get(&mut wbuf, &cfg, &job.id);
                inflight.push_back(job);
                sent += 1;
            }
            if !wbuf.is_empty() {
                match timeout(cfg.attempt_timeout, stream.write_all(&wbuf)).await {
                    Ok(Ok(())) => {}
                    _ => {
                        counters.conn_errors.fetch_add(1, Ordering::Relaxed);
                        requeue_inflight(&mut inflight, &mut retry, &cfg, &counters).await;
                        continue 'outer;
                    }
                }
            }

            if inflight.is_empty() {
                // Nothing in flight: either budget exhausted (reconnect) or
                // the queue is momentarily empty (block for the next job).
                if sent >= cfg.reqs_per_conn {
                    continue 'outer;
                }
                match queue.recv().await {
                    Ok(j) => {
                        pending = Some(j);
                        continue 'conn;
                    }
                    Err(_) => break 'outer,
                }
            }

            // READ: pull bytes, then parse out every complete response.
            rbuf.reserve(32 * 1024);
            match timeout(cfg.attempt_timeout, stream.read_buf(&mut rbuf)).await {
                Ok(Ok(n)) if n > 0 => {}
                _ => {
                    // timeout, EOF with unanswered requests, or socket error
                    counters.conn_errors.fetch_add(1, Ordering::Relaxed);
                    requeue_inflight(&mut inflight, &mut retry, &cfg, &counters).await;
                    continue 'outer;
                }
            }
            let mut throttled_here = false;
            loop {
                match try_parse_response(&mut rbuf) {
                    Ok(Some(resp)) => {
                        let close = resp.close;
                        let Some(job) = inflight.pop_front() else {
                            // response without a request: protocol corruption
                            counters.conn_errors.fetch_add(1, Ordering::Relaxed);
                            continue 'outer;
                        };
                        let (back, thr) = deliver(job, resp, &cfg, &counters).await;
                        throttled_here |= thr;
                        if let Some(c) = last_conn.as_mut() {
                            c.2 += 1;
                        }
                        if let Some(j) = back {
                            retry.push(j);
                        }
                        if close {
                            // Server is closing (the 100th response). Anything
                            // still pipelined behind it will never be answered.
                            requeue_inflight(&mut inflight, &mut retry, &cfg, &counters).await;
                            if throttled_here {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                            }
                            continue 'outer;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        sample_err(&counters, "(parse)", &format!("{e:#}"));
                        counters.conn_errors.fetch_add(1, Ordering::Relaxed);
                        requeue_inflight(&mut inflight, &mut retry, &cfg, &counters).await;
                        continue 'outer;
                    }
                }
            }
            if throttled_here {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            if sent >= cfg.reqs_per_conn && inflight.is_empty() {
                continue 'outer; // full budget served cleanly
            }
        }
    }
}

/// Push interrupted in-flight jobs onto the retry list (attempts+1), failing
/// out the ones that have exhausted their budget.
async fn requeue_inflight(
    inflight: &mut std::collections::VecDeque<Job>,
    retry: &mut Vec<Job>,
    cfg: &Cfg,
    counters: &Counters,
) {
    while let Some(mut job) = inflight.pop_front() {
        job.attempts += 1;
        counters.retry_events.fetch_add(1, Ordering::Relaxed);
        if job.attempts >= cfg.max_attempts {
            sample_err(counters, &job.id, "retries exhausted (conn errors)");
            counters.retryable_failure.fetch_add(1, Ordering::Relaxed);
            let _ = job
                .tx
                .send(Record::Failed(job.id, FailureKind::Retryable))
                .await;
        } else {
            retry.push(job);
        }
    }
}

// ----------------------------------------------------------------------------
// Writer: runs on a blocking thread, owns one parquet file for one shard.

fn run_writer(
    out_path: PathBuf,
    failed_path: PathBuf,
    mut rx: mpsc::Receiver<Record>,
    row_group_bytes: usize,
) -> Result<(u64, u64)> {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("blob_id", DataType::Utf8, false),
        Field::new("content", DataType::LargeBinary, false),
    ]));

    let props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::None)
        .set_encoding(Encoding::PLAIN)
        .set_max_row_group_row_count(Some(1_000_000_000))
        .set_write_batch_size(8192)
        .build();

    // Write to a sibling `<shard>.parquet.partial` and rename to the final name
    // ONLY after writer.close() flushes the footer + trailing PAR1 magic. A
    // process killed mid-shard therefore leaves a `.partial` (or nothing), never
    // a footer-less file under the real name -- so `parquet_is_complete` can tell
    // finished shards from interrupted ones and the run is safely resumable.
    let mut partial_os = out_path.clone().into_os_string();
    partial_os.push(".partial");
    let partial: PathBuf = partial_os.into();
    let file = std::fs::File::create(&partial).with_context(|| format!("create {partial:?}"))?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    let mut ids = StringBuilder::new();
    let mut contents = LargeBinaryBuilder::new();
    let mut pending_bytes: usize = 0;
    let mut rows_in_group: usize = 0;
    let mut total_rows: u64 = 0;
    let mut total_bytes: u64 = 0;

    let mut failed_sink: Option<std::io::BufWriter<std::fs::File>> = None;

    while let Some(rec) = rx.blocking_recv() {
        match rec {
            Record::Ok(id, data) => {
                ids.append_value(&id);
                contents.append_value(&data);
                pending_bytes += data.len();
                rows_in_group += 1;
                total_rows += 1;
                total_bytes += data.len() as u64;

                if pending_bytes >= row_group_bytes {
                    write_group(&mut writer, &mut ids, &mut contents, &schema)?;
                    pending_bytes = 0;
                    rows_in_group = 0;
                }
            }
            Record::Failed(id, kind) => {
                use std::io::Write;
                if failed_sink.is_none() {
                    let file = std::fs::File::create(&failed_path)
                        .with_context(|| format!("create failure details {failed_path:?}"))?;
                    failed_sink = Some(std::io::BufWriter::new(file));
                }
                let sink = failed_sink.as_mut().expect("failure sink initialized");
                writeln!(sink, "{id}\t{}", kind.as_str())
                    .with_context(|| format!("write failure details {failed_path:?}"))?;
            }
        }
    }

    if rows_in_group > 0 {
        write_group(&mut writer, &mut ids, &mut contents, &schema)?;
    }
    writer.close()?; // writes the parquet footer + trailing PAR1 magic
    std::fs::rename(&partial, &out_path)
        .with_context(|| format!("rename {partial:?} -> {out_path:?}"))?;
    if let Some(mut sink) = failed_sink {
        use std::io::Write;
        sink.flush()
            .with_context(|| format!("flush failure details {failed_path:?}"))?;
    }
    Ok((total_rows, total_bytes))
}

fn write_group(
    writer: &mut ArrowWriter<std::fs::File>,
    ids: &mut StringBuilder,
    contents: &mut LargeBinaryBuilder,
    schema: &SchemaRef,
) -> Result<()> {
    let id_arr: ArrayRef = Arc::new(ids.finish());
    let content_arr: ArrayRef = Arc::new(contents.finish());
    let batch = RecordBatch::try_new(schema.clone(), vec![id_arr, content_arr])?;
    writer.write(&batch)?;
    writer.flush()?; // close this row group at our chosen byte boundary
    Ok(())
}

// ----------------------------------------------------------------------------
// Per-shard pipeline: stream ids into the shared queue; the writer channel
// closes once the reader and every outstanding Job (each holds a tx clone)
// are done, which finalizes the parquet file.

async fn process_file(
    input: PathBuf,
    output: PathBuf,
    failed: PathBuf,
    queue: async_channel::Sender<Job>,
    writer_chan_cap: usize,
    row_group_bytes: usize,
    read_skip: usize,
) -> Result<()> {
    let (tx, rx) = mpsc::channel::<Record>(writer_chan_cap);
    let writer_handle =
        tokio::task::spawn_blocking(move || run_writer(output, failed, rx, row_group_bytes));

    let f = tokio::fs::File::open(&input)
        .await
        .with_context(|| format!("open {input:?}"))?;
    let reader: Box<dyn AsyncRead + Send + Unpin> = if is_zst(&input) {
        let mut dec = ZstdDecoder::new(BufReader::new(f));
        dec.multiple_members(true);
        Box::new(dec)
    } else {
        Box::new(f)
    };
    let mut reader = BufReader::with_capacity(256 * 1024, reader);

    // Benchmarking aid: skip past ids that recent runs already fetched (S3
    // serves recently-read objects ~2x faster, which poisons measurements).
    // Counts newlines chunk-wise instead of allocating a String per line.
    if read_skip > 0 {
        use tokio::io::AsyncBufReadExt as _;
        let mut remaining = read_skip;
        'skip: loop {
            let buf = reader.fill_buf().await?;
            if buf.is_empty() {
                break;
            }
            let mut consumed = buf.len();
            for (i, &b) in buf.iter().enumerate() {
                if b == b'\n' {
                    remaining -= 1;
                    if remaining == 0 {
                        consumed = i + 1;
                        reader.consume(consumed);
                        break 'skip;
                    }
                }
            }
            reader.consume(consumed);
        }
    }

    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        let id = line.trim();
        if id.is_empty() {
            continue;
        }
        let job = Job {
            id: id.to_string(),
            tx: tx.clone(),
            attempts: 0,
        };
        if queue.send(job).await.is_err() {
            break; // queue closed: shutting down
        }
    }

    drop(tx);
    let (rows, bytes) = writer_handle.await.context("writer join")??;
    eprintln!(
        "[shard done] {} rows={rows} bytes={bytes}",
        input.file_name().and_then(|s| s.to_str()).unwrap_or("?")
    );
    Ok(())
}

// ----------------------------------------------------------------------------

fn main() -> Result<()> {
    if env::args().nth(1).as_deref() == Some("--version") {
        println!("swh_downloader {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let tokio_workers = env_usize("TOKIO_WORKERS", 2);
    let rt = if tokio_workers == 0 {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(tokio_workers)
            .enable_all()
            .build()?
    };
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    // Install the rustls crypto provider up front so the TLS builder is
    // unambiguous (only used when SCHEME=https).
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    match rlimit::increase_nofile_limit(1_048_576) {
        Ok(n) => eprintln!("[init] nofile soft limit -> {n}"),
        Err(e) => eprintln!("[init] could not raise nofile limit ({e}); set `ulimit -n` high"),
    }

    let input_dir = expand(env_str("INPUT_DIR", "~/nt_swh_chunks"));
    let output_dir = expand(env_str("OUTPUT_DIR", "/mnt/temp/nt_swh_files"));
    let bucket = env_str("BUCKET", "softwareheritage");
    let scheme = env_str("SCHEME", "http");
    let fleet = env_usize("FLEET", env_usize("GLOBAL_CONCURRENCY", 1024));
    let pipeline = env_usize("PIPELINE", 16).max(1);
    let active_files = env_usize("ACTIVE_FILES", 16);
    let writer_chan_cap = env_usize("WRITER_CHAN_CAP", 1024);
    let row_group_bytes = env_usize("ROW_GROUP_MB", 192) * 1024 * 1024;
    let max_attempts = env_usize("MAX_ATTEMPTS", 6) as u8;
    let attempt_timeout = Duration::from_secs(env_usize("ATTEMPT_TIMEOUT_S", 15) as u64);
    let connect_timeout = Duration::from_secs(env_usize("CONNECT_TIMEOUT_S", 5) as u64);
    // S3 serves exactly 100 responses per connection, then closes.
    let reqs_per_conn = env_usize("REQS_PER_CONN", 100) as u32;
    let rcvbuf = env_usize("RCVBUF", 131072);
    let sndbuf = env_usize("SNDBUF", 16384);
    let dns_refresh = Duration::from_secs(env_usize("DNS_REFRESH_S", 2) as u64);
    let max_ips = env_usize("MAX_IPS", 512);
    let read_skip = env_usize("READ_SKIP", 0);
    let shard_count = env_usize("SHARD_COUNT", 1).max(1);
    let shard_index = env_usize("SHARD_INDEX", 0);

    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("create output dir {output_dir:?}"))?;

    // Virtual-hosted-style host (bucket name has no dots, so the
    // *.s3.amazonaws.com cert covers it when SCHEME=https).
    let host = format!("{bucket}.s3.amazonaws.com");
    let port: u16 = if scheme == "https" { 443 } else { 80 };
    let tls = if scheme == "https" {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let cc = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let name = rustls::pki_types::ServerName::try_from(host.clone())?;
        Some((tokio_rustls::TlsConnector::from(Arc::new(cc)), name))
    } else {
        None
    };

    let cfg = Arc::new(Cfg {
        req_tail: format!(" HTTP/1.1\r\nhost: {host}\r\n\r\n").into_bytes(),
        tls,
        reqs_per_conn,
        pipeline,
        max_attempts,
        attempt_timeout,
        connect_timeout,
        rcvbuf,
        sndbuf,
        bind_ip: env::var("BIND_IP").ok().and_then(|s| s.parse().ok()),
    });

    // Seed the IP set synchronously so workers always see >= 1 address.
    let ips = Arc::new(IpSet {
        entries: RwLock::new(Vec::new()),
    });
    {
        let seed: Vec<IpEntry> = tokio::net::lookup_host((host.as_str(), port))
            .await
            .with_context(|| format!("resolve {host}"))?
            .filter(|a| a.is_ipv4())
            .map(|addr| IpEntry {
                addr,
                ewma_us: AtomicU64::new(0),
            })
            .collect();
        anyhow::ensure!(!seed.is_empty(), "no A records for {host}");
        *ips.entries.write().unwrap() = seed;
    }
    tokio::spawn(harvest_ips(
        ips.clone(),
        host.clone(),
        port,
        dns_refresh,
        max_ips,
    ));

    let mut inputs: Vec<PathBuf> = std::fs::read_dir(&input_dir)
        .with_context(|| format!("read_dir {input_dir:?}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.ends_with(".zst") || n.ends_with(".txt"))
                .unwrap_or(false)
        })
        .collect();
    inputs.sort();
    if shard_count > 1 {
        inputs = inputs
            .into_iter()
            .enumerate()
            .filter(|(i, _)| i % shard_count == shard_index)
            .map(|(_, p)| p)
            .collect();
    }
    if inputs.is_empty() {
        anyhow::bail!("no input files for this partition under {input_dir:?}");
    }
    // Resume: skip shards already written completely (valid parquet footer). A
    // shard whose writer died mid-flight has column data but no footer, so it is
    // treated as incomplete and re-downloaded. This makes the whole run
    // idempotent -- re-running after ANY interruption (signal, OOM, reboot)
    // finishes only what's left and never clobbers a good file.
    let partition_total = inputs.len();
    inputs.retain(|p| !parquet_is_complete(&output_dir.join(format!("{}.parquet", shard_name(p)))));
    let skipped = partition_total - inputs.len();
    if inputs.is_empty() {
        eprintln!(
            "[init] partition {shard_index}/{shard_count}: all {partition_total} shard(s) already complete; nothing to do"
        );
        return Ok(());
    }
    eprintln!(
        "[init] {} shard(s) to (re)download (partition {shard_index}/{shard_count}; {skipped} already complete) | active={active_files} | fleet={fleet} | pipeline={pipeline} | scheme={scheme} | row_group={}MB",
        inputs.len(),
        row_group_bytes / 1024 / 1024
    );

    let (queue_tx, queue_rx) = async_channel::bounded::<Job>(fleet * pipeline * 2);
    let counters = Arc::new(Counters::default());

    // Graceful shutdown: on SIGINT/SIGTERM, stop pulling NEW shards into the
    // active set but let already-started shards run to completion and finalize
    // their parquet (footer written, atomic rename). Un-started shards are
    // simply not created -> the resume check picks them up next run. A second
    // signal force-exits (those in-flight shards stay `.partial` and re-run).
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{SignalKind, signal};
            let mut term = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(_) => return,
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            shutdown.store(true, Ordering::SeqCst);
            eprintln!(
                "[signal] shutdown requested: no new shards will start; finishing in-flight shards (signal again to force-exit)"
            );
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            eprintln!(
                "[signal] second signal: forcing exit; in-flight shards left .partial -- re-run to finish"
            );
            std::process::exit(130);
        });
    }

    let mut workers = JoinSet::new();
    for wid in 0..fleet {
        workers.spawn(worker(
            wid,
            queue_rx.clone(),
            ips.clone(),
            cfg.clone(),
            counters.clone(),
        ));
    }

    let prog = counters.clone();
    let prog_q = queue_tx.clone();
    let prog_ips = ips.clone();
    let progress = tokio::spawn(async move {
        let start = Instant::now();
        let mut last_ok = 0u64;
        let mut last_by = 0u64;
        let mut last_op = 0u64;
        let mut last_t = Instant::now();
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            let ok = prog.ok.load(Ordering::Relaxed);
            let by = prog.bytes.load(Ordering::Relaxed);
            let not_found = prog.not_found.load(Ordering::Relaxed);
            let retryable = prog.retryable_failure.load(Ordering::Relaxed);
            let permanent = prog.permanent_failure.load(Ordering::Relaxed);
            let retries = prog.retry_events.load(Ordering::Relaxed);
            let thr = prog.throttled.load(Ordering::Relaxed);
            let op = prog.conn_opens.load(Ordering::Relaxed);
            let ce = prog.conn_errors.load(Ordering::Relaxed);
            let now = Instant::now();
            let dt = now.duration_since(last_t).as_secs_f64().max(1e-6);
            let ops = (ok - last_ok) as f64 / dt;
            let mbps = (by - last_by) as f64 / dt / 1.0e6;
            let conn_s = (op - last_op) as f64 / dt;
            eprintln!(
                "[{:>6.0}s] success={ok} not_found={not_found} retryable_failure={retryable} permanent_failure={permanent} retries={retries} throttled={thr} | {ops:>8.0} obj/s | {mbps:>7.1} MB/s | conns {conn_s:>5.0}/s (tot {op}, err {ce}) | q={} ips={}",
                start.elapsed().as_secs_f64(),
                prog_q.len(),
                prog_ips.len(),
            );
            last_ok = ok;
            last_by = by;
            last_op = op;
            last_t = now;
        }
    });

    let file_sem = Arc::new(Semaphore::new(active_files));
    let mut readers = JoinSet::new();
    for input in inputs {
        let stem = shard_name(&input);
        let output = output_dir.join(format!("{stem}.parquet"));
        let failed = output_dir.join(format!("{stem}.failures.tsv"));
        let queue = queue_tx.clone();
        let fsem = file_sem.clone();
        let shutdown = shutdown.clone();

        readers.spawn(async move {
            let _file_permit = fsem.acquire_owned().await.expect("file sem closed");
            // Shutting down: don't start a new shard (resume handles it next run).
            if shutdown.load(Ordering::SeqCst) {
                return Ok::<(), anyhow::Error>(());
            }
            process_file(
                input,
                output,
                failed,
                queue,
                writer_chan_cap,
                row_group_bytes,
                read_skip,
            )
            .await
        });
    }

    while let Some(res) = readers.join_next().await {
        res.context("reader join")??;
    }
    // All shards fully written: stop the fleet.
    queue_tx.close();
    while workers.join_next().await.is_some() {}
    progress.abort();

    let success = counters.ok.load(Ordering::Relaxed);
    let not_found = counters.not_found.load(Ordering::Relaxed);
    let retryable_failure = counters.retryable_failure.load(Ordering::Relaxed);
    let permanent_failure = counters.permanent_failure.load(Ordering::Relaxed);
    let retry_events = counters.retry_events.load(Ordering::Relaxed);
    let throttled = counters.throttled.load(Ordering::Relaxed);
    let bytes = counters.bytes.load(Ordering::Relaxed);
    let conn_opens = counters.conn_opens.load(Ordering::Relaxed);
    let conn_errors = counters.conn_errors.load(Ordering::Relaxed);
    let metrics_path = expand(env_str(
        "METRICS_PATH",
        output_dir
            .join("download-metrics.json")
            .to_string_lossy()
            .as_ref(),
    ));
    write_metrics(
        &metrics_path,
        success,
        not_found,
        retryable_failure,
        permanent_failure,
        retry_events,
        throttled,
        bytes,
        conn_opens,
        conn_errors,
    )?;
    eprintln!(
        "[ALL DONE] success={success} not_found={not_found} retryable_failure={retryable_failure} permanent_failure={permanent_failure} retries={retry_events} throttled={throttled} conns={conn_opens} bytes={bytes} ({:.2} GiB)",
        bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_metrics(
    path: &Path,
    success: u64,
    not_found: u64,
    retryable_failure: u64,
    permanent_failure: u64,
    retry_events: u64,
    throttled: u64,
    bytes: u64,
    conn_opens: u64,
    conn_errors: u64,
) -> Result<()> {
    use std::io::Write;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut partial = path.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let classified = success + not_found + retryable_failure + permanent_failure;
    let mut file = std::fs::File::create(&partial)?;
    writeln!(
        file,
        "{{\"success\":{success},\"not_found\":{not_found},\"retryable_failure\":{retryable_failure},\"permanent_failure\":{permanent_failure},\"classified\":{classified},\"retry_events\":{retry_events},\"throttled\":{throttled},\"bytes\":{bytes},\"connection_opens\":{conn_opens},\"connection_errors\":{conn_errors}}}"
    )?;
    file.sync_all()?;
    std::fs::rename(&partial, path)?;
    Ok(())
}

// ----------------------------------------------------------------------------

fn env_str(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn expand(p: String) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(p)
}

/// Log the first handful of real errors (not 404s, which are expected and
/// counted separately) so throttling/timeouts surface immediately.
fn sample_err(counters: &Counters, path: &str, msg: &str) {
    if counters.err_samples.fetch_add(1, Ordering::Relaxed) < 20 {
        eprintln!("[error] {path}: {msg}");
    }
}

fn is_zst(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("zst"))
        .unwrap_or(false)
}

/// A parquet file is complete iff it ends with the 4-byte `PAR1` magic, which
/// `ArrowWriter::close()` writes only after the footer. A shard whose process
/// was killed mid-write has column data but no footer -> reported incomplete and
/// re-downloaded on the next run. Cheap (one seek + 4-byte read), so it is safe
/// to call once per shard at startup.
fn parquet_is_complete(path: &Path) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(len) = f.metadata().map(|m| m.len()) else {
        return false;
    };
    // Smallest legal parquet is "PAR1" header + footer + "PAR1" trailer.
    if len < 8 || f.seek(SeekFrom::End(-4)).is_err() {
        return false;
    }
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).is_ok() && &magic == b"PAR1"
}

/// Output shard name: strip a trailing `.zst` and then `.txt`
/// (so `bucket_0.txt.zst` -> `bucket_0`).
fn shard_name(path: &Path) -> String {
    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("shard");
    let name = name.strip_suffix(".zst").unwrap_or(name);
    let name = name.strip_suffix(".txt").unwrap_or(name);
    name.to_string()
}
