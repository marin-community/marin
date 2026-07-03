// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Authenticated ingress front for the finelog server.
//!
//! A globally shared finelog receives pushes from many controllers across the
//! internet, so it can no longer rely on being private behind one controller's
//! proxy. This module gates every RPC with an ordered stack of auth *layers*,
//! mirroring rigging's `server_auth` (each layer *allows*, *falls through*, or
//! *rejects*).
//!
//! The policy is **default-deny**: a request no layer *allows* is rejected
//! `Unauthenticated`. Auth is a stack of allow-layers on top of that deny —
//! there is no "empty means open" path. The interceptor is *always* installed
//! (see `ServerConfig`); the only variable is the layer list. When nothing is
//! configured the default is [`AuthPolicy::allow_localhost`] (loopback only), so
//! a bare finelog is reachable for local debugging but never open to its network.
//! Two layer kinds compose:
//! - [`AuthLayer::Jwt`] — a bearer whose HS256 signature verifies against one of a
//!   set of trusted per-cluster keys, and which has not expired, admits the
//!   request. This mirrors marin's own token model (`JwtTokenManager` mints
//!   `HS256` JWTs signed with a per-cluster HMAC key); each relaying controller
//!   mints a short-lived finelog-delegation JWT and the store verifies it against
//!   that cluster's key — the same JWT mechanism the control plane uses, so the
//!   log plane adds no second credential *system*. The store holds only
//!   delegation keys (a key that grants finelog access, not control-plane token
//!   minting), and checks signature + `exp` only (it cannot reach a controller's
//!   revocation table, so exposure is TTL-bounded). Every configured key admits
//!   equally — federation members are mutually trusted for the log plane.
//! - [`AuthLayer::Cidr`] — a request whose transport peer is in a trusted network
//!   is admitted without a token. This reproduces today's intra-cluster
//!   reachability explicitly: a global finelog that also serves its own cluster
//!   lists that cluster's loopback/VPC ranges (e.g. `127.0.0.0/8`, `10.0.0.0/8`)
//!   so local clients keep working without a JWT, while remote pushes must sign.
//!   CIDR matches the transport peer only (never a forwarded header), the same
//!   distrust of spoofable `X-Forwarded-For` that rigging's loopback check applies.
//!
//! The layers are walked in order: the first `Allow` admits, the first `Reject`
//! denies, and a request no layer claims is denied. Order matters — the CIDR
//! layer is placed first so a trusted-network client is admitted before the JWT
//! layer would reject a token it cannot verify (e.g. the home controller's
//! control-plane `worker_token`, which a global finelog holding only delegation
//! keys cannot check).

use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{ConnectInfo, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next as AxumNext;
use axum::response::{IntoResponse, Response};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use connectrpc::{
    async_trait, ConnectError, Interceptor, Next, RequestContext, UnaryRequest, UnaryResponse,
};
use hmac::{Hmac, Mac};
use serde::Deserialize;
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Accept a token whose `exp` is at most this far in the past, to tolerate small
/// clock skew between a minting controller and the store.
const EXP_LEEWAY_SECONDS: i64 = 60;

/// Reject an HS256 delegation secret shorter than this (a too-short HMAC key is a
/// misconfiguration, not a valid key). marin's own keys are `token_hex(32)` = 64
/// ASCII bytes, well above this floor.
const MIN_SECRET_BYTES: usize = 16;

/// One rule's verdict over a request, mirroring rigging's
/// AUTHENTICATED / ABSENT / REJECTED.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// This rule admits the request; stop and admit.
    Allow,
    /// This rule does not claim the request; try the next.
    Fallthrough,
    /// A credential was presented but is invalid; stop and deny.
    Reject,
}

/// An IPv4/IPv6 network in CIDR form (`10.0.0.0/8`, `2001:db8::/32`).
#[derive(Debug, Clone)]
pub struct Cidr {
    network: IpAddr,
    prefix_len: u8,
}

impl Cidr {
    /// Parse `"<addr>/<prefix_len>"`. A bare address (no `/`) is a host route
    /// (full-length prefix). Errors on a malformed address or an out-of-range
    /// prefix so a bad config fails the server at startup rather than silently
    /// admitting nothing.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let (addr_str, prefix_str) = match spec.split_once('/') {
            Some((a, p)) => (a, Some(p)),
            None => (spec, None),
        };
        let network: IpAddr = addr_str
            .parse()
            .map_err(|_| format!("invalid CIDR address {addr_str:?} in {spec:?}"))?;
        let max = if network.is_ipv4() { 32 } else { 128 };
        let prefix_len = match prefix_str {
            Some(p) => p
                .parse::<u8>()
                .map_err(|_| format!("invalid CIDR prefix {p:?} in {spec:?}"))?,
            None => max,
        };
        if prefix_len > max {
            return Err(format!(
                "CIDR prefix /{prefix_len} exceeds /{max} in {spec:?}"
            ));
        }
        Ok(Self {
            network,
            prefix_len,
        })
    }

    /// Whether `addr` falls within this network (same family, matching prefix).
    pub fn contains(&self, addr: IpAddr) -> bool {
        match (self.network, addr) {
            (IpAddr::V4(net), IpAddr::V4(ip)) => {
                prefix_matches(&net.octets(), &ip.octets(), self.prefix_len)
            }
            (IpAddr::V6(net), IpAddr::V6(ip)) => {
                prefix_matches(&net.octets(), &ip.octets(), self.prefix_len)
            }
            // A v4-mapped v6 peer (`::ffff:a.b.c.d`, common when a dual-stack
            // listener accepts a v4 client) is matched against a v4 rule on its
            // embedded v4 address.
            (IpAddr::V4(net), IpAddr::V6(ip)) => ip
                .to_ipv4_mapped()
                .is_some_and(|v4| prefix_matches(&net.octets(), &v4.octets(), self.prefix_len)),
            (IpAddr::V6(_), IpAddr::V4(_)) => false,
        }
    }
}

/// Whether the first `prefix_len` bits of `net` and `addr` are equal.
fn prefix_matches(net: &[u8], addr: &[u8], prefix_len: u8) -> bool {
    let mut bits = prefix_len as usize;
    for (n, a) in net.iter().zip(addr.iter()) {
        if bits == 0 {
            break;
        }
        if bits >= 8 {
            if n != a {
                return false;
            }
            bits -= 8;
        } else {
            let mask = 0xffu8 << (8 - bits);
            return (n & mask) == (a & mask);
        }
    }
    true
}

/// A trusted per-cluster delegation key: the cluster it authenticates and the
/// HS256 secret (the JWT signing key's raw bytes). `Debug` renders the cluster
/// but never the secret.
struct JwtKey {
    cluster: String,
    secret: Vec<u8>,
}

impl fmt::Debug for JwtKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JwtKey")
            .field("cluster", &self.cluster)
            .finish_non_exhaustive()
    }
}

/// Verifies HS256 JWTs against a set of trusted per-cluster delegation keys.
///
/// A token is accepted iff its signature verifies against one configured key
/// (constant-time HMAC compare) and its `exp` is in the future within
/// [`EXP_LEEWAY_SECONDS`]. Only delegation keys are held — never a controller's
/// control-plane signing key — so the store can verify a cluster's tokens
/// without gaining the power to mint control-plane ones. On success the matching
/// cluster is returned for attribution. Every configured delegation key admits
/// equally: federation members are mutually trusted for the log plane, so a
/// valid token may write under any namespace (per-cluster write isolation is not
/// enforced in v1).
#[derive(Debug)]
pub struct JwtVerifier {
    keys: Vec<JwtKey>,
}

impl JwtVerifier {
    /// Build from `(cluster, secret)` pairs. The secret is HMAC-keyed as raw
    /// ASCII bytes to match PyJWT (`jwt.encode(payload, signing_key, "HS256")`
    /// HMACs the signing-key STRING's bytes), so a hex `secrets.token_hex(32)`
    /// key is passed through verbatim, never hex-decoded. Rejects an empty
    /// cluster or a secret shorter than [`MIN_SECRET_BYTES`].
    fn new(keys: Vec<(String, String)>) -> Result<Self, String> {
        let mut compiled = Vec::with_capacity(keys.len());
        for (cluster, secret) in keys {
            if cluster.is_empty() {
                return Err("jwt key entry has an empty cluster".to_string());
            }
            if secret.len() < MIN_SECRET_BYTES {
                return Err(format!(
                    "jwt delegation secret for cluster {cluster:?} is too short \
                     ({} bytes; need >= {MIN_SECRET_BYTES})",
                    secret.len()
                ));
            }
            compiled.push(JwtKey {
                cluster,
                secret: secret.into_bytes(),
            });
        }
        Ok(Self { keys: compiled })
    }

    /// The cluster whose key verifies `token` (HS256 signature valid + unexpired
    /// at `now_unix` within the skew leeway), or `None` if none does. Decodes the
    /// JWS envelope once, then tries each trusted key against the same bytes.
    fn verify(&self, token: &str, now_unix: i64) -> Option<&str> {
        let parts = JwtParts::split(token)?;
        if !parts.is_hs256() {
            return None;
        }
        let claims = parts.claims()?;
        if claims.exp + EXP_LEEWAY_SECONDS < now_unix {
            return None; // expired beyond the skew allowance
        }
        self.keys
            .iter()
            .find(|k| parts.signature_valid(&k.secret))
            .map(|k| k.cluster.as_str())
    }
}

/// The declarative shape of one auth layer, deserialized from the
/// `FINELOG_AUTH_POLICY` JSON (== the finelog config's `auth:` list). List order
/// is evaluation order.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AuthLayerConfig {
    /// Admit a request whose transport peer is in one of these networks.
    Cidr { cidrs: Vec<String> },
    /// Admit a request bearing a JWT that verifies against one of these
    /// per-cluster delegation keys.
    Jwt { keys: Vec<JwtKeyConfig> },
}

#[derive(Debug, Deserialize)]
struct JwtKeyConfig {
    cluster: String,
    secret: String,
}

/// One compiled entry in the ordered auth stack.
#[derive(Debug)]
enum AuthLayer {
    Cidr(Vec<Cidr>),
    Jwt(JwtVerifier),
}

impl AuthLayer {
    /// Compile a config layer into its runtime form, validating CIDRs and keys.
    fn compile(config: AuthLayerConfig) -> Result<Self, String> {
        match config {
            AuthLayerConfig::Cidr { cidrs } => {
                if cidrs.is_empty() {
                    return Err("cidr layer has no networks".to_string());
                }
                let nets = cidrs
                    .iter()
                    .map(|s| Cidr::parse(s))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(AuthLayer::Cidr(nets))
            }
            AuthLayerConfig::Jwt { keys } => {
                if keys.is_empty() {
                    return Err("jwt layer has no keys".to_string());
                }
                let pairs = keys.into_iter().map(|k| (k.cluster, k.secret)).collect();
                Ok(AuthLayer::Jwt(JwtVerifier::new(pairs)?))
            }
        }
    }

    /// Evaluate this layer against extracted request facts: the bearer token
    /// (already stripped of `Bearer `) and the transport peer IP.
    fn check(&self, bearer: Option<&str>, peer_ip: Option<IpAddr>) -> Verdict {
        match self {
            AuthLayer::Cidr(nets) => match peer_ip {
                Some(ip) if nets.iter().any(|n| n.contains(ip)) => Verdict::Allow,
                _ => Verdict::Fallthrough,
            },
            // rigging JwtAuthenticator parity: no bearer → fall through (a later
            // layer may claim it); a valid bearer → allow; a present-but-invalid
            // bearer → reject, never downgraded to a weaker layer.
            AuthLayer::Jwt(verifier) => match bearer {
                None => Verdict::Fallthrough,
                Some(tok) => match verifier.verify(tok, now_unix()) {
                    Some(cluster) => {
                        tracing::trace!(cluster, "finelog: admitted jwt-authenticated request");
                        Verdict::Allow
                    }
                    None => Verdict::Reject,
                },
            },
        }
    }
}

/// The three dot-separated base64url segments of a JWT, plus the signing input.
struct JwtParts<'a> {
    header_b64: &'a str,
    payload_b64: &'a str,
    signature: Vec<u8>,
}

impl<'a> JwtParts<'a> {
    fn split(token: &'a str) -> Option<Self> {
        let mut it = token.split('.');
        let header_b64 = it.next()?;
        let payload_b64 = it.next()?;
        let sig_b64 = it.next()?;
        if it.next().is_some() {
            return None; // a JWS has exactly three segments
        }
        let signature = URL_SAFE_NO_PAD.decode(sig_b64).ok()?;
        Some(Self {
            header_b64,
            payload_b64,
            signature,
        })
    }

    /// Reject anything but HS256 up front (guards against `alg: none` and
    /// algorithm-confusion tokens, even though we only ever HMAC-verify).
    fn is_hs256(&self) -> bool {
        let Ok(bytes) = URL_SAFE_NO_PAD.decode(self.header_b64) else {
            return false;
        };
        let Ok(header) = serde_json::from_slice::<JwtHeader>(&bytes) else {
            return false;
        };
        header.alg.eq_ignore_ascii_case("HS256")
    }

    fn claims(&self) -> Option<JwtClaims> {
        let bytes = URL_SAFE_NO_PAD.decode(self.payload_b64).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Constant-time HMAC-SHA256 check of `header.payload` against `key`.
    fn signature_valid(&self, key: &[u8]) -> bool {
        let Ok(mut mac) = HmacSha256::new_from_slice(key) else {
            return false;
        };
        mac.update(self.header_b64.as_bytes());
        mac.update(b".");
        mac.update(self.payload_b64.as_bytes());
        mac.verify_slice(&self.signature).is_ok()
    }
}

#[derive(serde::Deserialize)]
struct JwtHeader {
    #[serde(default)]
    alg: String,
}

#[derive(serde::Deserialize)]
struct JwtClaims {
    exp: i64,
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// An ordered stack of auth layers with a **default-deny** terminal. Always
/// installed (see `ServerConfig`); the private default is [`allow_localhost`].
///
/// [`allow_localhost`]: AuthPolicy::allow_localhost
#[derive(Debug)]
pub struct AuthPolicy {
    layers: Vec<AuthLayer>,
}

impl AuthPolicy {
    /// The private default: admit loopback only, deny everything else. Used when
    /// no `auth:` policy is configured, so a bare finelog is reachable for local
    /// debugging but never open to its network.
    pub fn allow_localhost() -> Self {
        Self {
            layers: vec![AuthLayer::Cidr(vec![
                Cidr::parse("127.0.0.0/8").expect("valid loopback CIDR"),
                Cidr::parse("::1/128").expect("valid loopback CIDR"),
            ])],
        }
    }

    /// Parse an ordered layer list from JSON (the `FINELOG_AUTH_POLICY` value):
    /// `[{"type":"cidr","cidrs":[..]},{"type":"jwt","keys":[{"cluster":..,"secret":..}]}]`.
    /// An empty list is rejected (that is a total lockout — omit the policy
    /// entirely for the allow-localhost default). A malformed entry, CIDR, or
    /// too-short secret errors so the server fails at startup rather than silently
    /// mis-admitting.
    pub fn parse(json: &str) -> Result<Self, String> {
        let configs: Vec<AuthLayerConfig> =
            serde_json::from_str(json).map_err(|e| format!("invalid auth policy JSON: {e}"))?;
        if configs.is_empty() {
            return Err(
                "auth policy is an empty list (omit it for the allow-localhost default)"
                    .to_string(),
            );
        }
        let layers = configs
            .into_iter()
            .map(AuthLayer::compile)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    /// A one-line, secret-free description of the stack for a startup log line
    /// (e.g. `cidr[3], jwt[2]`).
    pub fn describe(&self) -> String {
        self.layers
            .iter()
            .map(|l| match l {
                AuthLayer::Cidr(nets) => format!("cidr[{}]", nets.len()),
                AuthLayer::Jwt(v) => format!("jwt[{}]", v.keys.len()),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The core decision over already-extracted request facts: the bearer token
    /// (no `Bearer ` prefix) and the transport peer IP. `true` = admit. Walks the
    /// stack: first `Allow` admits, first `Reject` denies, an unclaimed request is
    /// denied (default-deny). Shared by the Connect interceptor and the axum
    /// [`auth_gate`] middleware so both surfaces enforce the identical policy.
    fn admits(&self, bearer: Option<&str>, peer_ip: Option<IpAddr>) -> bool {
        for layer in &self.layers {
            match layer.check(bearer, peer_ip) {
                Verdict::Allow => return true,
                Verdict::Reject => return false,
                Verdict::Fallthrough => {}
            }
        }
        false
    }

    /// Connect-interceptor entry point: extract the bearer + peer IP from the RPC
    /// context and apply the policy.
    fn authorize(&self, ctx: &RequestContext) -> Result<(), ConnectError> {
        if self.admits(bearer_token(ctx), peer_ip(ctx)) {
            Ok(())
        } else {
            Err(unauthenticated())
        }
    }
}

fn unauthenticated() -> ConnectError {
    ConnectError::unauthenticated("finelog: request not authorized (no auth layer admitted it)")
}

/// The token from an `Authorization` header value, with or without the `Bearer `
/// scheme prefix. `None` when empty.
fn strip_bearer(header: &str) -> Option<&str> {
    let token = header.strip_prefix("Bearer ").unwrap_or(header).trim();
    (!token.is_empty()).then_some(token)
}

/// The bearer token from an RPC's `Authorization` header. `None` when absent or
/// empty.
fn bearer_token(ctx: &RequestContext) -> Option<&str> {
    strip_bearer(ctx.header("authorization")?.to_str().ok()?)
}

/// The request's transport peer IP. `axum::serve` records it as
/// `ConnectInfo<SocketAddr>` in the request extensions (see
/// `into_make_service_with_connect_info` at the server bind sites), which the
/// connect service copies into the `RequestContext`.
fn peer_ip(ctx: &RequestContext) -> Option<IpAddr> {
    ctx.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip())
}

/// Server interceptor enforcing an [`AuthPolicy`] over every unary RPC.
///
/// Always installed (see `build_connect_service`); the private default policy is
/// [`AuthPolicy::allow_localhost`]. Gates every method on both services — ingest
/// (`PushLogs`/`WriteRows`/`RegisterTable`) and reads (`FetchLogs`/`Query`) alike.
pub struct AuthInterceptor {
    policy: Arc<AuthPolicy>,
}

impl AuthInterceptor {
    pub fn new(policy: Arc<AuthPolicy>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl Interceptor for AuthInterceptor {
    async fn intercept_unary(
        &self,
        req: UnaryRequest,
        next: Next<'_>,
    ) -> Result<UnaryResponse, ConnectError> {
        self.policy.authorize(&req.ctx)?;
        next.run(req).await
    }
}

/// axum middleware enforcing the same [`AuthPolicy`] over a route group that does
/// NOT pass through the Connect interceptor chain — the `/debug/*` admin routes.
/// Those are plain axum routes (mounted before the Connect fallback), so without
/// this they would be reachable regardless of the policy; gating them here means
/// the admin surface obeys the identical stack (default-deny, loopback-only by
/// default) as every RPC. `/health` is deliberately left ungated for liveness
/// probes, and the SPA serves only static assets (its data arrives over gated
/// RPCs), so only the admin routes need this.
pub async fn auth_gate(
    State(policy): State<Arc<AuthPolicy>>,
    request: Request,
    next: AxumNext,
) -> Response {
    let bearer = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(strip_bearer);
    let peer_ip = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip());
    if policy.admits(bearer, peer_ip) {
        next.run(request).await
    } else {
        (StatusCode::UNAUTHORIZED, "finelog: unauthorized").into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::http::{Extensions, HeaderMap};
    use bytes::Bytes;
    use connectrpc::codec::CodecFormat;
    use connectrpc::interceptor::run_chain;
    use connectrpc::response::EncodedResponse;
    use connectrpc::spec::{Spec, StreamType};

    use super::*;

    fn cidr(spec: &str) -> Cidr {
        Cidr::parse(spec).unwrap()
    }

    #[test]
    fn cidr_ipv4_containment() {
        let net = cidr("10.0.0.0/8");
        assert!(net.contains("10.1.2.3".parse().unwrap()));
        assert!(net.contains("10.255.255.255".parse().unwrap()));
        assert!(!net.contains("11.0.0.1".parse().unwrap()));
        assert!(!net.contains("9.255.255.255".parse().unwrap()));
    }

    #[test]
    fn cidr_non_byte_aligned_prefix() {
        let net = cidr("192.168.16.0/20");
        assert!(net.contains("192.168.31.255".parse().unwrap()));
        assert!(!net.contains("192.168.32.0".parse().unwrap()));
    }

    #[test]
    fn cidr_host_route_and_ipv6() {
        assert!(cidr("127.0.0.1").contains("127.0.0.1".parse().unwrap()));
        assert!(!cidr("127.0.0.1").contains("127.0.0.2".parse().unwrap()));
        let v6 = cidr("2001:db8::/32");
        assert!(v6.contains("2001:db8:abcd::1".parse().unwrap()));
        assert!(!v6.contains("2001:db9::1".parse().unwrap()));
    }

    #[test]
    fn cidr_v4_mapped_v6_peer() {
        // A dual-stack listener reports a v4 client as ::ffff:10.0.0.5.
        let net = cidr("10.0.0.0/8");
        assert!(net.contains("::ffff:10.0.0.5".parse().unwrap()));
    }

    #[test]
    fn cidr_rejects_bad_spec() {
        assert!(Cidr::parse("not-an-ip/8").is_err());
        assert!(Cidr::parse("10.0.0.0/40").is_err());
    }

    /// A request context carrying an optional Authorization header and optional
    /// transport peer address.
    fn ctx(authorization: Option<&str>, peer: Option<SocketAddr>) -> RequestContext {
        let mut headers = HeaderMap::new();
        if let Some(value) = authorization {
            headers.insert("authorization", value.parse().unwrap());
        }
        let mut extensions = Extensions::new();
        if let Some(addr) = peer {
            extensions.insert(ConnectInfo(addr));
        }
        RequestContext::new(headers)
            .with_spec(Some(Spec::server(
                "/finelog.logging.LogService/PushLogs",
                StreamType::Unary,
            )))
            .with_path("/finelog.logging.LogService/PushLogs")
            .with_extensions(extensions)
    }

    // Delegation secrets (>= MIN_SECRET_BYTES) and the cluster names they map to.
    const KEY_A: &str = "delegation-key-cluster-a";
    const KEY_B: &str = "delegation-key-cluster-b";
    // Year 2286 / 1970+100s — fixed so exp checks never flake on wall-clock.
    const FUTURE: i64 = 9_999_999_999;
    const PAST: i64 = 100;

    /// Mint an HS256 JWT (`{"sub":"c","exp":<exp>}`) signed with `key`.
    fn mint(key: &str, exp: i64) -> String {
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(format!(r#"{{"sub":"c","exp":{exp}}}"#).as_bytes());
        let signing_input = format!("{header}.{payload}");
        let mut mac = HmacSha256::new_from_slice(key.as_bytes()).unwrap();
        mac.update(signing_input.as_bytes());
        let sig = URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());
        format!("{signing_input}.{sig}")
    }

    fn bearer(token: &str) -> String {
        format!("Bearer {token}")
    }

    fn verifier(pairs: &[(&str, &str)]) -> JwtVerifier {
        JwtVerifier::new(
            pairs
                .iter()
                .map(|(c, s)| (c.to_string(), s.to_string()))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn jwt_verifier_returns_cluster_rejects_forged_and_expired() {
        let v = verifier(&[("alpha", KEY_A), ("bravo", KEY_B)]);
        // Signed by a trusted key, unexpired → returns the matching cluster.
        assert_eq!(v.verify(&mint(KEY_A, FUTURE), 1_000), Some("alpha"));
        assert_eq!(v.verify(&mint(KEY_B, FUTURE), 1_000), Some("bravo"));
        // Signed by an untrusted key → None (signature fails against all keys).
        assert_eq!(
            v.verify(&mint("an-untrusted-signing-key", FUTURE), 1_000),
            None
        );
        // Trusted key but long-expired → None.
        assert_eq!(v.verify(&mint(KEY_A, PAST), 1_000), None);
        // Malformed → None, never panics.
        assert_eq!(v.verify("not-a-jwt", 1_000), None);
        assert_eq!(v.verify("a.b", 1_000), None);
    }

    #[test]
    fn jwt_verifier_accepts_pyjwt_minted_token() {
        // Cross-language contract: a token minted by PyJWT (`JwtTokenManager` on the
        // relaying controller) must verify here. Pinned so a change to either side's
        // base64url/HMAC/claims handling is caught. Minted with:
        //   jwt.encode({"sub":"marin","role":"finelog-relay","jti":"fixedjti0001",
        //               "iat":1000000000,"exp":4102444800}, KEY, algorithm="HS256")
        const KEY: &str = "delegation-key-0123456789abcdefX";
        const TOKEN: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJtYXJpbiIsInJvbGUiOiJmaW5lbG9nLXJlbGF5IiwianRpIjoiZml4ZWRqdGkwMDAxIiwiaWF0IjoxMDAwMDAwMDAwLCJleHAiOjQxMDI0NDQ4MDB9.kTVu3jf6JUbqdHe8WYswdHWzw7WBNT1NfyCxtMoiaPE";
        let v = verifier(&[("marin", KEY)]);
        // exp is 4102444800 (2100-01-01); verify at a 2020-era clock.
        assert_eq!(v.verify(TOKEN, 1_600_000_000), Some("marin"));
        // A one-byte-different key must not verify (guards against a trivial match).
        let wrong = verifier(&[("marin", "delegation-key-0123456789abcdefY")]);
        assert_eq!(wrong.verify(TOKEN, 1_600_000_000), None);
    }

    #[test]
    fn jwt_verifier_exp_leeway_tolerates_small_skew() {
        let v = verifier(&[("alpha", KEY_A)]);
        let now = 1_000_000;
        // Expired by less than the leeway → still accepted (clock skew).
        assert_eq!(
            v.verify(&mint(KEY_A, now - (EXP_LEEWAY_SECONDS - 5)), now),
            Some("alpha")
        );
        // Expired well beyond the leeway → rejected.
        assert_eq!(
            v.verify(&mint(KEY_A, now - (EXP_LEEWAY_SECONDS + 60)), now),
            None
        );
    }

    #[test]
    fn jwt_verifier_new_rejects_short_secret_and_empty_cluster() {
        assert!(JwtVerifier::new(vec![("alpha".into(), "short".into())]).is_err());
        assert!(JwtVerifier::new(vec![(String::new(), KEY_A.into())]).is_err());
        assert!(JwtVerifier::new(vec![("alpha".into(), KEY_A.into())]).is_ok());
    }

    #[test]
    fn jwt_verifier_rejects_alg_none_forgery() {
        // A token whose header claims `alg: none` (unsigned) must be rejected even
        // with an empty signature segment.
        let header = URL_SAFE_NO_PAD.encode(br#"{"alg":"none","typ":"JWT"}"#);
        let payload = URL_SAFE_NO_PAD.encode(format!(r#"{{"exp":{FUTURE}}}"#).as_bytes());
        let forged = format!("{header}.{payload}.");
        assert_eq!(verifier(&[("alpha", KEY_A)]).verify(&forged, 1_000), None);
    }

    fn jwt_policy_json(cluster: &str, secret: &str) -> String {
        format!(r#"[{{"type":"jwt","keys":[{{"cluster":"{cluster}","secret":"{secret}"}}]}}]"#)
    }

    fn stacked_policy_json(cidr: &str, cluster: &str, secret: &str) -> String {
        format!(
            r#"[{{"type":"cidr","cidrs":["{cidr}"]}},{{"type":"jwt","keys":[{{"cluster":"{cluster}","secret":"{secret}"}}]}}]"#
        )
    }

    fn token_policy() -> AuthPolicy {
        AuthPolicy::parse(&jwt_policy_json("alpha", KEY_A)).unwrap()
    }

    fn stacked_policy() -> AuthPolicy {
        AuthPolicy::parse(&stacked_policy_json("10.0.0.0/8", "alpha", KEY_A)).unwrap()
    }

    #[test]
    fn parse_rejects_empty_and_malformed() {
        // Empty list = total lockout; callers must omit the policy for the default.
        assert!(AuthPolicy::parse("[]").is_err());
        assert!(AuthPolicy::parse("not json").is_err());
        assert!(AuthPolicy::parse(r#"[{"type":"cidr","cidrs":["nope/8"]}]"#).is_err());
        assert!(AuthPolicy::parse(r#"[{"type":"cidr","cidrs":[]}]"#).is_err());
        assert!(AuthPolicy::parse(r#"[{"type":"jwt","keys":[]}]"#).is_err());
        // A too-short delegation secret fails the whole policy at parse time.
        assert!(AuthPolicy::parse(&jwt_policy_json("alpha", "short")).is_err());
    }

    #[test]
    fn allow_localhost_admits_loopback_denies_lan() {
        let p = AuthPolicy::allow_localhost();
        assert!(p
            .authorize(&ctx(None, Some("127.0.0.1:5555".parse().unwrap())))
            .is_ok());
        assert!(p
            .authorize(&ctx(None, Some("[::1]:5555".parse().unwrap())))
            .is_ok());
        assert!(p
            .authorize(&ctx(None, Some("10.1.2.3:5555".parse().unwrap())))
            .is_err());
        // No peer info → default-deny.
        assert!(p.authorize(&ctx(None, None)).is_err());
    }

    #[test]
    fn jwt_layer_admits_valid_denies_others() {
        let p = token_policy();
        assert!(p
            .authorize(&ctx(Some(&bearer(&mint(KEY_A, FUTURE))), None))
            .is_ok());
        // Present but forged → rejected (not downgraded).
        assert!(p
            .authorize(&ctx(
                Some(&bearer(&mint("an-untrusted-signing-key", FUTURE))),
                None
            ))
            .is_err());
        // Absent → rejected by the default-deny terminal.
        assert!(p.authorize(&ctx(None, None)).is_err());
    }

    #[test]
    fn cidr_first_admits_trusted_peer_over_unverifiable_token() {
        // Order (cidr before jwt) is load-bearing: a trusted-network client whose
        // token the jwt layer cannot verify is still admitted by CIDR, because the
        // CIDR layer wins before jwt would reject. This is the home controller's
        // control-plane worker_token case.
        let p = stacked_policy();
        let trusted: SocketAddr = "10.1.2.3:5555".parse().unwrap();
        let untrusted: SocketAddr = "192.0.2.1:5555".parse().unwrap();
        assert!(p.authorize(&ctx(None, Some(trusted))).is_ok());
        assert!(p
            .authorize(&ctx(
                Some(&bearer(&mint("an-untrusted-signing-key", FUTURE))),
                Some(trusted)
            ))
            .is_ok());
        // Out-of-VPC client must present a valid JWT.
        assert!(p
            .authorize(&ctx(Some(&bearer(&mint(KEY_A, FUTURE))), Some(untrusted)))
            .is_ok());
        assert!(p.authorize(&ctx(None, Some(untrusted))).is_err());
        assert!(p
            .authorize(&ctx(
                Some(&bearer(&mint("an-untrusted-signing-key", FUTURE))),
                Some(untrusted)
            ))
            .is_err());
    }

    fn ok_response() -> UnaryResponse {
        UnaryResponse::from_encoded(EncodedResponse::new(Bytes::new()), CodecFormat::Proto)
    }

    async fn run_through(
        policy: AuthPolicy,
        request: RequestContext,
    ) -> (bool, Result<UnaryResponse, ConnectError>) {
        let interceptor: Arc<dyn Interceptor> = Arc::new(AuthInterceptor::new(Arc::new(policy)));
        let chain: Vec<Arc<dyn Interceptor>> = vec![interceptor];
        let ran = Arc::new(AtomicUsize::new(0));
        let ran2 = Arc::clone(&ran);
        let req = UnaryRequest::new(request, Bytes::new(), CodecFormat::Proto);
        let result = run_chain(&chain, req, move |_req| {
            let ran2 = Arc::clone(&ran2);
            async move {
                ran2.fetch_add(1, Ordering::SeqCst);
                Ok(ok_response())
            }
        })
        .await;
        (ran.load(Ordering::SeqCst) == 1, result)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interceptor_admits_valid_token() {
        let (ran, result) = run_through(
            token_policy(),
            ctx(Some(&bearer(&mint(KEY_A, FUTURE))), None),
        )
        .await;
        assert!(result.is_ok());
        assert!(ran, "handler ran for an authenticated request");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interceptor_rejects_unauthenticated() {
        let (ran, result) = run_through(token_policy(), ctx(None, None)).await;
        assert_eq!(
            result.unwrap_err().code,
            connectrpc::ErrorCode::Unauthenticated
        );
        assert!(!ran, "handler never ran for an unauthenticated request");
    }
}
