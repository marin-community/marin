// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Authenticated ingress front for the finelog server.
//!
//! A hub finelog receives pushes from many per-cluster finelogs across the internet,
//! so it cannot rely on being private behind one controller's proxy. This module gates
//! every RPC with an ordered stack of auth layers, each of which allows, falls through,
//! or rejects a request (the same shape as rigging's `server_auth`).
//!
//! The policy is default-deny: a request no layer allows is rejected
//! `Unauthenticated`. Auth is a stack of allow-layers on top of that deny — there is
//! no "empty means open" path. The interceptor is always installed (see
//! `ServerConfig`); the only variable is the layer list. With nothing configured the
//! default is [`AuthPolicy::allow_localhost`] (loopback only), so a bare finelog is
//! reachable for local debugging but never open to its network. Two layer kinds
//! compose:
//! - [`AuthLayer::Jwt`] — a bearer whose EdDSA (Ed25519) signature verifies against
//!   one of a set of trusted per-cluster public keys, whose audience is exactly
//!   `finelog`, and which has not expired, admits the request. Each sending finelog
//!   mints a short-lived `aud="finelog"` JWT with its per-cluster private key and the
//!   hub verifies it against that cluster's public key — the same JWT mechanism the
//!   control plane uses, so the log plane adds no second credential system. The hub
//!   holds only public keys (which grant no minting power), verified via
//!   `jsonwebtoken`, and checks signature + `aud="finelog"` + `exp` only (it cannot
//!   reach a revocation table, so exposure is TTL-bounded). Requiring `aud="finelog"`
//!   is the load-bearing cross-plane guard (RFC 8725): a control-plane `aud="iris"`
//!   token, though signed by the same key, is rejected here. Each cluster may carry
//!   multiple public keys so a key rotation overlaps (old + new both verify). The
//!   matched key names the caller: see [`AuthIdentity`], which the ingest handlers use
//!   to stamp a pushed row's origin cluster.
//! - [`AuthLayer::Cidr`] — a request whose transport peer is in a trusted network is
//!   admitted without a token, so a finelog that also serves its own cluster lists that
//!   cluster's loopback/VPC ranges (e.g. `127.0.0.0/8`, `10.0.0.0/8`) and local clients
//!   reach it without a JWT while remote pushes must sign. CIDR matches the transport
//!   peer only, never a spoofable `X-Forwarded-For` value.
//!
//! A valid bearer whose configured key grants `trusted_collector` is
//! authoritative before network fallback. All other requests walk the layers in
//! order: the first `Allow` admits, the first `Reject` denies, and a request no
//! layer claims is denied. CIDR therefore stays first for a trusted-network
//! client whose token the log plane cannot verify (e.g. the home controller's
//! control-plane `worker_token`).

use std::collections::HashMap;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{ConnectInfo, Request, State};
use axum::http::StatusCode;
use axum::middleware::Next as AxumNext;
use axum::response::{IntoResponse, Response};
use connectrpc::{
    async_trait, ConnectError, Interceptor, Next, RequestContext, UnaryRequest, UnaryResponse,
};
use jsonwebtoken::{decode, Algorithm, DecodingKey, Validation};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The one audience this delegation plane accepts, and the one the forwarder mints
/// under. A token minted for any other plane (e.g. control-plane `aud="iris"`), even
/// under the same signing key, is rejected — the load-bearing cross-plane guard
/// (RFC 8725).
pub(crate) const FINELOG_AUDIENCE: &str = "finelog";
pub const TELEMETRY_AGENT_AUDIENCE: &str = "telemetry-agent";
pub const MAX_TELEMETRY_PRODUCER_TOKEN_LIFETIME_SECONDS: i64 = 3600;

/// Accept a token whose `exp` is at most this far in the past, to tolerate small
/// clock skew between a minting server and the hub.
const EXP_LEEWAY_SECONDS: u64 = 60;

/// Identity that an Iris producer credential authorizes the local telemetry
/// agent to stamp. These values come only from a verified token and the agent's
/// own registration, never from the producer telemetry body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetryProducerIdentity {
    pub cluster: String,
    pub iris_job_id: String,
    pub iris_task_id: String,
    pub attempt_id: i32,
    pub attempt_uid: String,
    pub worker_id: String,
    pub node_id: String,
}

/// Infrastructure identity the host agent was registered to serve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetryAgentRegistration {
    pub cluster: String,
    pub worker_id: String,
    pub node_id: String,
}

/// Verifies short-lived Iris producer credentials at the local agent boundary.
#[derive(Debug)]
pub struct TelemetryAgentVerifier {
    decoding_keys: Vec<DecodingKey>,
    validation: Validation,
    registration: TelemetryAgentRegistration,
}

impl TelemetryAgentVerifier {
    pub fn new(
        issuer: &str,
        public_keys: &[String],
        registration: TelemetryAgentRegistration,
    ) -> Result<Self, String> {
        if issuer.is_empty() {
            return Err("telemetry agent issuer must not be empty".to_string());
        }
        if public_keys.is_empty() {
            return Err("telemetry agent requires at least one Iris public key".to_string());
        }
        validate_nonempty_identity([
            ("registration.cluster", registration.cluster.as_str()),
            ("registration.worker_id", registration.worker_id.as_str()),
            ("registration.node_id", registration.node_id.as_str()),
        ])?;
        let decoding_keys = public_keys
            .iter()
            .map(|pem| {
                DecodingKey::from_ed_pem(pem.as_bytes())
                    .map_err(|error| format!("invalid telemetry agent Ed25519 public key: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut validation = Validation::new(Algorithm::EdDSA);
        validation.set_audience(&[TELEMETRY_AGENT_AUDIENCE]);
        validation.set_issuer(&[issuer]);
        validation.set_required_spec_claims(&["iss", "aud", "sub", "iat", "exp"]);
        validation.leeway = EXP_LEEWAY_SECONDS;
        Ok(Self {
            decoding_keys,
            validation,
            registration,
        })
    }

    pub fn verify(&self, token: &str) -> Result<TelemetryProducerIdentity, String> {
        let claims = self
            .decoding_keys
            .iter()
            .find_map(|key| {
                decode::<TelemetryProducerClaims>(token, key, &self.validation)
                    .ok()
                    .map(|decoded| decoded.claims)
            })
            .ok_or_else(|| "producer credential did not verify".to_string())?;
        validate_nonempty_identity([
            ("sub", claims.sub.as_str()),
            ("attempt_uid", claims.attempt_uid.as_str()),
            ("cluster", claims.cluster.as_str()),
            ("iris_job_id", claims.iris_job_id.as_str()),
            ("iris_task_id", claims.iris_task_id.as_str()),
            ("worker_id", claims.worker_id.as_str()),
            ("node_id", claims.node_id.as_str()),
        ])?;
        let lifetime = claims
            .exp
            .checked_sub(claims.iat)
            .ok_or_else(|| "producer credential lifetime overflows".to_string())?;
        if lifetime <= 0 {
            return Err("producer credential exp must be after iat".to_string());
        }
        if lifetime > MAX_TELEMETRY_PRODUCER_TOKEN_LIFETIME_SECONDS {
            return Err(format!(
                "producer credential lifetime exceeds {} seconds",
                MAX_TELEMETRY_PRODUCER_TOKEN_LIFETIME_SECONDS
            ));
        }
        if claims.attempt_id < 0 {
            return Err("producer credential attempt_id must be nonnegative".to_string());
        }
        if claims.sub != claims.attempt_uid {
            return Err("producer credential sub must equal attempt_uid".to_string());
        }
        let latest_iat = unix_now_seconds().saturating_add(EXP_LEEWAY_SECONDS as i64);
        if claims.iat > latest_iat {
            return Err("producer credential iat is in the future".to_string());
        }
        for (field, claimed, registered) in [
            (
                "cluster",
                claims.cluster.as_str(),
                self.registration.cluster.as_str(),
            ),
            (
                "worker_id",
                claims.worker_id.as_str(),
                self.registration.worker_id.as_str(),
            ),
            (
                "node_id",
                claims.node_id.as_str(),
                self.registration.node_id.as_str(),
            ),
        ] {
            if claimed != registered {
                return Err(format!(
                    "producer credential {field} {claimed:?} does not match agent registration {registered:?}"
                ));
            }
        }
        Ok(TelemetryProducerIdentity {
            cluster: claims.cluster,
            iris_job_id: claims.iris_job_id,
            iris_task_id: claims.iris_task_id,
            attempt_id: claims.attempt_id,
            attempt_uid: claims.attempt_uid,
            worker_id: claims.worker_id,
            node_id: claims.node_id,
        })
    }
}

fn validate_nonempty_identity<const N: usize>(fields: [(&str, &str); N]) -> Result<(), String> {
    if let Some((name, _)) = fields.into_iter().find(|(_, value)| value.is_empty()) {
        return Err(format!("{name} must not be empty"));
    }
    Ok(())
}

fn unix_now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after Unix epoch")
        .as_secs() as i64
}

#[derive(Debug, Deserialize)]
struct TelemetryProducerClaims {
    #[allow(dead_code)]
    iss: String,
    #[allow(dead_code)]
    aud: String,
    sub: String,
    iat: i64,
    exp: i64,
    cluster: String,
    iris_job_id: String,
    iris_task_id: String,
    attempt_id: i32,
    attempt_uid: String,
    worker_id: String,
    node_id: String,
}

/// Who a layer decided the request is. Placed in the request extensions by
/// [`AuthInterceptor`] and read by the ingest handlers to bind a pushed row's
/// origin `cluster` to the credential that carried it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthIdentity {
    /// Admitted by a token: this cluster's public key verified the bearer, so the
    /// writer may only claim this cluster as a row's origin.
    Jwt { cluster: String },
    /// Admitted by a key whose server-side configuration grants the
    /// `trusted_collector` role. Only this identity may preserve infrastructure
    /// enrichment that a telemetry agent stamped from producer credentials.
    TrustedCollector { cluster: String },
    /// Admitted by transport peer address. A trusted network carries no per-writer
    /// identity, so such a writer names its own origin cluster (typically empty —
    /// it is the store's own cluster writing locally).
    Network,
}

/// One rule's verdict over a request, mirroring rigging's
/// AUTHENTICATED / ABSENT / REJECTED.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    /// This rule admits the request as `identity`; stop and admit.
    Allow(AuthIdentity),
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

/// A trusted per-cluster delegation issuer: the cluster it authenticates and the
/// set of Ed25519 public keys that verify its tokens. A cluster carries more
/// than one key only across a rotation overlap (old + new both accepted). `Debug`
/// renders the cluster and key count but never key material (public keys are not
/// secret, but there is no reason to spill PEM into logs).
struct JwtKey {
    cluster: String,
    role: JwtKeyRole,
    decoding_keys: Vec<DecodingKey>,
}

impl fmt::Debug for JwtKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JwtKey")
            .field("cluster", &self.cluster)
            .field("role", &self.role)
            .field("public_keys", &self.decoding_keys.len())
            .finish_non_exhaustive()
    }
}

/// Verifies EdDSA (Ed25519) JWTs against a set of trusted per-cluster public keys.
///
/// A token is accepted iff its signature verifies against one configured public
/// key, its `aud` is exactly `finelog` (the cross-plane guard), and its `exp` is
/// in the future within [`EXP_LEEWAY_SECONDS`]. Only public keys are held — never
/// a controller's private signing key — so the store can verify a cluster's tokens
/// without gaining the power to mint any. On success the matching cluster is
/// returned for attribution. Federation members may write any namespace
/// (per-cluster write isolation is not enforced in v1), while the configured
/// role decides whether ordinary producer infrastructure fields are cleared or
/// signed collector enrichment is preserved.
#[derive(Debug)]
pub struct JwtVerifier {
    keys: Vec<JwtKey>,
    validation: Validation,
}

impl JwtVerifier {
    /// Build from `(cluster, role, public_keys)` tuples, each PEM an Ed25519
    /// SubjectPublicKeyInfo (`-----BEGIN PUBLIC KEY-----`). Rejects an empty
    /// cluster, a cluster with no public keys, or a PEM `jsonwebtoken` cannot parse
    /// as an Ed25519 public key, so a bad config fails the server at startup rather
    /// than silently admitting nothing.
    fn new(keys: Vec<(String, JwtKeyRole, Vec<String>)>) -> Result<Self, String> {
        let mut compiled = Vec::with_capacity(keys.len());
        for (cluster, role, pems) in keys {
            if cluster.is_empty() {
                return Err("jwt key entry has an empty cluster".to_string());
            }
            if pems.is_empty() {
                return Err(format!(
                    "jwt key entry for cluster {cluster:?} has no public keys"
                ));
            }
            let mut decoding_keys = Vec::with_capacity(pems.len());
            for pem in &pems {
                let key = DecodingKey::from_ed_pem(pem.as_bytes()).map_err(|e| {
                    format!(
                        "jwt public key for cluster {cluster:?} is not a valid Ed25519 PEM: {e}"
                    )
                })?;
                decoding_keys.push(key);
            }
            compiled.push(JwtKey {
                cluster,
                role,
                decoding_keys,
            });
        }
        Ok(Self {
            keys: compiled,
            validation: finelog_validation(),
        })
    }

    /// The cluster whose public key verifies `token` — a valid EdDSA signature,
    /// `aud="finelog"`, and unexpired within [`EXP_LEEWAY_SECONDS`] — or `None` if
    /// none does. `jsonwebtoken` validates signature, algorithm (EdDSA only, so an
    /// HS256/`alg:none` token is rejected as algorithm confusion), audience, and
    /// expiry in one pass; we try each trusted key and return the first that admits.
    fn verify(&self, token: &str) -> Option<(&str, JwtKeyRole)> {
        self.keys
            .iter()
            .find(|k| {
                k.decoding_keys
                    .iter()
                    .any(|dk| decode::<JwtClaims>(token, dk, &self.validation).is_ok())
            })
            .map(|k| (k.cluster.as_str(), k.role))
    }
}

fn validate_key_bindings(layers: &[AuthLayer]) -> Result<(), String> {
    let mut bindings: HashMap<[u8; 32], (String, JwtKeyRole)> = HashMap::new();
    for layer in layers {
        let AuthLayer::Jwt(verifier) = layer else {
            continue;
        };
        for key in &verifier.keys {
            for decoding_key in &key.decoding_keys {
                let fingerprint: [u8; 32] = Sha256::digest(decoding_key.as_bytes()).into();
                let binding = (key.cluster.clone(), key.role);
                if let Some(existing) = bindings.get(&fingerprint) {
                    if existing != &binding {
                        return Err(format!(
                            "one jwt public key has conflicting bindings {:?}/{:?} and {:?}/{:?}",
                            existing.0, existing.1, binding.0, binding.1
                        ));
                    }
                } else {
                    bindings.insert(fingerprint, binding);
                }
            }
        }
    }
    Ok(())
}

/// The `jsonwebtoken` validation policy: EdDSA only, `aud` must be exactly
/// `finelog`, `exp` and `aud` are required claims (a token missing either is
/// rejected), and a [`EXP_LEEWAY_SECONDS`] skew allowance on `exp`.
fn finelog_validation() -> Validation {
    let mut validation = Validation::new(Algorithm::EdDSA);
    validation.set_audience(&[FINELOG_AUDIENCE]);
    validation.set_required_spec_claims(&["exp", "aud"]);
    validation.leeway = EXP_LEEWAY_SECONDS;
    validation
}

/// The declarative shape of one auth layer, deserialized from the
/// `FINELOG_AUTH_POLICY` JSON (== the finelog config's `auth:` list). List order
/// is evaluation order.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AuthLayerConfig {
    /// Admit a request whose transport peer is in one of these networks.
    Cidr { cidrs: Vec<String> },
    /// Admit a request bearing an EdDSA JWT (`aud="finelog"`) that verifies
    /// against one of these per-cluster public keys.
    Jwt { keys: Vec<JwtKeyConfig> },
}

#[derive(Debug, Deserialize)]
struct JwtKeyConfig {
    cluster: String,
    role: JwtKeyRole,
    /// The cluster's Ed25519 public keys in PEM (SubjectPublicKeyInfo). A list so a
    /// key rotation can list old + new during the overlap window; both verify.
    public_keys: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JwtKeyRole {
    Cluster,
    TrustedCollector,
}

/// One compiled entry in the ordered auth stack.
#[derive(Debug)]
enum AuthLayer {
    Cidr(Vec<Cidr>),
    // Boxed: JwtVerifier dwarfs the Cidr variant, and an unboxed enum would carry that
    // footprint for every layer in the stack.
    Jwt(Box<JwtVerifier>),
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
                let pairs = keys
                    .into_iter()
                    .map(|k| (k.cluster, k.role, k.public_keys))
                    .collect();
                Ok(AuthLayer::Jwt(Box::new(JwtVerifier::new(pairs)?)))
            }
        }
    }

    /// Evaluate this layer against extracted request facts: the bearer token
    /// (already stripped of `Bearer `) and the transport peer IP.
    fn check(&self, bearer: Option<&str>, peer_ip: Option<IpAddr>) -> Verdict {
        match self {
            AuthLayer::Cidr(nets) => match peer_ip {
                Some(ip) if nets.iter().any(|n| n.contains(ip)) => {
                    Verdict::Allow(AuthIdentity::Network)
                }
                _ => Verdict::Fallthrough,
            },
            // rigging JwtAuthenticator parity: no bearer → fall through (a later
            // layer may claim it); a valid bearer → allow; a present-but-invalid
            // bearer → reject, never downgraded to a weaker layer.
            AuthLayer::Jwt(verifier) => match bearer {
                None => Verdict::Fallthrough,
                Some(tok) => match verifier.verify(tok) {
                    Some((cluster, role)) => {
                        tracing::trace!(cluster, "finelog: admitted jwt-authenticated request");
                        Verdict::Allow(match role {
                            JwtKeyRole::Cluster => AuthIdentity::Jwt {
                                cluster: cluster.to_string(),
                            },
                            JwtKeyRole::TrustedCollector => AuthIdentity::TrustedCollector {
                                cluster: cluster.to_string(),
                            },
                        })
                    }
                    None => Verdict::Reject,
                },
            },
        }
    }
}

/// The delegation-token claims finelog reads. `jsonwebtoken` validates the
/// signature, algorithm, `aud`, and `exp`; these fields are the subset finelog
/// deserializes for attribution and the audience check. The signer always emits
/// `iss`/`aud`/`sub`/`exp`, so all four are required (a token missing any is
/// rejected at deserialization).
#[derive(Deserialize)]
struct JwtClaims {
    #[allow(dead_code)]
    iss: String,
    #[allow(dead_code)]
    aud: String,
    #[allow(dead_code)]
    sub: String,
    #[allow(dead_code)]
    exp: i64,
}

/// An ordered stack of auth layers with a default-deny terminal. Always
/// installed (see `ServerConfig`); the private default is [`allow_localhost`].
/// A valid `trusted_collector` bearer is authoritative before the ordered walk
/// so a network layer cannot erase its signed infrastructure identity.
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
    /// `[{"type":"cidr","cidrs":[..]},{"type":"jwt","keys":[{"cluster":..,"public_keys":[..]}]}]`.
    /// An empty list is rejected (that is a total lockout — omit the policy
    /// entirely for the allow-localhost default). A malformed entry, CIDR, or
    /// unparseable public-key PEM errors so the server fails at startup rather than
    /// silently mis-admitting.
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
        validate_key_bindings(&layers)?;
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
    /// (no `Bearer ` prefix) and the transport peer IP. `Some(identity)` = admit.
    /// A verified `trusted_collector` bearer wins before the ordered walk. For
    /// every other request, first `Allow` admits, first `Reject` denies, and an
    /// unclaimed request is denied. Shared by the Connect interceptor and axum
    /// [`auth_gate`] middleware so both surfaces enforce the identical policy.
    pub(crate) fn admits(
        &self,
        bearer: Option<&str>,
        peer_ip: Option<IpAddr>,
    ) -> Option<AuthIdentity> {
        if let Some(token) = bearer {
            for layer in &self.layers {
                let AuthLayer::Jwt(verifier) = layer else {
                    continue;
                };
                if let Some((cluster, JwtKeyRole::TrustedCollector)) = verifier.verify(token) {
                    return Some(AuthIdentity::TrustedCollector {
                        cluster: cluster.to_string(),
                    });
                }
            }
        }
        for layer in &self.layers {
            match layer.check(bearer, peer_ip) {
                Verdict::Allow(identity) => return Some(identity),
                Verdict::Reject => return None,
                Verdict::Fallthrough => {}
            }
        }
        None
    }

    /// Connect-interceptor entry point: extract the bearer + peer IP from the RPC
    /// context and apply the policy, returning who the request authenticated as.
    fn authorize(&self, ctx: &RequestContext) -> Result<AuthIdentity, ConnectError> {
        self.admits(bearer_token(ctx), peer_ip(ctx))
            .ok_or_else(unauthenticated)
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
///
/// It also records the admitting [`AuthIdentity`] in the request extensions, which
/// connect carries through to the handler. `PushLogs` reads it to bind a pushed row's
/// origin `cluster` to the credential that carried it.
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
        mut req: UnaryRequest,
        next: Next<'_>,
    ) -> Result<UnaryResponse, ConnectError> {
        let identity = self.policy.authorize(&req.ctx)?;
        req.ctx.extensions_mut().insert(identity);
        next.run(req).await
    }
}

/// The identity the interceptor admitted this request as.
///
/// Absent only if the request reached a handler without traversing
/// [`AuthInterceptor`], which cannot happen for a registered RPC — every method on
/// both services sits behind it. A handler that needs the identity to make an
/// authorization decision must therefore treat `None` as a failure, not as a
/// permissive default.
pub fn request_identity(ctx: &RequestContext) -> Option<&AuthIdentity> {
    ctx.extensions().get::<AuthIdentity>()
}

/// axum middleware enforcing the same [`AuthPolicy`] over a route group that does
/// NOT pass through the Connect interceptor chain — `/debug/*` and REST telemetry.
/// Those are plain axum routes (mounted before the Connect fallback), so without
/// this they would be reachable regardless of the policy. The admitted identity
/// is inserted into request extensions so ingestion can stamp authoritative
/// origin fields. `/health` is deliberately left ungated for liveness probes,
/// and the SPA serves only static assets (its data arrives over gated RPCs).
pub async fn auth_gate(
    State(policy): State<Arc<AuthPolicy>>,
    mut request: Request,
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
    if let Some(identity) = policy.admits(bearer, peer_ip) {
        request.extensions_mut().insert(identity);
        next.run(request).await
    } else {
        crate::server::telemetry_ingest::auth_error_response(&request)
            .unwrap_or_else(|| (StatusCode::UNAUTHORIZED, "finelog: unauthorized").into_response())
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
    use tower::ServiceExt;

    use super::*;

    fn cidr(spec: &str) -> Cidr {
        Cidr::parse(spec).unwrap()
    }

    #[tokio::test]
    async fn axum_auth_gate_passes_the_admitted_identity_to_plain_handlers() {
        let app =
            axum::Router::new()
                .route(
                    "/",
                    axum::routing::get(
                        |axum::extract::Extension(identity): axum::extract::Extension<
                            AuthIdentity,
                        >| async move {
                            match identity {
                                AuthIdentity::Network => "network",
                                AuthIdentity::Jwt { .. } => "jwt",
                                AuthIdentity::TrustedCollector { .. } => "trusted_collector",
                            }
                        },
                    ),
                )
                .layer(axum::middleware::from_fn_with_state(
                    Arc::new(AuthPolicy::allow_localhost()),
                    auth_gate,
                ));
        let mut request = axum::http::Request::builder()
            .uri("/")
            .body(axum::body::Body::empty())
            .unwrap();
        request.extensions_mut().insert(ConnectInfo(
            "127.0.0.1:12345".parse::<SocketAddr>().unwrap(),
        ));
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64)
            .await
            .unwrap();
        assert_eq!(&body[..], b"network");
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

    // The shared Ed25519 test keypairs. A `mint(PRIV, ..)`-signed token is a
    // Rust-self-signed vector; the authoritative Python(PyJWT-EdDSA)↔Rust
    // cross-language vector lives in the shared conformance suite
    // (rigging.auth_vectors) — a self-signed vector suffices for these units.
    use crate::server::test_support::{PRIV_A, PRIV_B, PRIV_UNTRUSTED, PUB_A, PUB_B};

    // Year 2286 / 1970+100s — fixed so exp checks never flake on wall-clock
    // (jsonwebtoken validates `exp` against the real clock).
    const FUTURE: i64 = 9_999_999_999;
    const PAST: i64 = 100;

    fn unix_now() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }

    /// Mint an EdDSA JWT (`{"iss","aud","sub","exp"}`, header `kid`) signed with the
    /// Ed25519 private key `private_pem`, for audience `aud`.
    fn mint(private_pem: &str, aud: &str, exp: i64) -> String {
        #[derive(serde::Serialize)]
        struct Claims<'a> {
            iss: &'a str,
            aud: &'a str,
            sub: &'a str,
            exp: i64,
        }
        let claims = Claims {
            iss: "alpha",
            aud,
            sub: "relay",
            exp,
        };
        let mut header = jsonwebtoken::Header::new(Algorithm::EdDSA);
        header.kid = Some("test-kid".to_string());
        let key = jsonwebtoken::EncodingKey::from_ed_pem(private_pem.as_bytes()).unwrap();
        jsonwebtoken::encode(&header, &claims, &key).unwrap()
    }

    /// Mint an EdDSA JWT with the delegation-plane audience (`aud="finelog"`).
    fn mint_finelog(private_pem: &str, exp: i64) -> String {
        mint(private_pem, FINELOG_AUDIENCE, exp)
    }

    fn mint_producer(
        private_pem: &str,
        audience: &str,
        expires_at: i64,
        attempt_uid: &str,
        attempt_id: i32,
        worker_id: &str,
        node_id: &str,
    ) -> String {
        let claims = serde_json::json!({
            "iss": "iris",
            "aud": audience,
            "sub": "attempt-uid",
            "iat": unix_now(),
            "exp": expires_at,
            "cluster": "alpha",
            "iris_job_id": "job-1",
            "iris_task_id": "task-2",
            "attempt_id": attempt_id,
            "attempt_uid": attempt_uid,
            "worker_id": worker_id,
            "node_id": node_id,
        });
        mint_producer_claims(private_pem, &claims)
    }

    fn mint_producer_claims(private_pem: &str, claims: &serde_json::Value) -> String {
        let key = jsonwebtoken::EncodingKey::from_ed_pem(private_pem.as_bytes()).unwrap();
        jsonwebtoken::encode(&jsonwebtoken::Header::new(Algorithm::EdDSA), &claims, &key).unwrap()
    }

    fn agent_verifier() -> TelemetryAgentVerifier {
        TelemetryAgentVerifier::new(
            "iris",
            &[PUB_A.to_string()],
            TelemetryAgentRegistration {
                cluster: "alpha".to_string(),
                worker_id: "worker-4".to_string(),
                node_id: "node-5".to_string(),
            },
        )
        .unwrap()
    }

    #[test]
    fn telemetry_agent_verifies_signed_claims_and_binds_registration() {
        let verifier = agent_verifier();
        assert_eq!(
            verifier
                .verify(&mint_producer(
                    PRIV_A,
                    TELEMETRY_AGENT_AUDIENCE,
                    unix_now() + 300,
                    "attempt-uid",
                    3,
                    "worker-4",
                    "node-5",
                ))
                .unwrap(),
            TelemetryProducerIdentity {
                cluster: "alpha".to_string(),
                iris_job_id: "job-1".to_string(),
                iris_task_id: "task-2".to_string(),
                attempt_id: 3,
                attempt_uid: "attempt-uid".to_string(),
                worker_id: "worker-4".to_string(),
                node_id: "node-5".to_string(),
            }
        );

        assert!(verifier
            .verify(&mint_producer(
                PRIV_A,
                FINELOG_AUDIENCE,
                unix_now() + 300,
                "attempt-uid",
                3,
                "worker-4",
                "node-5",
            ))
            .is_err());
        assert!(verifier
            .verify(&mint_producer(
                PRIV_A,
                TELEMETRY_AGENT_AUDIENCE,
                unix_now() + 300,
                "attempt-uid",
                3,
                "other-worker",
                "node-5",
            ))
            .is_err());
        assert!(verifier
            .verify(&mint_producer(
                PRIV_A,
                TELEMETRY_AGENT_AUDIENCE,
                unix_now() + 300,
                "attempt-uid",
                3,
                "worker-4",
                "other-node",
            ))
            .is_err());
        assert!(verifier
            .verify(&mint_producer(
                PRIV_A,
                TELEMETRY_AGENT_AUDIENCE,
                PAST,
                "attempt-uid",
                3,
                "worker-4",
                "node-5",
            ))
            .is_err());
        assert!(verifier
            .verify(&mint_producer(
                PRIV_UNTRUSTED,
                TELEMETRY_AGENT_AUDIENCE,
                unix_now() + 300,
                "attempt-uid",
                3,
                "worker-4",
                "node-5",
            ))
            .is_err());
        assert!(verifier
            .verify(&mint_producer(
                PRIV_A,
                TELEMETRY_AGENT_AUDIENCE,
                unix_now() + 300,
                "different-attempt",
                3,
                "worker-4",
                "node-5",
            ))
            .is_err());
        assert!(verifier
            .verify(&mint_producer(
                PRIV_A,
                TELEMETRY_AGENT_AUDIENCE,
                unix_now() + MAX_TELEMETRY_PRODUCER_TOKEN_LIFETIME_SECONDS + 1,
                "attempt-uid",
                3,
                "worker-4",
                "node-5",
            ))
            .is_err());
        assert!(verifier
            .verify(&mint_producer(
                PRIV_A,
                TELEMETRY_AGENT_AUDIENCE,
                unix_now() + 300,
                "attempt-uid",
                -1,
                "worker-4",
                "node-5",
            ))
            .is_err());
    }

    #[test]
    fn telemetry_agent_rejects_overflow_and_incomplete_signed_claims() {
        let verifier = agent_verifier();
        let base = serde_json::json!({
            "iss": "iris",
            "aud": TELEMETRY_AGENT_AUDIENCE,
            "sub": "attempt-uid",
            "iat": unix_now(),
            "exp": unix_now() + 300,
            "cluster": "alpha",
            "iris_job_id": "job-1",
            "iris_task_id": "task-2",
            "attempt_id": 3,
            "attempt_uid": "attempt-uid",
            "worker_id": "worker-4",
            "node_id": "node-5",
        });

        let mut overflow = base.clone();
        overflow["iat"] = serde_json::json!(i64::MIN);
        overflow["exp"] = serde_json::json!(i64::MAX);
        assert!(verifier
            .verify(&mint_producer_claims(PRIV_A, &overflow))
            .is_err());

        for field in [
            "iss",
            "aud",
            "sub",
            "iat",
            "exp",
            "cluster",
            "iris_job_id",
            "iris_task_id",
            "attempt_id",
            "attempt_uid",
            "worker_id",
            "node_id",
        ] {
            let mut incomplete = base.clone();
            incomplete.as_object_mut().unwrap().remove(field);
            assert!(
                verifier
                    .verify(&mint_producer_claims(PRIV_A, &incomplete))
                    .is_err(),
                "missing {field} was accepted"
            );
        }
    }

    #[test]
    fn telemetry_agent_rejects_wrong_issuer_registration_and_future_iat() {
        let verifier = agent_verifier();
        let base = serde_json::json!({
            "iss": "iris",
            "aud": TELEMETRY_AGENT_AUDIENCE,
            "sub": "attempt-uid",
            "iat": unix_now(),
            "exp": unix_now() + 300,
            "cluster": "alpha",
            "iris_job_id": "job-1",
            "iris_task_id": "task-2",
            "attempt_id": 3,
            "attempt_uid": "attempt-uid",
            "worker_id": "worker-4",
            "node_id": "node-5",
        });
        for (field, value) in [
            ("iss", serde_json::json!("other-issuer")),
            ("cluster", serde_json::json!("other-cluster")),
            ("iat", serde_json::json!(unix_now() + 120)),
        ] {
            let mut claims = base.clone();
            claims[field] = value;
            if field == "iat" {
                claims["exp"] = serde_json::json!(unix_now() + 300);
            }
            assert!(
                verifier
                    .verify(&mint_producer_claims(PRIV_A, &claims))
                    .is_err(),
                "invalid {field} was accepted"
            );
        }
    }

    /// Mint an HS256 JWT that is otherwise well-formed (right `aud`, unexpired) —
    /// used to prove the EdDSA verifier rejects an algorithm-confusion token.
    fn mint_hs256(exp: i64) -> String {
        #[derive(serde::Serialize)]
        struct Claims<'a> {
            iss: &'a str,
            aud: &'a str,
            sub: &'a str,
            exp: i64,
        }
        let claims = Claims {
            iss: "alpha",
            aud: FINELOG_AUDIENCE,
            sub: "relay",
            exp,
        };
        let header = jsonwebtoken::Header::new(Algorithm::HS256);
        let key = jsonwebtoken::EncodingKey::from_secret(b"an-hmac-secret-an-attacker-picks");
        jsonwebtoken::encode(&header, &claims, &key).unwrap()
    }

    fn bearer(token: &str) -> String {
        format!("Bearer {token}")
    }

    /// A verifier trusting one public key per cluster.
    fn verifier(pairs: &[(&str, &str)]) -> JwtVerifier {
        JwtVerifier::new(
            pairs
                .iter()
                .map(|(c, pem)| (c.to_string(), JwtKeyRole::Cluster, vec![pem.to_string()]))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn jwt_verifier_returns_cluster_rejects_forged_and_expired() {
        let v = verifier(&[("alpha", PUB_A), ("bravo", PUB_B)]);
        // Signed by a trusted key, aud=finelog, unexpired → the matching cluster.
        assert_eq!(
            v.verify(&mint_finelog(PRIV_A, FUTURE)),
            Some(("alpha", JwtKeyRole::Cluster))
        );
        assert_eq!(
            v.verify(&mint_finelog(PRIV_B, FUTURE)),
            Some(("bravo", JwtKeyRole::Cluster))
        );
        // Signed by an untrusted key → None (signature fails against all keys).
        assert_eq!(v.verify(&mint_finelog(PRIV_UNTRUSTED, FUTURE)), None);
        // Trusted key but long-expired → None.
        assert_eq!(v.verify(&mint_finelog(PRIV_A, PAST)), None);
        // Malformed → None, never panics.
        assert_eq!(v.verify("not-a-jwt"), None);
        assert_eq!(v.verify("a.b"), None);
    }

    #[test]
    fn jwt_verifier_rejects_wrong_audience() {
        // The cross-plane guard (RFC 8725): a token signed by the SAME trusted key,
        // unexpired and well-formed, but minted for the control plane (aud="iris")
        // must NOT verify at finelog. Without this a control-plane token would replay
        // against the log plane.
        let v = verifier(&[("alpha", PUB_A)]);
        assert_eq!(v.verify(&mint(PRIV_A, "iris", FUTURE)), None);
        assert_eq!(v.verify(&mint(PRIV_A, "iris-peer", FUTURE)), None);
        // The delegation audience still verifies with the same key.
        assert_eq!(
            v.verify(&mint_finelog(PRIV_A, FUTURE)),
            Some(("alpha", JwtKeyRole::Cluster))
        );
    }

    #[test]
    fn jwt_verifier_rejects_hs256_alg_confusion() {
        // An HS256 token (right aud, unexpired) must be rejected: the validation is
        // EdDSA-only, so a symmetric token is algorithm confusion, never admitted.
        let v = verifier(&[("alpha", PUB_A)]);
        assert_eq!(v.verify(&mint_hs256(FUTURE)), None);
    }

    #[test]
    fn jwt_verifier_rejects_tampered_token() {
        // A valid token with its final signature character flipped must fail the
        // signature check.
        let v = verifier(&[("alpha", PUB_A)]);
        let token = mint_finelog(PRIV_A, FUTURE);
        let mut tampered = token.clone();
        let last = tampered.pop().unwrap();
        tampered.push(if last == 'A' { 'B' } else { 'A' });
        assert_ne!(tampered, token);
        assert_eq!(v.verify(&tampered), None);
    }

    #[test]
    fn jwt_verifier_rotation_accepts_either_key() {
        // A cluster mid-rotation lists both its old and new public keys; a token
        // signed by either private half verifies and attributes to that cluster.
        let v = JwtVerifier::new(vec![(
            "alpha".to_string(),
            JwtKeyRole::Cluster,
            vec![PUB_A.to_string(), PUB_B.to_string()],
        )])
        .unwrap();
        assert_eq!(
            v.verify(&mint_finelog(PRIV_A, FUTURE)),
            Some(("alpha", JwtKeyRole::Cluster))
        );
        assert_eq!(
            v.verify(&mint_finelog(PRIV_B, FUTURE)),
            Some(("alpha", JwtKeyRole::Cluster))
        );
        // A key outside the rotation set still fails.
        assert_eq!(v.verify(&mint_finelog(PRIV_UNTRUSTED, FUTURE)), None);
    }

    #[test]
    fn jwt_verifier_exp_leeway_tolerates_small_skew() {
        let v = verifier(&[("alpha", PUB_A)]);
        let now = unix_now();
        let leeway = EXP_LEEWAY_SECONDS as i64;
        // Expired by less than the leeway → still accepted (clock skew).
        assert_eq!(
            v.verify(&mint_finelog(PRIV_A, now - (leeway - 5))),
            Some(("alpha", JwtKeyRole::Cluster))
        );
        // Expired well beyond the leeway → rejected.
        assert_eq!(v.verify(&mint_finelog(PRIV_A, now - (leeway + 60))), None);
    }

    #[test]
    fn jwt_verifier_new_rejects_bad_pem_and_empty_fields() {
        // A PEM jsonwebtoken cannot parse as an Ed25519 public key fails the build.
        assert!(JwtVerifier::new(vec![(
            "alpha".into(),
            JwtKeyRole::Cluster,
            vec!["not-a-pem".into()]
        )])
        .is_err());
        // An empty cluster fails the build.
        assert!(JwtVerifier::new(vec![(
            String::new(),
            JwtKeyRole::Cluster,
            vec![PUB_A.into()]
        )])
        .is_err());
        // A cluster with no public keys fails the build.
        assert!(JwtVerifier::new(vec![("alpha".into(), JwtKeyRole::Cluster, vec![])]).is_err());
        assert!(JwtVerifier::new(vec![(
            "alpha".into(),
            JwtKeyRole::Cluster,
            vec![PUB_A.into()]
        )])
        .is_ok());
    }

    fn jwt_policy_json_with_role(cluster: &str, role: &str, pem: &str) -> String {
        serde_json::json!([
            {"type": "jwt", "keys": [{"cluster": cluster, "role": role, "public_keys": [pem]}]}
        ])
        .to_string()
    }

    fn jwt_policy_json(cluster: &str, pem: &str) -> String {
        jwt_policy_json_with_role(cluster, "cluster", pem)
    }

    fn stacked_policy_json(cidr: &str, cluster: &str, pem: &str) -> String {
        serde_json::json!([
            {"type": "cidr", "cidrs": [cidr]},
            {"type": "jwt", "keys": [{"cluster": cluster, "role": "cluster", "public_keys": [pem]}]}
        ])
        .to_string()
    }

    fn token_policy() -> AuthPolicy {
        AuthPolicy::parse(&jwt_policy_json("alpha", PUB_A)).unwrap()
    }

    fn stacked_policy() -> AuthPolicy {
        AuthPolicy::parse(&stacked_policy_json("10.0.0.0/8", "alpha", PUB_A)).unwrap()
    }

    #[test]
    fn parse_rejects_empty_and_malformed() {
        // Empty list = total lockout; callers must omit the policy for the default.
        assert!(AuthPolicy::parse("[]").is_err());
        assert!(AuthPolicy::parse("not json").is_err());
        assert!(AuthPolicy::parse(r#"[{"type":"cidr","cidrs":["nope/8"]}]"#).is_err());
        assert!(AuthPolicy::parse(r#"[{"type":"cidr","cidrs":[]}]"#).is_err());
        assert!(AuthPolicy::parse(r#"[{"type":"jwt","keys":[]}]"#).is_err());
        // An unparseable public-key PEM fails the whole policy at parse time.
        assert!(AuthPolicy::parse(&jwt_policy_json("alpha", "not-a-pem")).is_err());
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
            .authorize(&ctx(Some(&bearer(&mint_finelog(PRIV_A, FUTURE))), None))
            .is_ok());
        // Present but forged → rejected (not downgraded).
        assert!(p
            .authorize(&ctx(
                Some(&bearer(&mint_finelog(PRIV_UNTRUSTED, FUTURE))),
                None
            ))
            .is_err());
        // Absent → rejected by the default-deny terminal.
        assert!(p.authorize(&ctx(None, None)).is_err());
    }

    #[test]
    fn admitting_layer_names_the_identity_it_admitted() {
        // The ingest handlers bind a pushed row's origin cluster to this identity,
        // so which layer admitted -- and under which cluster -- is contract, not a
        // detail. A token identifies its cluster; a trusted network does not.
        let p = AuthPolicy::parse(&stacked_policy_json("10.0.0.0/8", "alpha", PUB_A)).unwrap();
        assert_eq!(
            p.authorize(&ctx(Some(&bearer(&mint_finelog(PRIV_A, FUTURE))), None))
                .unwrap(),
            AuthIdentity::Jwt {
                cluster: "alpha".to_string()
            }
        );
        assert_eq!(
            p.authorize(&ctx(None, Some("10.1.2.3:5555".parse().unwrap())))
                .unwrap(),
            AuthIdentity::Network
        );
    }

    #[test]
    fn signed_token_only_preserves_collector_identity_for_explicit_role() {
        let policy = AuthPolicy::parse(&jwt_policy_json_with_role(
            "alpha",
            "trusted_collector",
            PUB_A,
        ))
        .unwrap();
        assert_eq!(
            policy
                .authorize(&ctx(Some(&bearer(&mint_finelog(PRIV_A, FUTURE))), None))
                .unwrap(),
            AuthIdentity::TrustedCollector {
                cluster: "alpha".to_string()
            }
        );

        assert!(AuthPolicy::parse(&jwt_policy_json_with_role("alpha", "unknown", PUB_A)).is_err());
    }

    #[test]
    fn duplicate_public_key_cannot_choose_identity_by_policy_order() {
        let ambiguous_role = serde_json::json!([
            {"type": "jwt", "keys": [
                {"cluster": "alpha", "role": "cluster", "public_keys": [PUB_A]}
            ]},
            {"type": "jwt", "keys": [
                {"cluster": "alpha", "role": "trusted_collector", "public_keys": [PUB_A]}
            ]}
        ])
        .to_string();
        assert!(AuthPolicy::parse(&ambiguous_role).is_err());

        let ambiguous_cluster = serde_json::json!([
            {"type": "jwt", "keys": [
                {"cluster": "alpha", "role": "cluster", "public_keys": [PUB_A]},
                {"cluster": "bravo", "role": "cluster", "public_keys": [PUB_A]}
            ]}
        ])
        .to_string();
        assert!(AuthPolicy::parse(&ambiguous_cluster).is_err());

        let same_binding = serde_json::json!([
            {"type": "jwt", "keys": [
                {"cluster": "alpha", "role": "cluster", "public_keys": [PUB_A]}
            ]},
            {"type": "jwt", "keys": [
                {"cluster": "alpha", "role": "cluster", "public_keys": [PUB_A]}
            ]}
        ])
        .to_string();
        let policy = AuthPolicy::parse(&same_binding).unwrap();
        let signed = mint_finelog(PRIV_A, FUTURE);
        assert_eq!(
            policy
                .authorize(&ctx(Some(&bearer(&signed)), None))
                .unwrap(),
            AuthIdentity::Jwt {
                cluster: "alpha".to_string()
            }
        );
    }

    #[test]
    fn trusted_collector_bearer_precedes_network_fallback() {
        let policy = AuthPolicy::parse(
            &serde_json::json!([
                {"type": "cidr", "cidrs": ["10.0.0.0/8"]},
                {"type": "jwt", "keys": [
                    {"cluster": "alpha", "role": "trusted_collector", "public_keys": [PUB_A]}
                ]}
            ])
            .to_string(),
        )
        .unwrap();
        let peer = Some("10.1.2.3:5555".parse().unwrap());
        assert_eq!(
            policy
                .authorize(&ctx(Some(&bearer(&mint_finelog(PRIV_A, FUTURE))), peer))
                .unwrap(),
            AuthIdentity::TrustedCollector {
                cluster: "alpha".to_string()
            }
        );
        assert_eq!(
            policy.authorize(&ctx(None, peer)).unwrap(),
            AuthIdentity::Network
        );
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
                Some(&bearer(&mint_finelog(PRIV_UNTRUSTED, FUTURE))),
                Some(trusted)
            ))
            .is_ok());
        // Out-of-VPC client must present a valid JWT.
        assert!(p
            .authorize(&ctx(
                Some(&bearer(&mint_finelog(PRIV_A, FUTURE))),
                Some(untrusted)
            ))
            .is_ok());
        assert!(p.authorize(&ctx(None, Some(untrusted))).is_err());
        assert!(p
            .authorize(&ctx(
                Some(&bearer(&mint_finelog(PRIV_UNTRUSTED, FUTURE))),
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
            ctx(Some(&bearer(&mint_finelog(PRIV_A, FUTURE))), None),
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

    /// Translate one shared-vector layer into finelog's policy-JSON form. A `cidr`
    /// layer already matches finelog's shape (`{"type":"cidr","cidrs":[..]}`) and
    /// passes through unchanged; a `jwt` layer gains the fixed test public key
    /// [`PUB_A`] under a single "vector" cluster (the vector file's jwt verifier is
    /// mocked, so any trusted key works). An unknown type also passes through
    /// unchanged so finelog's own parser is the one that rejects it.
    fn translate_vector_layer(layer: &serde_json::Value) -> serde_json::Value {
        match layer.get("type").and_then(|v| v.as_str()) {
            Some("jwt") => serde_json::json!({
                "type": "jwt",
                "keys": [{"cluster": "vector", "role": "cluster", "public_keys": [PUB_A]}],
            }),
            _ => layer.clone(),
        }
    }

    /// Translate a whole shared-vector `stack` into finelog's `AuthPolicy::parse`
    /// JSON string.
    fn translate_vector_stack(stack: &[serde_json::Value]) -> String {
        let layers: Vec<serde_json::Value> = stack.iter().map(translate_vector_layer).collect();
        serde_json::Value::Array(layers).to_string()
    }

    /// Run the SHARED cross-language auth conformance vectors through finelog's
    /// engine and assert every allow/deny verdict matches.
    ///
    /// The vector file (`lib/rigging/src/rigging/auth_vectors.json`) is the single
    /// source of truth consumed by BOTH engines; the Python counterpart is
    /// `lib/rigging/tests/test_auth_vectors.py`. We load the file in place from its
    /// shared location (never copy it) so the two engines cannot drift.
    ///
    /// The jwt verifier is mocked per the file's `note`: `token == "valid"` maps to
    /// a real bearer signed by the trusted [`PRIV_A`] (verifies against [`PUB_A`],
    /// `aud="finelog"`); `"invalid"` maps to a bearer signed by [`PRIV_UNTRUSTED`]
    /// (present but unverifiable ⇒ Reject); `null` maps to no bearer. Only the
    /// verdict is checked — the Python-only `expect.matched` field does not apply here
    /// (finelog's `admits` returns a bare bool).
    #[test]
    fn conformance_vectors_match_rigging() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../rigging/src/rigging/auth_vectors.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read shared auth vectors at {}: {e}", path.display()));
        let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();

        for vector in doc["vectors"].as_array().unwrap() {
            let name = vector["name"].as_str().unwrap();
            let stack = vector["stack"].as_array().unwrap();
            let policy = AuthPolicy::parse(&translate_vector_stack(stack))
                .unwrap_or_else(|e| panic!("vector {name}: policy parse failed: {e}"));

            let request = &vector["request"];
            // Map the mocked token to a real finelog bearer (raw token, no `Bearer `
            // prefix, since `admits` takes the already-stripped credential).
            let bearer: Option<String> = match request["token"].as_str() {
                Some("valid") => Some(mint_finelog(PRIV_A, FUTURE)),
                Some("invalid") => Some(mint_finelog(PRIV_UNTRUSTED, FUTURE)),
                Some(other) => panic!("vector {name}: unexpected token {other:?}"),
                None => None,
            };
            let peer: SocketAddr = request["peer"].as_str().unwrap().parse().unwrap();

            let admitted = policy.admits(bearer.as_deref(), Some(peer.ip())).is_some();
            let expected = vector["expect"]["verdict"].as_str().unwrap() == "allow";
            assert_eq!(
                admitted,
                expected,
                "vector {name}: {}",
                vector["description"].as_str().unwrap_or("")
            );
        }

        for case in doc["parse_error_stacks"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let stack = case["stack"].as_array().unwrap();
            let policy_json = translate_vector_stack(stack);
            assert!(
                AuthPolicy::parse(&policy_json).is_err(),
                "parse_error case {name}: expected AuthPolicy::parse to reject {policy_json}"
            );
        }
    }
}
