// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{net::IpAddr, str::FromStr};

use axum::http::{header, HeaderMap, HeaderName};
use ipnet::IpNet;
use jsonwebtoken::jwk::{Jwk, JwkSet};
use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use moka::sync::Cache;
use serde::Deserialize;
use tokio::sync::Mutex;

const IAP_KEYS_TTL: Duration = Duration::from_secs(3600);
const IAP_FETCH_TIMEOUT: Duration = Duration::from_secs(10);
pub(crate) const IAP_ASSERTION_HEADER: HeaderName =
    HeaderName::from_static("x-goog-iap-jwt-assertion");

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeAuthMode {
    Permissive,
    Optional,
    Enforcing,
}

#[derive(Clone, Debug, Deserialize)]
pub struct NativeAuthConfig {
    pub mode: NativeAuthMode,
    pub issuers: Vec<String>,
    pub jwks: JwkSet,
    pub leeway_seconds: u64,
    pub cache_capacity: u64,
    pub cache_ttl_seconds: u64,
    #[serde(default)]
    pub trusted_cidrs: Vec<String>,
    pub control_audience: String,
    pub proxy_audience: String,
    pub proxy_scope: String,
    pub federation_audience: String,
    pub session_cookie: String,
    pub iap_public_keys_url: String,
    pub iap_issuer: String,
    #[serde(default)]
    pub iap_audience: Option<String>,
    #[serde(default)]
    pub federation_keys: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct VerifiedIdentity {
    pub endpoint: Option<String>,
    pub federation_peer: Option<String>,
    pub(crate) expires_at: u64,
}

#[derive(Debug)]
pub enum VerifyOutcome {
    Verified(VerifiedIdentity),
    Invalid,
    Anonymous,
}

#[derive(Default)]
pub(crate) struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
}

pub struct NativeVerifier {
    config: NativeAuthConfig,
    keys: Vec<(Option<String>, DecodingKey)>,
    federation_keys: Vec<(String, DecodingKey)>,
    iap: Option<IapVerifier>,
    cache: Cache<[u8; 32], VerifiedIdentity>,
    stats: Arc<CacheStats>,
    trusted_networks: Vec<IpNet>,
}

#[derive(Debug, Deserialize)]
struct Claims {
    aud: String,
    exp: u64,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UnverifiedClaims {
    #[serde(default)]
    aud: String,
    #[serde(default)]
    iss: String,
}

#[derive(Debug, Deserialize)]
struct FederationClaims {
    #[serde(rename = "aud")]
    _aud: String,
    exp: u64,
    #[serde(rename = "iss")]
    _iss: String,
    sub: String,
}

#[derive(Debug, Deserialize)]
struct IapClaims {
    #[serde(rename = "aud")]
    _aud: String,
    exp: u64,
    #[serde(rename = "iss")]
    _iss: String,
}

struct IapKeyCache {
    expires_at: Instant,
    keys: HashMap<String, DecodingKey>,
}

struct IapVerifier {
    audience: String,
    public_keys_url: String,
    issuer: String,
    client: reqwest::Client,
    cache: Mutex<IapKeyCache>,
}

impl NativeVerifier {
    pub(crate) fn new(config: NativeAuthConfig, stats: Arc<CacheStats>) -> Result<Self, String> {
        let keys = config
            .jwks
            .keys
            .iter()
            .map(key_entry)
            .collect::<Result<Vec<_>, _>>()?;
        let cache = Cache::builder()
            .max_capacity(config.cache_capacity)
            .time_to_live(Duration::from_secs(config.cache_ttl_seconds))
            .build();
        let trusted_networks = config
            .trusted_cidrs
            .iter()
            .map(|cidr| {
                IpNet::from_str(cidr)
                    .map_err(|error| format!("invalid trusted CIDR {cidr:?}: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let federation_keys = config
            .federation_keys
            .iter()
            .map(|(peer_id, pem)| {
                DecodingKey::from_ed_pem(pem.as_bytes())
                    .map(|key| (peer_id.clone(), key))
                    .map_err(|error| format!("invalid federation key for {peer_id:?}: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let iap = config
            .iap_audience
            .as_ref()
            .map(|audience| {
                IapVerifier::new(
                    audience.clone(),
                    config.iap_public_keys_url.clone(),
                    config.iap_issuer.clone(),
                )
            })
            .transpose()?;
        Ok(Self {
            config,
            keys,
            federation_keys,
            iap,
            cache,
            stats,
            trusted_networks,
        })
    }

    pub async fn verify_request(
        &self,
        headers: &HeaderMap,
        explicit_token: Option<&str>,
        peer_ip: IpAddr,
        direct_connection: bool,
    ) -> VerifyOutcome {
        let token = explicit_token
            .map(str::to_string)
            .or_else(|| bearer_token(headers))
            .or_else(|| session_cookie(headers, &self.config.session_cookie));
        if let Some(token) = token {
            return self.verify_cached(&token, || self.verify_token(&token));
        }
        if let Some(assertion) = headers
            .get(&IAP_ASSERTION_HEADER)
            .and_then(|value| value.to_str().ok())
        {
            let key = *blake3::hash(assertion.as_bytes()).as_bytes();
            if let Some(identity) = self.cached_identity(&key) {
                return VerifyOutcome::Verified(identity);
            }
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            let verified = match &self.iap {
                Some(iap) => iap.verify(assertion, self.config.leeway_seconds).await,
                None => Err("IAP assertion verification is not configured".to_string()),
            };
            return self.cache_result(key, verified);
        }
        if direct_connection
            && (peer_ip.is_loopback()
                || self
                    .trusted_networks
                    .iter()
                    .any(|network| network.contains(&peer_ip)))
        {
            return VerifyOutcome::Anonymous;
        }
        if matches!(
            self.config.mode,
            NativeAuthMode::Permissive | NativeAuthMode::Optional
        ) {
            VerifyOutcome::Anonymous
        } else {
            VerifyOutcome::Invalid
        }
    }

    fn verify_cached(
        &self,
        token: &str,
        verify: impl FnOnce() -> Result<VerifiedIdentity, String>,
    ) -> VerifyOutcome {
        let key = *blake3::hash(token.as_bytes()).as_bytes();
        if let Some(identity) = self.cached_identity(&key) {
            return VerifyOutcome::Verified(identity);
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        self.cache_result(key, verify())
    }

    fn cached_identity(&self, key: &[u8; 32]) -> Option<VerifiedIdentity> {
        let identity = self.cache.get(key)?;
        if unix_time()
            <= identity
                .expires_at
                .saturating_add(self.config.leeway_seconds)
        {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Some(identity);
        }
        self.cache.invalidate(key);
        None
    }

    fn cache_result(
        &self,
        key: [u8; 32],
        result: Result<VerifiedIdentity, String>,
    ) -> VerifyOutcome {
        match result {
            Ok(identity) => {
                self.cache.insert(key, identity.clone());
                VerifyOutcome::Verified(identity)
            }
            Err(_) if matches!(self.config.mode, NativeAuthMode::Permissive) => {
                VerifyOutcome::Anonymous
            }
            Err(_) => VerifyOutcome::Invalid,
        }
    }
}

impl CacheStats {
    pub(crate) fn snapshot(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }
}

impl NativeVerifier {
    fn verify_token(&self, token: &str) -> Result<VerifiedIdentity, String> {
        let unverified = jsonwebtoken::dangerous::insecure_decode::<UnverifiedClaims>(token)
            .map_err(|error| format!("invalid JWT claims: {error}"))?;
        if unverified.claims.aud == self.config.federation_audience {
            return self.verify_federation_token(token, &unverified.claims.iss);
        }
        if self.keys.is_empty() {
            return Err("native JWT verifier has no keys".to_string());
        }
        let header =
            decode_header(token).map_err(|error| format!("invalid JWT header: {error}"))?;
        let matched_keys = self
            .keys
            .iter()
            .filter(|(kid, _)| header.kid.is_some() && kid.as_ref() == header.kid.as_ref())
            .collect::<Vec<_>>();
        let candidate_keys = if matched_keys.is_empty() {
            self.keys.iter().collect::<Vec<_>>()
        } else {
            matched_keys
        };
        let mut last_error = None;
        for (_, key) in candidate_keys {
            let mut validation = Validation::new(Algorithm::EdDSA);
            validation.set_audience(&[&self.config.control_audience, &self.config.proxy_audience]);
            validation.set_issuer(&self.config.issuers);
            validation.leeway = self.config.leeway_seconds;
            validation.set_required_spec_claims(&["exp", "iat", "iss", "aud"]);
            match decode::<Claims>(token, key, &validation) {
                Ok(token_data) => return self.identity_from_claims(token_data.claims),
                Err(error) => last_error = Some(error),
            }
        }
        Err(format!("JWT verification failed: {last_error:?}"))
    }

    fn verify_federation_token(
        &self,
        token: &str,
        unverified_issuer: &str,
    ) -> Result<VerifiedIdentity, String> {
        let mut last_error = None;
        for (peer_id, key) in self
            .federation_keys
            .iter()
            .filter(|(peer_id, _)| peer_id == unverified_issuer)
        {
            let mut validation = Validation::new(Algorithm::EdDSA);
            validation.set_audience(&[&self.config.federation_audience]);
            validation.set_issuer(&[peer_id]);
            validation.leeway = self.config.leeway_seconds;
            validation.set_required_spec_claims(&["exp", "iat", "iss", "aud", "sub"]);
            match decode::<FederationClaims>(token, key, &validation) {
                Ok(token_data) if token_data.claims.sub == *peer_id => {
                    return Ok(VerifiedIdentity {
                        endpoint: None,
                        federation_peer: Some(peer_id.clone()),
                        expires_at: token_data.claims.exp,
                    })
                }
                Ok(_) => return Err("federation JWT subject does not match issuer".to_string()),
                Err(error) => last_error = Some(error),
            }
        }
        Err(format!(
            "federation JWT verification failed: {last_error:?}"
        ))
    }

    fn identity_from_claims(&self, claims: Claims) -> Result<VerifiedIdentity, String> {
        let proxy_audience = claims.aud == self.config.proxy_audience;
        let proxy_scope = claims.scope.as_deref() == Some(self.config.proxy_scope.as_str());
        if proxy_audience != proxy_scope {
            return Err("JWT audience/scope mismatch".to_string());
        }
        if claims.aud != self.config.control_audience && !proxy_audience {
            return Err("JWT audience is not accepted".to_string());
        }
        if proxy_scope && claims.endpoint.is_none() {
            return Err("proxy JWT has no endpoint claim".to_string());
        }
        Ok(VerifiedIdentity {
            endpoint: claims.endpoint.filter(|_| proxy_scope),
            federation_peer: None,
            expires_at: claims.exp,
        })
    }
}

fn key_entry(jwk: &Jwk) -> Result<(Option<String>, DecodingKey), String> {
    let key =
        DecodingKey::from_jwk(jwk).map_err(|error| format!("invalid JWT public key: {error}"))?;
    Ok((jwk.common.key_id.clone(), key))
}

fn bearer_token(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then(|| token.to_string())
}

fn session_cookie(headers: &HeaderMap, cookie_name: &str) -> Option<String> {
    let cookies = headers.get(header::COOKIE)?.to_str().ok()?;
    cookies.split(';').find_map(|cookie| {
        let (name, value) = cookie.trim().split_once('=')?;
        (name == cookie_name).then(|| value.to_string())
    })
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_secs()
}

impl IapVerifier {
    fn new(audience: String, public_keys_url: String, issuer: String) -> Result<Self, String> {
        if audience.is_empty() || public_keys_url.is_empty() || issuer.is_empty() {
            return Err("IAP audience, public-key URL, and issuer must not be empty".to_string());
        }
        let client = reqwest::Client::builder()
            .timeout(IAP_FETCH_TIMEOUT)
            .build()
            .map_err(|error| format!("failed to build IAP key client: {error}"))?;
        Ok(Self {
            audience,
            public_keys_url,
            issuer,
            client,
            cache: Mutex::new(IapKeyCache {
                expires_at: Instant::now(),
                keys: HashMap::new(),
            }),
        })
    }

    async fn verify(&self, token: &str, leeway_seconds: u64) -> Result<VerifiedIdentity, String> {
        let header =
            decode_header(token).map_err(|error| format!("invalid IAP JWT header: {error}"))?;
        if header.alg != Algorithm::ES256 {
            return Err("IAP JWT must use ES256".to_string());
        }
        let kid = header
            .kid
            .ok_or_else(|| "IAP JWT has no key id".to_string())?;
        let mut cache = self.cache.lock().await;
        if cache.expires_at <= Instant::now() || cache.keys.is_empty() {
            let keys = self
                .client
                .get(&self.public_keys_url)
                .send()
                .await
                .map_err(|error| format!("failed to fetch IAP public keys: {error}"))?
                .error_for_status()
                .map_err(|error| format!("IAP public-key endpoint failed: {error}"))?
                .json::<HashMap<String, String>>()
                .await
                .map_err(|error| format!("invalid IAP public-key response: {error}"))?
                .into_iter()
                .map(|(kid, pem)| {
                    DecodingKey::from_ec_pem(pem.as_bytes())
                        .map(|key| (kid, key))
                        .map_err(|error| format!("invalid IAP public key: {error}"))
                })
                .collect::<Result<HashMap<_, _>, _>>()?;
            cache.keys = keys;
            cache.expires_at = Instant::now() + IAP_KEYS_TTL;
        }
        let key = cache
            .keys
            .get(&kid)
            .ok_or_else(|| format!("IAP JWT key {kid:?} is unavailable"))?;
        let mut validation = Validation::new(Algorithm::ES256);
        validation.set_audience(&[&self.audience]);
        validation.set_issuer(&[&self.issuer]);
        validation.leeway = leeway_seconds;
        validation.set_required_spec_claims(&["exp", "iss", "aud"]);
        let claims = decode::<IapClaims>(token, key, &validation)
            .map_err(|error| format!("IAP JWT verification failed: {error}"))?
            .claims;
        Ok(VerifiedIdentity {
            endpoint: None,
            federation_peer: None,
            expires_at: claims.exp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{encode, EncodingKey, Header};

    const TEST_IAP_ISSUER: &str = "https://cloud.google.com/iap";
    use serde::Serialize;

    const PRIVATE_KEY: &[u8] = br#"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgWTFfCGljY6aw3Hrt
kHmPRiazukxPLb6ilpRAewjW8nihRANCAATDskChT+Altkm9X7MI69T3IUmrQU0L
950IxEzvw/x5BMEINRMrXLBJhqzO9Bm+d6JbqA21YQmd1Kt4RzLJR1W+
-----END PRIVATE KEY-----"#;
    const PUBLIC_KEY: &[u8] = br#"-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEw7JAoU/gJbZJvV+zCOvU9yFJq0FN
C/edCMRM78P8eQTBCDUTK1ywSYaszvQZvneiW6gNtWEJndSreEcyyUdVvg==
-----END PUBLIC KEY-----"#;

    #[derive(Serialize)]
    struct TestClaims<'a> {
        aud: &'a str,
        exp: u64,
        iss: &'a str,
    }

    async fn verifier() -> IapVerifier {
        let verifier = IapVerifier::new(
            "iap-audience".to_string(),
            "https://unused.example/iap-keys".to_string(),
            TEST_IAP_ISSUER.to_string(),
        )
        .unwrap();
        let mut cache = verifier.cache.lock().await;
        cache.keys.insert(
            "test-key".to_string(),
            DecodingKey::from_ec_pem(PUBLIC_KEY).unwrap(),
        );
        cache.expires_at = Instant::now() + Duration::from_secs(60);
        drop(cache);
        verifier
    }

    fn token(audience: &str) -> String {
        let mut header = Header::new(Algorithm::ES256);
        header.kid = Some("test-key".to_string());
        encode(
            &header,
            &TestClaims {
                aud: audience,
                exp: unix_time() + 60,
                iss: TEST_IAP_ISSUER,
            },
            &EncodingKey::from_ec_pem(PRIVATE_KEY).unwrap(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn iap_verifier_checks_signature_and_audience() {
        let verifier = verifier().await;

        assert!(verifier.verify(&token("iap-audience"), 0).await.is_ok());
        assert!(verifier.verify(&token("wrong-audience"), 0).await.is_err());
    }
}
