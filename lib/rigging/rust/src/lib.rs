// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! The signed grant contract shared by Iris and Finelog. Applications select
//! concrete actions and resources; neither token claims nor remote failures
//! grant native administrator, worker, or replication authority.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jsonwebtoken::{decode, decode_header, Algorithm, DecodingKey, Validation};
use serde::Deserialize;

pub const SHARED_GRANT_ROLE: &str = "shared-grant";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub issuer: String,
    pub public_keys: Vec<String>,
    pub authority_url: String,
    pub ca_bundle: Option<String>,
    pub timeout: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Permission {
    pub service: String,
    pub actions: Vec<String>,
    pub resource: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Access {
    pub iss: String,
    pub aud: String,
    pub sub: String,
    pub act: HashMap<String, String>,
    pub owner: String,
    pub grant_id: String,
    pub policy_revision: String,
    pub grant_version: u64,
    pub permissions: Vec<Permission>,
    pub iat: u64,
    pub exp: u64,
    pub nbf: Option<u64>,
    pub jti: String,
}

pub struct Verifier {
    keys: Vec<DecodingKey>,
    issuer: String,
    audience: String,
}

impl Verifier {
    pub fn new(config: &Config, audience: &str) -> Result<Self, String> {
        if config.issuer.is_empty() || config.public_keys.is_empty() || audience.is_empty() {
            return Err("shared grants require an issuer, public keys, and audience".into());
        }
        let keys = config
            .public_keys
            .iter()
            .map(|pem| DecodingKey::from_ed_pem(pem.as_bytes()).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            keys,
            issuer: config.issuer.clone(),
            audience: audience.into(),
        })
    }

    pub fn verify(&self, token: &str) -> Result<Access, String> {
        let header = decode_header(token).map_err(|_| "invalid grant header")?;
        if header.alg != Algorithm::EdDSA || header.typ.as_deref() != Some("at+jwt") {
            return Err("grant requires EdDSA and at+jwt".into());
        }
        let mut validation = Validation::new(Algorithm::EdDSA);
        validation.set_issuer(&[&self.issuer]);
        validation.set_audience(&[&self.audience]);
        validation.set_required_spec_claims(&["exp", "iat", "iss", "aud", "sub"]);
        validation.leeway = 0;
        validation.validate_nbf = true;
        let access = self
            .keys
            .iter()
            .find_map(|key| {
                decode::<Access>(token, key, &validation)
                    .ok()
                    .map(|data| data.claims)
            })
            .ok_or("invalid or expired grant")?;
        if access.grant_version != 1
            || access.aud != self.audience
            || [
                &access.sub,
                &access.owner,
                &access.grant_id,
                &access.policy_revision,
                &access.jti,
            ]
            .iter()
            .any(|s| s.is_empty())
            || access.act.len() != 1
            || access.act.get("sub") != Some(&access.sub)
            || access.iat > now()
            || access.exp <= now()
            || access.exp <= access.iat
            || access.permissions.is_empty()
            || access.permissions.iter().any(|p| {
                p.service != self.audience
                    || p.actions.is_empty()
                    || p.actions.iter().any(|a| a.is_empty() || a.contains('*'))
                    || !canonical_resource(&p.resource, true)
            })
        {
            return Err("invalid grant identity or bounds".into());
        }
        Ok(access)
    }
}

impl Access {
    pub fn allows(&self, action: &str, resource: &str) -> bool {
        canonical_resource(resource, false)
            && self.permissions.iter().any(|p| {
                p.service == self.aud
                    && p.actions.iter().any(|a| a == action)
                    && resource.starts_with(&p.resource)
            })
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before Unix epoch")
        .as_secs()
}

fn canonical_resource(resource: &str, prefix: bool) -> bool {
    if resource.is_empty()
        || resource.chars().any(|c| c <= ' ' || "%\\*?#".contains(c))
        || (prefix && !resource.ends_with('/'))
    {
        return false;
    }
    let path = if let Some(rest) = resource.strip_prefix("s3://") {
        let Some((bucket, path)) = rest.split_once('/') else {
            return false;
        };
        if bucket.is_empty()
            || !bucket
                .bytes()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'.' || c == b'-')
            || !bucket.as_bytes()[0].is_ascii_alphanumeric()
        {
            return false;
        }
        format!("/{path}")
    } else {
        if !resource.starts_with('/') {
            return false;
        }
        resource.to_string()
    };
    !path.contains("//") && !path.split('/').any(|p| p == "." || p == "..")
}

#[derive(Debug)]
pub enum CheckError {
    Denied,
    Unavailable,
}

pub struct Client {
    verifier: Verifier,
    client: reqwest::Client,
    endpoint: String,
}

#[derive(Deserialize)]
struct Decision {
    grant_id: String,
    actor: String,
    owner: String,
    policy_revision: String,
    expires_at: f64,
}

impl Client {
    pub fn verify(&self, token: &str) -> Result<Access, String> {
        self.verifier.verify(token)
    }

    pub fn new(config: &Config, audience: &str) -> Result<Self, String> {
        let origin = url::Url::parse(&config.authority_url).map_err(|e| e.to_string())?;
        if !origin.username().is_empty()
            || origin.password().is_some()
            || origin.query().is_some()
            || origin.fragment().is_some()
            || origin.path() != "/"
            || origin.host_str().is_none()
            || !(origin.scheme() == "https"
                || (origin.scheme() == "http"
                    && matches!(origin.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))))
            || !config.timeout.is_finite()
            || config.timeout <= 0.0
        {
            return Err("shared authorization requires an HTTPS origin outside loopback".into());
        }
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs_f64(config.timeout));
        if let Some(path) = &config.ca_bundle {
            let pem = std::fs::read(path).map_err(|e| e.to_string())?;
            builder = builder.add_root_certificate(
                reqwest::Certificate::from_pem(&pem).map_err(|e| e.to_string())?,
            );
        }
        Ok(Self {
            verifier: Verifier::new(config, audience)?,
            client: builder.build().map_err(|e| e.to_string())?,
            endpoint: format!(
                "{}/v1/authorize",
                config.authority_url.trim_end_matches('/')
            ),
        })
    }

    pub async fn authorize(
        &self,
        token: &str,
        action: &str,
        resource: &str,
    ) -> Result<Access, CheckError> {
        let access = self
            .verifier
            .verify(token)
            .map_err(|_| CheckError::Denied)?;
        if !access.allows(action, resource) {
            return Err(CheckError::Denied);
        }
        let response = self.client.post(&self.endpoint).bearer_auth(token)
            .json(&serde_json::json!({"audience": access.aud, "action": action, "resource": resource}))
            .send().await.map_err(|_| CheckError::Unavailable)?;
        if matches!(response.status().as_u16(), 401 | 403) {
            return Err(CheckError::Denied);
        }
        if response.status().as_u16() != 200 {
            return Err(CheckError::Unavailable);
        }
        let decision: Decision = response.json().await.map_err(|_| CheckError::Unavailable)?;
        if decision.grant_id != access.grant_id
            || decision.actor != access.sub
            || decision.owner != access.owner
            || decision.policy_revision != access.policy_revision
            || !decision.expires_at.is_finite()
            || (access.exp as f64) > decision.expires_at
        {
            return Err(CheckError::Unavailable);
        }
        self.verifier.verify(token).map_err(|_| CheckError::Denied)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_and_rust_agree_on_signed_grants_and_resource_bounds() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../src/rigging/grant_vectors.json")).unwrap();
        let config = Config {
            issuer: fixture["issuer"].as_str().unwrap().into(),
            public_keys: serde_json::from_value(fixture["public_keys"].clone()).unwrap(),
            authority_url: "http://127.0.0.1:1".into(),
            ca_bundle: None,
            timeout: 1.0,
        };
        let verifier = Verifier::new(&config, fixture["audience"].as_str().unwrap()).unwrap();
        for vector in fixture["vectors"].as_array().unwrap() {
            let result = verifier.verify(vector["token"].as_str().unwrap());
            assert_eq!(
                result.is_ok(),
                vector["valid"].as_bool().unwrap(),
                "{}",
                vector["name"]
            );
            if let Ok(access) = result {
                for operation in fixture["operations"].as_array().unwrap() {
                    assert_eq!(
                        access.allows(
                            operation["action"].as_str().unwrap(),
                            operation["resource"].as_str().unwrap()
                        ),
                        operation["allowed"].as_bool().unwrap(),
                        "{operation}"
                    );
                }
            }
        }
    }
}
