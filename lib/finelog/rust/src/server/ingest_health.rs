//! Registration health for the namespaces this process ingests into itself.
//!
//! A namespace this server owns (today `telemetry_v1`) must be registered
//! before it can accept a row, and that registration is re-driven on every boot
//! from the catalog's persisted schema. When it fails, every write to that
//! namespace fails with it — for as long as the binary and the catalog
//! disagree, across restarts. That is invisible from `/health` alone, which
//! answers "is this process listening", so a deploy of a wedged binary passes
//! its health gate and the ingest outage is left for a human to notice on a
//! stale dashboard.
//!
//! This is the missing signal. [`IngestHealth`] records the outcome of each
//! owned namespace's registration; `/health` reports `degraded` in its body
//! while any of them is unregistered, and `/api/server` carries the per-
//! namespace detail. The status code stays 200 either way: `/health` is also
//! the Kubernetes liveness and readiness probe, and restarting or
//! de-endpointing a server whose disagreement survives restarts would turn a
//! partial outage into a total one.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Body `/health` returns while every owned namespace is registered.
pub const HEALTH_OK: &str = "ok";

/// Where one owned namespace's registration stands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(
    tag = "state",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum RegistrationState {
    /// Declared but not yet attempted, or an attempt is in flight.
    Pending,
    Registered,
    Failed {
        error: String,
        /// When *this process* first saw the registration fail. Read against
        /// `process.startedAtUnix` it says whether the namespace has been
        /// unavailable since boot or broke later.
        since_unix: i64,
        attempts: u64,
    },
}

/// One namespace's registration state, as reported by `/api/server`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NamespaceRegistration {
    pub namespace: String,
    #[serde(flatten)]
    pub state: RegistrationState,
}

/// The registration state of every namespace this process ingests into itself.
#[derive(Debug, Default)]
pub struct IngestHealth {
    states: Mutex<BTreeMap<String, RegistrationState>>,
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0)
}

impl IngestHealth {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare that this process owns `namespace`, before its first
    /// registration attempt. Reporting it as pending from the moment the
    /// router is built closes the window where a health gate polls between the
    /// listener binding and the registration failing, and reads `ok`.
    pub fn declare_owned(&self, namespace: &str) {
        self.states
            .lock()
            .unwrap()
            .entry(namespace.to_string())
            .or_insert(RegistrationState::Pending);
    }

    pub fn record_registered(&self, namespace: &str) {
        self.states
            .lock()
            .unwrap()
            .insert(namespace.to_string(), RegistrationState::Registered);
    }

    /// Record a failed registration attempt. Repeated failures keep the
    /// original `since_unix` and count up, so the age of the wedge survives the
    /// retry every write drives.
    pub fn record_failure(&self, namespace: &str, error: &str) {
        let mut states = self.states.lock().unwrap();
        let state = states
            .entry(namespace.to_string())
            .or_insert(RegistrationState::Pending);
        match state {
            RegistrationState::Failed {
                error: recorded,
                attempts,
                ..
            } => {
                recorded.clear();
                recorded.push_str(error);
                *attempts += 1;
            }
            _ => {
                *state = RegistrationState::Failed {
                    error: error.to_string(),
                    since_unix: now_unix(),
                    attempts: 1,
                }
            }
        }
    }

    pub fn snapshot(&self) -> Vec<NamespaceRegistration> {
        self.states
            .lock()
            .unwrap()
            .iter()
            .map(|(namespace, state)| NamespaceRegistration {
                namespace: namespace.clone(),
                state: state.clone(),
            })
            .collect()
    }

    /// The `/health` body: [`HEALTH_OK`], or a one-line summary naming every
    /// namespace that cannot currently accept rows.
    pub fn health_body(&self) -> String {
        let unavailable: Vec<String> = self
            .states
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(namespace, state)| match state {
                RegistrationState::Registered => None,
                RegistrationState::Pending => Some(format!("{namespace}: registration pending")),
                RegistrationState::Failed { error, .. } => {
                    Some(format!("{namespace}: registration failed: {error}"))
                }
            })
            .collect();
        if unavailable.is_empty() {
            return HEALTH_OK.to_string();
        }
        format!("degraded: {}", unavailable.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_body_reports_ok_only_once_every_owned_namespace_is_registered() {
        let health = IngestHealth::new();
        assert_eq!(
            health.health_body(),
            HEALTH_OK,
            "nothing owned, nothing to report"
        );

        health.declare_owned("telemetry_v1");
        assert!(health.health_body().contains("registration pending"));

        health.record_registered("telemetry_v1");
        assert_eq!(health.health_body(), HEALTH_OK);
    }

    #[test]
    fn repeated_failures_keep_the_first_observation_time_and_count_up() {
        let health = IngestHealth::new();
        health.declare_owned("telemetry_v1");
        health.record_failure("telemetry_v1", "projection mismatch");
        let first = health.snapshot();
        health.record_failure("telemetry_v1", "projection mismatch again");

        let RegistrationState::Failed {
            since_unix: first_since,
            ..
        } = first[0].state.clone()
        else {
            panic!("expected a recorded failure, got {:?}", first[0].state);
        };
        let RegistrationState::Failed {
            error,
            since_unix,
            attempts,
        } = health.snapshot()[0].state.clone()
        else {
            panic!("expected a recorded failure");
        };
        assert_eq!(since_unix, first_since);
        assert_eq!(attempts, 2);
        assert_eq!(error, "projection mismatch again");
        assert!(health.health_body().contains("projection mismatch again"));
    }

    #[test]
    fn a_later_success_clears_a_recorded_failure() {
        let health = IngestHealth::new();
        health.record_failure("telemetry_v1", "boom");
        health.record_registered("telemetry_v1");
        assert_eq!(health.health_body(), HEALTH_OK);
        assert_eq!(health.snapshot()[0].state, RegistrationState::Registered);
    }
}
