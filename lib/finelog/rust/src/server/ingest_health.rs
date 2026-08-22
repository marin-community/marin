//! Registration health for the namespaces this process ingests into itself.
//!
//! `telemetry_v1` must be registered before it accepts a row, and the
//! registration is re-driven from the catalog on every boot, so a schema the
//! catalog rejects wedges ingest across restarts. `/health` reports that in its
//! body; `/api/server` carries the per-namespace detail.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// Body `/health` returns while every owned namespace is registered.
pub const HEALTH_OK: &str = "ok";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(
    tag = "state",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum RegistrationState {
    /// Declared, no attempt has succeeded yet.
    Pending,
    Registered,
    Failed {
        error: String,
        /// When this process first saw the registration fail. Compare against
        /// `process.startedAtUnix` to tell a wedge at boot from a later one.
        since_unix: i64,
        attempts: u64,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NamespaceRegistration {
    pub namespace: String,
    #[serde(flatten)]
    pub state: RegistrationState,
}

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

    /// Mark `namespace` pending, before its first registration attempt. A
    /// health gate polling between the listener binding and the first attempt
    /// then reads `pending`, not `ok`.
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

    /// Record a failed attempt, keeping the first `since_unix` and counting up.
    /// Every write retries the registration, so attempts accumulate fast.
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

    /// The `/health` body: [`HEALTH_OK`], or a line naming each namespace that
    /// cannot accept rows.
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
        assert_eq!(health.health_body(), HEALTH_OK);

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
