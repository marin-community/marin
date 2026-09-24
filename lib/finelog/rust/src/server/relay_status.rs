// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Hub-resident relay heartbeat state.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use crate::proto::finelog::stats::{
    RelayNamespaceStatus, RelaySenderStatus, ReportRelayStatusRequest,
};

#[derive(Default)]
pub struct RelayStatusRegistry {
    senders: Mutex<HashMap<String, SenderState>>,
}

#[derive(Clone)]
struct SenderState {
    status: RelaySenderStatus,
    retired_boot_ids: HashSet<String>,
}

impl RelayStatusRegistry {
    /// Record a complete sender snapshot and return its stored receipt time.
    ///
    /// A stale sequence or a report from a retired boot keeps the prior receipt
    /// time so rejected reports cannot refresh the sender heartbeat.
    pub fn report(
        &self,
        cluster: String,
        report: ReportRelayStatusRequest,
        received_at_ms: i64,
    ) -> Result<i64, String> {
        let boot_id = required_nonempty(report.boot_id, "boot_id")?;
        let report_sequence = report
            .report_sequence
            .filter(|sequence| *sequence > 0)
            .ok_or_else(|| "report_sequence must be greater than zero".to_string())?;
        let target = required_nonempty(report.target, "target")?;
        let mut namespace_names = HashSet::new();
        let mut namespaces = Vec::with_capacity(report.namespaces.len());

        let mut senders = self.senders.lock().unwrap();
        let previous = senders.get(&cluster);
        if let Some(previous) = previous {
            if previous.retired_boot_ids.contains(&boot_id) {
                return Ok(previous.status.received_at_ms.unwrap_or(received_at_ms));
            }
            if previous.status.boot_id.as_deref() == Some(boot_id.as_str())
                && previous.status.report_sequence.unwrap_or_default() >= report_sequence
            {
                return Ok(previous.status.received_at_ms.unwrap_or(received_at_ms));
            }
        }

        for namespace in report.namespaces {
            let name = required_nonempty(namespace.namespace, "namespace")?;
            if !namespace_names.insert(name.clone()) {
                return Err(format!("duplicate namespace {name:?}"));
            }
            let visible_high_water = namespace
                .visible_high_water
                .ok_or_else(|| format!("visible_high_water required for {name:?}"))?;
            let published_high_water = namespace
                .published_high_water
                .ok_or_else(|| format!("published_high_water required for {name:?}"))?;
            let prior = previous.and_then(|sender| {
                sender
                    .status
                    .namespaces
                    .iter()
                    .find(|namespace| namespace.namespace.as_deref() == Some(name.as_str()))
            });
            let publication_progress_at_ms = progress_time(
                prior.and_then(|status| status.published_high_water),
                published_high_water,
                prior.and_then(|status| status.publication_progress_at_ms),
                received_at_ms,
            );
            let cursor_progress_at_ms = optional_progress_time(
                prior.and_then(|status| status.settled_cursor),
                namespace.settled_cursor,
                prior.and_then(|status| status.cursor_progress_at_ms),
                received_at_ms,
            );
            namespaces.push(RelayNamespaceStatus {
                namespace: Some(name),
                visible_high_water: Some(visible_high_water),
                published_high_water: Some(published_high_water),
                settled_cursor: namespace.settled_cursor,
                publication_progress_at_ms: Some(publication_progress_at_ms),
                cursor_progress_at_ms: Some(cursor_progress_at_ms),
                ..Default::default()
            });
        }
        namespaces.sort_by(|left, right| left.namespace.cmp(&right.namespace));

        let mut retired_boot_ids = previous
            .map(|state| state.retired_boot_ids.clone())
            .unwrap_or_default();
        if let Some(previous_boot_id) = previous.and_then(|state| state.status.boot_id.as_ref()) {
            if previous_boot_id != &boot_id {
                retired_boot_ids.insert(previous_boot_id.clone());
            }
        }
        senders.insert(
            cluster.clone(),
            SenderState {
                status: RelaySenderStatus {
                    cluster: Some(cluster),
                    boot_id: Some(boot_id),
                    report_sequence: Some(report_sequence),
                    target: Some(target),
                    received_at_ms: Some(received_at_ms),
                    namespaces,
                    ..Default::default()
                },
                retired_boot_ids,
            },
        );
        Ok(received_at_ms)
    }

    pub fn list(&self) -> Vec<RelaySenderStatus> {
        let mut statuses: Vec<_> = self
            .senders
            .lock()
            .unwrap()
            .values()
            .map(|state| state.status.clone())
            .collect();
        statuses.sort_by(|left, right| left.cluster.cmp(&right.cluster));
        statuses
    }
}

fn required_nonempty(value: Option<String>, field: &str) -> Result<String, String> {
    value
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{field} required"))
}

fn progress_time(previous: Option<i64>, current: i64, prior_time: Option<i64>, now: i64) -> i64 {
    if previous.is_some_and(|value| current <= value) {
        return prior_time.unwrap_or(now);
    }
    now
}

fn optional_progress_time(
    previous: Option<i64>,
    current: Option<i64>,
    prior_time: Option<i64>,
    now: i64,
) -> i64 {
    if current.is_some_and(|value| previous.is_none_or(|prior| value > prior)) {
        return now;
    }
    prior_time.unwrap_or(now)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::finelog::stats::ReportRelayNamespaceStatus;

    fn report(
        boot_id: &str,
        sequence: u64,
        published: i64,
        cursor: Option<i64>,
    ) -> ReportRelayStatusRequest {
        ReportRelayStatusRequest {
            boot_id: Some(boot_id.to_string()),
            report_sequence: Some(sequence),
            target: Some("https://hub".to_string()),
            namespaces: vec![ReportRelayNamespaceStatus {
                namespace: Some("telemetry_v1.node_agent".to_string()),
                visible_high_water: Some(published + 1),
                published_high_water: Some(published),
                settled_cursor: cursor,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn progress_times_only_move_when_watermarks_advance() {
        let registry = RelayStatusRegistry::default();
        registry
            .report("cw-a".to_string(), report("a", 1, 10, None), 100)
            .unwrap();
        registry
            .report("cw-a".to_string(), report("a", 2, 10, None), 200)
            .unwrap();
        let unchanged = registry.list().pop().unwrap().namespaces.pop().unwrap();
        assert_eq!(unchanged.publication_progress_at_ms, Some(100));
        assert_eq!(unchanged.cursor_progress_at_ms, Some(100));

        registry
            .report("cw-a".to_string(), report("a", 3, 11, Some(8)), 300)
            .unwrap();
        let advanced = registry.list().pop().unwrap().namespaces.pop().unwrap();
        assert_eq!(advanced.publication_progress_at_ms, Some(300));
        assert_eq!(advanced.cursor_progress_at_ms, Some(300));
    }

    #[test]
    fn stale_reports_and_retired_boots_do_not_refresh_the_heartbeat() {
        let registry = RelayStatusRegistry::default();
        registry
            .report("cw-a".to_string(), report("a", 2, 10, Some(8)), 100)
            .unwrap();
        assert_eq!(
            registry.report("cw-a".to_string(), report("a", 1, 12, Some(9)), 200),
            Ok(100)
        );
        registry
            .report("cw-a".to_string(), report("b", 1, 12, Some(9)), 300)
            .unwrap();
        assert_eq!(
            registry.report("cw-a".to_string(), report("a", 3, 13, Some(10)), 400),
            Ok(300)
        );
        assert_eq!(registry.list()[0].received_at_ms, Some(300));
    }

    #[test]
    fn complete_snapshot_removes_namespaces_missing_from_the_next_report() {
        let registry = RelayStatusRegistry::default();
        registry
            .report("cw-a".to_string(), report("a", 1, 10, Some(8)), 100)
            .unwrap();
        let empty = ReportRelayStatusRequest {
            boot_id: Some("a".to_string()),
            report_sequence: Some(2),
            target: Some("https://hub".to_string()),
            ..Default::default()
        };
        registry.report("cw-a".to_string(), empty, 200).unwrap();
        assert!(registry.list()[0].namespaces.is_empty());
    }
}
