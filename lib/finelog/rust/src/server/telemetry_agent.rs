// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Trust-boundary normalization shared by telemetry agent ingest and replay.

use std::collections::HashMap;

use buffa::{Message, MessageField};
use sha2::{Digest, Sha256};

use crate::proto::finelog::telemetry::{
    TelemetryArtifactV1, TelemetryBatchV1, TelemetryEventV1, TelemetryLogV1, TelemetryMetricV1,
    TelemetryRecordV1, TelemetryResourceV1,
};
use crate::server::auth::TelemetryProducerIdentity;
use crate::store::telemetry_catalog::{catalog, CATALOG_VERSION};

const PRODUCER_REQUEST_DIGEST_DOMAIN: &[u8] = b"marin.telemetry-agent.request.v1\0";
const CANONICAL_PAYLOAD_DIGEST_DOMAIN: &[u8] = b"marin.telemetry-agent.canonical.v1\0";
const SCHEMA_VERSION: i32 = 1;

/// An agent-normalized batch whose exact protobuf bytes are safe to persist and replay.
#[derive(Debug, Clone)]
pub struct CanonicalAgentBatch {
    pub batch: TelemetryBatchV1,
    pub body: Vec<u8>,
    pub payload_sha256: String,
}

/// Bind an exact producer request to the verified infrastructure identity that sent it.
pub fn producer_request_digest(
    identity: &TelemetryProducerIdentity,
    content_type: &str,
    body: &[u8],
) -> String {
    let mut digest = Sha256::new();
    digest.update(PRODUCER_REQUEST_DIGEST_DOMAIN);
    for value in [
        identity.cluster.as_bytes(),
        identity.iris_job_id.as_bytes(),
        identity.iris_task_id.as_bytes(),
        identity.attempt_id.to_string().as_bytes(),
        identity.attempt_uid.as_bytes(),
        identity.worker_id.as_bytes(),
        identity.node_id.as_bytes(),
        content_type.as_bytes(),
        body,
    ] {
        digest_field(&mut digest, value);
    }
    format!("{:x}", digest.finalize())
}

/// Stamp verified claims, clear producer-owned infrastructure claims, recompute
/// metric series identity, and encode one deterministic protobuf envelope.
pub fn canonicalize_agent_batch(
    mut batch: TelemetryBatchV1,
    identity: &TelemetryProducerIdentity,
) -> Result<CanonicalAgentBatch, String> {
    reject_unknown_fields(&batch)?;
    if batch.schema_version != Some(SCHEMA_VERSION) {
        return Err(format!(
            "schema_version must be {SCHEMA_VERSION}, got {:?}",
            batch.schema_version
        ));
    }
    if batch.catalog_version.as_deref() != Some(CATALOG_VERSION) {
        return Err(format!(
            "catalog_version must be {CATALOG_VERSION:?}, got {:?}",
            batch.catalog_version
        ));
    }
    if batch.batch_id.as_deref().is_none_or(str::is_empty) {
        return Err("batch_id must not be empty".to_string());
    }
    if batch.records.is_empty() {
        return Err("batch must contain at least one record".to_string());
    }

    let mut delivery_lane = None;
    for (index, record) in batch.records.iter_mut().enumerate() {
        let delivery_class = record
            .delivery_class
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("records[{index}].delivery_class must not be empty"))?;
        if let Some(expected) = delivery_lane {
            if delivery_class != expected {
                return Err(format!(
                    "batch mixes delivery classes {expected:?} and {delivery_class:?}"
                ));
            }
        } else {
            delivery_lane = Some(delivery_class);
        }
        validate_record_delivery(record, index)?;

        let resource = record
            .resource
            .take()
            .ok_or_else(|| format!("records[{index}].resource is required"))?;
        record.resource = MessageField::some(canonical_resource(resource, identity)?);
        if let Some(metric) = record.metric.as_option_mut() {
            let resource = record.resource.as_option().expect("resource was just set");
            metric.series_id = Some(crate::server::telemetry_ingest::canonical_series_id(
                metric.scope.as_deref().unwrap_or(""),
                metric.name.as_deref().unwrap_or(""),
                resource,
                metric.device_uid.as_deref(),
                metric.device_type.as_deref(),
                &metric.attributes,
            ));
        }
    }
    crate::server::telemetry_ingest::validate_canonical_agent_batch(&batch)
        .map_err(|error| format!("canonical batch validation failed: {error}"))?;

    let body = deterministic_batch_bytes(&batch);
    let mut digest = Sha256::new();
    digest.update(CANONICAL_PAYLOAD_DIGEST_DOMAIN);
    digest.update(&body);
    Ok(CanonicalAgentBatch {
        batch,
        body,
        payload_sha256: format!("{:x}", digest.finalize()),
    })
}

fn canonical_resource(
    resource: TelemetryResourceV1,
    identity: &TelemetryProducerIdentity,
) -> Result<TelemetryResourceV1, String> {
    let TelemetryResourceV1 {
        service_name,
        service_instance_id,
        role,
        root_run_uid,
        service_version,
        run_id_alias,
        iris_job_id: _,
        iris_task_id: _,
        task_index: _,
        attempt_id: _,
        attempt_uid: _,
        worker_id: _,
        node_id: _,
        pod_uid: _,
        container_id: _,
        rank,
        process_index,
        actor_id,
        engine_id,
        repository,
        git_revision,
        image_digest,
        model_id,
        model_revision,
        policy_step,
        owner,
        experiment_issue,
        cluster: _,
        entity_authority: _,
        entity_type: _,
        entity_uid: _,
        __buffa_unknown_fields,
    } = resource;
    if !__buffa_unknown_fields.is_empty() {
        return Err("resource contains unknown fields".to_string());
    }
    Ok(TelemetryResourceV1 {
        // Producer/service fields are preserved.
        service_name,
        service_instance_id,
        role,
        root_run_uid,
        service_version,
        run_id_alias,
        rank,
        process_index,
        actor_id,
        engine_id,
        repository,
        git_revision,
        image_digest,
        model_id,
        model_revision,
        policy_step,
        owner,
        experiment_issue,
        // Infrastructure fields are stamped only from verified producer claims.
        iris_job_id: Some(identity.iris_job_id.clone()),
        iris_task_id: Some(identity.iris_task_id.clone()),
        attempt_id: Some(identity.attempt_id),
        attempt_uid: Some(identity.attempt_uid.clone()),
        worker_id: Some(identity.worker_id.clone()),
        node_id: Some(identity.node_id.clone()),
        cluster: Some(identity.cluster.clone()),
        // Infrastructure fields without an authenticated v1 claim are cleared.
        task_index: None,
        pod_uid: None,
        container_id: None,
        entity_authority: None,
        entity_type: None,
        entity_uid: None,
        __buffa_unknown_fields: Default::default(),
    })
}

fn validate_record_delivery(record: &TelemetryRecordV1, index: usize) -> Result<(), String> {
    let actual = record.delivery_class.as_deref().unwrap_or("");
    let expected = match record.signal.as_deref() {
        Some("metric") => {
            let metric = record
                .metric
                .as_option()
                .ok_or_else(|| format!("records[{index}].metric is required"))?;
            catalog()
                .metric(
                    metric.scope.as_deref().unwrap_or(""),
                    metric.name.as_deref().unwrap_or(""),
                )
                .map(|descriptor| descriptor.delivery_class.as_str())
                .ok_or_else(|| format!("records[{index}].metric is not in the catalog"))?
        }
        Some("event") => {
            let event = record
                .event
                .as_option()
                .ok_or_else(|| format!("records[{index}].event is required"))?;
            catalog()
                .event(event.event_name.as_deref().unwrap_or(""))
                .map(|descriptor| descriptor.delivery_class.as_str())
                .ok_or_else(|| format!("records[{index}].event is not in the catalog"))?
        }
        Some("log") if record.log.is_set() => "buffered",
        Some("artifact") if record.artifact.is_set() => "durable",
        Some(signal) => return Err(format!("records[{index}] has invalid signal {signal:?}")),
        None => return Err(format!("records[{index}].signal is required")),
    };
    if actual != expected {
        return Err(format!(
            "records[{index}].delivery_class must be {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

fn reject_unknown_fields(batch: &TelemetryBatchV1) -> Result<(), String> {
    if !batch.__buffa_unknown_fields.is_empty() {
        return Err("batch contains unknown fields".to_string());
    }
    for (index, record) in batch.records.iter().enumerate() {
        if !record.__buffa_unknown_fields.is_empty() {
            return Err(format!("records[{index}] contains unknown fields"));
        }
        for (name, unknown) in [
            (
                "resource",
                record
                    .resource
                    .as_option()
                    .map(|value| &value.__buffa_unknown_fields),
            ),
            (
                "metric",
                record
                    .metric
                    .as_option()
                    .map(|value| &value.__buffa_unknown_fields),
            ),
            (
                "event",
                record
                    .event
                    .as_option()
                    .map(|value| &value.__buffa_unknown_fields),
            ),
            (
                "log",
                record
                    .log
                    .as_option()
                    .map(|value| &value.__buffa_unknown_fields),
            ),
            (
                "artifact",
                record
                    .artifact
                    .as_option()
                    .map(|value| &value.__buffa_unknown_fields),
            ),
        ] {
            if unknown.is_some_and(|fields| !fields.is_empty()) {
                return Err(format!("records[{index}].{name} contains unknown fields"));
            }
        }
    }
    Ok(())
}

fn deterministic_batch_bytes(batch: &TelemetryBatchV1) -> Vec<u8> {
    let mut scalar = batch.clone();
    let records = std::mem::take(&mut scalar.records);
    scalar.__buffa_unknown_fields.clear();
    let mut output = scalar.encode_to_vec();
    for record in &records {
        append_message_field(&mut output, 4, &deterministic_record_bytes(record));
    }
    output
}

fn deterministic_record_bytes(record: &TelemetryRecordV1) -> Vec<u8> {
    let mut scalar = record.clone();
    let resource = scalar.resource.take();
    let metric = scalar.metric.take();
    let event = scalar.event.take();
    let log = scalar.log.take();
    let artifact = scalar.artifact.take();
    scalar.__buffa_unknown_fields.clear();
    let mut output = scalar.encode_to_vec();
    if let Some(resource) = resource {
        append_message_field(&mut output, 5, &resource.encode_to_vec());
    }
    if let Some(metric) = metric {
        append_message_field(&mut output, 6, &deterministic_metric_bytes(&metric));
    }
    if let Some(event) = event {
        append_message_field(&mut output, 7, &deterministic_event_bytes(&event));
    }
    if let Some(log) = log {
        append_message_field(&mut output, 8, &deterministic_log_bytes(&log));
    }
    if let Some(artifact) = artifact {
        append_message_field(&mut output, 9, &deterministic_artifact_bytes(&artifact));
    }
    output
}

fn deterministic_metric_bytes(metric: &TelemetryMetricV1) -> Vec<u8> {
    deterministic_map_message(metric, 16, |scalar| &mut scalar.attributes)
}

fn deterministic_event_bytes(event: &TelemetryEventV1) -> Vec<u8> {
    deterministic_map_message(event, 8, |scalar| &mut scalar.attributes)
}

fn deterministic_log_bytes(log: &TelemetryLogV1) -> Vec<u8> {
    deterministic_map_message(log, 6, |scalar| &mut scalar.attributes)
}

fn deterministic_artifact_bytes(artifact: &TelemetryArtifactV1) -> Vec<u8> {
    deterministic_map_message(artifact, 10, |scalar| &mut scalar.attributes)
}

fn deterministic_map_message<T>(
    message: &T,
    field_number: u32,
    attributes: impl FnOnce(&mut T) -> &mut HashMap<String, String>,
) -> Vec<u8>
where
    T: Message + Clone,
{
    let mut scalar = message.clone();
    let mut entries: Vec<_> = std::mem::take(attributes(&mut scalar))
        .into_iter()
        .collect();
    entries.sort();
    let mut output = scalar.encode_to_vec();
    for (key, value) in entries {
        let mut entry = Vec::with_capacity(key.len() + value.len() + 8);
        append_string_field(&mut entry, 1, &key);
        append_string_field(&mut entry, 2, &value);
        append_message_field(&mut output, field_number, &entry);
    }
    output
}

fn append_string_field(output: &mut Vec<u8>, field_number: u32, value: &str) {
    append_message_field(output, field_number, value.as_bytes());
}

fn append_message_field(output: &mut Vec<u8>, field_number: u32, value: &[u8]) {
    encode_varint(((field_number as u64) << 3) | 2, output);
    encode_varint(value.len() as u64, output);
    output.extend_from_slice(value);
}

fn encode_varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn digest_field(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn producer_identity() -> TelemetryProducerIdentity {
        TelemetryProducerIdentity {
            cluster: "trusted-cluster".to_string(),
            iris_job_id: "trusted-job".to_string(),
            iris_task_id: "trusted-task".to_string(),
            attempt_id: 42,
            attempt_uid: "trusted-attempt".to_string(),
            worker_id: "trusted-worker".to_string(),
            node_id: "trusted-node".to_string(),
        }
    }

    fn spoofed_resource() -> TelemetryResourceV1 {
        TelemetryResourceV1 {
            service_name: Some("producer-service".to_string()),
            service_instance_id: Some("producer-instance".to_string()),
            role: Some("trainer".to_string()),
            root_run_uid: Some("root-run".to_string()),
            service_version: Some("version".to_string()),
            run_id_alias: Some("alias".to_string()),
            iris_job_id: Some("spoof-job".to_string()),
            iris_task_id: Some("spoof-task".to_string()),
            task_index: Some(9),
            attempt_id: Some(9),
            attempt_uid: Some("spoof-attempt".to_string()),
            worker_id: Some("spoof-worker".to_string()),
            node_id: Some("spoof-node".to_string()),
            pod_uid: Some("spoof-pod".to_string()),
            container_id: Some("spoof-container".to_string()),
            rank: Some(2),
            process_index: Some(3),
            actor_id: Some("actor".to_string()),
            engine_id: Some("engine".to_string()),
            repository: Some("repo".to_string()),
            git_revision: Some("revision".to_string()),
            image_digest: Some("image".to_string()),
            model_id: Some("model".to_string()),
            model_revision: Some("model-revision".to_string()),
            policy_step: Some(7),
            owner: Some("owner".to_string()),
            experiment_issue: Some(204),
            cluster: Some("spoof-cluster".to_string()),
            entity_authority: Some("spoof-authority".to_string()),
            entity_type: Some("spoof-type".to_string()),
            entity_uid: Some("spoof-uid".to_string()),
            ..Default::default()
        }
    }

    fn metric_batch(attributes: HashMap<String, String>) -> TelemetryBatchV1 {
        TelemetryBatchV1 {
            schema_version: Some(SCHEMA_VERSION),
            catalog_version: Some(CATALOG_VERSION.to_string()),
            batch_id: Some("producer-batch".to_string()),
            records: vec![TelemetryRecordV1 {
                record_index: Some(0),
                signal: Some("metric".to_string()),
                event_ts_unix_nano: Some(1),
                observed_ts_unix_nano: Some(2),
                delivery_class: Some("coalescing".to_string()),
                resource: MessageField::some(spoofed_resource()),
                metric: MessageField::some(TelemetryMetricV1 {
                    scope: Some("telemetry.runtime".to_string()),
                    name: Some("queue_records".to_string()),
                    description: Some(
                        "Telemetry records waiting for background export".to_string(),
                    ),
                    unit: Some("{record}".to_string()),
                    instrument_kind: Some("gauge".to_string()),
                    temporality: Some("unspecified".to_string()),
                    series_id: Some("spoof-series".to_string()),
                    value: Some(1.0),
                    attributes,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn event_batch(attributes: HashMap<String, String>) -> TelemetryBatchV1 {
        TelemetryBatchV1 {
            schema_version: Some(SCHEMA_VERSION),
            catalog_version: Some(CATALOG_VERSION.to_string()),
            batch_id: Some("producer-batch".to_string()),
            records: vec![TelemetryRecordV1 {
                record_index: Some(0),
                signal: Some("event".to_string()),
                event_ts_unix_nano: Some(1),
                observed_ts_unix_nano: Some(2),
                delivery_class: Some("durable".to_string()),
                resource: MessageField::some(spoofed_resource()),
                event: MessageField::some(TelemetryEventV1 {
                    event_name: Some("telemetry.runtime.gap".to_string()),
                    severity_number: Some(17),
                    severity_text: Some("ERROR".to_string()),
                    attributes,
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn canonicalization_exhaustively_stamps_and_clears_infrastructure_identity() {
        let canonical =
            canonicalize_agent_batch(metric_batch(HashMap::new()), &producer_identity()).unwrap();
        let record = &canonical.batch.records[0];
        let resource = record.resource.as_option().unwrap();

        assert_eq!(resource.service_name.as_deref(), Some("producer-service"));
        assert_eq!(
            resource.service_instance_id.as_deref(),
            Some("producer-instance")
        );
        assert_eq!(resource.role.as_deref(), Some("trainer"));
        assert_eq!(resource.root_run_uid.as_deref(), Some("root-run"));
        assert_eq!(resource.service_version.as_deref(), Some("version"));
        assert_eq!(resource.run_id_alias.as_deref(), Some("alias"));
        assert_eq!(resource.rank, Some(2));
        assert_eq!(resource.process_index, Some(3));
        assert_eq!(resource.actor_id.as_deref(), Some("actor"));
        assert_eq!(resource.engine_id.as_deref(), Some("engine"));
        assert_eq!(resource.repository.as_deref(), Some("repo"));
        assert_eq!(resource.git_revision.as_deref(), Some("revision"));
        assert_eq!(resource.image_digest.as_deref(), Some("image"));
        assert_eq!(resource.model_id.as_deref(), Some("model"));
        assert_eq!(resource.model_revision.as_deref(), Some("model-revision"));
        assert_eq!(resource.policy_step, Some(7));
        assert_eq!(resource.owner.as_deref(), Some("owner"));
        assert_eq!(resource.experiment_issue, Some(204));

        assert_eq!(resource.cluster.as_deref(), Some("trusted-cluster"));
        assert_eq!(resource.iris_job_id.as_deref(), Some("trusted-job"));
        assert_eq!(resource.iris_task_id.as_deref(), Some("trusted-task"));
        assert_eq!(resource.attempt_id, Some(42));
        assert_eq!(resource.attempt_uid.as_deref(), Some("trusted-attempt"));
        assert_eq!(resource.worker_id.as_deref(), Some("trusted-worker"));
        assert_eq!(resource.node_id.as_deref(), Some("trusted-node"));

        assert_eq!(resource.task_index, None);
        assert_eq!(resource.pod_uid, None);
        assert_eq!(resource.container_id, None);
        assert_eq!(resource.entity_authority, None);
        assert_eq!(resource.entity_type, None);
        assert_eq!(resource.entity_uid, None);

        let metric = record.metric.as_option().unwrap();
        assert_eq!(
            metric.series_id.as_deref(),
            Some(
                crate::server::telemetry_ingest::canonical_series_id(
                    "telemetry.runtime",
                    "queue_records",
                    resource,
                    None,
                    None,
                    &HashMap::new(),
                )
                .as_str()
            )
        );
        assert_ne!(metric.series_id.as_deref(), Some("spoof-series"));
    }

    #[test]
    fn canonical_protobuf_and_both_digests_are_stable_and_identity_bound() {
        let mut left = HashMap::new();
        left.insert("reason".to_string(), "overflow".to_string());
        left.insert("dropped_records".to_string(), "2".to_string());
        let mut right = HashMap::new();
        right.insert("dropped_records".to_string(), "2".to_string());
        right.insert("reason".to_string(), "overflow".to_string());
        let identity = producer_identity();

        let left = canonicalize_agent_batch(event_batch(left), &identity).unwrap();
        let right = canonicalize_agent_batch(event_batch(right), &identity).unwrap();
        assert_eq!(left.body, right.body);
        assert_eq!(left.payload_sha256, right.payload_sha256);
        let mut encoded = left.body.as_slice();
        assert_eq!(TelemetryBatchV1::decode(&mut encoded).unwrap(), left.batch);
        assert_eq!(
            producer_request_digest(&identity, "application/x-protobuf", b"body"),
            producer_request_digest(&identity, "application/x-protobuf", b"body")
        );

        let mut other = identity.clone();
        other.cluster = "other-cluster".to_string();
        assert_ne!(
            producer_request_digest(&identity, "application/x-protobuf", b"body"),
            producer_request_digest(&other, "application/x-protobuf", b"body")
        );
        assert_ne!(
            producer_request_digest(&identity, "application/x-protobuf", b"body"),
            producer_request_digest(&identity, "application/json", b"body")
        );
    }

    #[test]
    fn canonicalization_rejects_mixed_or_nested_delivery_policy() {
        let mut batch = metric_batch(HashMap::new());
        let mut event = batch.records[0].clone();
        event.record_index = Some(1);
        event.signal = Some("event".to_string());
        event.metric = MessageField::none();
        event.event = MessageField::some(TelemetryEventV1 {
            event_name: Some("telemetry.runtime.gap".to_string()),
            severity_number: Some(17),
            severity_text: Some("ERROR".to_string()),
            ..Default::default()
        });
        event.delivery_class = Some("durable".to_string());
        batch.records.push(event);
        assert!(canonicalize_agent_batch(batch, &producer_identity())
            .unwrap_err()
            .contains("mixes delivery classes"));

        let mut wrong = metric_batch(HashMap::new());
        wrong.records[0].delivery_class = Some("buffered".to_string());
        assert!(canonicalize_agent_batch(wrong, &producer_identity())
            .unwrap_err()
            .contains("must be"));
    }

    #[test]
    fn canonicalization_runs_full_hub_semantic_validation_before_encoding() {
        let mut missing_value = metric_batch(HashMap::new());
        missing_value.records[0]
            .metric
            .modify(|metric| metric.value = None);
        assert!(
            canonicalize_agent_batch(missing_value, &producer_identity())
                .unwrap_err()
                .contains("scalar metrics require one finite value")
        );

        let mut multiple_signals = event_batch(HashMap::new());
        multiple_signals.records[0].metric = MessageField::some(TelemetryMetricV1::default());
        assert!(
            canonicalize_agent_batch(multiple_signals, &producer_identity())
                .unwrap_err()
                .contains("exactly one populated signal")
        );
    }
}
