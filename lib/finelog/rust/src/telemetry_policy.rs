//! Stable telemetry stream names and their server-owned storage layout.

pub(crate) const TELEMETRY_NAMESPACE: &str = "telemetry_v1";
const GIBIBYTE: i64 = 1024 * 1024 * 1024;
const DEFAULT_STREAM_MAX_BYTES: i64 = 2 * GIBIBYTE;
pub(crate) const LEVANTER_NAMESPACE: &str = "telemetry_v1.levanter";
pub(crate) const NODE_AGENT_NAMESPACE: &str = "telemetry_v1.node_agent";
pub(crate) const IRIS_RPC_NAMESPACE: &str = "telemetry_v1.iris.rpc";
pub(crate) const VLLM_NAMESPACE: &str = "telemetry_v1.vllm";
pub(crate) const ZEPHYR_NAMESPACE: &str = "telemetry_v1.zephyr";
pub(crate) const LEGACY_NAMESPACE: &str = "telemetry_v1.legacy";
pub(crate) const LEVANTER_STATUS_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.levanter.status";
pub(crate) const LEVANTER_DETAIL_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.levanter.detail";
pub(crate) const NODE_AGENT_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.node_agent";
pub(crate) const IRIS_RPC_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.iris_rpc";
pub(crate) const VLLM_STORAGE_NAMESPACE: &str = "telemetry_storage_v1.vllm";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TelemetryStorageShard {
    pub logical_namespace: &'static str,
    pub storage_namespace: &'static str,
    pub max_bytes: i64,
}

pub(crate) const TELEMETRY_STORAGE_SHARDS: [TelemetryStorageShard; 5] = [
    TelemetryStorageShard {
        logical_namespace: LEVANTER_NAMESPACE,
        storage_namespace: LEVANTER_STATUS_STORAGE_NAMESPACE,
        max_bytes: 22 * GIBIBYTE,
    },
    TelemetryStorageShard {
        logical_namespace: LEVANTER_NAMESPACE,
        storage_namespace: LEVANTER_DETAIL_STORAGE_NAMESPACE,
        max_bytes: 10 * GIBIBYTE,
    },
    TelemetryStorageShard {
        logical_namespace: NODE_AGENT_NAMESPACE,
        storage_namespace: NODE_AGENT_STORAGE_NAMESPACE,
        max_bytes: 15 * GIBIBYTE,
    },
    TelemetryStorageShard {
        logical_namespace: IRIS_RPC_NAMESPACE,
        storage_namespace: IRIS_RPC_STORAGE_NAMESPACE,
        max_bytes: GIBIBYTE,
    },
    TelemetryStorageShard {
        logical_namespace: VLLM_NAMESPACE,
        storage_namespace: VLLM_STORAGE_NAMESPACE,
        max_bytes: 2 * GIBIBYTE,
    },
];

const LEVANTER_STATUS_NAMES: [&str; 5] = [
    "global_step",
    "phase",
    "progress_time_seconds",
    "step",
    "train_loss",
];

// Remove superseded entries after every forwarder advances past its last legacy
// segment and the corresponding hub shards are empty.
const LEGACY_STORAGE_NAMESPACES: [(&str, &str); 12] = [
    (LEVANTER_NAMESPACE, LEVANTER_NAMESPACE),
    ("telemetry_v1.levanter.priority", LEVANTER_NAMESPACE),
    ("telemetry_v1.levanter.bulk", LEVANTER_NAMESPACE),
    ("telemetry_v1.levanter.standard", LEVANTER_NAMESPACE),
    ("telemetry_v1.levanter.core", LEVANTER_NAMESPACE),
    ("telemetry_v1.levanter.extra", LEVANTER_NAMESPACE),
    (NODE_AGENT_NAMESPACE, NODE_AGENT_NAMESPACE),
    ("telemetry_v1.node_agent.standard", NODE_AGENT_NAMESPACE),
    (IRIS_RPC_NAMESPACE, IRIS_RPC_NAMESPACE),
    ("telemetry_v1.iris.rpc.standard", IRIS_RPC_NAMESPACE),
    (VLLM_NAMESPACE, VLLM_NAMESPACE),
    ("telemetry_v1.vllm.standard", VLLM_NAMESPACE),
];

pub(crate) fn migration_source_namespaces() -> impl Iterator<Item = &'static str> {
    std::iter::once(TELEMETRY_NAMESPACE).chain(
        LEGACY_STORAGE_NAMESPACES
            .iter()
            .map(|(storage, _logical)| *storage),
    )
}

pub(crate) fn migration_source_logical_namespace(namespace: &str) -> Option<&'static str> {
    LEGACY_STORAGE_NAMESPACES
        .iter()
        .find_map(|(storage, logical)| (*storage == namespace).then_some(*logical))
}

pub(crate) fn ingest_storage_namespace(
    logical_namespace: &str,
    record_name: &str,
) -> Option<String> {
    if logical_namespace == LEVANTER_NAMESPACE {
        return Some(
            if LEVANTER_STATUS_NAMES.contains(&record_name) {
                LEVANTER_STATUS_STORAGE_NAMESPACE
            } else {
                LEVANTER_DETAIL_STORAGE_NAMESPACE
            }
            .to_string(),
        );
    }
    if let Some(namespace) = TELEMETRY_STORAGE_SHARDS.iter().find_map(|shard| {
        (shard.logical_namespace == logical_namespace).then_some(shard.storage_namespace)
    }) {
        return Some(namespace.to_string());
    }
    is_semantic_namespace(logical_namespace).then(|| logical_namespace.to_string())
}

/// Apply the telemetry layout policy to one row.
///
/// A client-selected semantic namespace is authoritative. The root namespace is
/// the temporary old-client form: its semantic namespace is inferred from the
/// complete row before the ordinary semantic-to-physical policy is applied.
/// Physical namespaces are accepted unchanged so Finelog forwarders can relay
/// data already laid out by an edge server.
pub(crate) fn telemetry_storage_namespace(
    requested_namespace: &str,
    service: &str,
    record_name: &str,
) -> Option<String> {
    if requested_namespace.starts_with("telemetry_storage_v1.") {
        return storage_max_bytes(requested_namespace).map(|_| requested_namespace.to_string());
    }
    let logical_namespace = if requested_namespace == TELEMETRY_NAMESPACE {
        legacy_logical_namespace(service, record_name)
    } else {
        migration_source_logical_namespace(requested_namespace).unwrap_or(requested_namespace)
    };
    ingest_storage_namespace(logical_namespace, record_name)
}

/// Classify a row written before semantic namespaces were available.
///
/// The legacy root carries no client-selected namespace, so the owning service
/// is the only semantic boundary that can be recovered. Iris controller rows
/// are narrower: only the native RPC and proxy metric families belong to the
/// `iris.rpc` stream. Unknown services enter the bounded legacy stream.
pub(crate) fn legacy_logical_namespace(service: &str, record_name: &str) -> &'static str {
    match service {
        "levanter" => LEVANTER_NAMESPACE,
        "iris-node-agent" => NODE_AGENT_NAMESPACE,
        "iris-controller"
            if record_name.starts_with("rpc_") || record_name.starts_with("proxy_") =>
        {
            IRIS_RPC_NAMESPACE
        }
        "vllm" => VLLM_NAMESPACE,
        "zephyr" => ZEPHYR_NAMESPACE,
        _ => LEGACY_NAMESPACE,
    }
}

pub(crate) fn storage_max_bytes(storage_namespace: &str) -> Option<i64> {
    if storage_namespace == TELEMETRY_NAMESPACE {
        return Some(50 * GIBIBYTE);
    }
    TELEMETRY_STORAGE_SHARDS
        .iter()
        .find_map(|shard| (shard.storage_namespace == storage_namespace).then_some(shard.max_bytes))
        .or_else(|| is_semantic_namespace(storage_namespace).then_some(DEFAULT_STREAM_MAX_BYTES))
}

pub(crate) fn logical_namespace_for_storage(storage_namespace: &str) -> Option<&'static str> {
    TELEMETRY_STORAGE_SHARDS.iter().find_map(|shard| {
        (shard.storage_namespace == storage_namespace).then_some(shard.logical_namespace)
    })
}

pub(crate) fn is_forwarded_telemetry_namespace(namespace: &str) -> bool {
    namespace == TELEMETRY_NAMESPACE
        || storage_max_bytes(namespace).is_some()
        || migration_source_logical_namespace(namespace).is_some()
        || logical_namespace_for_storage(namespace).is_some()
}

fn is_semantic_namespace(namespace: &str) -> bool {
    let Some(scope) = namespace.strip_prefix("telemetry_v1.") else {
        return false;
    };
    if namespace.len() > 64
        || LEGACY_STORAGE_NAMESPACES
            .iter()
            .any(|(legacy, logical)| legacy != logical && *legacy == namespace)
    {
        return false;
    }
    scope.split('.').all(|component| {
        let mut chars = component.chars();
        chars.next().is_some_and(|first| first.is_ascii_lowercase())
            && chars.all(|character| {
                character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_rows_use_service_and_metric_policy() {
        assert_eq!(
            telemetry_storage_namespace(TELEMETRY_NAMESPACE, "levanter", "train_loss").as_deref(),
            Some(LEVANTER_STATUS_STORAGE_NAMESPACE)
        );
        assert_eq!(
            telemetry_storage_namespace(TELEMETRY_NAMESPACE, "iris-controller", "rpc_latency_ms")
                .as_deref(),
            Some(IRIS_RPC_STORAGE_NAMESPACE)
        );
        assert_eq!(
            telemetry_storage_namespace(TELEMETRY_NAMESPACE, "rigging", "queue_depth").as_deref(),
            Some(LEGACY_NAMESPACE)
        );
        assert_eq!(
            telemetry_storage_namespace(TELEMETRY_NAMESPACE, "zephyr", "progress_time_seconds")
                .as_deref(),
            Some(ZEPHYR_NAMESPACE)
        );
    }

    #[test]
    fn explicit_semantic_namespaces_are_authoritative() {
        assert_eq!(
            telemetry_storage_namespace("telemetry_v1.rigging.scheduler", "levanter", "train_loss")
                .as_deref(),
            Some("telemetry_v1.rigging.scheduler")
        );
        assert_eq!(
            telemetry_storage_namespace(LEVANTER_NAMESPACE, "unexpected-service", "train_loss")
                .as_deref(),
            Some(LEVANTER_STATUS_STORAGE_NAMESPACE)
        );
    }

    #[test]
    fn forwarders_preserve_physical_layout() {
        assert_eq!(
            telemetry_storage_namespace(
                LEVANTER_DETAIL_STORAGE_NAMESPACE,
                "levanter",
                "train_loss"
            )
            .as_deref(),
            Some(LEVANTER_DETAIL_STORAGE_NAMESPACE)
        );
        assert!(is_forwarded_telemetry_namespace(
            "telemetry_v1.levanter.extra"
        ));
    }
}
