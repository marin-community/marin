//! Stats-service error types and their Connect code mapping.
//!
//! The mapping is load-bearing:
//!
//! - `SchemaConflict` -> `failed_precondition` (NOT `already_exists`)
//! - `SchemaValidation` / `InvalidNamespace` -> `invalid_argument`
//! - `NamespaceNotFound` -> `not_found`
//! - `QueryResultTooLarge` -> `resource_exhausted`
//! - `Internal` -> `internal`

use connectrpc::ConnectError;

/// Domain errors raised by the store/catalog layer.
#[derive(Debug, Clone)]
pub enum StatsError {
    /// Requested schema differs from the registered one in a non-additive way
    /// (type change, new non-nullable column).
    SchemaConflict(String),
    /// A schema or write batch is structurally invalid (missing ordering key,
    /// unknown column type, reserved column).
    SchemaValidation(String),
    /// Namespace name fails the regex or path-containment check, or a drop is
    /// in flight.
    InvalidNamespace(String),
    /// Named namespace is not registered.
    NamespaceNotFound(String),
    /// Query result exceeds the size cap.
    QueryResultTooLarge(String),
    /// A durability await exceeded its budget (write not durable in time).
    DeadlineExceeded(String),
    /// Unexpected internal failure.
    Internal(String),
}

impl std::fmt::Display for StatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatsError::SchemaConflict(m) => write!(f, "{m}"),
            StatsError::SchemaValidation(m) => write!(f, "{m}"),
            StatsError::InvalidNamespace(m) => write!(f, "{m}"),
            StatsError::NamespaceNotFound(m) => write!(f, "{m}"),
            StatsError::QueryResultTooLarge(m) => write!(f, "{m}"),
            StatsError::DeadlineExceeded(m) => write!(f, "{m}"),
            StatsError::Internal(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for StatsError {}

impl From<StatsError> for ConnectError {
    fn from(err: StatsError) -> ConnectError {
        match err {
            StatsError::SchemaConflict(m) => ConnectError::failed_precondition(m),
            StatsError::SchemaValidation(m) => ConnectError::invalid_argument(m),
            StatsError::InvalidNamespace(m) => ConnectError::invalid_argument(m),
            StatsError::NamespaceNotFound(m) => ConnectError::not_found(m),
            StatsError::QueryResultTooLarge(m) => ConnectError::resource_exhausted(m),
            StatsError::DeadlineExceeded(m) => ConnectError::deadline_exceeded(m),
            StatsError::Internal(m) => ConnectError::internal(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use connectrpc::ErrorCode;

    fn code_of(err: StatsError) -> ErrorCode {
        ConnectError::from(err).code
    }

    #[test]
    fn schema_conflict_maps_to_failed_precondition_not_already_exists() {
        assert_eq!(
            code_of(StatsError::SchemaConflict("x".into())),
            ErrorCode::FailedPrecondition
        );
    }

    #[test]
    fn validation_and_invalid_map_to_invalid_argument() {
        assert_eq!(
            code_of(StatsError::SchemaValidation("x".into())),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            code_of(StatsError::InvalidNamespace("x".into())),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn not_found_maps_to_not_found() {
        assert_eq!(
            code_of(StatsError::NamespaceNotFound("x".into())),
            ErrorCode::NotFound
        );
    }

    #[test]
    fn too_large_maps_to_resource_exhausted() {
        assert_eq!(
            code_of(StatsError::QueryResultTooLarge("x".into())),
            ErrorCode::ResourceExhausted
        );
    }

    #[test]
    fn internal_maps_to_internal() {
        assert_eq!(
            code_of(StatsError::Internal("x".into())),
            ErrorCode::Internal
        );
    }
}
