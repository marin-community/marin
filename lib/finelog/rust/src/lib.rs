//! Native finelog server library.

pub mod proto {
    connectrpc::include_generated!();
}

pub mod errors;
pub(crate) mod ingestion;
pub(crate) mod ingestion_policy;
pub(crate) mod json;
pub mod migrations;
pub mod query;
pub mod server;
pub mod store;
pub(crate) mod telemetry_policy;

#[cfg(test)]
pub mod test_support;
