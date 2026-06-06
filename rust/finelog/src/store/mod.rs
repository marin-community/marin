//! Storage engine for the native finelog server.
//!
//! Provides the metadata layer (schema types + conversions, namespace name
//! validation, storage policy, the rusqlite catalog sidecar), the data path
//! (RAM buffer, parquet segments, query), and the `Store` orchestration the
//! RPC handlers sit on.

pub mod adopt;
pub mod catalog;
pub mod compaction;
pub mod ipc;
pub mod log_read;
pub mod namespace;
pub mod namespace_name;
pub mod policy;
pub mod ram_buffer;
pub mod reconcile;
pub mod remote;
pub mod schema;
pub mod segment;
// The orchestration module is named `store`; the re-export below gives callers
// `finelog::store::Store` without the extra path.
#[allow(clippy::module_inception)]
pub mod store;
pub mod types;

pub use store::Store;
