//! Native finelog server library.
//!
//! Phase 0 of the Rust rewrite (see
//! `.agents/projects/2026-06-02_finelog_rust.md`): proto codegen + a bootable
//! server with `/health`. The two RPC services are wired bottom-up in later
//! phases; until then their handlers return `Unimplemented`.

pub mod proto {
    connectrpc::include_generated!();
}

pub mod errors;
pub mod query;
pub mod server;
pub mod store;
