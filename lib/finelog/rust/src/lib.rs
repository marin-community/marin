//! Native finelog server library.

pub mod proto {
    connectrpc::include_generated!();
}

pub mod errors;
pub mod query;
pub mod server;
pub mod store;

#[cfg(test)]
pub mod test_support;
