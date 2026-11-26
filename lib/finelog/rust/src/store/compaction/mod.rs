//! Native leveled compaction: planner (pure policy), config, k-way merge, and
//! the parquet read/merge/write executor.
//!
//! The executor performs the read/merge/write of segments via a native arrow
//! k-way merge (`merge.rs`). Two drivers consume the jobs `planner::plan`
//! produces: `object_driver` compacts a table's immutable objects under a
//! maintenance lease, and `local_driver` compacts the local files of a legacy
//! table. Both run the same executor.

#[cfg(test)]
mod backfill_bench;
pub mod config;
pub mod executor;
pub mod local_driver;
pub mod merge;
pub mod object_driver;
pub mod planner;
pub mod staging;
