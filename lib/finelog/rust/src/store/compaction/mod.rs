//! Native leveled compaction: planner (pure policy), config, k-way merge, and
//! the parquet read/merge/write executor.
//!
//! The executor performs the read/merge/write of segments via a native arrow
//! k-way merge (`merge.rs`). The per-namespace maintenance task, `commit_swap`,
//! eviction, and remote sync consume `CompactionJob`s produced by
//! `planner::plan` and applied by `executor`.

pub mod config;
pub mod executor;
pub mod merge;
pub mod planner;
