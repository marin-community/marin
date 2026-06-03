//! Native leveled compaction: planner (pure policy), config, k-way merge, and
//! the parquet read/merge/write executor.
//!
//! Port of `compactor.py` (planner) + the `_run_job`/`_apply_merge`/
//! `_apply_level_bump` execution from `log_namespace.py`, with the DuckDB COPY
//! replaced by a native arrow k-way merge (`merge.rs`). The per-namespace
//! maintenance task, `commit_swap`, eviction, and remote sync (4d/4e) consume
//! `CompactionJob`s produced by `planner::plan` and applied by `executor`.

pub mod config;
pub mod executor;
pub mod merge;
pub mod planner;
