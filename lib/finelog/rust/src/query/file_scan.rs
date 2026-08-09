//! Shared rewrites over DataFusion Parquet scan files.

use std::sync::Arc;

use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::ExecutionPlan;

/// Apply `rewrite` to every file in a Parquet scan, preserving its grouping.
///
/// Non-Parquet plans are returned unchanged.
pub fn rewrite_parquet_files(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl Fn(&PartitionedFile) -> PartitionedFile,
) -> Arc<dyn ExecutionPlan> {
    let Some(exec) = plan.as_any().downcast_ref::<DataSourceExec>() else {
        return plan;
    };
    let Some(config) = exec.data_source().as_any().downcast_ref::<FileScanConfig>() else {
        return plan;
    };
    let groups = config
        .file_groups
        .iter()
        .map(|group| FileGroup::new(group.files().iter().map(&rewrite).collect()))
        .collect();
    let config = FileScanConfigBuilder::from(config.clone())
        .with_file_groups(groups)
        .build();
    DataSourceExec::from_data_source(config)
}
