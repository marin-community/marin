//! Shared Parquet input readers and rewrites over DataFusion scan files.

use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};

use crate::errors::StatsError;

/// Fetch the usual object footer in one ranged read, with a second request
/// when the metadata exceeds this hint.
pub const REMOTE_FOOTER_SIZE_HINT: usize = 512 * 1024;

/// Read one exact Parquet input through the same object-store registry used by
/// queries. Range reads decode into batches without creating a local cache file.
pub async fn read_parquet_batches(
    source: &str,
    byte_size: u64,
    batch_rows: usize,
) -> Result<Vec<RecordBatch>, StatsError> {
    let url = ListingTableUrl::parse(source)
        .map_err(|error| StatsError::Internal(format!("parse Parquet input {source}: {error}")))?;
    let backend = super::shared_runtime_env()
        .object_store(url.object_store())
        .map_err(|error| {
            StatsError::Internal(format!("resolve Parquet input {source}: {error}"))
        })?;
    let reader = ParquetObjectReader::new(backend, url.prefix().clone())
        .with_file_size(byte_size)
        .with_footer_size_hint(REMOTE_FOOTER_SIZE_HINT);
    let builder = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .map_err(|error| StatsError::Internal(format!("open Parquet input {source}: {error}")))?;
    let stream = builder
        .with_batch_size(batch_rows)
        .build()
        .map_err(|error| StatsError::Internal(format!("build Parquet input {source}: {error}")))?;
    stream
        .try_collect()
        .await
        .map_err(|error| StatsError::Internal(format!("read Parquet input {source}: {error}")))
}

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

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, Int64Array};
    use arrow::compute::concat_batches;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::ObjectStoreExt;

    use super::*;

    #[tokio::test]
    async fn parquet_batches_use_the_registered_backend_and_preserve_rows() {
        let batch = RecordBatch::try_from_iter([(
            "seq",
            Arc::new(Int64Array::from(vec![3, 1, 2])) as ArrayRef,
        )])
        .unwrap();
        let bytes = crate::store::segment::write_segment(&batch).unwrap();
        let size = bytes.len() as u64;
        let backend = Arc::new(InMemory::new());
        backend
            .put(&Path::from("prefix/rows.parquet"), bytes.into())
            .await
            .unwrap();
        super::super::register_scan_object_store("s3://finelog-reader-test", backend).unwrap();

        let batches = read_parquet_batches("s3://finelog-reader-test/prefix/rows.parquet", size, 1)
            .await
            .unwrap();
        assert_eq!(concat_batches(&batch.schema(), &batches).unwrap(), batch);
    }
}
