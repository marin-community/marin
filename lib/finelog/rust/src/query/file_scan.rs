//! Shared Parquet input readers and rewrites over DataFusion scan files.

use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;
use parquet::arrow::async_reader::{
    AsyncFileReader, ParquetObjectReader, ParquetRecordBatchStreamBuilder,
};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::ParquetMetaData;

use crate::errors::StatsError;
use crate::store::object_store::{CachePins, ObjectReference, ObjectStore};

/// Fetch the usual object footer in one ranged read, with a second request
/// when the metadata exceeds this hint.
pub const REMOTE_FOOTER_SIZE_HINT: usize = 512 * 1024;

/// Open an exact Parquet input through the object-store registry used by queries.
pub async fn open_parquet(source: &str, byte_size: u64) -> Result<ParquetInput, StatsError> {
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
    let builder =
        ParquetRecordBatchStreamBuilder::new(Box::new(reader) as Box<dyn AsyncFileReader>)
            .await
            .map_err(|error| {
                StatsError::Internal(format!("open Parquet input {source}: {error}"))
            })?;
    Ok(ParquetInput {
        builder,
        cache_guard: None,
    })
}

/// Open an immutable input without filling cache misses. Cached files stay
/// pinned against eviction until their metadata or batches have been consumed.
pub async fn open_cached_parquet(
    store: Arc<dyn ObjectStore>,
    reference: &ObjectReference,
) -> Result<ParquetInput, StatsError> {
    let guard = CachePins::default();
    let cached = store.pin_cached_path(reference, &guard).await?.is_some();
    let reader = super::object_reader::ObjectRangeReader::new(store, reference.clone());
    let builder =
        ParquetRecordBatchStreamBuilder::new(Box::new(reader) as Box<dyn AsyncFileReader>)
            .await
            .map_err(|error| StatsError::Internal(format!("open immutable Parquet: {error}")))?;
    Ok(ParquetInput {
        builder,
        cache_guard: cached.then_some(guard),
    })
}

pub struct ParquetInput {
    builder: ParquetRecordBatchStreamBuilder<Box<dyn AsyncFileReader>>,
    cache_guard: Option<CachePins>,
}

impl ParquetInput {
    pub fn metadata(&self) -> &ParquetMetaData {
        self.builder.metadata()
    }

    pub fn is_cached(&self) -> bool {
        self.cache_guard.is_some()
    }

    /// Decode selected columns, skipping requested names absent from the file.
    pub async fn read_batches(
        self,
        batch_rows: usize,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, StatsError> {
        let Self {
            mut builder,
            cache_guard: _guard,
        } = self;
        if let Some(wanted) = columns {
            let schema = builder.parquet_schema();
            let indices =
                (0..schema.num_columns()).filter(|&i| wanted.contains(&schema.column(i).name()));
            let mask = ProjectionMask::leaves(schema, indices);
            builder = builder.with_projection(mask);
        }
        let stream = builder
            .with_batch_size(batch_rows)
            .build()
            .map_err(|error| StatsError::Internal(format!("build Parquet input: {error}")))?;
        stream
            .try_collect()
            .await
            .map_err(|error| StatsError::Internal(format!("read Parquet input: {error}")))
    }
}

/// Apply `rewrite` to every file in a Parquet scan, preserving its grouping.
///
/// Non-Parquet plans are returned unchanged.
pub fn rewrite_parquet_files(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl Fn(&PartitionedFile) -> PartitionedFile,
) -> Arc<dyn ExecutionPlan> {
    rewrite_parquet_scans(plan, |config| {
        let groups = config
            .file_groups
            .iter()
            .map(|group| FileGroup::new(group.files().iter().map(&rewrite).collect()))
            .collect();
        FileScanConfigBuilder::from(config.clone())
            .with_file_groups(groups)
            .build()
    })
}

fn rewrite_parquet_scans(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl Fn(&FileScanConfig) -> FileScanConfig,
) -> Arc<dyn ExecutionPlan> {
    plan.transform_up(|plan| {
        let Some(exec) = plan.as_any().downcast_ref::<DataSourceExec>() else {
            return Ok(Transformed::no(plan));
        };
        let Some(config) = exec.data_source().as_any().downcast_ref::<FileScanConfig>() else {
            return Ok(Transformed::no(plan));
        };
        Ok(Transformed::yes(DataSourceExec::from_data_source(rewrite(
            config,
        ))))
    })
    .expect("rewriting scan inputs preserves schemas and child arity")
    .data
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
        let batch = RecordBatch::try_from_iter([
            ("seq", Arc::new(Int64Array::from(vec![3, 1, 2])) as ArrayRef),
            (
                "unused",
                Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
            ),
        ])
        .unwrap();
        let bytes = crate::store::segment::write_segment(&batch).unwrap();
        let size = bytes.len() as u64;
        let backend = Arc::new(InMemory::new());
        backend
            .put(&Path::from("prefix/rows.parquet"), bytes.into())
            .await
            .unwrap();
        super::super::register_scan_object_store("s3://finelog-reader-test", backend).unwrap();

        let batches = open_parquet("s3://finelog-reader-test/prefix/rows.parquet", size)
            .await
            .unwrap()
            .read_batches(1, None)
            .await
            .unwrap();
        assert_eq!(concat_batches(&batch.schema(), &batches).unwrap(), batch);
        let batches = open_parquet("s3://finelog-reader-test/prefix/rows.parquet", size)
            .await
            .unwrap()
            .read_batches(1, Some(&["seq", "absent"]))
            .await
            .unwrap();
        let expected = batch.project(&[0]).unwrap();
        assert_eq!(
            concat_batches(&expected.schema(), &batches).unwrap(),
            expected
        );
    }
}
