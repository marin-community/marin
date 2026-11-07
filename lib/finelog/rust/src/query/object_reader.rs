//! Parquet adapters over Finelog's cache-aware immutable-object interface.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use bytes::Bytes;
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::listing::PartitionedFile;
use datafusion::execution::cache::cache_manager::CachedFileMetadataEntry;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_datasource_parquet::{
    CachedParquetMetaData, ParquetFileMetrics, ParquetFileReaderFactory,
};
use futures::{future::BoxFuture, FutureExt};
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};

use crate::store::object_store::{CachePins, ObjectReference, ObjectStore};

pub(crate) struct ObjectRangeReader {
    scope: Option<Arc<super::read_scope::ReadScope>>,
    cache_pins: CachePins,
    store: Arc<dyn ObjectStore>,
    reference: ObjectReference,
    metadata_size_hint: Option<usize>,
    metrics: Option<ParquetFileMetrics>,
    object_meta: Option<object_store::ObjectMeta>,
}

impl ObjectRangeReader {
    pub fn new(store: Arc<dyn ObjectStore>, reference: ObjectReference) -> Self {
        Self {
            scope: None,
            cache_pins: CachePins::default(),
            store,
            reference,
            metadata_size_hint: Some(super::file_scan::REMOTE_FOOTER_SIZE_HINT),
            metrics: None,
            object_meta: None,
        }
    }
}

impl AsyncFileReader for ObjectRangeReader {
    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        if let Some(scope) = &self.scope {
            scope.range_reads.fetch_add(1, Ordering::Relaxed);
            scope.range_bytes.fetch_add(
                ranges.iter().map(|range| range.end - range.start).sum(),
                Ordering::Relaxed,
            );
        }
        if let Some(metrics) = &self.metrics {
            metrics.bytes_scanned.add(
                ranges
                    .iter()
                    .map(|range| (range.end - range.start) as usize)
                    .sum::<usize>(),
            );
        }
        async move {
            self.store
                .read_ranges(&self.reference, &ranges)
                .await
                .map_err(|error| ParquetError::External(Box::new(error)))?
                .ok_or_else(|| {
                    ParquetError::General(format!("missing object {}", self.reference.id.as_str()))
                })
        }
        .boxed()
    }

    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        if let Some(scope) = &self.scope {
            scope.range_reads.fetch_add(1, Ordering::Relaxed);
            scope
                .range_bytes
                .fetch_add(range.end - range.start, Ordering::Relaxed);
        }
        if let Some(metrics) = &self.metrics {
            metrics
                .bytes_scanned
                .add((range.end - range.start) as usize);
        }
        async move {
            self.store
                .read_range(&self.reference, range)
                .await
                .map_err(|error| ParquetError::External(Box::new(error)))?
                .ok_or_else(|| {
                    ParquetError::General(format!("missing object {}", self.reference.id.as_str()))
                })
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        async move {
            self.store
                .pin_cached_path(&self.reference, &self.cache_pins)
                .await
                .map_err(|error| ParquetError::External(Box::new(error)))?;
            let cache = super::shared_runtime_env()
                .cache_manager
                .get_file_metadata_cache();
            if let Some(meta) = &self.object_meta {
                if let Some(entry) = cache
                    .get(&meta.location)
                    .filter(|entry| entry.is_valid_for(meta))
                {
                    if let Some(metadata) = entry
                        .file_metadata
                        .as_any()
                        .downcast_ref::<CachedParquetMetaData>()
                    {
                        return Ok(Arc::clone(metadata.parquet_metadata()));
                    }
                }
            }
            let size = self.reference.version.byte_size;
            let mut reader = ParquetMetaDataReader::new()
                .with_prefetch_hint(self.metadata_size_hint)
                .with_metadata_options(options.map(|options| options.metadata_options().clone()));
            if self.object_meta.is_some() {
                reader = reader.with_page_index_policy(PageIndexPolicy::Optional);
            } else if let Some(options) = options {
                reader = reader
                    .with_column_index_policy(options.column_index_policy())
                    .with_offset_index_policy(options.offset_index_policy());
            }
            let metadata = Arc::new(reader.load_and_finish(&mut *self, size).await?);
            if let Some(meta) = &self.object_meta {
                cache.put(
                    &meta.location,
                    CachedFileMetadataEntry::new(
                        meta.clone(),
                        Arc::new(CachedParquetMetaData::new(Arc::clone(&metadata))),
                    ),
                );
            }
            Ok(metadata)
        }
        .boxed()
    }
}

impl Drop for ObjectRangeReader {
    fn drop(&mut self) {
        if let Some(metrics) = &self.metrics {
            metrics
                .scan_efficiency_ratio
                .add_part(metrics.bytes_scanned.value());
            metrics
                .scan_efficiency_ratio
                .set_total(self.reference.version.byte_size as usize);
        }
    }
}

pub(crate) struct ObjectReaders {
    pub scope: Arc<super::read_scope::ReadScope>,
    pub store: Arc<dyn ObjectStore>,
    pub references: BTreeMap<String, ObjectReference>,
}

impl std::fmt::Debug for ObjectReaders {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObjectReaders")
            .field("objects", &self.references.len())
            .finish()
    }
}

impl ParquetFileReaderFactory for ObjectReaders {
    fn create_reader(
        &self,
        partition_index: usize,
        file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Box<dyn AsyncFileReader + Send>> {
        let reference = self
            .references
            .get(file.object_meta.location.as_ref())
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "scan has no immutable reference for {}",
                    file.object_meta.location
                ))
            })?;
        let mut reader = ObjectRangeReader::new(Arc::clone(&self.store), reference.clone());
        reader.scope = Some(Arc::clone(&self.scope));
        reader.metrics = Some(ParquetFileMetrics::new(
            partition_index,
            file.object_meta.location.as_ref(),
            metrics,
        ));
        reader.object_meta = Some(file.object_meta);
        reader.metadata_size_hint = metadata_size_hint;
        Ok(Box::new(reader))
    }
}
