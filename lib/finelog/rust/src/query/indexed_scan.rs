//! Snapshot-bound index preparation at DataFusion's lazy file-opening boundary.

use std::any::Any;
use std::collections::{BTreeSet, HashMap};
use std::fmt::{self, Formatter};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use datafusion::common::config::ConfigOptions;
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::{DisplayFormatType, SortOrderPushdownResult};
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_stream::{FileOpenFuture, FileOpener};
use datafusion_datasource::table_schema::TableSchema;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource_parquet::ParquetFileReaderFactory;
use futures::StreamExt;

use crate::indices::format::SectionKind;
use crate::indices::input::IndexReadScope;
use crate::indices::projection::parse_projection_reference;
use crate::indices::registry::OpenedIndexSummary;
use crate::indices::IndexRegistry;
use crate::query::exact_prune;
use crate::query::object_reader::ObjectReaders;
use crate::query::read_scope::ReadScope;
use crate::query::trigram_prune::{self, StringRange, SubstringPredicate};
use crate::store::object_store::{ObjectReference, ObjectStore, QueryCacheAdmission};
use crate::store::table::query_view::SegmentObjects;
use crate::store::table_state::LocalArtifacts;

const OPTIONAL_INDEX_BUDGET: Duration = Duration::from_millis(500);

/// Attached to an explicit file descriptor until the opener consumes it.
#[derive(Clone)]
pub(super) struct SegmentFile {
    pub objects: SegmentObjects,
    pub artifacts: LocalArtifacts,
}

#[derive(Clone)]
pub(super) struct ScanInputs {
    pub scope: Arc<ReadScope>,
    pub store: Arc<dyn ObjectStore>,
    pub indices: Arc<IndexRegistry>,
    pub readers: Arc<ObjectReaders>,
    pub admission: Arc<QueryCacheAdmission>,
    pub needles: HashMap<String, SubstringPredicate>,
    pub key_ranges: HashMap<String, StringRange>,
    pub exact: HashMap<String, Vec<String>>,
    pub required_columns: BTreeSet<String>,
}

/// Compose index opening with Parquet's optimizer and decoder implementations.
#[derive(Clone)]
pub(super) struct IndexedParquetSource {
    inner: Arc<dyn FileSource>,
    inputs: Arc<ScanInputs>,
}

impl IndexedParquetSource {
    pub fn new(inner: Arc<dyn FileSource>, inputs: Arc<ScanInputs>) -> Self {
        Self { inner, inputs }
    }

    fn wrap(&self, inner: Arc<dyn FileSource>) -> Arc<dyn FileSource> {
        Arc::new(Self::new(inner, Arc::clone(&self.inputs)))
    }
}

impl FileSource for IndexedParquetSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn object_store::ObjectStore>,
        config: &FileScanConfig,
        partition: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        Ok(Arc::new(IndexedFileOpener {
            inner: self
                .inner
                .create_file_opener(object_store, config, partition)?,
            inputs: Arc::clone(&self.inputs),
            metrics: self.inner.metrics().clone(),
            partition,
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_schema(&self) -> &TableSchema {
        self.inner.table_schema()
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        self.wrap(self.inner.with_batch_size(batch_size))
    }

    fn filter(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.inner.filter()
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        self.inner.projection()
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        self.inner.metrics()
    }

    fn file_type(&self) -> &str {
        self.inner.file_type()
    }

    fn fmt_extra(&self, format: DisplayFormatType, formatter: &mut Formatter) -> fmt::Result {
        self.inner.fmt_extra(format, formatter)?;
        write!(formatter, ", indexed_open=true")
    }

    fn supports_repartitioning(&self) -> bool {
        // Index masks and projection replacements describe whole source files.
        false
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let mut inputs = self.inputs.as_ref().clone();
        for filter in &filters {
            required_physical_columns(filter.as_ref(), &mut inputs.required_columns);
        }
        let inputs = Arc::new(inputs);
        let result = self.inner.try_pushdown_filters(filters, config)?;
        Ok(FilterPushdownPropagation {
            filters: result.filters,
            updated_node: result
                .updated_node
                .map(|inner| Arc::new(Self::new(inner, inputs)) as Arc<dyn FileSource>),
        })
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>> {
        Ok(self
            .inner
            .try_pushdown_projection(projection)?
            .map(|inner| self.wrap(inner)))
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
        properties: &EquivalenceProperties,
    ) -> Result<SortOrderPushdownResult<Arc<dyn FileSource>>> {
        Ok(self
            .inner
            .try_pushdown_sort(order, properties)?
            .map(|inner| self.wrap(inner)))
    }
}

fn required_physical_columns(expression: &dyn PhysicalExpr, columns: &mut BTreeSet<String>) {
    if let Some(column) = expression.as_any().downcast_ref::<Column>() {
        columns.insert(column.name().to_owned());
    }
    for child in expression.children() {
        required_physical_columns(child.as_ref(), columns);
    }
}

struct IndexedFileOpener {
    inner: Arc<dyn FileOpener>,
    inputs: Arc<ScanInputs>,
    metrics: ExecutionPlanMetricsSet,
    partition: usize,
}

enum SegmentRead {
    Skip,
    Source(PartitionedFile),
    Projection {
        projection: PartitionedFile,
        source: Box<PartitionedFile>,
    },
}

impl FileOpener for IndexedFileOpener {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture> {
        let inputs = Arc::clone(&self.inputs);
        let inner = Arc::clone(&self.inner);
        let metrics = self.metrics.clone();
        let partition = self.partition;
        Ok(Box::pin(async move {
            let _preparation = inputs.scope.prepare();
            let prepared = super::run_before_deadline(
                inputs.scope.deadline(),
                prepare_segment(file, &inputs, partition, &metrics),
            )
            .await?;
            let batches = super::run_before_deadline(
                inputs.scope.deadline(),
                open_segment(inner, prepared, &inputs),
            )
            .await?;
            Ok(futures::stream::try_unfold(batches, move |mut batches| {
                let inputs = Arc::clone(&inputs);
                async move {
                    match super::run_before_deadline(inputs.scope.deadline(), async {
                        batches.next().await.transpose()
                    })
                    .await?
                    {
                        Some(batch) => Ok(Some((batch, batches))),
                        None => Ok(None),
                    }
                }
            })
            .boxed())
        }))
    }
}

async fn prepare_segment(
    mut file: PartitionedFile,
    inputs: &ScanInputs,
    partition: usize,
    metrics: &ExecutionPlanMetricsSet,
) -> Result<SegmentRead> {
    let segment = file
        .extensions
        .as_ref()
        .and_then(|extension| extension.downcast_ref::<SegmentFile>())
        .ok_or_else(|| DataFusionError::Internal("indexed scan has no segment reference".into()))?
        .clone();
    file.extensions = None;
    let deadline = inputs.scope.deadline().map_or(
        tokio::time::Instant::now() + OPTIONAL_INDEX_BUDGET,
        |deadline| deadline.min(tokio::time::Instant::now() + OPTIONAL_INDEX_BUDGET),
    );
    let index_scope = IndexReadScope {
        deadline,
        stats: Arc::clone(&inputs.scope.index_reads),
    };
    let opened = if inputs.needles.is_empty() && inputs.exact.is_empty() {
        None
    } else {
        inputs
            .indices
            .open_summary_async(&segment.artifacts, &index_scope)
            .await
    };
    let mut spans: Option<(Vec<bool>, usize)> = None;
    let mut runs = None;
    if let Some(opened) = &opened {
        let key_column = opened.header.sections.iter().find_map(|section| {
            crate::indices::parse_trigram_coverage(&section.coverage)
                .filter(|coverage| !coverage.key_column.is_empty())
                .map(|coverage| coverage.key_column)
        });
        let mut columns =
            trigram_prune::ordered_trigram_needles(&inputs.needles, &inputs.key_ranges);
        columns.sort_by_key(|(column, _)| key_column.as_deref() != Some(*column));
        for (column, needles) in columns {
            let Some((coverage, index)) = inputs
                .indices
                .summary_trigram_async(opened, column, &index_scope)
                .await
            else {
                continue;
            };
            if !coverage.key_column.is_empty()
                && inputs
                    .key_ranges
                    .get(&coverage.key_column)
                    .is_some_and(|range| {
                        !coverage.key_band_overlaps(range.lo.as_deref(), range.hi.as_deref())
                    })
            {
                return Ok(SegmentRead::Skip);
            }
            let keep = needles.keep_mask(&index, coverage.span_count as usize);
            match &mut spans {
                Some((current, span_rows))
                    if current.len() == keep.len() && *span_rows == coverage.span_rows as usize =>
                {
                    for (current, keep) in current.iter_mut().zip(keep) {
                        *current &= keep;
                    }
                }
                None => spans = Some((keep, coverage.span_rows as usize)),
                Some(_) => continue,
            }
            if spans
                .as_ref()
                .is_some_and(|(keep, _)| keep.iter().all(|keep| !keep))
            {
                return Ok(SegmentRead::Skip);
            }
        }
        if let Some((projection, reference)) = projection_file(
            &file, &segment, opened, inputs, partition, metrics, deadline,
        )
        .await
        {
            warm_selected_source(inputs, &reference).await?;
            return Ok(SegmentRead::Projection {
                projection,
                source: Box::new(file),
            });
        }
        if exact_prune::exact_postings_may_cover(&opened.header, &inputs.exact) {
            if let Some(index) = inputs
                .indices
                .summary_exact_async(opened, &index_scope)
                .await
            {
                let selected = exact_prune::selected_runs(&index, &inputs.exact);
                if let Some(selected) = selected {
                    if selected.is_empty() {
                        return Ok(SegmentRead::Skip);
                    }
                    let rows = selected.iter().map(|run| run.len).sum();
                    if exact_prune::postings_are_selective(rows, index.total_rows) {
                        runs = Some(selected);
                    }
                }
            }
        }
    }
    warm_selected_source(inputs, &segment.objects.source).await?;
    if spans.is_none() && runs.is_none() {
        return Ok(SegmentRead::Source(file));
    }
    let metadata = inputs
        .readers
        .create_reader(partition, file.clone(), None, metrics)?
        .get_metadata(None)
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))?;
    if opened.as_ref().is_some_and(|opened| {
        crate::store::segment::segment_id_of(&metadata) != Some(opened.header.binding.segment_id)
    }) {
        return Ok(SegmentRead::Source(file));
    }
    let rows: Vec<usize> = metadata
        .row_groups()
        .iter()
        .map(|group| group.num_rows() as usize)
        .collect();
    let mut access = spans
        .and_then(|(keep, span_rows)| trigram_prune::span_access_plan(&keep, span_rows, &rows));
    if let Some(runs) = runs {
        let exact = exact_prune::runs_access_plan(&runs, &rows);
        access = Some(
            access
                .as_ref()
                .and_then(|current| exact_prune::intersect_access_plans(current, &exact))
                .unwrap_or(exact),
        );
    }
    if let Some(access) = access {
        file.extensions = Some(Arc::new(access));
    }
    Ok(SegmentRead::Source(file))
}

async fn open_segment(
    inner: Arc<dyn FileOpener>,
    prepared: SegmentRead,
    inputs: &ScanInputs,
) -> Result<futures::stream::BoxStream<'static, Result<arrow::array::RecordBatch>>> {
    let source = match prepared {
        SegmentRead::Skip => {
            inputs.scope.pruned.fetch_add(1, Ordering::Relaxed);
            return Ok(futures::stream::empty().boxed());
        }
        SegmentRead::Source(file) => file,
        SegmentRead::Projection { projection, source } => {
            inputs.scope.source_opens.fetch_add(1, Ordering::Relaxed);
            // Retain the source until the projection produces its first row.
            // A cached footer does not prove that its payload is still readable.
            let first: Result<_> = async {
                let mut batches = inner.open(projection)?.await?;
                while let Some(batch) = futures::TryStreamExt::try_next(&mut batches).await? {
                    if batch.num_rows() > 0 {
                        return Ok(Some((batch, batches)));
                    }
                }
                Ok(None)
            }
            .await;
            match first {
                Ok(Some((batch, batches))) => {
                    return Ok(futures::stream::once(async move { Ok(batch) })
                        .chain(batches)
                        .boxed());
                }
                Ok(None) => return Ok(futures::stream::empty().boxed()),
                Err(error) if super::is_read_deadline(&error) => return Err(error),
                Err(error) => {
                    inputs
                        .scope
                        .index_reads
                        .unavailable
                        .fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(%error, "covering projection unavailable before first row; scan source");
                }
            }
            let reference = inputs
                .readers
                .references
                .get(source.object_meta.location.as_ref())
                .ok_or_else(|| {
                    DataFusionError::Internal("projection fallback has no source reference".into())
                })?;
            warm_selected_source(inputs, reference).await?;
            *source
        }
    };
    inputs.scope.source_opens.fetch_add(1, Ordering::Relaxed);
    inner.open(source)?.await
}

async fn warm_selected_source(inputs: &ScanInputs, reference: &ObjectReference) -> Result<()> {
    if inputs
        .store
        .cached_path(reference)
        .await
        .map_err(|error| DataFusionError::External(Box::new(error)))?
        .is_none()
    {
        inputs.store.warm_query(reference, &inputs.admission);
    }
    Ok(())
}

async fn projection_file(
    source: &PartitionedFile,
    segment: &SegmentFile,
    opened: &OpenedIndexSummary,
    inputs: &ScanInputs,
    partition: usize,
    metrics: &ExecutionPlanMetricsSet,
    deadline: tokio::time::Instant,
) -> Option<(PartitionedFile, ObjectReference)> {
    let projection = opened
        .header
        .sections
        .iter()
        .filter(|section| section.kind == SectionKind::CoveringProjection)
        .filter_map(|section| parse_projection_reference(&section.coverage))
        .find(|projection| {
            exact_prune::projection_covers(projection, &inputs.exact, &inputs.required_columns)
        })?;
    let reference = ObjectReference::try_from(
        segment
            .objects
            .artifacts
            .projections
            .get(&projection.descriptor.name)?,
    )
    .ok()?;
    if reference.version.byte_size != projection.file_bytes {
        return None;
    }
    let path = inputs.store.planned_local_path(&reference.id).ok()?;
    let location = ListingTableUrl::parse(path.to_string_lossy().as_ref())
        .ok()?
        .prefix()
        .clone();
    let mut file = source.clone();
    file.object_meta.location = location;
    file.object_meta.size = reference.version.byte_size;
    file.object_meta.last_modified = std::time::SystemTime::UNIX_EPOCH.into();
    file.object_meta.e_tag = reference.version.e_tag.clone();
    file.object_meta.version = reference.version.provider_version.clone();
    file.range = None;
    file.statistics = None;
    file.ordering = None;
    file.extensions = None;
    file.metadata_size_hint = None;
    let mut reader = inputs
        .readers
        .create_reader(partition, file.clone(), None, metrics)
        .ok()?;
    let metadata = tokio::time::timeout_at(deadline, reader.get_metadata(None))
        .await
        .ok()?
        .ok()?;
    if metadata.file_metadata().num_rows() as u64 != projection.descriptor.row_count
        || crate::store::segment::segment_id_of(&metadata).map(|identity| identity.to_string())
            != Some(projection.file_segment_id)
    {
        return None;
    }
    Some((file, reference))
}
