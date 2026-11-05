//! Per-namespace DataFusion `TableProvider` over sealed parquet segments.
//!
//! A namespace's queryable data is the snapshot of its **sealed** local segment
//! files (NOT the in-RAM buffer). The durability contract (WriteRows/PushLogs
//! ack only after L0 seal+persist) makes this complete for RPC clients without
//! unioning the RAM buffer.
//!
//! An empty segment list yields a typed-empty table carrying the registered
//! arrow schema (incl. the implicit `seq` column).
//!
//! ## Read-visibility seam
//!
//! The provider holds a *snapshot* of the segments a read may open (see
//! `Namespace::query_snapshot`). A legacy table snapshots paths captured under
//! the namespace insertion lock; compaction takes the query-visibility write
//! side before unlinking a file, so a query that snapshotted the pre-compaction
//! paths keeps scanning the files it captured.
//!
//! An object-backed table instead snapshots exact object references planned
//! from an immutable `TableSnapshot`. Those objects are immutable and never
//! unlinked while referenced, so the scan prunes by the pinned metadata first
//! and reads selected objects from the cache or backing store.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DFSchema, Result as DFResult};
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfigBuilder};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::MemTable;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource_parquet::source::ParquetSource;

use crate::indices::{IndexRegistry, SegmentArtifacts};
use crate::partition_policy::{PhysicalPartitionPolicy, SegmentPartition};
use crate::query::file_scan::REMOTE_FOOTER_SIZE_HINT;
use crate::query::indexed_scan::{IndexedParquetSource, ScanInputs, SegmentFile};
use crate::store::object_store::{ObjectReference, ObjectStore, QueryCacheAdmission};
use crate::store::table::query_view::SegmentObjectMap;

/// A live namespace as one DataFusion table.
///
/// Object-backed tables plan explicit files from the snapshot. Legacy tables
/// use a filesystem `ListingTable`; empty tables retain the registered schema.
#[derive(Debug)]
pub struct NamespaceProvider {
    read_deadline: Option<tokio::time::Instant>,
    schema: SchemaRef,
    inner: Inner,
    /// The snapshotted sealed segment paths, retained so scans can locate each
    /// segment's typed index bundle. Empty for the typed-empty
    /// (no-segments) case.
    segment_paths: Vec<String>,
    segment_key_column: Option<String>,
    segment_key_bounds: BTreeMap<String, (String, String)>,
    partition_policy: Option<&'static dyn PhysicalPartitionPolicy>,
    segment_partitions: BTreeMap<String, SegmentPartition>,
    /// Exact per-segment `seq` bounds, captured with the path snapshot.
    segment_seq_bounds: BTreeMap<String, (i64, i64)>,
    /// Exact per-segment row counts. These let an unfiltered LIMIT scan avoid
    /// opening segments after a prefix that already contains enough rows.
    segment_row_counts: BTreeMap<String, u64>,
    indices: Arc<IndexRegistry>,
    /// The artifacts each snapshotted segment advertises, captured with the
    /// path snapshot. A scan opens exactly these references and never derives a
    /// bundle filename from a segment filename.
    segment_artifacts: Arc<SegmentArtifacts>,
    exact_postings_policy: Option<BTreeMap<String, Vec<String>>>,
    segment_indexes_enabled: bool,
    /// Set for an object-backed table: the immutable objects behind the planned
    /// paths, resolved only for the segments a scan selects.
    object_sources: Option<ObjectSources>,
}

impl std::fmt::Debug for ObjectSources {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObjectSources")
            .field("segments", &self.segments.len())
            .finish()
    }
}

/// The immutable objects an object-backed table's planned paths resolve to.
struct ObjectSources {
    store: Arc<dyn ObjectStore>,
    segments: SegmentObjectMap,
}

impl ObjectSources {
    fn reader_factory(
        &self,
        paths: &[String],
        scope: Arc<super::read_scope::ReadScope>,
    ) -> DFResult<Arc<super::object_reader::ObjectReaders>> {
        let mut references = BTreeMap::new();
        for objects in paths.iter().filter_map(|path| self.segments.get(path)) {
            let artifacts = objects
                .artifacts
                .projections
                .values()
                // Projection references are optional derived state. The opener
                // validates a selected projection before replacing its source.
                .filter_map(|reference| ObjectReference::try_from(reference).ok());
            for reference in std::iter::once(objects.source.clone()).chain(artifacts) {
                let local = self
                    .store
                    .planned_local_path(&reference.id)
                    .map_err(|error| {
                        datafusion::error::DataFusionError::External(Box::new(error))
                    })?;
                let mut locations = vec![local.to_string_lossy().into_owned()];
                locations.extend(self.store.remote_scan_url(&reference.id));
                for location in locations {
                    let url = ListingTableUrl::parse(location)?;
                    references.insert(url.prefix().to_string(), reference.clone());
                }
            }
        }
        Ok(Arc::new(super::object_reader::ObjectReaders {
            scope,
            store: Arc::clone(&self.store),
            references,
        }))
    }
}

#[derive(Debug)]
enum Inner {
    Listing(Arc<ListingTable>),
    Empty(Arc<MemTable>),
}

/// Retype `schema`'s top-level `Utf8` columns as `Utf8View`.
///
/// Parquet stores strings the same way either way, so this only chooses the
/// in-memory layout the scan produces. `Utf8View` lets the reader emit 16-byte
/// views over the decompressed pages rather than copying every value into one
/// contiguous buffer, and lets grouping and comparison settle most rows on the
/// inline 4-byte prefix. On a `GROUP BY name` over a low-cardinality column that
/// is worth roughly 1.8x.
///
/// Nested string types (a `Map`'s keys and values) are left alone: the win comes
/// from the wide top-level columns, and retyping map children would churn the
/// duck-typed `json_*` path for no measured gain.
///
/// [`crate::query::normalize_result`] converts back at the response boundary, so
/// the layout never reaches a client.
fn view_typed_schema(schema: &SchemaRef) -> SchemaRef {
    if !schema
        .fields()
        .iter()
        .any(|f| f.data_type() == &DataType::Utf8)
    {
        return Arc::clone(schema);
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Utf8 => f.as_ref().clone().with_data_type(DataType::Utf8View),
            _ => f.as_ref().clone(),
        })
        .collect();
    Arc::new(ArrowSchema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    ))
}

impl NamespaceProvider {
    async fn scan_objects(
        &self,
        sources: &ObjectSources,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let paths = self.segment_paths_for_scan(filters, limit);
        if paths.is_empty() {
            return MemTable::try_new(Arc::clone(&self.schema), vec![vec![]])?
                .scan(state, projection, filters, limit)
                .await;
        }
        let admission = state
            .config()
            .get_extension::<QueryCacheAdmission>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(
                    "query cache admission is missing from SessionConfig".into(),
                )
            })?;
        let scope = state
            .config()
            .get_extension::<super::read_scope::ReadScope>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(
                    "read scope is missing from SessionConfig".into(),
                )
            })?;
        scope.constrain(self.read_deadline);
        scope
            .candidates
            .fetch_add(paths.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let readers = sources.reader_factory(&paths, Arc::clone(&scope))?;
        let mut exact = crate::query::exact_prune::values_by_column(filters);
        if let Some(policy) = &self.exact_postings_policy {
            exact.retain(|column, values| {
                policy.get(column).is_some_and(|indexed| {
                    values
                        .iter()
                        .all(|value| indexed.binary_search(value).is_ok())
                })
            });
        }
        let mut required_columns: BTreeSet<_> = projection
            .map(|columns| {
                columns
                    .iter()
                    .map(|&column| self.schema.field(column).name().clone())
                    .collect()
            })
            .unwrap_or_else(|| {
                self.schema
                    .fields()
                    .iter()
                    .map(|field| field.name().clone())
                    .collect()
            });
        required_columns.extend(
            filters
                .iter()
                .flat_map(Expr::column_refs)
                .map(|column| column.name.clone()),
        );
        let inputs = Arc::new(ScanInputs {
            scope,
            store: Arc::clone(&sources.store),
            indices: Arc::clone(&self.indices),
            readers: Arc::clone(&readers),
            admission,
            needles: if self.segment_indexes_enabled {
                crate::query::trigram_prune::substring_needles_by_column(filters)
            } else {
                HashMap::new()
            },
            key_ranges: crate::query::trigram_prune::string_column_ranges(filters),
            exact: if self.segment_indexes_enabled {
                exact
            } else {
                HashMap::new()
            },
            required_columns,
        });
        let mut parquet = ParquetSource::new(Arc::clone(&self.schema))
            .with_metadata_size_hint(REMOTE_FOOTER_SIZE_HINT)
            .with_parquet_file_reader_factory(readers);
        if let Some(predicate) = filters.iter().cloned().reduce(Expr::and) {
            let schema = DFSchema::try_from(self.schema.as_ref().clone())?;
            parquet = parquet.with_predicate(state.create_physical_expr(predicate, &schema)?);
        }
        let source = Arc::new(IndexedParquetSource::new(Arc::new(parquet), inputs));
        let partitions = paths.len().min(state.config().target_partitions()).max(1);
        let mut groups = vec![Vec::new(); partitions];
        for (index, path) in paths.iter().enumerate() {
            let objects = sources.segments.get(path).ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "missing object reference for {path}"
                ))
            })?;
            let location = ListingTableUrl::parse(path)?.prefix().to_string();
            let mut file = PartitionedFile::new(location, objects.source.version.byte_size);
            file.object_meta.e_tag = objects.source.version.e_tag.clone();
            file.object_meta.version = objects.source.version.provider_version.clone();
            file.extensions = Some(Arc::new(SegmentFile {
                objects: objects.clone(),
                artifacts: self
                    .segment_artifacts
                    .get(path)
                    .cloned()
                    .unwrap_or_default(),
            }));
            groups[index % partitions].push(file);
        }
        let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
            .with_file_groups(groups.into_iter().map(FileGroup::new).collect())
            .with_limit(limit)
            .with_projection_indices(projection.cloned())?
            .build();
        Ok(DataSourceExec::from_data_source(config))
    }

    pub fn segment_paths(&self) -> &[String] {
        &self.segment_paths
    }

    pub fn indices(&self) -> &Arc<IndexRegistry> {
        &self.indices
    }

    pub fn segment_artifacts(&self) -> &Arc<SegmentArtifacts> {
        &self.segment_artifacts
    }

    fn listing_table(schema: SchemaRef, segment_paths: &[String]) -> DFResult<Arc<ListingTable>> {
        let urls: Vec<ListingTableUrl> = segment_paths
            .iter()
            .map(|path| ListingTableUrl::parse(format!("file://{path}")))
            .collect::<DFResult<Vec<_>>>()?;
        let options =
            ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
        let config = ListingTableConfig::new_with_multi_paths(urls)
            .with_listing_options(options)
            .with_schema(schema);
        Ok(Arc::new(ListingTable::try_new(config)?))
    }

    fn segment_paths_for_filters(&self, filters: &[Expr]) -> Vec<String> {
        let int_ranges = crate::query::predicate::int_column_ranges(filters);
        let string_ranges = crate::query::trigram_prune::string_column_ranges(filters);
        let int_key_range = self
            .segment_key_column
            .as_ref()
            .and_then(|column| int_ranges.get(column));
        let string_key_range = self
            .segment_key_column
            .as_ref()
            .and_then(|column| string_ranges.get(column));
        let exact_values = crate::query::exact_prune::values_by_column(filters);
        let partition_candidates = self
            .partition_policy
            .and_then(|policy| policy.partitions_for_exact_values(&exact_values));
        let seq_range = int_ranges.get("seq");
        let paths = self
            .segment_paths
            .iter()
            .filter(|path| {
                let key_matches =
                    self.segment_key_bounds
                        .get(path.as_str())
                        .is_none_or(|(minimum, maximum)| {
                            let int_matches = int_key_range.is_none_or(|range| {
                                let Ok(minimum) = minimum.parse::<i64>() else {
                                    return true;
                                };
                                let Ok(maximum) = maximum.parse::<i64>() else {
                                    return true;
                                };
                                minimum > maximum || range.overlaps(minimum, maximum)
                            });
                            let string_matches = string_key_range.is_none_or(|range| {
                                let minimum = minimum.as_bytes();
                                let maximum = maximum.as_bytes();
                                minimum > maximum
                                    || (range.lo.as_deref().is_none_or(|lower| maximum >= lower)
                                        && range.hi.as_deref().is_none_or(|upper| minimum <= upper))
                            });
                            int_matches && string_matches
                        });
                let seq_matches = seq_range.is_none_or(|range| {
                    self.segment_seq_bounds
                        .get(path.as_str())
                        .is_none_or(|&(minimum, maximum)| {
                            minimum > maximum || range.overlaps(minimum, maximum)
                        })
                });
                let partition_matches = partition_candidates.as_ref().is_none_or(|candidates| {
                    let Some(policy) = self.partition_policy else {
                        return true;
                    };
                    self.segment_partitions
                        .get(path.as_str())
                        .is_none_or(|partition| {
                            !policy.is_current_partition(partition)
                                || candidates.contains(partition)
                        })
                });
                key_matches && seq_matches && partition_matches
            })
            .cloned()
            .collect::<Vec<_>>();
        if paths.len() != self.segment_paths.len() {
            tracing::debug!(
                segments_total = self.segment_paths.len(),
                segments_selected = paths.len(),
                "scoped segment planning to key range, seq range, and physical partitions"
            );
        }
        paths
    }

    fn segment_paths_for_scan(&self, filters: &[Expr], limit: Option<usize>) -> Vec<String> {
        let paths = self.segment_paths_for_filters(filters);
        let Some(limit) = limit else {
            return paths;
        };
        if !filters.is_empty() {
            return paths;
        }
        if limit == 0 {
            return Vec::new();
        }

        let mut rows = 0_u64;
        for (index, path) in paths.iter().enumerate() {
            let Some(&segment_rows) = self.segment_row_counts.get(path) else {
                // A missing count cannot prove any prefix satisfies the LIMIT.
                return paths;
            };
            rows = rows.saturating_add(segment_rows);
            if rows >= limit as u64 {
                return paths[..=index].to_vec();
            }
        }
        paths
    }

    /// Build a provider from the registered arrow `schema` and a snapshot of
    /// sealed segment file paths.
    ///
    /// `segment_paths` are absolute local filesystem paths
    /// (`{ns_dir}/seg_L*_*.parquet`). Each is registered individually (rather
    /// than listing a directory) so the scan sees exactly the snapshotted set —
    /// no re-listing, and compaction can't slip a new file in.
    pub fn build(
        schema: SchemaRef,
        segment_paths: &[String],
        indices: Arc<IndexRegistry>,
    ) -> DFResult<NamespaceProvider> {
        let schema = view_typed_schema(&schema);
        if segment_paths.is_empty() {
            let mem = MemTable::try_new(Arc::clone(&schema), vec![vec![]])?;
            return Ok(NamespaceProvider {
                read_deadline: None,
                schema,
                inner: Inner::Empty(Arc::new(mem)),
                segment_paths: Vec::new(),
                segment_key_column: None,
                segment_key_bounds: BTreeMap::new(),
                partition_policy: None,
                segment_partitions: BTreeMap::new(),
                segment_seq_bounds: BTreeMap::new(),
                segment_row_counts: BTreeMap::new(),
                indices,
                segment_artifacts: Arc::new(SegmentArtifacts::new()),
                exact_postings_policy: None,
                segment_indexes_enabled: true,
                object_sources: None,
            });
        }

        let listing = Self::listing_table(Arc::clone(&schema), segment_paths)?;
        Ok(NamespaceProvider {
            read_deadline: None,
            schema,
            inner: Inner::Listing(listing),
            segment_paths: segment_paths.to_vec(),
            segment_key_column: None,
            segment_key_bounds: BTreeMap::new(),
            partition_policy: None,
            segment_partitions: BTreeMap::new(),
            segment_seq_bounds: BTreeMap::new(),
            segment_row_counts: BTreeMap::new(),
            indices,
            segment_artifacts: Arc::new(SegmentArtifacts::new()),
            exact_postings_policy: None,
            segment_indexes_enabled: true,
            object_sources: None,
        })
    }

    /// Attach the immutable objects the planned paths resolve to.
    ///
    /// With these attached the provider's paths need not exist when the plan is
    /// built: planning uses pinned metadata and execution opens selected files
    /// through the unified object reader.
    pub fn with_object_sources(
        mut self,
        store: Arc<dyn ObjectStore>,
        segments: SegmentObjectMap,
    ) -> Self {
        if !segments.is_empty() {
            self.indices = Arc::new(
                self.indices
                    .with_object_sources(Arc::clone(&store), &segments),
            );
            self.object_sources = Some(ObjectSources { store, segments });
        }
        self
    }

    /// Attach exact key bounds captured with the segment path snapshot.
    /// Paths missing from `bounds` remain queryable for scan safety.
    pub fn with_segment_key_bounds(
        mut self,
        key_column: impl Into<String>,
        bounds: BTreeMap<String, (String, String)>,
    ) -> Self {
        self.segment_key_column = Some(key_column.into());
        self.segment_key_bounds = bounds;
        self
    }

    /// Attach exact per-segment `seq` bounds captured with the path snapshot.
    ///
    /// Segments partition the sequence space disjointly, so a `seq`-bounded
    /// predicate (a forwarder cursor read, a tail query) selects only the
    /// segments whose range it overlaps. An object-backed scan opens only
    /// the selected segments.
    pub fn with_segment_seq_bounds(mut self, bounds: BTreeMap<String, (i64, i64)>) -> Self {
        self.segment_seq_bounds = bounds;
        self
    }

    /// Attach exact row counts captured with the segment path snapshot.
    pub fn with_segment_row_counts(mut self, row_counts: BTreeMap<String, u64>) -> Self {
        self.segment_row_counts = row_counts;
        self
    }

    pub(crate) fn with_read_deadline(mut self, deadline: Option<tokio::time::Instant>) -> Self {
        self.read_deadline = deadline;
        self
    }

    pub(crate) fn read_deadline(&self) -> Option<tokio::time::Instant> {
        self.read_deadline
    }

    /// Attach hidden physical partitions captured with the path snapshot.
    pub fn with_segment_partitions(
        mut self,
        policy: Option<&'static dyn PhysicalPartitionPolicy>,
        partitions: BTreeMap<String, SegmentPartition>,
    ) -> Self {
        self.partition_policy = policy;
        self.segment_partitions = partitions;
        self
    }

    /// Attach the artifact references the snapshotted segments advertise.
    ///
    /// A segment missing from `artifacts` advertises nothing, so its scan reads
    /// the source Parquet.
    pub fn with_segment_artifacts(mut self, artifacts: SegmentArtifacts) -> Self {
        self.segment_artifacts = Arc::new(artifacts);
        self
    }

    /// Supply the registered values for which segment indexes may contain exact postings.
    pub fn with_exact_postings_policy(mut self, mut policy: BTreeMap<String, Vec<String>>) -> Self {
        for values in policy.values_mut() {
            values.sort();
            values.dedup();
        }
        self.exact_postings_policy = Some(policy);
        self
    }

    /// A provider whose segments advertise the sidecars they carry locally.
    #[cfg(test)]
    pub fn build_with_local_artifacts(
        schema: SchemaRef,
        segment_paths: &[String],
    ) -> DFResult<NamespaceProvider> {
        Ok(
            Self::build(schema, segment_paths, crate::indices::test_index_registry())?
                .with_segment_artifacts(crate::indices::sidecar_artifacts(segment_paths)),
        )
    }

    /// Enable or disable every derived segment index for this provider.
    ///
    /// Source Parquet remains authoritative. A disabled managed policy ignores
    /// any derived files that an older schema left beside its segments.
    pub fn with_segment_indexes_enabled(mut self, enabled: bool) -> Self {
        self.segment_indexes_enabled = enabled;
        self
    }
}

#[async_trait]
impl TableProvider for NamespaceProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // Inexact: DataFusion re-checks the filters, but the parquet scan can
        // still prune row groups from them.
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        if let Some(sources) = &self.object_sources {
            return self
                .scan_objects(sources, state, projection, filters, limit)
                .await;
        }
        match &self.inner {
            Inner::Listing(t) => {
                let mut segment_paths = self.segment_paths_for_scan(filters, limit);
                let needles = crate::query::trigram_prune::substring_needles_by_column(filters);
                let key_ranges = crate::query::trigram_prune::string_column_ranges(filters);
                let integer_ranges = crate::query::predicate::int_column_ranges(filters);
                let admission = state.config().get_extension::<QueryCacheAdmission>();
                if self.segment_indexes_enabled && !needles.is_empty() && !integer_ranges.is_empty()
                {
                    segment_paths = tokio::task::spawn_blocking(move || {
                        segment_paths
                            .into_iter()
                            .filter(|path| {
                                crate::store::segment::segment_integer_bounds(std::path::Path::new(
                                    path,
                                ))
                                .is_none_or(|bounds| {
                                    integer_ranges.iter().all(|(column, range)| {
                                        bounds.get(column).is_none_or(|&(minimum, maximum)| {
                                            range.overlaps(minimum, maximum)
                                        })
                                    })
                                })
                            })
                            .collect()
                    })
                    .await
                    .map_err(|error| {
                        datafusion::error::DataFusionError::Execution(format!(
                            "Parquet range prune task join: {error}"
                        ))
                    })?;
                }
                let mut trigram_pruning = None;
                if self.segment_indexes_enabled && !needles.is_empty() {
                    let indices = Arc::clone(&self.indices);
                    let artifacts = Arc::clone(&self.segment_artifacts);
                    let paths = segment_paths;
                    let prune_needles = needles.clone();
                    let pruning = tokio::task::spawn_blocking(move || {
                        crate::query::trigram_prune::plan_segments(
                            &paths,
                            &prune_needles,
                            &key_ranges,
                            &indices,
                            &artifacts,
                        )
                    })
                    .await
                    .map_err(|error| {
                        datafusion::error::DataFusionError::Execution(format!(
                            "trigram segment prune task join: {error}"
                        ))
                    })?;
                    segment_paths = pruning.paths.clone();
                    trigram_pruning = Some(pruning);
                }
                // Delegate to DataFusion's parquet scan (which keeps the existing
                // range / min-max row-group pruning), then layer bundle-backed
                // filtered projections or access plans onto its files.
                let plan = if segment_paths.is_empty() {
                    MemTable::try_new(Arc::clone(&self.schema), vec![vec![]])?
                        .scan(state, projection, filters, limit)
                        .await?
                } else if segment_paths.len() == self.segment_paths.len() {
                    t.scan(state, projection, filters, limit).await?
                } else {
                    Self::listing_table(Arc::clone(&self.schema), &segment_paths)?
                        .scan(state, projection, filters, limit)
                        .await?
                };
                if !self.segment_indexes_enabled {
                    return Ok(plan);
                }
                let mut exact = crate::query::exact_prune::values_by_column(filters);
                if let Some(policy) = &self.exact_postings_policy {
                    exact.retain(|column, values| {
                        policy.get(column).is_some_and(|indexed| {
                            values
                                .iter()
                                .all(|value| indexed.binary_search(value).is_ok())
                        })
                    });
                }
                if needles.is_empty() && exact.is_empty() {
                    return Ok(plan);
                }
                // Key ranges (incl. the analyzer's synthesized prefix bounds) scope
                // which segments' sections the prune reads — cheap expr inspection,
                // done here before the blocking work.
                let mut required_columns: BTreeSet<String> = projection
                    .map(|indices| {
                        indices
                            .iter()
                            .map(|&index| self.schema.field(index).name().clone())
                            .collect()
                    })
                    .unwrap_or_else(|| {
                        self.schema
                            .fields()
                            .iter()
                            .map(|field| field.name().clone())
                            .collect()
                    });
                required_columns.extend(
                    filters
                        .iter()
                        .flat_map(Expr::column_refs)
                        .map(|column| column.name.clone()),
                );
                // Bundle + footer reads are blocking, so run pruning off the
                // async worker.
                let indices = Arc::clone(&self.indices);
                let artifacts = Arc::clone(&self.segment_artifacts);
                tokio::task::spawn_blocking(move || {
                    let _pins = admission;
                    let plan = match trigram_pruning {
                        Some(pruning) => {
                            crate::query::trigram_prune::apply_pruning(plan, &pruning, &indices)
                        }
                        None => plan,
                    };
                    crate::query::exact_prune::apply(
                        plan,
                        &segment_paths,
                        &exact,
                        &required_columns,
                        &indices,
                        &artifacts,
                    )
                })
                .await
                .map_err(|e| {
                    datafusion::error::DataFusionError::Execution(format!(
                        "trigram prune task join: {e}"
                    ))
                })
            }
            Inner::Empty(t) => t.scan(state, projection, filters, limit).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::os::unix::fs::FileExt;
    use std::sync::Arc;

    use arrow::array::{Array, Int64Array, ListArray, StringArray};

    use crate::indices::trigram::SIDECAR_SPAN_ROWS;
    use crate::query::string_values::StringValues;
    use arrow::datatypes::{DataType, Field, Int64Type, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::datasource::physical_plan::FileScanConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::execution::FunctionRegistry;
    use datafusion::logical_expr::{col, expr::ScalarFunction, lit};
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::levanter_metrics_policy::LEVANTER_RUN_PARTITION_POLICY;
    use crate::partition_policy::PhysicalPartitionPolicy;
    use crate::store::object_store::ObjectId;
    use crate::store::segment::write_segment_to_dir;

    fn tempdir(tag: &str) -> std::path::PathBuf {
        crate::test_support::unique_dir(tag)
    }

    /// Store-form worker arrow schema: seq, worker_id, mem_bytes.
    fn worker_arrow() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("mem_bytes", DataType::Int64, false),
        ]))
    }

    fn worker_batch(first_seq: i64, ids: Vec<&str>, mem: Vec<i64>) -> RecordBatch {
        let n = ids.len() as i64;
        RecordBatch::try_new(
            worker_arrow(),
            vec![
                Arc::new(Int64Array::from_iter_values(first_seq..first_seq + n)),
                Arc::new(StringArray::from(ids)),
                Arc::new(Int64Array::from(mem)),
            ],
        )
        .unwrap()
    }

    struct ObjectScanFixture {
        provider: NamespaceProvider,
        remote: Arc<crate::test_support::FaultInjectingObjectStore>,
        ids: Vec<ObjectId>,
        root: std::path::PathBuf,
    }

    async fn sparse_object_fixture(indexed: bool) -> ObjectScanFixture {
        let dir = tempdir("lazy_filtered_limit");
        let remote = crate::test_support::FaultInjectingObjectStore::new(Arc::new(
            crate::store::object_store::build_remote_object_store(
                dir.join("remote").to_str().unwrap(),
            )
            .unwrap()
            .unwrap(),
        ));
        let store = Arc::new(
            crate::store::object_store::CachedObjectStore::new(
                remote.clone(),
                dir.join("cache"),
                None,
            )
            .unwrap(),
        );
        let mut paths = Vec::new();
        let mut objects = BTreeMap::new();
        let mut ids = Vec::new();
        let mut local_artifacts = BTreeMap::new();
        for index in 0..32 {
            let id = ObjectId::table("workers", &format!("objects/{index:03}.parquet")).unwrap();
            let batch = worker_batch(
                index * 128 + 1,
                vec![if index == 1 { "inference" } else { "heartbeat" }; 128],
                (0..128).collect(),
            );
            let bytes = crate::store::segment::write_segment(&batch).unwrap();
            let version = if index < 2 {
                remote.write(&id, bytes.clone().into()).await.unwrap()
            } else {
                // The suffix is deliberately absent. Planning it must cost no
                // I/O, and satisfying LIMIT must not require opening it all.
                crate::store::object_store::ObjectVersion {
                    e_tag: None,
                    provider_version: None,
                    byte_size: bytes.len() as u64,
                    local_value: None,
                }
            };
            let path = store
                .planned_local_path(&id)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            let mut artifacts = crate::store::table_state::ArtifactReferences::default();
            if indexed && index < 2 {
                let source_path = dir.join(format!("indexed-{index}.parquet"));
                std::fs::write(&source_path, &bytes).unwrap();
                crate::indices::write_segment_index(
                    &source_path,
                    &[batch],
                    &crate::indices::SegmentIndexConfig::from_policies(
                        ["worker_id"],
                        &[],
                        &[],
                        None,
                    ),
                )
                .unwrap();
                let built = crate::indices::sidecar_artifacts(&[source_path.to_string_lossy()]);
                let local = built.values().next().unwrap();
                let bundle_id =
                    ObjectId::table("workers", &format!("indices/{index:03}.fidx")).unwrap();
                let version = remote
                    .write(
                        &bundle_id,
                        bytes::Bytes::from(std::fs::read(local.bundle.as_ref().unwrap()).unwrap()),
                    )
                    .await
                    .unwrap();
                artifacts.binding = local.binding.clone();
                artifacts.bundle = Some(
                    crate::proto::finelog::stats::ObjectRef::default()
                        .with_object_id(bundle_id.as_str())
                        .with_byte_size(version.byte_size),
                );
                local_artifacts.insert(
                    path.clone(),
                    crate::store::table_state::LocalArtifacts {
                        binding: local.binding.clone(),
                        bundle: Some(store.planned_local_path(&bundle_id).unwrap()),
                        projections: BTreeMap::new(),
                    },
                );
            }
            paths.push(path.clone());
            ids.push(id.clone());
            objects.insert(
                path,
                crate::store::table::query_view::SegmentObjects {
                    source: ObjectReference { id, version },
                    artifacts,
                },
            );
        }
        let provider = NamespaceProvider::build(
            worker_arrow(),
            &paths,
            crate::indices::test_index_registry(),
        )
        .unwrap()
        .with_segment_artifacts(local_artifacts)
        .with_object_sources(store, objects);
        ObjectScanFixture {
            provider,
            remote,
            ids,
            root: dir,
        }
    }

    fn object_scan_context(provider: NamespaceProvider) -> SessionContext {
        let config = crate::query::make_ctx()
            .copied_config()
            .with_target_partitions(1)
            .with_extension(Arc::new(QueryCacheAdmission::new(0, 0)));
        let ctx = SessionContext::new_with_config(config);
        crate::query::udf::register_scalar_udfs(&ctx);
        ctx.register_table("workers", Arc::new(provider)).unwrap();
        ctx
    }

    #[tokio::test]
    async fn object_scan_plans_without_reads_and_stops_after_exact_matches() {
        let ObjectScanFixture {
            provider,
            remote,
            ids,
            root: dir,
        } = sparse_object_fixture(false).await;
        let ctx = object_scan_context(provider);
        remote.clear_calls();
        let sql = "SELECT mem_bytes FROM workers WHERE worker_id LIKE '%inference%' AND mem_bytes % 2 = 0 LIMIT 7 OFFSET 3";
        ctx.sql(&format!("EXPLAIN {sql}"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert!(remote
            .keys_for(crate::test_support::ObjectOp::ReadRange)
            .is_empty());
        assert!(remote
            .keys_for(crate::test_support::ObjectOp::Read)
            .is_empty());
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let values: Vec<_> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(values, vec![6, 8, 10, 12, 14, 16, 18]);
        let read: BTreeSet<_> = remote
            .keys_for(crate::test_support::ObjectOp::ReadRange)
            .into_iter()
            .collect();
        assert!(read.contains(ids[0].as_str()));
        assert!(read.contains(ids[1].as_str()));
        assert!(read
            .iter()
            .all(|key| ids[..3].iter().any(|id| id.as_str() == key)));
        assert!(ctx
            .sql("SELECT count(*) FROM workers")
            .await
            .unwrap()
            .collect()
            .await
            .is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn each_file_gets_an_optional_index_budget_after_an_earlier_timeout() {
        let ObjectScanFixture {
            provider,
            remote,
            ids,
            root,
        } = sparse_object_fixture(true).await;
        // Only two files are candidates; the second source is absent, so its
        // usable index must still exclude it after the first index times out.
        let bounds = BTreeMap::from_iter(provider.segment_paths().iter().enumerate().map(
            |(index, path)| {
                (
                    path.clone(),
                    (index as i64 * 128 + 1, (index as i64 + 1) * 128),
                )
            },
        ));
        let provider = provider.with_segment_seq_bounds(bounds);
        remote.delete(&ids[1]).await.unwrap();
        let first_bundle = ObjectId::table("workers", "indices/000.fidx").unwrap();
        let second_bundle = ObjectId::table("workers", "indices/001.fidx").unwrap();
        let gate = crate::test_support::FaultGate::new();
        remote.arm(crate::test_support::ObjectFault::new(
            crate::test_support::ObjectOp::ReadRange,
            crate::test_support::ObjectPattern::EndsWith(first_bundle.as_str().into()),
            crate::test_support::FaultAction::Park(Arc::clone(&gate)),
        ));
        let ctx = object_scan_context(provider);
        let frame = ctx
            .sql("SELECT mem_bytes FROM workers WHERE seq <= 256 AND worker_id LIKE '%unicorn%'")
            .await
            .unwrap();
        tokio::time::pause();
        let query = tokio::spawn(async move { frame.collect().await });
        gate.entered().await;
        tokio::time::advance(std::time::Duration::from_millis(501)).await;
        tokio::time::resume();
        gate.release();
        let rows = query.await.unwrap().unwrap();
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        assert!(!remote.ranges_for(&ids[0]).is_empty());
        assert!(!remote.ranges_for(&second_bundle).is_empty());
        assert!(remote.ranges_for(&ids[1]).is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn object_scan_retention_deadline_includes_time_spent_planning() {
        let ObjectScanFixture {
            provider,
            remote,
            root,
            ..
        } = sparse_object_fixture(false).await;
        let provider = provider.with_read_deadline(Some(
            tokio::time::Instant::now() + std::time::Duration::from_secs(1),
        ));
        let ctx = object_scan_context(provider);
        remote.clear_calls();
        let plan = ctx
            .sql("SELECT mem_bytes FROM workers LIMIT 1")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        tokio::time::advance(std::time::Duration::from_secs(2)).await;
        let error = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .unwrap_err();
        assert!(crate::query::is_read_deadline(&error));
        assert!(remote
            .keys_for(crate::test_support::ObjectOp::ReadRange)
            .is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cancelling_a_parked_object_scan_does_not_block_a_later_read() {
        let ObjectScanFixture {
            provider,
            remote,
            ids,
            root,
        } = sparse_object_fixture(false).await;
        let gate = crate::test_support::FaultGate::new();
        remote.arm(crate::test_support::ObjectFault::new(
            crate::test_support::ObjectOp::ReadRange,
            crate::test_support::ObjectPattern::EndsWith(ids[0].as_str().to_string()),
            crate::test_support::FaultAction::Park(Arc::clone(&gate)),
        ));
        let ctx = object_scan_context(provider);
        let frame = ctx
            .sql("SELECT mem_bytes FROM workers LIMIT 1")
            .await
            .unwrap();
        let query = tokio::spawn(async move { frame.collect().await });
        gate.entered().await;
        query.abort();
        assert!(query.await.unwrap_err().is_cancelled());
        gate.release();
        let rows = ctx
            .sql("SELECT mem_bytes FROM workers LIMIT 1")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_run_id_prunes_current_partitions_but_keeps_transition_files() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("run_id", DataType::Utf8, false),
        ]));
        let paths = vec![
            "/tmp/current-run-a.parquet".to_string(),
            "/tmp/current-run-b.parquet".to_string(),
            "/tmp/unpartitioned.parquet".to_string(),
            "/tmp/old-spec.parquet".to_string(),
        ];
        let partition_for = |run_id: &str| {
            LEVANTER_RUN_PARTITION_POLICY
                .partitions_for_exact_values(&HashMap::from([(
                    "run_id".to_string(),
                    vec![run_id.to_string()],
                )]))
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
        };
        let mut old_spec = partition_for("run-a");
        old_spec.spec_id = 0;
        let provider = NamespaceProvider::build_with_local_artifacts(schema, &paths)
            .unwrap()
            .with_segment_partitions(
                Some(&LEVANTER_RUN_PARTITION_POLICY),
                BTreeMap::from([
                    (paths[0].clone(), partition_for("run-a")),
                    (paths[1].clone(), partition_for("run-b")),
                    (paths[3].clone(), old_spec),
                ]),
            );

        assert_eq!(
            provider.segment_paths_for_filters(&[col("run_id").eq(lit("run-a"))]),
            vec![paths[0].clone(), paths[2].clone(), paths[3].clone()]
        );
        assert_eq!(provider.segment_paths_for_filters(&[]), paths);
    }

    #[test]
    fn unfiltered_limit_selects_the_smallest_sufficient_segment_prefix() {
        let paths: Vec<String> = (0..3)
            .map(|index| format!("/planned/seg_{index}.parquet"))
            .collect();
        let provider = NamespaceProvider::build_with_local_artifacts(worker_arrow(), &paths)
            .unwrap()
            .with_segment_row_counts(BTreeMap::from([
                (paths[0].clone(), 40),
                (paths[1].clone(), 70),
                (paths[2].clone(), 100),
            ]));

        assert!(provider.segment_paths_for_scan(&[], Some(0)).is_empty());
        assert_eq!(provider.segment_paths_for_scan(&[], Some(100)), paths[..2]);
        assert_eq!(provider.segment_paths_for_scan(&[], None), paths);
    }

    #[test]
    fn limit_pruning_is_conservative_with_filters_or_missing_counts() {
        let paths: Vec<String> = (0..3)
            .map(|index| format!("/planned/seg_{index}.parquet"))
            .collect();
        let provider = NamespaceProvider::build_with_local_artifacts(worker_arrow(), &paths)
            .unwrap()
            .with_segment_row_counts(BTreeMap::from([
                (paths[0].clone(), 100),
                (paths[2].clone(), 100),
            ]));

        assert_eq!(provider.segment_paths_for_scan(&[], Some(101)), paths);
        assert_eq!(
            provider.segment_paths_for_scan(&[col("mem_bytes").gt(lit(0_i64))], Some(1)),
            paths
        );
    }

    /// A `seq`-bounded predicate selects segments with overlapping ranges.
    /// A segment without recorded bounds is always selected.
    #[tokio::test]
    async fn a_seq_range_selects_overlapping_segments() {
        let schema = worker_arrow();
        let paths: Vec<String> = (0..3)
            .map(|i| format!("/planned/seg_{i}.parquet"))
            .collect();
        let provider = NamespaceProvider::build_with_local_artifacts(schema, &paths)
            .unwrap()
            .with_segment_seq_bounds(BTreeMap::from([
                (paths[0].clone(), (1, 100)),
                (paths[1].clone(), (101, 200)),
                // paths[2] has no recorded bounds and must never be pruned.
            ]));

        let range = col("seq")
            .gt_eq(lit(150i64))
            .and(col("seq").lt_eq(lit(160i64)));
        assert_eq!(
            provider.segment_paths_for_filters(&[range]),
            vec![paths[1].clone(), paths[2].clone()]
        );
        assert_eq!(
            provider.segment_paths_for_filters(&[col("seq").gt(lit(500i64))]),
            vec![paths[2].clone()]
        );
        assert_eq!(provider.segment_paths_for_filters(&[]), paths);
    }

    #[tokio::test]
    async fn empty_namespace_scans_zero_rows_typed() {
        let schema = worker_arrow();
        let provider =
            NamespaceProvider::build_with_local_artifacts(Arc::clone(&schema), &[]).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.worker"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql("SELECT * FROM \"iris.worker\"")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
        // Typed empty: the registered column set incl. `seq` survives.
        let result_schema = ctx
            .sql("SELECT * FROM \"iris.worker\"")
            .await
            .unwrap()
            .schema()
            .as_arrow()
            .clone();
        let names: Vec<&str> = result_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, vec!["seq", "worker_id", "mem_bytes"]);
    }

    #[tokio::test]
    async fn sealed_segments_scan_with_projection_and_order() {
        let dir = tempdir("scan");
        // Two segments out of seq order to prove the listing reads both.
        write_segment_to_dir(
            &dir,
            0,
            1,
            &worker_batch(1, vec!["w-1", "w-2"], vec![100, 200]),
        )
        .unwrap();
        write_segment_to_dir(&dir, 0, 3, &worker_batch(3, vec!["w-3"], vec![300])).unwrap();
        let paths: Vec<String> = crate::store::segment::discover_segments(&dir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        assert_eq!(paths.len(), 2);

        let provider =
            NamespaceProvider::build_with_local_artifacts(worker_arrow(), &paths).unwrap();
        let ctx = SessionContext::new();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.worker"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql("SELECT worker_id, mem_bytes FROM \"iris.worker\" ORDER BY worker_id")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let ids = first_column_strings(&batches);
        assert_eq!(ids, vec!["w-1", "w-2", "w-3"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn exact_predicate_uses_complete_filtered_projections() {
        let dir = tempdir("exact_projection");
        let batch = worker_batch(1, vec!["w-1", "w-2", "w-3"], vec![100, 200, 300]);
        let (path, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let config = crate::indices::exact::ExactIndexConfig {
            column: "worker_id".to_string(),
            exact_values: vec!["w-2".to_string()],
            value_counts: false,
        };
        crate::indices::write_segment_index(
            &path,
            &[batch],
            &crate::indices::SegmentIndexConfig::from_policies(
                Vec::<String>::new(),
                &[config],
                &[crate::store::schema::CoveringProjection::new(
                    "workers",
                    "worker_id",
                    ["w-2"],
                    ["seq", "worker_id", "mem_bytes"],
                )],
                None,
            ),
        )
        .unwrap();

        let provider = NamespaceProvider::build_with_local_artifacts(
            worker_arrow(),
            &[path.to_string_lossy().into_owned()],
        )
        .unwrap();
        let ctx = SessionContext::new();
        let state = ctx.state();
        let plan = provider
            .scan(&state, None, &[col("worker_id").eq(lit("w-2"))], None)
            .await
            .unwrap();
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let config = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        let files: Vec<_> = config
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .collect();
        assert_eq!(files.len(), 1);
        assert!(files[0]
            .object_meta
            .location
            .as_ref()
            .ends_with(".fidx.workers.parquet"));
        assert!(files[0].extensions.is_none());

        ctx.register_table("workers", Arc::new(provider)).unwrap();
        let batches = ctx
            .sql("SELECT worker_id FROM workers WHERE worker_id = 'w-2'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(first_column_strings(&batches), vec!["w-2"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn remote_projection_reads_and_missing_payload_falls_back_to_source() {
        let dir = tempdir("remote_projection_fallback");
        let batch = worker_batch(1, vec!["w-1", "w-2", "w-3"], vec![100, 200, 300]);
        let (path, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        crate::indices::write_segment_index(
            &path,
            &[batch],
            &crate::indices::SegmentIndexConfig::from_policies(
                Vec::<String>::new(),
                &[crate::indices::exact::ExactIndexConfig {
                    column: "worker_id".into(),
                    exact_values: vec!["w-2".into()],
                    value_counts: false,
                }],
                &[crate::store::schema::CoveringProjection::new(
                    "workers",
                    "worker_id",
                    ["w-2"],
                    ["seq", "worker_id", "mem_bytes"],
                )],
                None,
            ),
        )
        .unwrap();
        let artifacts = crate::indices::sidecar_artifacts(&[path.to_string_lossy()]);
        let local = artifacts.values().next().unwrap();
        let remote = crate::test_support::FaultInjectingObjectStore::new(Arc::new(
            crate::store::object_store::build_remote_object_store(
                dir.join("remote").to_str().unwrap(),
            )
            .unwrap()
            .unwrap(),
        ));
        let store = Arc::new(
            crate::store::object_store::CachedObjectStore::new(
                remote.clone(),
                dir.join("cache"),
                None,
            )
            .unwrap(),
        );
        let source_id = ObjectId::table("workers", "objects/source.parquet").unwrap();
        let bundle_id = ObjectId::table("workers", "indices/source.fidx").unwrap();
        let projection_id = ObjectId::table("workers", "indices/workers.parquet").unwrap();
        let mut refs = Vec::new();
        for (id, artifact) in [
            (&bundle_id, local.bundle.as_ref().unwrap()),
            (&projection_id, &local.projections["workers"]),
        ] {
            let version = remote
                .write(id, bytes::Bytes::from(std::fs::read(artifact).unwrap()))
                .await
                .unwrap();
            refs.push(
                crate::proto::finelog::stats::ObjectRef::default()
                    .with_object_id(id.as_str())
                    .with_byte_size(version.byte_size),
            );
        }
        let source = ObjectReference {
            id: source_id.clone(),
            version: crate::store::object_store::ObjectVersion {
                e_tag: None,
                provider_version: None,
                local_value: None,
                byte_size: std::fs::metadata(&path).unwrap().len(),
            },
        };
        let planned = store
            .planned_local_path(&source_id)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let mut fields = worker_arrow().fields().to_vec();
        fields.push(Arc::new(Field::new("added", DataType::Int64, true)));
        let provider = NamespaceProvider::build(
            Arc::new(ArrowSchema::new(fields)),
            std::slice::from_ref(&planned),
            crate::indices::test_index_registry(),
        )
        .unwrap()
        .with_segment_artifacts(BTreeMap::from([(
            planned.clone(),
            crate::store::table_state::LocalArtifacts {
                binding: local.binding.clone(),
                bundle: Some(store.planned_local_path(&bundle_id).unwrap()),
                projections: BTreeMap::from([(
                    "workers".into(),
                    store.planned_local_path(&projection_id).unwrap(),
                )]),
            },
        )]))
        .with_object_sources(
            store,
            BTreeMap::from([(
                planned,
                crate::store::table::query_view::SegmentObjects {
                    source,
                    artifacts: crate::store::table_state::ArtifactReferences {
                        binding: local.binding.clone(),
                        bundle: Some(refs[0].clone()),
                        projections: BTreeMap::from([("workers".into(), refs[1].clone())]),
                    },
                },
            )]),
        );
        let ctx = object_scan_context(provider);
        let sql = "SELECT mem_bytes FROM workers WHERE worker_id = 'w-2'";
        remote.clear_calls();
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            200
        );
        assert!(!remote.ranges_for(&projection_id).is_empty());
        assert!(
            remote.ranges_for(&source_id).is_empty(),
            "the required source is deliberately absent"
        );

        // Keep the projection footer cached while removing its payload. An open
        // must fall back until it has actually produced a row.
        remote.delete(&projection_id).await.unwrap();
        remote
            .write(
                &source_id,
                bytes::Bytes::from(std::fs::read(&path).unwrap()),
            )
            .await
            .unwrap();
        remote.clear_calls();
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            200
        );
        assert!(!remote.ranges_for(&source_id).is_empty());
        remote.clear_calls();
        let batches = ctx
            .sql("SELECT mem_bytes, added FROM workers WHERE worker_id = 'w-2' AND added IS NULL")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        assert!(batches[0].column(1).is_null(0));
        assert!(!remote.ranges_for(&source_id).is_empty());
        assert!(
            remote.ranges_for(&projection_id).is_empty(),
            "a projection must cover residual filter columns"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn key_range_excludes_disjoint_segments_before_projection_planning() {
        let dir = tempdir("key_range_projection");
        let old = worker_batch(1, vec!["w-2"], vec![10]);
        let recent = worker_batch(2, vec!["w-2"], vec![110]);
        let unknown = worker_batch(3, vec!["w-2"], vec![120]);
        let (old_path, _) = write_segment_to_dir(&dir, 1, 1, &old).unwrap();
        let (recent_path, _) = write_segment_to_dir(&dir, 1, 2, &recent).unwrap();
        let (unknown_path, _) = write_segment_to_dir(&dir, 1, 3, &unknown).unwrap();
        let index_config = crate::indices::SegmentIndexConfig::from_policies(
            Vec::<String>::new(),
            &[crate::indices::exact::ExactIndexConfig {
                column: "worker_id".to_string(),
                exact_values: vec!["w-2".to_string()],
                value_counts: false,
            }],
            &[crate::store::schema::CoveringProjection::new(
                "workers",
                "worker_id",
                ["w-2"],
                ["seq", "worker_id", "mem_bytes"],
            )],
            None,
        );
        for (path, batch) in [
            (&old_path, &old),
            (&recent_path, &recent),
            (&unknown_path, &unknown),
        ] {
            crate::indices::write_segment_index(path, std::slice::from_ref(batch), &index_config)
                .unwrap();
        }
        let paths = [&old_path, &recent_path, &unknown_path]
            .map(|path| path.to_string_lossy().into_owned());
        let provider = NamespaceProvider::build_with_local_artifacts(worker_arrow(), &paths)
            .unwrap()
            .with_segment_key_bounds(
                "mem_bytes",
                BTreeMap::from([
                    (paths[0].clone(), ("10".to_string(), "10".to_string())),
                    (paths[1].clone(), ("110".to_string(), "110".to_string())),
                ]),
            );
        let filters = [
            col("worker_id").eq(lit("w-2")),
            col("mem_bytes").gt_eq(lit(100_i64)),
            col("mem_bytes").lt(lit(200_i64)),
        ];
        let ctx = SessionContext::new();
        let plan = provider
            .scan(&ctx.state(), None, &filters, None)
            .await
            .unwrap();
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let config = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        let locations: Vec<&str> = config
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .map(|file| file.object_meta.location.as_ref())
            .collect();
        let old_projection = crate::indices::exact::named_projection_path(&old_path, "workers");
        let old_projection = old_projection.file_name().unwrap().to_str().unwrap();
        assert_eq!(locations.len(), 2);
        assert!(locations
            .iter()
            .all(|location| location.ends_with(".fidx.workers.parquet")));
        assert!(locations
            .iter()
            .all(|location| !location.ends_with(old_projection)));

        ctx.register_table("workers", Arc::new(provider)).unwrap();
        let batches = ctx
            .sql(
                "SELECT mem_bytes FROM workers WHERE worker_id = 'w-2' \
                 AND mem_bytes >= 100 AND mem_bytes < 200 ORDER BY mem_bytes",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let values = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, vec![110, 120]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn string_key_range_prunes_disjoint_segments() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
        ]));
        let paths = vec![
            "/planned/a.parquet".to_string(),
            "/planned/b.parquet".to_string(),
            "/planned/unknown.parquet".to_string(),
        ];
        let provider = NamespaceProvider::build_with_local_artifacts(schema, &paths)
            .unwrap()
            .with_segment_key_bounds(
                "key",
                BTreeMap::from([
                    (
                        paths[0].clone(),
                        ("/job/a/0".to_string(), "/job/a/9".to_string()),
                    ),
                    (
                        paths[1].clone(),
                        ("/job/b/0".to_string(), "/job/b/9".to_string()),
                    ),
                ]),
            );

        assert_eq!(
            provider.segment_paths_for_filters(&[col("key").eq(lit("/job/b/4"))]),
            vec![paths[1].clone(), paths[2].clone()]
        );
    }

    #[tokio::test]
    async fn prefix_trigram_prunes_a_whole_segment_before_parquet_planning() {
        let dir = tempdir("prefix_segment_prune");
        let unrelated = worker_batch(1, vec!["/other/job/0:0"], vec![10]);
        let matching = worker_batch(2, vec!["/marin/hero/job/0:0"], vec![20]);
        let (unrelated_path, _) = write_segment_to_dir(&dir, 1, 1, &unrelated).unwrap();
        let (matching_path, _) = write_segment_to_dir(&dir, 1, 2, &matching).unwrap();
        let index_config = crate::indices::SegmentIndexConfig::from_policies(
            ["worker_id"],
            &[],
            &[],
            Some("worker_id".to_string()),
        );
        for (path, batch) in [(&unrelated_path, &unrelated), (&matching_path, &matching)] {
            crate::indices::write_segment_index(path, std::slice::from_ref(batch), &index_config)
                .unwrap();
        }
        let paths =
            [&unrelated_path, &matching_path].map(|path| path.to_string_lossy().into_owned());
        let artifacts = crate::indices::sidecar_artifacts(&paths);
        std::fs::remove_file(&unrelated_path).unwrap();

        let ctx = crate::query::make_ctx();
        let prefix = ctx.udf("prefix").unwrap();
        let filter = Expr::ScalarFunction(ScalarFunction::new_udf(
            prefix,
            vec![col("worker_id"), lit("/marin/hero/")],
        ));
        let plan = NamespaceProvider::build(
            worker_arrow(),
            &paths,
            crate::indices::test_index_registry(),
        )
        .unwrap()
        .with_segment_artifacts(artifacts)
        .scan(&ctx.state(), None, &[filter], None)
        .await
        .unwrap();
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let selected: Vec<_> = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .unwrap()
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .map(|file| file.object_meta.location.as_ref())
            .collect();
        assert_eq!(selected.len(), 1);
        assert!(selected[0].ends_with(matching_path.file_name().unwrap().to_str().unwrap()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn missing_projection_falls_back_only_for_that_segment() {
        let dir = tempdir("exact_projection_fallback");
        let first = worker_batch(1, vec!["w-1", "w-2"], vec![100, 200]);
        let second = worker_batch(3, vec!["w-2", "w-3"], vec![300, 400]);
        let (first_path, _) = write_segment_to_dir(&dir, 1, 1, &first).unwrap();
        let (second_path, _) = write_segment_to_dir(&dir, 1, 3, &second).unwrap();
        let config = crate::indices::exact::ExactIndexConfig {
            column: "worker_id".to_string(),
            exact_values: vec!["w-2".to_string()],
            value_counts: false,
        };
        let index_config = crate::indices::SegmentIndexConfig::from_policies(
            Vec::<String>::new(),
            &[config],
            &[crate::store::schema::CoveringProjection::new(
                "workers",
                "worker_id",
                ["w-2"],
                ["seq", "worker_id", "mem_bytes"],
            )],
            None,
        );
        crate::indices::write_segment_index(&first_path, &[first], &index_config).unwrap();
        crate::indices::write_segment_index(&second_path, &[second], &index_config).unwrap();
        std::fs::remove_file(crate::indices::exact::named_projection_path(
            &second_path,
            "workers",
        ))
        .unwrap();
        let paths = vec![
            first_path.to_string_lossy().into_owned(),
            second_path.to_string_lossy().into_owned(),
        ];
        let provider =
            NamespaceProvider::build_with_local_artifacts(worker_arrow(), &paths).unwrap();
        let ctx = SessionContext::new();
        let plan = provider
            .scan(&ctx.state(), None, &[col("worker_id").eq(lit("w-2"))], None)
            .await
            .unwrap();
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let config = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        let locations: Vec<&str> = config
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .map(|file| file.object_meta.location.as_ref())
            .collect();
        assert_eq!(
            locations
                .iter()
                .filter(|location| location.ends_with(".fidx.workers.parquet"))
                .count(),
            1
        );
        assert_eq!(
            locations
                .iter()
                .filter(|location| location.ends_with(".parquet"))
                .count(),
            2
        );

        ctx.register_table("workers", Arc::new(provider)).unwrap();
        let batches = ctx
            .sql("SELECT worker_id FROM workers WHERE worker_id = 'w-2' ORDER BY seq")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(first_column_strings(&batches), vec!["w-2", "w-2"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn uncovered_exact_value_does_not_read_postings_payload() {
        let dir = tempdir("uncovered_exact_value");
        let batch = worker_batch(1, vec!["w-1", "w-2"], vec![100, 200]);
        let (path, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let config = crate::indices::exact::ExactIndexConfig {
            column: "worker_id".to_string(),
            exact_values: vec!["w-2".to_string()],
            value_counts: false,
        };
        crate::indices::write_segment_index(
            &path,
            &[batch],
            &crate::indices::SegmentIndexConfig::from_policies(
                Vec::<String>::new(),
                &[config],
                &[],
                None,
            ),
        )
        .unwrap();

        let bundle_path = crate::indices::format::bundle_path(&path);
        let header = crate::indices::format::read_header(&bundle_path).unwrap();
        let postings = header
            .sections
            .iter()
            .find(|section| section.kind == crate::indices::format::SectionKind::ExactPostings)
            .unwrap();
        std::fs::File::options()
            .write(true)
            .open(&bundle_path)
            .unwrap()
            .write_all_at(&[0xff], postings.offset)
            .unwrap();

        let indices = crate::indices::test_index_registry();
        let paths = vec![path.to_string_lossy().into_owned()];
        let provider = NamespaceProvider::build(worker_arrow(), &paths, Arc::clone(&indices))
            .unwrap()
            .with_segment_artifacts(crate::indices::sidecar_artifacts(&paths));
        let plan = provider
            .scan(
                &SessionContext::new().state(),
                None,
                &[col("worker_id").eq(lit("w-missing"))],
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            indices.cache().corruption_counts().sections,
            0,
            "header coverage should reject the predicate before reading the corrupt payload"
        );
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let config = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        assert!(config
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .all(|file| file.extensions.is_none()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn registered_policy_skips_uncovered_legacy_postings_payload() {
        let dir = tempdir("legacy_uncovered_exact_value");
        let batch = worker_batch(1, vec!["w-1", "w-2"], vec![100, 200]);
        let (path, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let config = crate::indices::exact::ExactIndexConfig {
            column: "worker_id".to_string(),
            exact_values: vec!["w-2".to_string()],
            value_counts: false,
        };
        crate::indices::write_segment_index(
            &path,
            &[batch],
            &crate::indices::SegmentIndexConfig::from_policies(
                Vec::<String>::new(),
                &[config],
                &[],
                None,
            ),
        )
        .unwrap();

        let bundle_path = crate::indices::format::bundle_path(&path);
        let header = crate::indices::format::read_header(&bundle_path).unwrap();
        let postings = header
            .sections
            .iter()
            .find(|section| section.kind == crate::indices::format::SectionKind::ExactPostings)
            .unwrap();
        let payload =
            crate::indices::format::read_section(&bundle_path, &header, postings.id.as_str())
                .unwrap();
        let legacy_bundle = crate::indices::format::serialize(
            &header.binding,
            &[crate::indices::format::SectionInput {
                id: postings.id.clone(),
                kind: postings.kind,
                method_version: 1,
                exactness: postings.exactness,
                coverage: Vec::new(),
                payload,
            }],
        )
        .unwrap();
        std::fs::write(&bundle_path, legacy_bundle).unwrap();
        let legacy_header = crate::indices::format::read_header(&bundle_path).unwrap();
        let legacy_postings = legacy_header
            .sections
            .iter()
            .find(|section| section.kind == crate::indices::format::SectionKind::ExactPostings)
            .unwrap();
        std::fs::File::options()
            .write(true)
            .open(&bundle_path)
            .unwrap()
            .write_all_at(&[0xff], legacy_postings.offset)
            .unwrap();

        let indices = crate::indices::test_index_registry();
        let paths = vec![path.to_string_lossy().into_owned()];
        let provider = NamespaceProvider::build(worker_arrow(), &paths, Arc::clone(&indices))
            .unwrap()
            .with_segment_artifacts(crate::indices::sidecar_artifacts(&paths))
            .with_exact_postings_policy(BTreeMap::from([(
                "worker_id".to_string(),
                vec!["w-2".to_string()],
            )]));
        let plan = provider
            .scan(
                &SessionContext::new().state(),
                None,
                &[col("worker_id").eq(lit("w-missing"))],
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            indices.cache().corruption_counts().sections,
            0,
            "the registered policy should reject the predicate before reading legacy postings"
        );
        let exec = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec");
        let config = exec
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        assert!(config
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .all(|file| file.extensions.is_none()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn native_map_column_scans_and_json_get_reads_it() {
        // A sealed segment carrying a native Map<Utf8,Utf8> column reads back
        // through the ListingTable provider as a MapArray, and the duck-typed
        // json_get UDF filters/groups it — the whole storage-flip point.
        use crate::query::udf::register_scalar_udfs;
        use crate::store::schema::map_utf8_utf8_type;
        use arrow::array::{MapBuilder, MapFieldNames, StringBuilder};

        let dir = tempdir("map_scan");
        let schema: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("labels", map_utf8_utf8_type(), true),
        ]));

        let names = MapFieldNames {
            entry: "entries".to_string(),
            key: "key".to_string(),
            value: "value".to_string(),
        };
        let mut mb = MapBuilder::new(Some(names), StringBuilder::new(), StringBuilder::new());
        for scope in ["fleet", "fleet", "local"] {
            mb.keys().append_value("scope");
            mb.values().append_value(scope);
            mb.append(true).unwrap();
        }
        let labels = mb.finish();
        assert_eq!(labels.data_type(), &map_utf8_utf8_type());

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(1..=3)),
                Arc::new(labels),
            ],
        )
        .unwrap();
        write_segment_to_dir(&dir, 0, 1, &batch).unwrap();
        let paths: Vec<String> = crate::store::segment::discover_segments(&dir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        let provider = NamespaceProvider::build_with_local_artifacts(schema, &paths).unwrap();
        let ctx = SessionContext::new();
        register_scalar_udfs(&ctx);
        ctx.register_table(
            datafusion::common::TableReference::bare("probes"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql(
                "SELECT count(*) AS n FROM probes \
                 WHERE json_get(labels, 'scope') = 'fleet'",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let n = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(n.value(0), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn two_providers_join() {
        let wdir = tempdir("join_w");
        let tdir = tempdir("join_t");
        write_segment_to_dir(
            &wdir,
            0,
            1,
            &worker_batch(1, vec!["w-1", "w-2"], vec![100, 200]),
        )
        .unwrap();

        // task table: seq, worker_id, task_count
        let task_arrow: SchemaRef = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("task_count", DataType::Int64, false),
        ]));
        let task_batch = RecordBatch::try_new(
            Arc::clone(&task_arrow),
            vec![
                Arc::new(Int64Array::from_iter_values(1..3)),
                Arc::new(StringArray::from(vec!["w-1", "w-2"])),
                Arc::new(Int64Array::from(vec![10_i64, 20])),
            ],
        )
        .unwrap();
        write_segment_to_dir(&tdir, 0, 1, &task_batch).unwrap();

        let wpaths: Vec<String> = crate::store::segment::discover_segments(&wdir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let tpaths: Vec<String> = crate::store::segment::discover_segments(&tdir)
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();

        let ctx = SessionContext::new();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.worker"),
            Arc::new(
                NamespaceProvider::build_with_local_artifacts(worker_arrow(), &wpaths).unwrap(),
            ),
        )
        .unwrap();
        ctx.register_table(
            datafusion::common::TableReference::bare("iris.task"),
            Arc::new(NamespaceProvider::build_with_local_artifacts(task_arrow, &tpaths).unwrap()),
        )
        .unwrap();

        let batches = ctx
            .sql(
                "SELECT w.mem_bytes, t.task_count FROM \"iris.worker\" w \
                 JOIN \"iris.task\" t USING (worker_id) ORDER BY w.mem_bytes",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
        std::fs::remove_dir_all(&wdir).ok();
        std::fs::remove_dir_all(&tdir).ok();
    }

    /// The first column of `batches` as strings, in row order — what a
    /// single-column projection assertion compares against.
    fn first_column_strings(batches: &[RecordBatch]) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| {
                let column = b.column(0);
                let c = StringValues::new(column).expect("string column");
                (0..column.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Log-form schema: seq, key, data (the columns the trigram prune touches).
    fn log_arrow() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
            Field::new("data", DataType::Utf8, false),
        ]))
    }

    /// Write one segment whose `column` ("key" or "data") is a full span of
    /// `filler` followed by `span1`, so source span 0 lacks the needle and span 1
    /// carries it. The other string column is constant, which makes any prune
    /// attributable to `column`. Returns the segment path.
    fn write_two_span_log_segment(
        dir: &std::path::Path,
        column: &str,
        filler: &str,
        span1: &[&str],
    ) -> String {
        let mut varying: Vec<String> = (0..SIDECAR_SPAN_ROWS).map(|_| filler.to_string()).collect();
        varying.extend(span1.iter().map(|s| s.to_string()));
        let n = varying.len();
        let (keys, data) = match column {
            "key" => (varying, vec!["idle heartbeat ok".to_string(); n]),
            "data" => (vec!["/system/controller".to_string(); n], varying),
            other => panic!("unsupported varying column {other:?}"),
        };
        let batch = RecordBatch::try_new(
            log_arrow(),
            vec![
                Arc::new(Int64Array::from_iter_values(1..=n as i64)),
                Arc::new(StringArray::from(keys)),
                Arc::new(StringArray::from(data)),
            ],
        )
        .unwrap();
        let (path, _) = write_segment_to_dir(dir, 1, 1, &batch).unwrap();
        crate::indices::write_segment_index(
            &path,
            &[batch],
            &crate::indices::SegmentIndexConfig::from_policies(
                ["key", "data"],
                &[],
                &[],
                Some("key".to_string()),
            ),
        )
        .unwrap();
        path.to_string_lossy().into_owned()
    }

    /// Assert the scan injected a trigram access plan selecting exactly
    /// `span1_rows`. These segments fit one byte-sized row group, so the prune
    /// lands as a row selection inside it, not a skipped row group.
    fn assert_prunes_to_span1(
        plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        span1_rows: usize,
    ) {
        use datafusion::datasource::physical_plan::FileScanConfig;
        use datafusion::datasource::source::DataSourceExec;
        use datafusion_datasource_parquet::{ParquetAccessPlan, RowGroupAccess};

        let cfg = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec")
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        let mut checked = 0;
        for group in &cfg.file_groups {
            for pf in group.files() {
                let ap = pf
                    .extensions
                    .as_ref()
                    .and_then(|e| e.downcast_ref::<ParquetAccessPlan>())
                    .expect("trigram access plan attached to the partitioned file");
                let [RowGroupAccess::Selection(selection)] = ap.inner() else {
                    panic!("expected one row group carrying a row selection: {ap:?}");
                };
                assert_eq!(
                    selection.row_count(),
                    span1_rows,
                    "only the needle's span may be selected"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 1,
            "exactly one partitioned file with an access plan"
        );
    }

    /// Whether any expression in `plan` casts the `data` column. Walks the tree
    /// rather than the rendered text, which qualifies column names inconsistently
    /// across plan stages and would make a substring check pass vacuously.
    fn casts_the_data_column(plan: &datafusion::logical_expr::LogicalPlan) -> bool {
        let mut found = false;
        plan.apply(|node| {
            for e in node.expressions() {
                e.apply(|sub| {
                    if let Expr::Cast(cast) = sub {
                        if matches!(cast.expr.as_ref(), Expr::Column(c) if c.name == "data") {
                            found = true;
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
                .unwrap();
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        found
    }

    #[tokio::test]
    async fn string_predicates_read_the_scan_layout_without_casting() {
        // The scan reads strings as Utf8View, and the finelog UDFs accept that
        // layout. A UDF with an exact-Utf8 signature would instead make the
        // planner cast the whole column ahead of the predicate — materializing
        // every value precisely to throw most of them away.
        let dir = tempdir("no_cast");
        let path =
            write_two_span_log_segment(&dir, "data", "idle heartbeat ok", &["needle here"; 4]);
        let provider =
            NamespaceProvider::build_with_local_artifacts(log_arrow(), std::slice::from_ref(&path))
                .unwrap();
        assert_eq!(
            provider
                .schema()
                .field_with_name("data")
                .unwrap()
                .data_type(),
            &DataType::Utf8View,
        );

        let ctx = crate::query::make_ctx();
        ctx.register_table(
            datafusion::common::TableReference::bare("log"),
            Arc::new(provider),
        )
        .unwrap();
        for predicate in [
            "contains(data, 'needle')",
            "prefix(data, 'needle')",
            "regexp_matches(data, 'needle')",
            "json_get(data, 'k') IS NOT NULL",
        ] {
            let plan = ctx
                .sql(&format!("SELECT seq FROM \"log\" WHERE {predicate}"))
                .await
                .unwrap()
                .into_optimized_plan()
                .unwrap();
            assert!(
                !casts_the_data_column(&plan),
                "{predicate} planned a whole-column cast:\n{}",
                plan.display_indent()
            );
        }

        // The detector must actually fire on a cast, or the assertions above pass
        // for the wrong reason.
        let cast_plan = ctx
            .sql("SELECT seq FROM \"log\" WHERE CAST(data AS int) = 1")
            .await
            .unwrap()
            .into_optimized_plan()
            .unwrap();
        assert!(casts_the_data_column(&cast_plan));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn integer_pruning_distinguishes_scalar_from_nested_list_element() {
        let dir = tempdir("nested_integer_bounds");
        let list = ListArray::from_iter_primitive::<Int64Type, _, _>([Some(vec![Some(1)])]);
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("values", list.data_type().clone(), true),
            Field::new("item", DataType::Int64, false),
            Field::new("data", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(list),
                Arc::new(Int64Array::from(vec![100])),
                Arc::new(StringArray::from(vec!["DEBUG matching scalar"])),
            ],
        )
        .unwrap();
        let (path, _) = write_segment_to_dir(&dir, 1, 1, &batch).unwrap();
        let provider = NamespaceProvider::build_with_local_artifacts(
            schema,
            &[path.to_string_lossy().into_owned()],
        )
        .unwrap();
        let ctx = crate::query::make_ctx();
        ctx.register_table("nested", Arc::new(provider)).unwrap();
        let batches = ctx
            .sql("SELECT data FROM nested WHERE item = 100 AND contains(data, 'DEBUG')")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            first_column_strings(&batches),
            vec!["DEBUG matching scalar"]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn integer_range_skips_old_indexes_and_keeps_unknown_statistics() {
        let dir = tempdir("integer_range_trigram");
        let old = worker_batch(1, vec!["DEBUG-old"], vec![10]);
        let recent = worker_batch(2, vec!["DEBUG-recent"], vec![200]);
        let unknown = worker_batch(3, vec!["DEBUG-unknown"], vec![300]);
        let (old_path, _) = write_segment_to_dir(&dir, 1, 1, &old).unwrap();
        let (recent_path, _) = write_segment_to_dir(&dir, 1, 2, &recent).unwrap();
        let unknown_path = dir.join("unknown.parquet");
        let properties = parquet::file::properties::WriterProperties::builder()
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
            .build();
        let mut writer = parquet::arrow::ArrowWriter::try_new(
            std::fs::File::create(&unknown_path).unwrap(),
            worker_arrow(),
            Some(properties),
        )
        .unwrap();
        writer.write(&unknown).unwrap();
        writer.close().unwrap();
        let index_config = crate::indices::SegmentIndexConfig::from_policies(
            ["worker_id"],
            &[],
            &[],
            Some("worker_id".to_string()),
        );
        for (path, batch) in [
            (&old_path, &old),
            (&recent_path, &recent),
            (&unknown_path, &unknown),
        ] {
            crate::indices::write_segment_index(path, std::slice::from_ref(batch), &index_config)
                .unwrap();
        }
        // An excluded segment's corrupted index must never be read.
        let bundle = crate::indices::format::bundle_path(&old_path);
        let header = crate::indices::format::read_header(&bundle).unwrap();
        let section = header.section("trigram:worker_id").unwrap();
        std::fs::File::options()
            .write(true)
            .open(bundle)
            .unwrap()
            .write_all_at(&[0xff; 8], section.offset)
            .unwrap();
        let paths = [&old_path, &recent_path, &unknown_path]
            .map(|path| path.to_string_lossy().into_owned());
        let indices = crate::indices::test_index_registry();
        let provider = NamespaceProvider::build(worker_arrow(), &paths, Arc::clone(&indices))
            .unwrap()
            .with_segment_artifacts(crate::indices::sidecar_artifacts(&paths));
        let ctx = crate::query::make_ctx();
        ctx.register_table("workers", Arc::new(provider)).unwrap();
        let batches = ctx
            .sql(
                "SELECT worker_id FROM workers WHERE mem_bytes >= 100 \
             AND contains(worker_id, 'DEBUG') ORDER BY mem_bytes",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            first_column_strings(&batches),
            vec!["DEBUG-recent", "DEBUG-unknown"]
        );
        assert_eq!(indices.cache().corruption_counts().sections, 0);
        assert_eq!(indices.cache().load_stats().section_attempts, 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn trigram_masks_survive_cache_eviction_during_source_planning() {
        let dir = tempdir("trigram_mask_eviction");
        let path = write_two_span_log_segment(&dir, "data", "idle heartbeat ok", &["DEBUG prompt"]);
        let paths = vec![path.clone()];
        let artifacts = crate::indices::sidecar_artifacts(&paths);
        let indices = crate::indices::test_index_registry();
        let needles = HashMap::from([(
            "data".to_string(),
            crate::query::trigram_prune::SubstringPredicate::Contains(vec!["DEBUG".to_string()]),
        )]);
        let pruning = crate::query::trigram_prune::plan_segments(
            &paths,
            &needles,
            &HashMap::new(),
            &indices,
            &artifacts,
        );
        // Evict and remove the derived file after planning; the source remains readable.
        let bundle = artifacts[&path].bundle.as_ref().unwrap();
        indices.invalidate(bundle);
        std::fs::remove_file(bundle).unwrap();
        let provider = NamespaceProvider::build(log_arrow(), &paths, Arc::clone(&indices))
            .unwrap()
            .with_segment_indexes_enabled(false);
        let ctx = crate::query::make_ctx();
        let source_plan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
        let plan = crate::query::trigram_prune::apply_pruning(source_plan, &pruning, &indices);
        assert_prunes_to_span1(&plan, 1);
        let batches = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .unwrap();
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            1
        );
        assert_eq!(indices.cache().load_stats().section_attempts, 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn contains_query_returns_matches_and_prunes_row_groups() {
        let dir = tempdir("contains_prune");
        // The needle lives only in row group 1 (rows 2 and 4 of the tail).
        let needle = "Bootstrap completed for TPU-xyz";
        let rg1 = vec![
            "idle heartbeat ok",
            "E0601 Bootstrap completed for TPU-xyz started",
            "idle heartbeat ok",
            "another Bootstrap completed for TPU-xyz here",
        ];
        let rg1_rows = rg1.len();
        let path = write_two_span_log_segment(&dir, "data", "idle heartbeat ok", &rg1);

        // 1) End-to-end correctness: the contains() query returns exactly the two
        //    matching rows (and prunes row group 0 along the way).
        let ctx = crate::query::make_ctx();
        let provider =
            NamespaceProvider::build_with_local_artifacts(log_arrow(), std::slice::from_ref(&path))
                .unwrap();
        ctx.register_table(
            datafusion::common::TableReference::bare("log"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql(&format!(
                "SELECT data FROM \"log\" WHERE contains(data, '{needle}') ORDER BY seq"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let got = first_column_strings(&batches);
        assert_eq!(
            got,
            vec![
                "E0601 Bootstrap completed for TPU-xyz started".to_string(),
                "another Bootstrap completed for TPU-xyz here".to_string(),
            ],
            "contains() must return exactly the matching rows"
        );

        // 2) Evidence of pruning: the injected access plan skips row group 0 and
        //    keeps row group 1.
        let state = ctx.state();
        let udf = ctx.udf("contains").unwrap();
        let filter =
            Expr::ScalarFunction(ScalarFunction::new_udf(udf, vec![col("data"), lit(needle)]));
        let probe = NamespaceProvider::build_with_local_artifacts(log_arrow(), &[path]).unwrap();
        let plan = probe.scan(&state, None, &[filter], None).await.unwrap();
        assert_prunes_to_span1(&plan, rg1_rows);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn inference_message_alternatives_prune_without_losing_either_branch() {
        let dir = tempdir("inference_message_alternatives");
        let path = write_two_span_log_segment(
            &dir,
            "data",
            "idle heartbeat ok",
            &["model cache hit", "download complete"],
        );
        let ctx = crate::query::make_ctx();
        let provider =
            NamespaceProvider::build_with_local_artifacts(log_arrow(), std::slice::from_ref(&path))
                .unwrap();
        ctx.register_table("logs", Arc::new(provider)).unwrap();
        let frame = ctx.sql("SELECT data FROM logs WHERE contains(data, 'model cache') OR contains(data, 'download complete') ORDER BY seq").await.unwrap();
        assert_eq!(
            first_column_strings(&frame.collect().await.unwrap()),
            vec!["model cache hit", "download complete"]
        );
        // A branch without a usable substring constraint must retain its rows.
        let rows = ctx
            .sql(
                "SELECT data FROM logs WHERE contains(data, 'model cache') OR seq = 1 ORDER BY seq",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            first_column_strings(&rows),
            vec!["idle heartbeat ok", "model cache hit"]
        );
        let udf = ctx.udf("contains").unwrap();
        let filter = Expr::ScalarFunction(ScalarFunction::new_udf(
            udf.clone(),
            vec![col("data"), lit("model cache")],
        ))
        .or(Expr::ScalarFunction(ScalarFunction::new_udf(
            udf,
            vec![col("data"), lit("download complete")],
        )));
        let probe = NamespaceProvider::build_with_local_artifacts(log_arrow(), &[path]).unwrap();
        let plan = probe
            .scan(&ctx.state(), None, &[filter], None)
            .await
            .unwrap();
        assert_prunes_to_span1(&plan, 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn cold_index_excludes_missing_source_before_open() {
        let dir = tempdir("cold_index_before_source");
        let path =
            write_two_span_log_segment(&dir, "data", "idle heartbeat ok", &["ordinary shutdown"]);
        let artifacts = crate::indices::sidecar_artifacts(std::slice::from_ref(&path));
        let binding = artifacts[&path].binding.clone();
        let bundle_path = artifacts[&path].bundle.as_ref().unwrap();
        let header = crate::indices::format::read_header(bundle_path).unwrap();
        let mut sections: Vec<_> = header
            .sections
            .iter()
            .map(|section| crate::indices::format::SectionInput {
                id: section.id.clone(),
                kind: section.kind,
                method_version: section.method_version,
                exactness: section.exactness,
                coverage: section.coverage.clone(),
                payload: crate::indices::format::read_section(bundle_path, &header, &section.id)
                    .unwrap(),
            })
            .collect();
        // A large unrelated section must never be transferred for this query.
        sections.push(crate::indices::format::SectionInput {
            id: "trigram:unused".into(),
            kind: crate::indices::format::SectionKind::TrigramBloom,
            method_version: 1,
            exactness: crate::indices::format::Exactness::Lossy,
            coverage: header.sections[0].coverage.clone(),
            payload: vec![0; 1024 * 1024],
        });
        let bundle_bytes = bytes::Bytes::from(
            crate::indices::format::serialize(&header.binding, &sections).unwrap(),
        );
        let remote = Arc::new(
            crate::store::object_store::build_remote_object_store(
                dir.join("remote").to_str().unwrap(),
            )
            .unwrap()
            .unwrap(),
        );
        let remote = crate::test_support::FaultInjectingObjectStore::new(remote);
        let bundle_id = ObjectId::table("logs", "indices/one.fidx").unwrap();
        let bundle_version = remote
            .write(&bundle_id, bundle_bytes.clone())
            .await
            .unwrap();
        let bundle_ref = crate::proto::finelog::stats::ObjectRef::default()
            .with_object_id(bundle_id.as_str())
            .with_byte_size(bundle_version.byte_size);
        let source = ObjectReference {
            id: ObjectId::table("logs", "objects/one.parquet").unwrap(),
            version: crate::store::object_store::ObjectVersion {
                e_tag: None,
                provider_version: None,
                byte_size: std::fs::metadata(&path).unwrap().len(),
                local_value: None,
            },
        };
        // The source is absent; its index must prune it before the opener reads it.
        let store = Arc::new(
            crate::store::object_store::CachedObjectStore::new(
                remote.clone(),
                dir.join("cache"),
                None,
            )
            .unwrap(),
        );
        let planned = store
            .planned_local_path(&source.id)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let planned_bundle = store.planned_local_path(&bundle_id).unwrap();
        let provider = NamespaceProvider::build(
            log_arrow(),
            std::slice::from_ref(&planned),
            crate::indices::test_index_registry(),
        )
        .unwrap()
        .with_segment_seq_bounds(BTreeMap::from([(planned.clone(), (1, 16385))]))
        .with_segment_artifacts(BTreeMap::from([(
            planned.clone(),
            crate::store::table_state::LocalArtifacts {
                binding: binding.clone(),
                bundle: Some(planned_bundle.clone()),
                projections: BTreeMap::new(),
            },
        )]))
        .with_object_sources(
            store.clone(),
            BTreeMap::from([(
                planned,
                crate::store::table::query_view::SegmentObjects {
                    source: source.clone(),
                    artifacts: crate::store::table_state::ArtifactReferences {
                        binding,
                        bundle: Some(bundle_ref),
                        projections: BTreeMap::new(),
                    },
                },
            )]),
        );
        let mut config = crate::query::make_ctx()
            .copied_config()
            .with_extension(Arc::new(QueryCacheAdmission::new(0, 0)));
        config.options_mut().execution.parquet.pushdown_filters = false;
        let ctx = datafusion::prelude::SessionContext::new_with_config(config);
        crate::query::udf::register_scalar_udfs(&ctx);
        let provider = Arc::new(provider);
        ctx.register_table("logs", provider.clone()).unwrap();
        let rows = ctx.sql("SELECT * FROM logs WHERE contains(data, 'model cache') OR contains(data, 'download complete')").await.unwrap().collect().await.unwrap();
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        assert!(remote
            .keys_for(crate::test_support::ObjectOp::Read)
            .is_empty());
        let transferred: u64 = remote
            .ranges_for(&bundle_id)
            .iter()
            .map(|range| range.end - range.start)
            .sum();
        assert!(transferred > 0 && transferred < bundle_bytes.len() as u64 / 2);
        provider.indices.invalidate(&planned_bundle);
        remote.clear_calls();
        let rows = ctx.sql("SELECT * FROM logs WHERE key LIKE '%/inference-%' AND contains(data, 'ordinary shutdown')").await.unwrap().collect().await.unwrap();
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        let header = crate::indices::format::parse_header(&bundle_bytes).unwrap();
        let data = header.section("trigram:data").unwrap();
        assert!(
            remote
                .ranges_for(&bundle_id)
                .iter()
                .all(|range| { range.end <= data.offset || range.start >= data.offset + data.len }),
            "a rejecting key section must avoid reading the data section"
        );
        remote
            .write(
                &source.id,
                bytes::Bytes::from(std::fs::read(&path).unwrap()),
            )
            .await
            .unwrap();
        let filter = Expr::ScalarFunction(ScalarFunction::new_udf(
            ctx.udf("contains").unwrap(),
            vec![col("data"), lit("ordinary shutdown")],
        ));
        remote.clear_calls();
        let plan = provider
            .scan(&ctx.state(), None, &[filter], None)
            .await
            .unwrap();
        assert!(
            remote
                .keys_for(crate::test_support::ObjectOp::ReadRange)
                .is_empty(),
            "planning must not open the source or its index"
        );
        // Collect the raw scan without a residual FilterExec: only the index's
        // row selection can exclude the 16,384 idle rows here.
        let rows = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .unwrap();
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        assert!(
            !store.planned_local_path(&source.id).unwrap().exists(),
            "zero allowance must keep source scans remote"
        );
        drop(ctx);
        store
            .write(
                &source.id,
                bytes::Bytes::from(std::fs::read(&path).unwrap()),
            )
            .await
            .unwrap();
        assert!(
            !planned_bundle.exists(),
            "index range reads must not fill a whole bundle"
        );
        remote.clear_calls();
        let ctx = crate::query::make_ctx();
        ctx.register_table("logs", provider).unwrap();
        let rows = ctx
            .sql("SELECT * FROM logs WHERE seq > 100000 AND contains(data, 'model cache')")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        assert!(
            remote
                .keys_for(crate::test_support::ObjectOp::Read)
                .is_empty(),
            "snapshot sequence bounds must exclude the segment without storage reads"
        );
        assert!(remote
            .keys_for(crate::test_support::ObjectOp::ReadRange)
            .is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn disabled_segment_indexes_ignore_existing_bundle() {
        let dir = tempdir("disabled_segment_indexes");
        let needle = "Bootstrap completed for TPU-xyz";
        let path = write_two_span_log_segment(
            &dir,
            "data",
            "idle heartbeat ok",
            &["Bootstrap completed for TPU-xyz"],
        );
        let ctx = crate::query::make_ctx();
        let udf = ctx.udf("contains").unwrap();
        let filter =
            Expr::ScalarFunction(ScalarFunction::new_udf(udf, vec![col("data"), lit(needle)]));
        let plan =
            NamespaceProvider::build_with_local_artifacts(log_arrow(), std::slice::from_ref(&path))
                .unwrap()
                .with_segment_indexes_enabled(false)
                .scan(&ctx.state(), None, &[filter], None)
                .await
                .unwrap();
        let config = plan
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .expect("scan returns a parquet DataSourceExec")
            .data_source()
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("a FileScanConfig");
        assert!(config
            .file_groups
            .iter()
            .flat_map(|group| group.files())
            .all(|file| file.extensions.is_none()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn like_substring_query_prunes_row_groups() {
        // `data LIKE '%needle%'` must prune like `contains(data, 'needle')`: the
        // expression survives the simplifier as `Expr::Like` and the prune
        // extracts the framed substring. Asserts both the matching rows and the
        // injected skip of the needle-free row group 0.
        let dir = tempdir("like_prune");
        let needle = "Bootstrap completed for TPU-xyz";
        let rg1 = vec![
            "idle heartbeat ok",
            "E0601 Bootstrap completed for TPU-xyz started",
            "idle heartbeat ok",
        ];
        let rg1_rows = rg1.len();
        let path = write_two_span_log_segment(&dir, "data", "idle heartbeat ok", &rg1);

        let ctx = crate::query::make_ctx();
        let provider =
            NamespaceProvider::build_with_local_artifacts(log_arrow(), std::slice::from_ref(&path))
                .unwrap();
        ctx.register_table(
            datafusion::common::TableReference::bare("log"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql(&format!(
                "SELECT data FROM \"log\" WHERE data LIKE '%{needle}%' ORDER BY seq"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let got = first_column_strings(&batches);
        assert_eq!(
            got,
            vec!["E0601 Bootstrap completed for TPU-xyz started".to_string()],
            "LIKE must return exactly the matching row"
        );

        // The injected access plan skips the needle-free row group 0.
        let plan = NamespaceProvider::build_with_local_artifacts(log_arrow(), &[path])
            .unwrap()
            .scan(
                &ctx.state(),
                None,
                std::slice::from_ref(
                    &datafusion::prelude::col("data")
                        .like(datafusion::prelude::lit(format!("%{needle}%"))),
                ),
                None,
            )
            .await
            .unwrap();
        assert_prunes_to_span1(&plan, rg1_rows);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn regex_query_returns_matches_and_prunes_row_groups() {
        let dir = tempdir("regex_prune");
        let pattern = r"Bootstrap.*TPU-[a-z]+";
        let rg1 = vec![
            "idle heartbeat ok",
            "E0601 Bootstrap completed for TPU-xyz started",
            "Bootstrap stopped for GPU-xyz",
        ];
        let rg1_rows = rg1.len();
        let path = write_two_span_log_segment(&dir, "data", "idle heartbeat ok", &rg1);

        let ctx = crate::query::make_ctx();
        let provider =
            NamespaceProvider::build_with_local_artifacts(log_arrow(), std::slice::from_ref(&path))
                .unwrap();
        ctx.register_table(
            datafusion::common::TableReference::bare("log"),
            Arc::new(provider),
        )
        .unwrap();
        let batches = ctx
            .sql(&format!(
                "SELECT data FROM \"log\" WHERE regexp_matches(data, '{pattern}') ORDER BY seq"
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            first_column_strings(&batches),
            vec!["E0601 Bootstrap completed for TPU-xyz started".to_string()]
        );

        let udf = ctx.udf("regexp_matches").unwrap();
        let filter = Expr::ScalarFunction(ScalarFunction::new_udf(
            udf,
            vec![col("data"), lit(pattern)],
        ));
        let plan = NamespaceProvider::build_with_local_artifacts(log_arrow(), &[path])
            .unwrap()
            .scan(&ctx.state(), None, &[filter], None)
            .await
            .unwrap();
        assert_prunes_to_span1(&plan, rg1_rows);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn key_substring_query_prunes_row_groups() {
        // The job sits mid-key, so the `(key, seq)` sort and its min/max
        // statistics cannot bound it. Only the key's trigram section prunes.
        let dir = tempdir("key_substring_prune");
        let filler_key = "/power/other-run-coord/other-run/0:0";
        let span1_key = "/power/hs-final2-adoptedbatch-coord/grug-train/3:0";
        let span1_rows = 5;
        let path =
            write_two_span_log_segment(&dir, "key", filler_key, &vec![span1_key; span1_rows]);

        let ctx = crate::query::make_ctx();
        ctx.register_table(
            datafusion::common::TableReference::bare("log"),
            Arc::new(
                NamespaceProvider::build_with_local_artifacts(
                    log_arrow(),
                    std::slice::from_ref(&path),
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let batches = ctx
            .sql("SELECT key FROM \"log\" WHERE key LIKE '%final2-adoptedbatch%' ORDER BY seq")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            first_column_strings(&batches),
            vec![span1_key.to_string(); span1_rows],
            "the key substring query must return exactly the matching job's rows"
        );

        let plan = NamespaceProvider::build_with_local_artifacts(log_arrow(), &[path])
            .unwrap()
            .scan(
                &ctx.state(),
                None,
                std::slice::from_ref(&col("key").like(lit("%final2-adoptedbatch%"))),
                None,
            )
            .await
            .unwrap();
        assert_prunes_to_span1(&plan, span1_rows);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn non_contains_query_leaves_plan_unchanged() {
        // A query with no contains() filter must not be rewritten — the hot path
        // pays nothing. The returned plan is the untouched ListingTable scan.
        let dir = tempdir("no_contains");
        let path =
            write_two_span_log_segment(&dir, "data", "idle heartbeat ok", &["one match here"]);
        let ctx = crate::query::make_ctx();
        let provider = NamespaceProvider::build_with_local_artifacts(log_arrow(), &[path]).unwrap();
        let state = ctx.state();
        let plan = provider
            .scan(&state, None, &[col("seq").gt(lit(0_i64))], None)
            .await
            .unwrap();
        // No access-plan extension is attached when there is no contains() filter.
        let exec = plan.as_any().downcast_ref::<DataSourceExec>().unwrap();
        let cfg = exec
            .data_source()
            .as_any()
            .downcast_ref::<datafusion::datasource::physical_plan::FileScanConfig>()
            .unwrap();
        for group in &cfg.file_groups {
            for pf in group.files() {
                assert!(
                    pf.extensions.is_none(),
                    "no prune extension on the hot path"
                );
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
