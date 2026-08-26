//! Versioned physical partitioning inside a logical namespace.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use arrow::array::{Array, StringArray, UInt32Array};
use arrow::compute::take;
use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;

/// A hidden partition value stamped on a physical segment.
///
/// The spec id makes layout changes additive: queries prune segments written by
/// the current spec and conservatively retain unpartitioned or older-spec files.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct SegmentPartition {
    pub spec_id: u32,
    pub values: BTreeMap<String, String>,
}

impl SegmentPartition {
    pub fn value(&self, field: &str) -> Option<&str> {
        self.values.get(field).map(String::as_str)
    }
}

/// Sorted rows destined for one physical partition stream.
#[derive(Debug)]
pub struct PartitionedBatches {
    pub partition: SegmentPartition,
    pub batches: Vec<RecordBatch>,
}

/// Programmable physical layout for a logical namespace.
pub trait PhysicalPartitionPolicy: std::fmt::Debug + Sync {
    fn is_current_partition(&self, partition: &SegmentPartition) -> bool;

    /// Split already-sorted batches without changing order within an output.
    fn partition_batches(
        &self,
        batches: &[RecordBatch],
    ) -> Result<Vec<PartitionedBatches>, StatsError>;

    /// Partitions selected by exact query predicates, or `None` when the query
    /// cannot be pruned safely.
    fn partitions_for_exact_values(
        &self,
        exact_values: &HashMap<String, Vec<String>>,
    ) -> Option<BTreeSet<SegmentPartition>>;

    /// Relative directory for an L1+ segment carrying `partition`.
    ///
    /// Partition metadata remains exact even when this path uses a bounded
    /// bucket. L0 never calls this method: it stays flat and unpartitioned until
    /// compaction sorts and partitions it.
    fn segment_directory(&self, partition: &SegmentPartition) -> PathBuf;
}

/// An exact hidden partition over one required UTF-8 column.
#[derive(Clone, Copy, Debug)]
pub struct StringIdentityPartitionPolicy {
    pub spec_id: u32,
    pub column: &'static str,
    pub partition_field: &'static str,
    pub directory_prefix: &'static str,
    pub directory_buckets: u32,
}

impl StringIdentityPartitionPolicy {
    fn partition(&self, value: &str) -> SegmentPartition {
        SegmentPartition {
            spec_id: self.spec_id,
            values: BTreeMap::from([(self.partition_field.to_string(), value.to_string())]),
        }
    }

    fn bucket(&self, value: &str) -> u32 {
        assert!(self.directory_buckets > 0);
        let digest = Sha256::digest(value.as_bytes());
        u32::from_be_bytes(digest[..4].try_into().expect("sha256 prefix is four bytes"))
            % self.directory_buckets
    }
}

impl PhysicalPartitionPolicy for StringIdentityPartitionPolicy {
    fn is_current_partition(&self, partition: &SegmentPartition) -> bool {
        partition.spec_id == self.spec_id
            && partition.values.len() == 1
            && partition.value(self.partition_field).is_some()
    }

    fn partition_batches(
        &self,
        batches: &[RecordBatch],
    ) -> Result<Vec<PartitionedBatches>, StatsError> {
        partition_string_batches(self.column, batches, |value| self.partition(value))
    }

    fn partitions_for_exact_values(
        &self,
        exact_values: &HashMap<String, Vec<String>>,
    ) -> Option<BTreeSet<SegmentPartition>> {
        let values = exact_values.get(self.column)?;
        Some(values.iter().map(|value| self.partition(value)).collect())
    }

    fn segment_directory(&self, partition: &SegmentPartition) -> PathBuf {
        assert!(self.is_current_partition(partition));
        let value = partition
            .value(self.partition_field)
            .expect("current identity partition carries its field");
        Path::new(self.directory_prefix).join(format!("{:02}", self.bucket(value)))
    }
}

/// Final path for a segment under a namespace directory.
///
/// L0 is deliberately flat regardless of any partition metadata. L1+ follows
/// the active physical policy when the footer carries a current partition.
pub fn segment_path(
    namespace_dir: &Path,
    filename: &str,
    level: i32,
    partition: Option<&SegmentPartition>,
    policy: Option<&dyn PhysicalPartitionPolicy>,
) -> PathBuf {
    if level == 0 {
        return namespace_dir.join(filename);
    }
    match (partition, policy) {
        (Some(partition), Some(policy)) if policy.is_current_partition(partition) => namespace_dir
            .join(policy.segment_directory(partition))
            .join(filename),
        _ => namespace_dir.join(filename),
    }
}

pub(crate) fn select_rows(
    batch: &RecordBatch,
    row_indices: Vec<u32>,
) -> Result<RecordBatch, StatsError> {
    let indices = UInt32Array::from(row_indices);
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| StatsError::Internal(format!("select record batch rows: {error}")))?;
    RecordBatch::try_new(batch.schema(), columns)
        .map_err(|error| StatsError::Internal(format!("build selected record batch: {error}")))
}

fn partition_string_batches(
    column: &str,
    batches: &[RecordBatch],
    partition_for: impl Fn(&str) -> SegmentPartition,
) -> Result<Vec<PartitionedBatches>, StatsError> {
    let mut outputs: BTreeMap<SegmentPartition, Vec<RecordBatch>> = BTreeMap::new();
    let mut cached_partitions: HashMap<String, SegmentPartition> = HashMap::new();
    for batch in batches {
        let values = batch
            .column_by_name(column)
            .and_then(|column| column.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "physical partition policy requires a UTF-8 {:?} column",
                    column
                ))
            })?;
        let mut indices: BTreeMap<SegmentPartition, Vec<u32>> = BTreeMap::new();
        for row in 0..batch.num_rows() {
            if values.is_null(row) {
                return Err(StatsError::SchemaValidation(format!(
                    "physical partition policy requires non-null {:?} values",
                    column
                )));
            }
            let value = values.value(row);
            let partition = cached_partitions
                .entry(value.to_string())
                .or_insert_with(|| partition_for(value));
            indices
                .entry(partition.clone())
                .or_default()
                .push(row as u32);
        }
        for (partition, row_indices) in indices {
            let output = select_rows(batch, row_indices)?;
            outputs.entry(partition).or_default().push(output);
        }
    }
    Ok(outputs
        .into_iter()
        .map(|(partition, batches)| PartitionedBatches { partition, batches })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    const IDENTITY_POLICY: StringIdentityPartitionPolicy = StringIdentityPartitionPolicy {
        spec_id: 4,
        column: "name",
        partition_field: "name",
        directory_prefix: "name",
        directory_buckets: 32,
    };

    fn batch(names: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(names.to_vec()))],
        )
        .unwrap()
    }

    #[test]
    fn split_preserves_row_order_within_each_partition() {
        let input = batch(&["train_loss", "step", "train_loss", "global_step"]);
        let outputs = IDENTITY_POLICY.partition_batches(&[input]).unwrap();
        let train = IDENTITY_POLICY.partition("train_loss");
        let train_output = outputs
            .iter()
            .find(|output| output.partition == train)
            .unwrap();
        let names = train_output.batches[0]
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            names.iter().collect::<Vec<_>>(),
            vec![Some("train_loss"), Some("train_loss")]
        );
    }

    #[test]
    fn exact_values_map_to_the_same_partitions_as_ingestion() {
        let exact = HashMap::from([(
            "name".to_string(),
            vec!["train_loss".to_string(), "step".to_string()],
        )]);
        assert_eq!(
            IDENTITY_POLICY.partitions_for_exact_values(&exact).unwrap(),
            BTreeSet::from([
                IDENTITY_POLICY.partition("train_loss"),
                IDENTITY_POLICY.partition("step"),
            ])
        );
        assert!(IDENTITY_POLICY
            .partitions_for_exact_values(&HashMap::new())
            .is_none());
    }

    #[test]
    fn identity_partition_keeps_the_exact_value_out_of_the_namespace() {
        let outputs = IDENTITY_POLICY
            .partition_batches(&[batch(&["run/with/slashes", "run+long"])])
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .any(|output| { output.partition.value("name") == Some("run/with/slashes") }));
        assert_eq!(
            IDENTITY_POLICY
                .partitions_for_exact_values(&HashMap::from([(
                    "name".to_string(),
                    vec!["run+long".to_string()],
                )]))
                .unwrap(),
            BTreeSet::from([IDENTITY_POLICY.partition("run+long")])
        );
    }

    #[test]
    fn physical_path_is_bounded_while_partition_metadata_stays_exact() {
        let first = IDENTITY_POLICY.partition("run/with/slashes");
        let second = IDENTITY_POLICY.partition("a different run");
        let first_path = segment_path(
            Path::new("levanter.metrics"),
            "seg_L1_0001.parquet",
            1,
            Some(&first),
            Some(&IDENTITY_POLICY),
        );
        assert_eq!(first.value("name"), Some("run/with/slashes"));
        assert_eq!(first_path.components().count(), 4);
        assert_eq!(
            first_path
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .file_name()
                .unwrap(),
            "name"
        );
        assert_ne!(first, second);
        assert_eq!(
            segment_path(
                Path::new("levanter.metrics"),
                "seg_L0_0001.parquet",
                0,
                Some(&first),
                Some(&IDENTITY_POLICY),
            ),
            Path::new("levanter.metrics/seg_L0_0001.parquet")
        );
    }
}
