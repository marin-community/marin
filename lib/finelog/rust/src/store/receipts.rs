//! Durable idempotency receipt manifests.
//!
//! Each sealed L0 writes one checksummed manifest containing every receipt in
//! that segment. Manifests live independently from Parquet so compaction and
//! data retention cannot erase retry history. They are retained until the
//! namespace is explicitly dropped. L0 footer metadata bridges the crash
//! window between the segment rename and its manifest becoming durable.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::store::segment::{discover_segments, read_segment_receipts};
use crate::store::types::{parse_seg_filename, BatchReceipt};

const RECEIPT_DIRECTORY: &str = "_receipts";
const RECEIPT_MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct ReceiptManifest {
    version: u32,
    segment: String,
    receipts_sha256: String,
    receipts: Vec<BatchReceipt>,
}

#[derive(Debug)]
pub struct ReceiptRecovery {
    pub receipts: Vec<BatchReceipt>,
    pub manifest_count: usize,
    pub footer_repairs: usize,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn receipt_dir(namespace_dir: &Path) -> PathBuf {
    namespace_dir.join(RECEIPT_DIRECTORY)
}

fn segment_name(segment_path: &Path) -> Result<&str, StatsError> {
    segment_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            StatsError::Internal(format!(
                "segment path {} has no UTF-8 filename",
                segment_path.display()
            ))
        })
}

fn receipt_path(namespace_dir: &Path, segment: &str) -> PathBuf {
    receipt_dir(namespace_dir).join(format!("{segment}.receipts.json"))
}

fn sync_directory(path: &Path) -> Result<(), StatsError> {
    std::fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            StatsError::Internal(format!("fsync directory {}: {error}", path.display()))
        })
}

fn receipt_checksum(receipts: &[BatchReceipt]) -> Result<String, StatsError> {
    let encoded = serde_json::to_vec(receipts).map_err(|error| {
        StatsError::Internal(format!("encode receipt manifest checksum input: {error}"))
    })?;
    Ok(sha256_hex(&encoded))
}

fn decode_manifest(path: &Path) -> Result<ReceiptManifest, StatsError> {
    let bytes = std::fs::read(path).map_err(|error| {
        StatsError::Internal(format!("read receipt manifest {}: {error}", path.display()))
    })?;
    let manifest: ReceiptManifest = serde_json::from_slice(&bytes).map_err(|error| {
        StatsError::Internal(format!(
            "decode receipt manifest {}: {error}",
            path.display()
        ))
    })?;
    if manifest.version != RECEIPT_MANIFEST_VERSION {
        return Err(StatsError::Internal(format!(
            "receipt manifest {} has unsupported version {}",
            path.display(),
            manifest.version
        )));
    }
    let expected_name = format!("{}.receipts.json", manifest.segment);
    if path.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str()) {
        return Err(StatsError::Internal(format!(
            "receipt manifest {} does not match segment {:?}",
            path.display(),
            manifest.segment
        )));
    }
    let checksum = receipt_checksum(&manifest.receipts)?;
    if checksum != manifest.receipts_sha256 {
        return Err(StatsError::Internal(format!(
            "receipt manifest {} checksum mismatch",
            path.display()
        )));
    }
    Ok(manifest)
}

fn insert_receipt(
    receipts: &mut BTreeMap<String, BatchReceipt>,
    receipt: BatchReceipt,
    source: &Path,
) -> Result<(), StatsError> {
    if let Some(existing) = receipts.get(&receipt.batch_id) {
        if existing != &receipt {
            return Err(StatsError::Internal(format!(
                "conflicting durable receipts for batch {:?} while reading {}",
                receipt.batch_id,
                source.display()
            )));
        }
        return Ok(());
    }
    receipts.insert(receipt.batch_id.clone(), receipt);
    Ok(())
}

/// Persist one immutable, checksummed receipt manifest for a sealed L0.
///
/// Returns `true` when a new manifest was created and `false` when an identical
/// manifest already existed.
pub fn write_receipt_manifest(
    namespace_dir: &Path,
    segment_path: &Path,
    receipts: &[BatchReceipt],
) -> Result<bool, StatsError> {
    let segment = segment_name(segment_path)?;
    let directory = receipt_dir(namespace_dir);
    if !directory.exists() {
        std::fs::create_dir_all(&directory).map_err(|error| {
            StatsError::Internal(format!(
                "create receipt directory {}: {error}",
                directory.display()
            ))
        })?;
        sync_directory(namespace_dir)?;
    }
    let final_path = receipt_path(namespace_dir, segment);
    let manifest = ReceiptManifest {
        version: RECEIPT_MANIFEST_VERSION,
        segment: segment.to_string(),
        receipts_sha256: receipt_checksum(receipts)?,
        receipts: receipts.to_vec(),
    };
    if final_path.exists() {
        let existing = decode_manifest(&final_path)?;
        if existing.segment != manifest.segment
            || existing.receipts_sha256 != manifest.receipts_sha256
            || existing.receipts != manifest.receipts
        {
            return Err(StatsError::Internal(format!(
                "receipt manifest conflict for segment {segment:?}"
            )));
        }
        return Ok(false);
    }
    let bytes = serde_json::to_vec(&manifest).map_err(|error| {
        StatsError::Internal(format!(
            "encode receipt manifest for segment {segment:?}: {error}"
        ))
    })?;
    let staging_path = final_path.with_extension("json.tmp");
    {
        let mut file = std::fs::File::create(&staging_path).map_err(|error| {
            StatsError::Internal(format!(
                "create receipt staging file {}: {error}",
                staging_path.display()
            ))
        })?;
        file.write_all(&bytes).map_err(|error| {
            StatsError::Internal(format!(
                "write receipt staging file {}: {error}",
                staging_path.display()
            ))
        })?;
        file.sync_all().map_err(|error| {
            StatsError::Internal(format!(
                "fsync receipt staging file {}: {error}",
                staging_path.display()
            ))
        })?;
    }
    std::fs::rename(&staging_path, &final_path).map_err(|error| {
        StatsError::Internal(format!(
            "rename receipt manifest {} -> {}: {error}",
            staging_path.display(),
            final_path.display()
        ))
    })?;
    sync_directory(&directory)?;
    Ok(true)
}

/// Rebuild receipt truth from independent manifests and surviving L0 footers.
///
/// Startup is O(manifests + L0 filenames + receipts). It reads a Parquet footer
/// only when an L0 lacks a validated manifest. Footer-only receipts are
/// immediately re-homed; legacy L0s get an empty marker and are not rescanned.
pub fn recover_receipts(namespace_dir: &Path) -> Result<ReceiptRecovery, StatsError> {
    let mut receipts = BTreeMap::new();
    let mut manifested_segments = BTreeSet::new();
    let mut manifest_count = 0;
    let directory = receipt_dir(namespace_dir);
    if directory.exists() {
        let entries = std::fs::read_dir(&directory).map_err(|error| {
            StatsError::Internal(format!(
                "read receipt directory {}: {error}",
                directory.display()
            ))
        })?;
        for entry in entries {
            let path = entry
                .map_err(|error| {
                    StatsError::Internal(format!(
                        "read receipt directory entry {}: {error}",
                        directory.display()
                    ))
                })?
                .path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let manifest = decode_manifest(&path)?;
            manifest_count += 1;
            manifested_segments.insert(manifest.segment.clone());
            for receipt in manifest.receipts {
                insert_receipt(&mut receipts, receipt, &path)?;
            }
        }
    }

    let segments = discover_segments(namespace_dir);
    let mut footer_repairs = 0;
    for segment in &segments {
        let name = segment_name(segment)?;
        let Some((level, _)) = parse_seg_filename(name) else {
            continue;
        };
        if level != 0 || manifested_segments.contains(name) {
            continue;
        }
        let footer_receipts = read_segment_receipts(segment)?;
        for receipt in &footer_receipts {
            insert_receipt(&mut receipts, receipt.clone(), segment)?;
        }
        if write_receipt_manifest(namespace_dir, segment, &footer_receipts)? {
            manifest_count += 1;
            footer_repairs += 1;
        }
    }
    Ok(ReceiptRecovery {
        receipts: receipts.into_values().collect(),
        manifest_count,
        footer_repairs,
    })
}

#[cfg(test)]
pub fn receipt_manifest_count(namespace_dir: &Path) -> usize {
    std::fs::read_dir(receipt_dir(namespace_dir))
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| {
                    entry
                        .path()
                        .extension()
                        .and_then(|extension| extension.to_str())
                        == Some("json")
                })
                .count()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::store::segment::{write_segment_to_dir, write_segment_to_dir_with_receipts};

    fn tempdir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("finelog_receipts_{tag}_{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1_i64, 2]))]).unwrap()
    }

    fn receipt(batch_id: &str, first_seq: i64) -> BatchReceipt {
        BatchReceipt {
            batch_id: batch_id.to_string(),
            payload_sha256: format!("digest-{batch_id}"),
            rows_written: 1,
            first_seq,
            last_seq: first_seq,
        }
    }

    #[test]
    fn one_manifest_batches_receipts_and_survives_segment_deletion() {
        let dir = tempdir("batched");
        let receipts = vec![receipt("a", 1), receipt("b", 2)];
        let (segment, _) =
            write_segment_to_dir_with_receipts(&dir, 0, 1, &batch(), &receipts).unwrap();

        let first = recover_receipts(&dir).unwrap();
        assert_eq!(first.footer_repairs, 1);
        assert_eq!(first.manifest_count, 1);
        assert_eq!(first.receipts, receipts);
        assert_eq!(receipt_manifest_count(&dir), 1);

        std::fs::remove_file(segment).unwrap();
        let second = recover_receipts(&dir).unwrap();
        assert_eq!(second.footer_repairs, 0);
        assert_eq!(second.manifest_count, 1);
        assert_eq!(second.receipts, receipts);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn legacy_l0_gets_empty_marker_and_is_not_rescanned() {
        let dir = tempdir("legacy");
        write_segment_to_dir(&dir, 0, 1, &batch()).unwrap();

        let first = recover_receipts(&dir).unwrap();
        assert_eq!(first.footer_repairs, 1);
        assert_eq!(first.manifest_count, 1);
        assert!(first.receipts.is_empty());

        let second = recover_receipts(&dir).unwrap();
        assert_eq!(second.footer_repairs, 0);
        assert_eq!(second.manifest_count, 1);
        assert!(second.receipts.is_empty());
        std::fs::remove_dir_all(dir).ok();
    }
}
