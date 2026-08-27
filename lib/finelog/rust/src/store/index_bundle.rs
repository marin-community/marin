//! Typed, checksummed index metadata bound to one immutable Parquet segment.
//!
//! The bundle contains small index sections only. Queryable covering projections
//! stay as separate Parquet files and are referenced by descriptor sections.
//! Missing or malformed bundles are optional derived state: readers return
//! `None` and the query scans the source segment.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::store::trigram::ByteReader;

const MAGIC: &[u8; 4] = b"FIDX";
const VERSION: u8 = 1;
const FIXED_PREFIX_LEN: usize = 4 + 1 + 1 + 4;
const CHECKSUM_LEN: usize = 32;
const TEMP_SUFFIX: &str = ".tmp";
const MAX_HEADER_LEN: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChecksumAlgorithm {
    Sha256 = 1,
}

impl ChecksumAlgorithm {
    fn from_byte(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Sha256),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SectionKind {
    TrigramBloom = 1,
    ExactPostings = 2,
    ValueCounts = 3,
    CoveringProjection = 4,
    GroupExtrema = 5,
}

impl SectionKind {
    fn from_byte(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::TrigramBloom),
            2 => Some(Self::ExactPostings),
            3 => Some(Self::ValueCounts),
            4 => Some(Self::CoveringProjection),
            5 => Some(Self::GroupExtrema),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::TrigramBloom => "trigram_bloom",
            Self::ExactPostings => "exact_postings",
            Self::ValueCounts => "value_counts",
            Self::CoveringProjection => "covering_projection",
            Self::GroupExtrema => "group_extrema",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Exactness {
    Lossy = 1,
    ExactRows = 2,
    Covering = 3,
    ExactAggregate = 4,
}

impl Exactness {
    fn from_byte(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Lossy),
            2 => Some(Self::ExactRows),
            3 => Some(Self::Covering),
            4 => Some(Self::ExactAggregate),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lossy => "lossy",
            Self::ExactRows => "exact_rows",
            Self::Covering => "covering",
            Self::ExactAggregate => "exact_aggregate",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentBinding {
    pub segment_id: Uuid,
    pub row_count: u64,
    pub schema_fingerprint: [u8; CHECKSUM_LEN],
    pub policy_fingerprint: [u8; CHECKSUM_LEN],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionInput {
    pub id: String,
    pub kind: SectionKind,
    pub method_version: u8,
    pub exactness: Exactness,
    /// Method-specific concrete coverage, such as a column and value set.
    pub coverage: Vec<u8>,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionEntry {
    pub id: String,
    pub kind: SectionKind,
    pub method_version: u8,
    pub exactness: Exactness,
    pub checksum_algorithm: ChecksumAlgorithm,
    pub coverage: Vec<u8>,
    pub offset: u64,
    pub len: u64,
    checksum: [u8; CHECKSUM_LEN],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleHeader {
    pub binding: SegmentBinding,
    pub bundle_len: u64,
    pub checksum_algorithm: ChecksumAlgorithm,
    pub sections: Vec<SectionEntry>,
}

impl BundleHeader {
    pub fn section(&self, id: &str) -> Option<&SectionEntry> {
        self.sections.iter().find(|section| section.id == id)
    }

    pub fn matches(&self, segment_id: Uuid, row_count: u64) -> bool {
        self.binding.segment_id == segment_id && self.binding.row_count == row_count
    }
}

pub fn bundle_path(parquet_path: &Path) -> PathBuf {
    let mut path = parquet_path.as_os_str().to_os_string();
    path.push(".fidx");
    PathBuf::from(path)
}

pub fn fingerprint(bytes: &[u8]) -> [u8; CHECKSUM_LEN] {
    Sha256::digest(bytes).into()
}

pub fn serialize(binding: &SegmentBinding, sections: &[SectionInput]) -> Option<Vec<u8>> {
    let mut ids = BTreeSet::new();
    if sections.iter().any(|section| !ids.insert(&section.id)) {
        return None;
    }
    let directory_len = sections.iter().try_fold(0_usize, |total, section| {
        let entry_len = 1
            + 1
            + 1
            + 1
            + 2
            + section.id.len()
            + 4
            + section.coverage.len()
            + 8
            + 8
            + CHECKSUM_LEN;
        total.checked_add(entry_len)
    })?;
    let header_len = FIXED_PREFIX_LEN
        .checked_add(8 + 16 + 8 + CHECKSUM_LEN + CHECKSUM_LEN + 2)?
        .checked_add(directory_len)?
        .checked_add(CHECKSUM_LEN)?;
    if header_len > MAX_HEADER_LEN {
        return None;
    }
    let payload_len = sections.iter().try_fold(0_usize, |total, section| {
        total.checked_add(section.payload.len())
    })?;
    let bundle_len = header_len.checked_add(payload_len)?;
    let header_len_u32 = u32::try_from(header_len).ok()?;
    let bundle_len_u64 = u64::try_from(bundle_len).ok()?;
    let section_count = u16::try_from(sections.len()).ok()?;

    let mut out = Vec::with_capacity(bundle_len);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(ChecksumAlgorithm::Sha256 as u8);
    out.extend_from_slice(&header_len_u32.to_le_bytes());
    out.extend_from_slice(&bundle_len_u64.to_le_bytes());
    out.extend_from_slice(binding.segment_id.as_bytes());
    out.extend_from_slice(&binding.row_count.to_le_bytes());
    out.extend_from_slice(&binding.schema_fingerprint);
    out.extend_from_slice(&binding.policy_fingerprint);
    out.extend_from_slice(&section_count.to_le_bytes());

    let mut offset = u64::try_from(header_len).ok()?;
    for section in sections {
        out.push(section.kind as u8);
        out.push(section.method_version);
        out.push(section.exactness as u8);
        out.push(ChecksumAlgorithm::Sha256 as u8);
        put_bytes_u16(&mut out, section.id.as_bytes())?;
        put_bytes_u32(&mut out, &section.coverage)?;
        out.extend_from_slice(&offset.to_le_bytes());
        let len = u64::try_from(section.payload.len()).ok()?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&fingerprint(&section.payload));
        offset = offset.checked_add(len)?;
    }
    out.extend_from_slice(&fingerprint(&out));
    debug_assert_eq!(out.len(), header_len);
    for section in sections {
        out.extend_from_slice(&section.payload);
    }
    debug_assert_eq!(out.len(), bundle_len);
    Some(out)
}

pub fn parse_header(bytes: &[u8]) -> Option<BundleHeader> {
    let mut input = ByteReader::new(bytes);
    if input.take(4)? != MAGIC || input.u8()? != VERSION {
        return None;
    }
    let checksum_algorithm = ChecksumAlgorithm::from_byte(input.u8()?)?;
    let header_len = input.u32()? as usize;
    let bundle_len = input.u64()?;
    if header_len < FIXED_PREFIX_LEN + CHECKSUM_LEN
        || header_len > bytes.len()
        || bundle_len < header_len as u64
    {
        return None;
    }
    let segment_id = Uuid::from_slice(input.take(16)?).ok()?;
    let row_count = input.u64()?;
    let schema_fingerprint = input.array::<CHECKSUM_LEN>()?;
    let policy_fingerprint = input.array::<CHECKSUM_LEN>()?;
    let section_count = input.u16()? as usize;
    let mut sections = Vec::new();
    sections.try_reserve(section_count).ok()?;
    let mut ids = BTreeSet::new();
    let mut expected_offset = header_len as u64;
    for _ in 0..section_count {
        let kind = SectionKind::from_byte(input.u8()?)?;
        let method_version = input.u8()?;
        let exactness = Exactness::from_byte(input.u8()?)?;
        let section_checksum_algorithm = ChecksumAlgorithm::from_byte(input.u8()?)?;
        let id = String::from_utf8(take_bytes_u16(&mut input)?.to_vec()).ok()?;
        if !ids.insert(id.clone()) {
            return None;
        }
        let coverage = take_bytes_u32(&mut input)?.to_vec();
        let offset = input.u64()?;
        let len = input.u64()?;
        let checksum = input.array::<CHECKSUM_LEN>()?;
        let end = offset.checked_add(len)?;
        if offset != expected_offset || end > bundle_len {
            return None;
        }
        expected_offset = end;
        sections.push(SectionEntry {
            id,
            kind,
            method_version,
            exactness,
            checksum_algorithm: section_checksum_algorithm,
            coverage,
            offset,
            len,
            checksum,
        });
    }
    if bytes
        .len()
        .checked_sub(input.remaining())?
        .checked_add(CHECKSUM_LEN)?
        != header_len
    {
        return None;
    }
    let expected_directory_checksum = input.array::<CHECKSUM_LEN>()?;
    if fingerprint(&bytes[..header_len - CHECKSUM_LEN]) != expected_directory_checksum {
        return None;
    }
    if expected_offset != bundle_len {
        return None;
    }
    Some(BundleHeader {
        binding: SegmentBinding {
            segment_id,
            row_count,
            schema_fingerprint,
            policy_fingerprint,
        },
        bundle_len,
        checksum_algorithm,
        sections,
    })
}

pub fn peek_header_len(prefix: &[u8]) -> Option<usize> {
    if prefix.len() < FIXED_PREFIX_LEN || &prefix[..4] != MAGIC || prefix[4] != VERSION {
        return None;
    }
    Some(u32::from_le_bytes(prefix[6..10].try_into().ok()?) as usize)
}

pub fn read_header(path: &Path) -> Option<BundleHeader> {
    let file = File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    let mut prefix = [0_u8; FIXED_PREFIX_LEN];
    file.read_exact_at(&mut prefix, 0).ok()?;
    let header_len = peek_header_len(&prefix)?;
    if header_len > MAX_HEADER_LEN || header_len as u64 > file_len {
        return None;
    }
    let mut bytes = vec![0_u8; header_len];
    file.read_exact_at(&mut bytes, 0).ok()?;
    let header = parse_header(&bytes)?;
    (header.bundle_len == file_len).then_some(header)
}

pub fn read_section(path: &Path, header: &BundleHeader, id: &str) -> Option<Vec<u8>> {
    let section = header.section(id)?;
    let len = usize::try_from(section.len).ok()?;
    let mut payload = vec![0_u8; len];
    let file = File::open(path).ok()?;
    file.read_exact_at(&mut payload, section.offset).ok()?;
    (fingerprint(&payload) == section.checksum).then_some(payload)
}

pub fn write_bundle(
    parquet_path: &Path,
    binding: &SegmentBinding,
    sections: &[SectionInput],
) -> std::io::Result<PathBuf> {
    let bytes = serialize(binding, sections).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "index bundle is too large",
        )
    })?;
    let final_path = bundle_path(parquet_path);
    let staging = staging_path(parquet_path);
    let mut file = File::create(&staging)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    std::fs::rename(&staging, &final_path)?;
    if let Some(parent) = final_path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(final_path)
}

pub fn staging_path(parquet_path: &Path) -> PathBuf {
    let final_path = bundle_path(parquet_path);
    let mut staging = final_path.as_os_str().to_os_string();
    staging.push(TEMP_SUFFIX);
    PathBuf::from(staging)
}

fn put_bytes_u16(out: &mut Vec<u8>, bytes: &[u8]) -> Option<()> {
    out.extend_from_slice(&u16::try_from(bytes.len()).ok()?.to_le_bytes());
    out.extend_from_slice(bytes);
    Some(())
}

fn put_bytes_u32(out: &mut Vec<u8>, bytes: &[u8]) -> Option<()> {
    out.extend_from_slice(&u32::try_from(bytes.len()).ok()?.to_le_bytes());
    out.extend_from_slice(bytes);
    Some(())
}

fn take_bytes_u16<'a>(input: &mut ByteReader<'a>) -> Option<&'a [u8]> {
    let len = input.u16()? as usize;
    input.take(len)
}

fn take_bytes_u32<'a>(input: &mut ByteReader<'a>) -> Option<&'a [u8]> {
    let len = input.u32()? as usize;
    input.take(len)
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    fn binding() -> SegmentBinding {
        SegmentBinding {
            segment_id: Uuid::from_u128(7),
            row_count: 42,
            schema_fingerprint: fingerprint(b"schema"),
            policy_fingerprint: fingerprint(b"policy"),
        }
    }

    fn sections() -> Vec<SectionInput> {
        vec![
            SectionInput {
                id: "trigram:name".to_string(),
                kind: SectionKind::TrigramBloom,
                method_version: 2,
                exactness: Exactness::Lossy,
                coverage: b"name\0span_rows=16384".to_vec(),
                payload: b"bloom payload".to_vec(),
            },
            SectionInput {
                id: "counts:service".to_string(),
                kind: SectionKind::ValueCounts,
                method_version: 1,
                exactness: Exactness::ExactAggregate,
                coverage: b"service".to_vec(),
                payload: b"count payload".to_vec(),
            },
            SectionInput {
                id: "exact-postings".to_string(),
                kind: SectionKind::ExactPostings,
                method_version: 1,
                exactness: Exactness::ExactRows,
                coverage: b"name".to_vec(),
                payload: b"postings payload".to_vec(),
            },
            SectionInput {
                id: "projection:training-status".to_string(),
                kind: SectionKind::CoveringProjection,
                method_version: 1,
                exactness: Exactness::Covering,
                coverage: b"projection descriptor".to_vec(),
                payload: b"projection reference payload".to_vec(),
            },
        ]
    }

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "finelog_index_bundle_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn bundle_round_trip_reads_sections_independently() {
        let parquet = temp_path("round_trip.parquet");
        let path = write_bundle(&parquet, &binding(), &sections()).unwrap();
        let header = read_header(&path).unwrap();

        assert!(header.matches(Uuid::from_u128(7), 42));
        assert_eq!(
            read_section(&path, &header, "trigram:name").as_deref(),
            Some(b"bloom payload".as_slice())
        );
        assert_eq!(
            read_section(&path, &header, "counts:service").as_deref(),
            Some(b"count payload".as_slice())
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn corruption_is_isolated_for_every_section_kind() {
        let inputs = sections();
        for corrupt_id in inputs.iter().map(|section| section.id.as_str()) {
            let parquet = temp_path(&format!("section_corruption_{corrupt_id}.parquet"));
            let path = write_bundle(&parquet, &binding(), &inputs).unwrap();
            let header = read_header(&path).unwrap();
            let corrupt = header.section(corrupt_id).unwrap();
            let file = File::options().read(true).write(true).open(&path).unwrap();
            let mut byte = [0_u8; 1];
            file.read_exact_at(&mut byte, corrupt.offset).unwrap();
            byte[0] ^= 1;
            file.write_all_at(&byte, corrupt.offset).unwrap();

            assert!(read_section(&path, &header, corrupt_id).is_none());
            for intact_id in inputs
                .iter()
                .map(|section| section.id.as_str())
                .filter(|id| *id != corrupt_id)
            {
                assert!(
                    read_section(&path, &header, intact_id).is_some(),
                    "corrupting {corrupt_id} disabled {intact_id}"
                );
            }
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn corrupt_directory_disables_the_bundle() {
        let mut bytes = serialize(&binding(), &sections()).unwrap();
        let header_len = peek_header_len(&bytes).unwrap();
        bytes[header_len - CHECKSUM_LEN - 1] ^= 1;

        assert!(parse_header(&bytes[..header_len]).is_none());
    }

    #[test]
    fn truncated_bundle_is_rejected_before_section_reads() {
        let parquet = temp_path("truncated.parquet");
        let path = write_bundle(&parquet, &binding(), &sections()).unwrap();
        let mut bytes = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();
        bytes.pop();
        File::create(&path).unwrap().write_all(&bytes).unwrap();

        assert!(read_header(&path).is_none());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn oversized_declared_header_is_rejected_before_allocation() {
        let path = temp_path("oversized_header.fidx");
        let mut prefix = [0_u8; FIXED_PREFIX_LEN];
        prefix[..4].copy_from_slice(MAGIC);
        prefix[4] = VERSION;
        prefix[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
        File::create(&path).unwrap().write_all(&prefix).unwrap();

        assert!(read_header(&path).is_none());
        std::fs::remove_file(path).ok();
    }
}
