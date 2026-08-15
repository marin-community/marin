use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use xxhash_rust::xxh3::Xxh3;

const SIGNATURE_WORDS: usize = 64;
const SIGNATURE_BITS: usize = SIGNATURE_WORDS * u64::BITS as usize;
const SIGNATURE_HASHES: usize = 4;
const SIGNATURE_HASH_BITS: usize = 12;
const SIGNATURE_BIT_MASK: u64 = SIGNATURE_BITS as u64 - 1;
const FINGERPRINT_BITS: u32 = 24;
const FINGERPRINT_BYTES: usize = FINGERPRINT_BITS as usize / u8::BITS as usize;
const FINGERPRINT_MASK: u32 = (1 << FINGERPRINT_BITS) - 1;
const FINGERPRINT_CAPACITY: usize = 1_333;
const FINGERPRINT_DATA_BYTES: usize = FINGERPRINT_CAPACITY * FINGERPRINT_BYTES;
const FINGERPRINT_SERIALIZED_SIZE: usize = 64 + FINGERPRINT_DATA_BYTES;

#[pyclass(frozen)]
pub struct TokenNgrams {
    normalized: String,
    token_starts: Vec<u32>,
    chars: usize,
    lines: usize,
    ngram_size: usize,
    window_size: usize,
    hashes: Vec<u64>,
    window_starts: Vec<u32>,
}

#[pyclass(frozen)]
pub struct TokenNgramSignature {
    token_count: usize,
    ngram_size: usize,
    bitmap: [u64; SIGNATURE_WORDS],
}

#[pyclass(frozen)]
pub struct TokenNgramFingerprintSignature {
    chars: usize,
    lines: usize,
    token_count: usize,
    ngram_count: usize,
    ngram_size: usize,
    normalized_sequence_hash: u128,
    theta: u32,
    fingerprint_count: usize,
    fingerprints: [u8; FINGERPRINT_DATA_BYTES],
}

fn normalize_text(text: String) -> Result<(String, Vec<u32>), &'static str> {
    let lower = text.to_lowercase();
    let mut normalized = String::with_capacity(lower.len());
    let mut token_starts = Vec::new();
    for token in lower.split_whitespace() {
        if !token_starts.is_empty() {
            normalized.push(' ');
        }
        token_starts
            .push(u32::try_from(normalized.len()).map_err(|_| "normalized text exceeds 4 GiB")?);
        normalized.push_str(token);
    }
    Ok((normalized, token_starts))
}

impl TokenNgrams {
    fn from_text(text: String, ngram_size: usize) -> Result<Self, &'static str> {
        Self::from_text_with_hasher(text, ngram_size, Self::hash_window)
    }

    fn from_text_with_hasher<F>(
        text: String,
        ngram_size: usize,
        hash_window: F,
    ) -> Result<Self, &'static str>
    where
        F: Fn(&Self, usize) -> u64,
    {
        let chars = text.chars().count();
        let lines = text.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let (normalized, token_starts) = normalize_text(text)?;

        if token_starts.is_empty() {
            return Ok(Self {
                normalized,
                token_starts,
                chars,
                lines,
                ngram_size,
                window_size: 0,
                hashes: Vec::new(),
                window_starts: Vec::new(),
            });
        }

        let window_size = token_starts.len().min(ngram_size);
        let window_count = token_starts.len() - window_size + 1;
        let mut ngrams = Self {
            normalized,
            token_starts,
            chars,
            lines,
            ngram_size,
            window_size,
            hashes: Vec::with_capacity(window_count),
            window_starts: Vec::with_capacity(window_count),
        };
        let mut windows: Vec<(u64, u32)> = (0..window_count)
            .map(|start| (hash_window(&ngrams, start), start as u32))
            .collect();
        windows.sort_unstable_by_key(|(hash, _)| *hash);

        let mut hash_group_start = 0;
        for (hash, start) in windows {
            if ngrams.hashes.last() != Some(&hash) {
                hash_group_start = ngrams.hashes.len();
            }
            let repeated = ngrams.window_starts[hash_group_start..]
                .iter()
                .any(|existing| ngrams.windows_equal(start as usize, *existing as usize));
            if repeated {
                continue;
            }
            ngrams.hashes.push(hash);
            ngrams.window_starts.push(start);
        }
        Ok(ngrams)
    }

    fn token(&self, index: usize) -> &str {
        let start = self.token_starts[index] as usize;
        let end = self
            .token_starts
            .get(index + 1)
            .map_or(self.normalized.len(), |next| *next as usize - 1);
        &self.normalized[start..end]
    }

    fn hash_window(&self, start: usize) -> u64 {
        let mut hasher = Xxh3::new();
        hasher.update(&(self.window_size as u64).to_le_bytes());
        for index in start..start + self.window_size {
            let token = self.token(index);
            hasher.update(&(token.len() as u64).to_le_bytes());
            hasher.update(token.as_bytes());
        }
        hasher.digest()
    }

    fn windows_equal(&self, left_start: usize, right_start: usize) -> bool {
        (0..self.window_size)
            .all(|offset| self.token(left_start + offset) == self.token(right_start + offset))
    }

    fn window_equals_other(&self, start: usize, other: &Self, other_start: usize) -> bool {
        self.window_size == other.window_size
            && (0..self.window_size)
                .all(|offset| self.token(start + offset) == other.token(other_start + offset))
    }

    fn intersection_size_with(&self, other: &Self) -> usize {
        if self.hashes.is_empty() || other.hashes.is_empty() || self.ngram_size != other.ngram_size
        {
            return 0;
        }

        let mut left_index = 0;
        let mut right_index = 0;
        let mut shared = 0;
        while left_index < self.hashes.len() && right_index < other.hashes.len() {
            let left_hash = self.hashes[left_index];
            let right_hash = other.hashes[right_index];
            if left_hash < right_hash {
                left_index += 1;
                continue;
            }
            if right_hash < left_hash {
                right_index += 1;
                continue;
            }

            let left_end =
                self.hashes[left_index..].partition_point(|hash| *hash == left_hash) + left_index;
            let right_end = other.hashes[right_index..].partition_point(|hash| *hash == right_hash)
                + right_index;
            for left_start in &self.window_starts[left_index..left_end] {
                if other.window_starts[right_index..right_end]
                    .iter()
                    .any(|right_start| {
                        self.window_equals_other(*left_start as usize, other, *right_start as usize)
                    })
                {
                    shared += 1;
                }
            }
            left_index = left_end;
            right_index = right_end;
        }
        shared
    }

    fn signature_value(&self) -> TokenNgramSignature {
        let mut signature = TokenNgramSignature {
            token_count: self.token_starts.len(),
            ngram_size: self.ngram_size,
            bitmap: [0; SIGNATURE_WORDS],
        };
        for hash in &self.hashes {
            signature.insert(*hash);
        }
        signature
    }

    fn fingerprint_signature_value(&self) -> TokenNgramFingerprintSignature {
        TokenNgramFingerprintSignature::from_hashes(
            self.chars,
            self.lines,
            self.token_starts.len(),
            self.ngram_size,
            xxhash_rust::xxh3::xxh3_128(self.normalized.as_bytes()),
            &self.hashes,
        )
    }
}

impl TokenNgramSignature {
    fn from_text(text: String, ngram_size: usize) -> Result<Self, &'static str> {
        let chars = text.chars().count();
        let lines = text.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let (normalized, token_starts) = normalize_text(text)?;
        let token_count = token_starts.len();
        let mut signature = Self {
            token_count,
            ngram_size,
            bitmap: [0; SIGNATURE_WORDS],
        };
        if token_count == 0 {
            return Ok(signature);
        }

        let window_size = token_count.min(ngram_size);
        let window_count = token_count - window_size + 1;
        let ngrams = TokenNgrams {
            normalized,
            token_starts,
            chars,
            lines,
            ngram_size,
            window_size,
            hashes: Vec::new(),
            window_starts: Vec::new(),
        };
        for start in 0..window_count {
            signature.insert(ngrams.hash_window(start));
        }
        Ok(signature)
    }

    fn insert(&mut self, hash: u64) {
        for index in 0..SIGNATURE_HASHES {
            let bit = ((hash >> (index * SIGNATURE_HASH_BITS)) & SIGNATURE_BIT_MASK) as usize;
            self.bitmap[bit / u64::BITS as usize] |= 1 << (bit % u64::BITS as usize);
        }
    }

    fn may_be_subset_of_value(&self, other: &Self) -> bool {
        self.ngram_size == other.ngram_size
            && self
                .bitmap
                .iter()
                .zip(&other.bitmap)
                .all(|(member, representative)| member & !representative == 0)
    }
}

impl TokenNgramFingerprintSignature {
    fn from_text(text: String, ngram_size: usize) -> Result<Self, &'static str> {
        let ngrams = TokenNgrams::from_text(text, ngram_size)?;
        Ok(ngrams.fingerprint_signature_value())
    }

    fn from_hashes(
        chars: usize,
        lines: usize,
        token_count: usize,
        ngram_size: usize,
        normalized_sequence_hash: u128,
        hashes: &[u64],
    ) -> Self {
        let mut sorted: Vec<u32> = hashes
            .iter()
            .map(|hash| *hash as u32 & FINGERPRINT_MASK)
            .collect();
        sorted.sort_unstable();
        sorted.dedup();
        let theta = sorted
            .get(FINGERPRINT_CAPACITY)
            .map_or(u32::MAX, |_| sorted[FINGERPRINT_CAPACITY - 1]);
        sorted.truncate(FINGERPRINT_CAPACITY);
        let fingerprint_count = sorted.len();
        let mut fingerprints = [0; FINGERPRINT_DATA_BYTES];
        for (index, fingerprint) in sorted.into_iter().enumerate() {
            let offset = index * FINGERPRINT_BYTES;
            fingerprints[offset..offset + FINGERPRINT_BYTES]
                .copy_from_slice(&fingerprint.to_le_bytes()[..FINGERPRINT_BYTES]);
        }
        Self {
            chars,
            lines,
            token_count,
            ngram_count: hashes.len(),
            ngram_size,
            normalized_sequence_hash,
            theta,
            fingerprint_count,
            fingerprints,
        }
    }

    fn may_be_subset_of_value(&self, other: &Self) -> bool {
        if self.ngram_size != other.ngram_size
            || self.chars > other.chars
            || self.ngram_count > other.ngram_count
        {
            return false;
        }

        let member_end = (0..self.fingerprint_count)
            .take_while(|index| self.fingerprint_at(*index) <= other.theta)
            .count();
        let mut member_index = 0;
        let mut representative_index = 0;
        while member_index < member_end && representative_index < other.fingerprint_count {
            let member_fingerprint = self.fingerprint_at(member_index);
            let representative_fingerprint = other.fingerprint_at(representative_index);
            if member_fingerprint == representative_fingerprint {
                member_index += 1;
                representative_index += 1;
            } else if member_fingerprint > representative_fingerprint {
                representative_index += 1;
            } else {
                return false;
            }
        }
        member_index == member_end
    }

    fn fingerprint_at(&self, index: usize) -> u32 {
        let offset = index * FINGERPRINT_BYTES;
        u32::from_le_bytes([
            self.fingerprints[offset],
            self.fingerprints[offset + 1],
            self.fingerprints[offset + 2],
            0,
        ])
    }

    fn serialized_value(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FINGERPRINT_SERIALIZED_SIZE);
        for value in [
            self.chars,
            self.lines,
            self.token_count,
            self.ngram_count,
            self.ngram_size,
        ] {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&self.normalized_sequence_hash.to_le_bytes());
        bytes.extend_from_slice(&self.theta.to_le_bytes());
        bytes.extend_from_slice(&(self.fingerprint_count as u32).to_le_bytes());
        bytes.extend_from_slice(&self.fingerprints);
        debug_assert_eq!(bytes.len(), FINGERPRINT_SERIALIZED_SIZE);
        bytes
    }

    fn from_serialized(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != FINGERPRINT_SERIALIZED_SIZE {
            return Err("invalid fingerprint signature byte length");
        }
        let mut offset = 0;
        let mut read_u64 = || {
            let value = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
            value
        };
        let chars = read_u64() as usize;
        let lines = read_u64() as usize;
        let token_count = read_u64() as usize;
        let ngram_count = read_u64() as usize;
        let ngram_size = read_u64() as usize;
        let normalized_sequence_hash =
            u128::from_le_bytes(bytes[offset..offset + 16].try_into().unwrap());
        offset += 16;
        let theta = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let fingerprint_count =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if fingerprint_count > FINGERPRINT_CAPACITY {
            return Err("invalid fingerprint count");
        }
        let mut fingerprints = [0; FINGERPRINT_DATA_BYTES];
        fingerprints.copy_from_slice(&bytes[offset..]);
        Ok(Self {
            chars,
            lines,
            token_count,
            ngram_count,
            ngram_size,
            normalized_sequence_hash,
            theta,
            fingerprint_count,
            fingerprints,
        })
    }
}

#[pymethods]
impl TokenNgramSignature {
    #[new]
    fn new(py: Python<'_>, text: String, ngram_size: usize) -> PyResult<Self> {
        if ngram_size == 0 {
            return Err(PyValueError::new_err("ngram_size must be positive"));
        }
        py.detach(move || Self::from_text(text, ngram_size))
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn token_count(&self) -> usize {
        self.token_count
    }

    fn may_be_subset_of(&self, other: PyRef<'_, Self>) -> bool {
        self.may_be_subset_of_value(&other)
    }
}

#[pymethods]
impl TokenNgramFingerprintSignature {
    #[new]
    fn new(py: Python<'_>, text: String, ngram_size: usize) -> PyResult<Self> {
        if ngram_size == 0 {
            return Err(PyValueError::new_err("ngram_size must be positive"));
        }
        py.detach(move || Self::from_text(text, ngram_size))
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn chars(&self) -> usize {
        self.chars
    }

    #[getter]
    fn lines(&self) -> usize {
        self.lines
    }

    #[getter]
    fn token_count(&self) -> usize {
        self.token_count
    }

    #[getter]
    fn ngram_count(&self) -> usize {
        self.ngram_count
    }

    #[getter]
    fn normalized_sequence_hash(&self) -> u128 {
        self.normalized_sequence_hash
    }

    fn may_be_subset_of(&self, other: PyRef<'_, Self>) -> bool {
        self.may_be_subset_of_value(&other)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.serialized_value())
    }

    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Self::from_serialized(bytes).map_err(PyValueError::new_err)
    }
}

#[pymethods]
impl TokenNgrams {
    #[new]
    fn new(py: Python<'_>, text: String, ngram_size: usize) -> PyResult<Self> {
        if ngram_size == 0 {
            return Err(PyValueError::new_err("ngram_size must be positive"));
        }
        py.detach(move || Self::from_text(text, ngram_size))
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn token_count(&self) -> usize {
        self.token_starts.len()
    }

    fn __len__(&self) -> usize {
        self.hashes.len()
    }

    fn intersection_size(&self, other: PyRef<'_, Self>) -> usize {
        self.intersection_size_with(&other)
    }

    fn signature(&self) -> TokenNgramSignature {
        self.signature_value()
    }
}

#[cfg(test)]
mod tests {
    use super::{TokenNgramFingerprintSignature, TokenNgramSignature, TokenNgrams};

    fn constant_hash(_: &TokenNgrams, _: usize) -> u64 {
        1
    }

    #[test]
    fn hash_collisions_preserve_exact_set_semantics() {
        let left =
            TokenNgrams::from_text_with_hasher("one two three two three".into(), 2, constant_hash)
                .unwrap();
        let right =
            TokenNgrams::from_text_with_hasher("zero two three five".into(), 2, constant_hash)
                .unwrap();

        assert_eq!(left.hashes.len(), 3);
        assert_eq!(right.hashes.len(), 3);
        assert_eq!(left.intersection_size_with(&right), 1);
        assert_eq!(right.intersection_size_with(&left), 1);
    }

    #[test]
    fn signature_never_rejects_an_exact_subset() {
        let member = TokenNgramSignature::from_text("one two three".into(), 2).unwrap();
        let representative =
            TokenNgramSignature::from_text("zero one two three four".into(), 2).unwrap();

        assert!(member.may_be_subset_of_value(&representative));
        assert!(!representative.may_be_subset_of_value(&member));
    }

    #[test]
    fn fingerprint_signature_fits_fixed_budget() {
        assert!(std::mem::size_of::<TokenNgramFingerprintSignature>() <= 4_096);
    }

    #[test]
    fn fingerprint_signature_serialization_round_trips() {
        let signature =
            TokenNgramFingerprintSignature::from_text("one\ntwo three".into(), 2).unwrap();
        let restored =
            TokenNgramFingerprintSignature::from_serialized(&signature.serialized_value()).unwrap();

        assert_eq!(restored.chars, signature.chars);
        assert_eq!(restored.lines, signature.lines);
        assert_eq!(restored.token_count, signature.token_count);
        assert_eq!(restored.ngram_count, signature.ngram_count);
        assert_eq!(restored.fingerprints, signature.fingerprints);
    }
}
