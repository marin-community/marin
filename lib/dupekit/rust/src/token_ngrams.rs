use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use xxhash_rust::xxh3::Xxh3;

const SIGNATURE_WORDS: usize = 64;
const SIGNATURE_BITS: usize = SIGNATURE_WORDS * u64::BITS as usize;
const SIGNATURE_HASHES: usize = 4;
const SIGNATURE_HASH_BITS: usize = 12;
const SIGNATURE_BIT_MASK: u64 = SIGNATURE_BITS as u64 - 1;

#[pyclass(frozen)]
pub struct TokenNgrams {
    normalized: String,
    token_starts: Vec<u32>,
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
        let (normalized, token_starts) = normalize_text(text)?;

        if token_starts.is_empty() {
            return Ok(Self {
                normalized,
                token_starts,
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
}

impl TokenNgramSignature {
    fn from_text(text: String, ngram_size: usize) -> Result<Self, &'static str> {
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
    use super::{TokenNgramSignature, TokenNgrams};

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
}
