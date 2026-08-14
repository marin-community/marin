use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use xxhash_rust::xxh3::Xxh3;

#[pyclass(frozen)]
pub struct TokenNgrams {
    normalized: String,
    tokens: Vec<Range<usize>>,
    ngram_size: usize,
    window_size: usize,
    starts_by_hash: HashMap<u64, Vec<usize>>,
    len: usize,
}

impl TokenNgrams {
    fn from_text(text: String, ngram_size: usize) -> Self {
        let normalized = text.to_lowercase();
        let base = normalized.as_ptr() as usize;
        let tokens: Vec<Range<usize>> = normalized
            .split_whitespace()
            .map(|token| {
                let start = token.as_ptr() as usize - base;
                start..start + token.len()
            })
            .collect();
        if tokens.is_empty() {
            return Self {
                normalized,
                tokens,
                ngram_size,
                window_size: 0,
                starts_by_hash: HashMap::new(),
                len: 0,
            };
        }

        let window_size = tokens.len().min(ngram_size);
        let window_count = tokens.len() - window_size + 1;
        let mut ngrams = Self {
            normalized,
            tokens,
            ngram_size,
            window_size,
            starts_by_hash: HashMap::with_capacity(window_count),
            len: 0,
        };
        for start in 0..window_count {
            let hash = ngrams.hash_window(start);
            let repeated = ngrams.starts_by_hash.get(&hash).is_some_and(|starts| {
                starts
                    .iter()
                    .any(|existing| ngrams.windows_equal(start, *existing))
            });
            if repeated {
                continue;
            }
            ngrams.starts_by_hash.entry(hash).or_default().push(start);
            ngrams.len += 1;
        }
        ngrams
    }

    fn token(&self, index: usize) -> &str {
        &self.normalized[self.tokens[index].clone()]
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
        (0..self.window_size)
            .all(|offset| self.token(start + offset) == other.token(other_start + offset))
    }
}

#[pymethods]
impl TokenNgrams {
    #[new]
    fn new(py: Python<'_>, text: String, ngram_size: usize) -> PyResult<Self> {
        if ngram_size == 0 {
            return Err(PyValueError::new_err("ngram_size must be positive"));
        }
        Ok(py.detach(move || Self::from_text(text, ngram_size)))
    }

    #[getter]
    fn token_count(&self) -> usize {
        self.tokens.len()
    }

    fn __len__(&self) -> usize {
        self.len
    }

    fn intersection_size(&self, other: PyRef<'_, Self>) -> usize {
        if self.len == 0 || other.len == 0 || self.ngram_size != other.ngram_size {
            return 0;
        }
        let (left, right) = if self.starts_by_hash.len() <= other.starts_by_hash.len() {
            (self, &*other)
        } else {
            (&*other, self)
        };
        let mut shared = 0;
        for (hash, left_starts) in &left.starts_by_hash {
            let Some(right_starts) = right.starts_by_hash.get(hash) else {
                continue;
            };
            for left_start in left_starts {
                if right_starts
                    .iter()
                    .any(|right_start| left.window_equals_other(*left_start, right, *right_start))
                {
                    shared += 1;
                }
            }
        }
        shared
    }
}
