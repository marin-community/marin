use blake2::{Blake2b512, Digest};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use xxhash_rust::xxh3;

#[derive(Clone, Copy, Debug, PartialEq)]
// Generate __eq__ and __hash__
#[pyclass(eq, eq_int)]
pub enum HashAlgorithm {
    Blake2b,
    Blake3,
    Xxh3_128,
    Xxh3_64,
}

impl HashAlgorithm {
    pub fn hash_to_hex(&self, input: &[u8]) -> String {
        match self {
            HashAlgorithm::Blake2b => {
                let mut hasher = Blake2b512::new();
                hasher.update(input);
                hex::encode(hasher.finalize())
            }
            HashAlgorithm::Blake3 => hex::encode(blake3::hash(input).as_bytes()),
            HashAlgorithm::Xxh3_128 => format!("{:032x}", xxh3::xxh3_128(input)),
            HashAlgorithm::Xxh3_64 => format!("{:016x}", xxh3::xxh3_64(input)),
        }
    }
}

#[pyfunction]
pub fn hash_blake2(text: &[u8]) -> Vec<u8> {
    let mut hasher = Blake2b512::new();
    hasher.update(text);
    hasher.finalize().to_vec()
}

#[pyfunction]
pub fn hash_blake3(text: &[u8]) -> [u8; 32] {
    // Dereference to copy the bytes from the helper struct into a pure array
    *blake3::hash(text).as_bytes()
}

#[pyfunction]
pub fn hash_xxh3_128(text: &[u8]) -> u128 {
    xxh3::xxh3_128(text)
}

#[pyfunction]
pub fn hash_xxh3_64(text: &[u8]) -> u64 {
    xxh3::xxh3_64(text)
}

#[pyfunction]
// Bound<'_, ...>: Smart pointer to allow safe access to Python's internal memory.
pub fn hash_xxh3_64_batch(texts: Vec<Bound<'_, PyBytes>>) -> Vec<u64> {
    let mut results = Vec::with_capacity(texts.len());
    for text in texts {
        results.push(xxh3::xxh3_64(text.as_bytes()));
    }
    results
}
