use arrow::array::{Array, ListArray, ListBuilder, StringArray, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use regex::Regex;
use std::sync::Arc;
use xxhash_rust::xxh3;

/// Clean text using the SlimPajama text cleaning process.
/// 1. Lowercase
/// 2. Remove punctuation
/// 3. Replace multiple whitespace with single space
/// 4. Trim
pub fn clean_text(arr: &StringArray) -> PyResult<Arc<StringArray>> {
    let mut builder = StringBuilder::with_capacity(arr.len(), arr.len() * 50);
    let whitespace_re = Regex::new(r"\s+").map_err(|e| PyValueError::new_err(e.to_string()))?;
    let punctuation: &[char] = &[
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<',
        '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
    ];

    for i in 0..arr.len() {
        if arr.is_null(i) {
            builder.append_null();
            continue;
        }

        let text = arr.value(i);
        let lower = text.to_lowercase();
        let no_punct: String = lower.chars().filter(|c| !punctuation.contains(c)).collect();
        let normalized = whitespace_re.replace_all(&no_punct, " ");
        builder.append_value(normalized.trim());
    }

    Ok(Arc::new(builder.finish()))
}

/// Fused operation: Shingling -> Hashing -> Permutation -> Min extraction
pub fn compute_minhash(
    arr: &StringArray,
    num_perms: usize,
    ngram_size: usize,
    seed: u64,
) -> PyResult<Arc<dyn Array>> {
    // Generate permutations using Duplodocus strategy (Single u128 coefficient)
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut coeffs = Vec::with_capacity(num_perms);

    for _ in 0..num_perms {
        // Duplodocus ensures coefficients are odd to preserve properties of the permutation group
        let mut c = rng.gen::<u128>();
        if c % 2 == 0 {
            c = c.wrapping_add(1);
        }
        coeffs.push(c);
    }

    let mut values_builder = UInt64Builder::with_capacity(arr.len() * num_perms);
    let mut list_builder = ListBuilder::new(values_builder);

    for i in 0..arr.len() {
        if arr.is_null(i) {
            list_builder.append_null();
            continue;
        }

        let text = arr.value(i);
        let chars: Vec<char> = text.chars().collect();
        let mut signature = vec![u64::MAX; num_perms];

        if chars.len() < ngram_size {
            let hash = xxh3::xxh3_64(text.as_bytes()) as u128;
            update_signature(&mut signature, hash, &coeffs);
        } else {
            for window in chars.windows(ngram_size) {
                let s: String = window.iter().collect();
                let hash = xxh3::xxh3_64(s.as_bytes()) as u128;
                update_signature(&mut signature, hash, &coeffs);
            }
        }
        list_builder.values().append_slice(&signature);
        list_builder.append(true);
    }
    Ok(Arc::new(list_builder.finish()))
}

#[inline(always)]
fn update_signature(signature: &mut [u64], hash: u128, coeffs: &[u128]) {
    // Logic: (hash * coeff) >> 64. Similar to Duplodocus
    for (sig_val, &coeff) in signature.iter_mut().zip(coeffs) {
        let permuted_hash = (hash.wrapping_mul(coeff) >> 64) as u64;
        if permuted_hash < *sig_val {
            *sig_val = permuted_hash;
        }
    }
}

pub fn compute_lsh(input_col: &dyn Array, num_bands: usize) -> PyResult<Arc<dyn Array>> {
    let list_arr = input_col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| {
            PyValueError::new_err("Input to MinHashLSH must be a ListArray of UInt64")
        })?;

    let values_arr = list_arr
        .values()
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .ok_or_else(|| PyValueError::new_err("Inner array must be UInt64"))?;

    let mut out_values_builder = UInt64Builder::with_capacity(list_arr.len() * num_bands);
    let mut out_list_builder = ListBuilder::new(out_values_builder);

    for i in 0..list_arr.len() {
        if list_arr.is_null(i) {
            out_list_builder.append_null();
            continue;
        }

        let start = list_arr.value_offsets()[i] as usize;
        let end = list_arr.value_offsets()[i + 1] as usize;
        let sig_len = end - start;

        if sig_len == 0 {
            // Empty signature
            out_list_builder.append(true);
            continue;
        }

        if sig_len % num_bands != 0 {
            return Err(PyValueError::new_err(format!(
                "Signature length {} is not divisible by num_bands {}",
                sig_len, num_bands
            )));
        }

        let rows_per_band = sig_len / num_bands;
        let slice = &values_arr.values()[start..end];

        for band_idx in 0..num_bands {
            let band_start = band_idx * rows_per_band;
            let band_end = band_start + rows_per_band;
            let band_data = &slice[band_start..band_end];
            let band_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    band_data.as_ptr() as *const u8,
                    band_data.len() * std::mem::size_of::<u64>(),
                )
            };
            let bucket_hash = xxh3::xxh3_64(band_bytes);
            out_list_builder.values().append_value(bucket_hash);
        }
        out_list_builder.append(true);
    }

    Ok(Arc::new(out_list_builder.finish()))
}
