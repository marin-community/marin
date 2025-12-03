use pyo3::prelude::*;

mod bloom;
mod hashing;
mod marshaling;

use bloom::Bloom;
use hashing::HashAlgorithm;

#[pymodule]
fn dupekit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Bloom>()?;
    m.add_class::<HashAlgorithm>()?;

    // Hashing functions
    m.add_function(wrap_pyfunction!(hashing::hash_blake2, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::hash_blake3, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::hash_xxh3_128, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::hash_xxh3_64, m)?)?;
    m.add_function(wrap_pyfunction!(hashing::hash_xxh3_64_batch, m)?)?;

    // Marshaling Benchmarks
    m.add_function(wrap_pyfunction!(marshaling::process_native, m)?)?;
    m.add_function(wrap_pyfunction!(marshaling::process_arrow_batch, m)?)?;
    m.add_function(wrap_pyfunction!(marshaling::process_rust_structs, m)?)?;
    m.add_function(wrap_pyfunction!(marshaling::process_dicts_batch, m)?)?;
    m.add_function(wrap_pyfunction!(marshaling::process_dicts_loop, m)?)?;
    m.add_class::<marshaling::Document>()?;

    Ok(())
}
