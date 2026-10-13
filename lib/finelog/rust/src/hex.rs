//! Lowercase hex rendering for digests that appear in object keys and IDs.

/// Render `bytes` as lowercase hex, two characters per byte.
pub(crate) fn encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
