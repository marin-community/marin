// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Fixtures shared by tests across the crate's modules. Module-specific ones (a served
//! store, a Connect client) live beside the module they exercise.

/// A fresh directory under the system temp dir, unique per call.
pub fn unique_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "finelog_{tag}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
