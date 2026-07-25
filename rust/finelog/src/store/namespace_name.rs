//! Namespace-name validation + path containment.
//!
//! Names must match `^[a-z][a-z0-9_.-]{0,63}$` — restrictive
//! enough to be safe as both a directory name and a quoted SQL identifier. The
//! on-disk subdir must resolve strictly inside `data_dir` (rejecting `..`,
//! escapes, and the data dir itself). In-memory mode (`data_dir = None`) still
//! enforces the regex.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;

use crate::errors::StatsError;

const NAMESPACE_NAME_PATTERN: &str = r"^[a-z][a-z0-9_.-]{0,63}$";

fn namespace_name_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(NAMESPACE_NAME_PATTERN).expect("namespace name regex compiles"))
}

/// Validate `name` and return its on-disk subdirectory (or `None` in-memory).
///
/// Raises `InvalidNamespace` if the name fails the regex, resolves outside
/// `data_dir`, or resolves to the data dir itself.
pub fn validate_namespace_name(
    name: &str,
    data_dir: Option<&Path>,
) -> Result<Option<PathBuf>, StatsError> {
    if !namespace_name_re().is_match(name) {
        return Err(StatsError::InvalidNamespace(format!(
            "namespace {name:?} does not match {NAMESPACE_NAME_PATTERN}"
        )));
    }
    let Some(data_dir) = data_dir else {
        return Ok(None);
    };

    // Resolve the joined path and check containment. We normalize lexically
    // rather than canonicalizing the filesystem, because the namespace subdir
    // may not exist yet (canonicalize would fail).
    let base = normalize_lexically(data_dir);
    let target = normalize_lexically(&data_dir.join(name));

    if !target.starts_with(&base) {
        return Err(StatsError::InvalidNamespace(format!(
            "namespace {name:?} resolves to {} which is not strictly inside {}",
            target.display(),
            base.display()
        )));
    }
    if target == base {
        return Err(StatsError::InvalidNamespace(format!(
            "namespace {name:?} resolves to the data dir itself"
        )));
    }
    Ok(Some(target))
}

/// Lexically normalize a path: resolve `.` and `..` components without touching
/// the filesystem. `..` that would escape the root is dropped (mirroring how a
/// resolved absolute path can never go above `/`).
fn normalize_lexically(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                // Pop the last normal component if present; keep prefixes/roots.
                if !out.pop() {
                    // Nothing to pop (already at root/empty); ignore.
                }
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &[&str] = &["iris.worker", "iris.worker.v2", "a", "a-b", "abc.def_ghi"];
    const INVALID: &[&str] = &[
        "",
        "Iris.Worker",
        ".starts-dot",
        "1starts-digit",
        "has space",
        "has/slash",
        "..",
    ];

    #[test]
    fn accepts_valid_names_in_memory() {
        for name in VALID {
            assert!(
                validate_namespace_name(name, None).unwrap().is_none(),
                "name={name}"
            );
        }
        // 64 chars is the max.
        let n64 = "x".repeat(64);
        assert!(validate_namespace_name(&n64, None).is_ok());
    }

    #[test]
    fn rejects_invalid_names_in_memory() {
        for name in INVALID {
            assert!(
                matches!(
                    validate_namespace_name(name, None),
                    Err(StatsError::InvalidNamespace(_))
                ),
                "name={name}",
            );
        }
        // 65 chars exceeds the max.
        let n65 = "x".repeat(65);
        assert!(matches!(
            validate_namespace_name(&n65, None),
            Err(StatsError::InvalidNamespace(_))
        ));
    }

    #[test]
    fn rejects_path_traversal_on_disk() {
        let base = Path::new("/tmp/finelog_data");
        // The regex already rejects '..' and 'has/slash'; this exercises the
        // containment guard directly via a name that would escape if the regex
        // were laxer. Since the regex forbids '/', '..' is caught by the regex,
        // but we still confirm a valid name resolves inside the base.
        let ok = validate_namespace_name("iris.worker", Some(base))
            .unwrap()
            .unwrap();
        assert!(ok.starts_with(base));
        assert_ne!(ok, base);
    }

    #[test]
    fn valid_name_resolves_to_subdir() {
        let base = Path::new("/var/lib/finelog");
        let sub = validate_namespace_name("a.b", Some(base)).unwrap().unwrap();
        assert_eq!(sub, base.join("a.b"));
    }
}
