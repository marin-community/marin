//! Crash-safe local file operations shared by object-store implementations.

use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use super::ObjectVersion;
use crate::errors::StatsError;

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), StatsError> {
    let parent = path.parent().ok_or_else(|| {
        StatsError::Internal(format!("local object {} has no parent", path.display()))
    })?;
    std::fs::create_dir_all(parent).map_err(|error| {
        StatsError::Internal(format!(
            "create local object parent {}: {error}",
            parent.display()
        ))
    })?;
    let staging = path.with_extension(format!("tmp-{}-{}", std::process::id(), next_nonce()));
    let mut staging_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staging)
        .map_err(|error| {
            StatsError::Internal(format!(
                "create local object staging {}: {error}",
                staging.display()
            ))
        })?;
    let published = staging_file
        .write_all(bytes)
        .map_err(|error| {
            StatsError::Internal(format!(
                "write local object staging {}: {error}",
                staging.display()
            ))
        })
        .and_then(|()| {
            staging_file.sync_all().map_err(|error| {
                StatsError::Internal(format!(
                    "fsync local object staging {}: {error}",
                    staging.display()
                ))
            })
        })
        .and_then(|()| {
            std::fs::rename(&staging, path).map_err(|error| {
                StatsError::Internal(format!(
                    "publish local object {} -> {}: {error}",
                    staging.display(),
                    path.display()
                ))
            })
        });
    if let Err(error) = published {
        // A failed write (e.g. a full disk) must not leave the staging file
        // behind: staging files are exempt from cache eviction.
        return match std::fs::remove_file(&staging) {
            Ok(()) => Err(error),
            Err(cleanup) if cleanup.kind() == std::io::ErrorKind::NotFound => Err(error),
            Err(cleanup) => Err(StatsError::Internal(format!(
                "{error}; failed to remove local object staging {}: {cleanup}",
                staging.display()
            ))),
        };
    }
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            StatsError::Internal(format!(
                "fsync local object parent {}: {error}",
                parent.display()
            ))
        })
}

pub(crate) fn compare_and_swap(
    path: &Path,
    expected_value: Option<&[u8]>,
    bytes: &[u8],
) -> Result<ObjectVersion, StatsError> {
    let parent = path.parent().ok_or_else(|| {
        StatsError::Internal(format!("object pointer {} has no parent", path.display()))
    })?;
    std::fs::create_dir_all(parent).map_err(|error| {
        StatsError::Internal(format!(
            "create object pointer parent {}: {error}",
            parent.display()
        ))
    })?;
    let lock_path = path.with_extension("lock");
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|error| {
            StatsError::Internal(format!(
                "open object pointer lock {}: {error}",
                lock_path.display()
            ))
        })?;
    // SAFETY: `lock` owns this file descriptor until the function returns.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(StatsError::Internal(format!(
            "lock object pointer {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        )));
    }

    let current = match std::fs::read(path) {
        Ok(current) => Some(current),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(StatsError::Internal(format!(
                "read object pointer {}: {error}",
                path.display()
            )))
        }
    };
    if current.as_deref() != expected_value {
        return Err(StatsError::SchemaConflict(format!(
            "object pointer {} changed concurrently",
            path.display()
        )));
    }

    atomic_write(path, bytes)?;
    Ok(ObjectVersion {
        e_tag: None,
        provider_version: None,
        byte_size: bytes.len() as u64,
        local_value: Some(bytes::Bytes::copy_from_slice(bytes)),
    })
}

fn next_nonce() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}
