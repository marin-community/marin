//! Per-namespace storage retention policy.
//!
//! Fields left `None` inherit the cluster-wide defaults. proto3 zero <-> `None`
//! (inherit). The re-register "empty == keep existing" decision lives in the
//! store layer, not here.

use crate::proto::finelog::stats::{StoragePolicy as ProtoStoragePolicy, StoragePolicyView};

/// Per-namespace retention overrides.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoragePolicy {
    pub max_segments: Option<i32>,
    pub max_bytes: Option<i64>,
    pub max_age_seconds: Option<i64>,
}

impl StoragePolicy {
    /// True iff every field is `None` (inherit-all).
    pub fn is_empty(&self) -> bool {
        self.max_segments.is_none() && self.max_bytes.is_none() && self.max_age_seconds.is_none()
    }

    /// Decode a wire policy. proto3 zero means "inherit"; an absent
    /// `storage_policy` MessageField (`None` view) is identical to all-zero.
    pub fn from_proto_view(view: Option<&StoragePolicyView>) -> StoragePolicy {
        let Some(p) = view else {
            return StoragePolicy::default();
        };
        StoragePolicy {
            max_segments: nonzero_i32(p.max_segments.unwrap_or(0)),
            max_bytes: nonzero_i64(p.max_bytes.unwrap_or(0)),
            max_age_seconds: nonzero_i64(p.max_age_seconds.unwrap_or(0)),
        }
    }

    /// Encode for the wire. `None` round-trips as proto3's zero.
    pub fn to_proto_owned(&self) -> ProtoStoragePolicy {
        ProtoStoragePolicy {
            max_segments: Some(self.max_segments.unwrap_or(0)),
            max_bytes: Some(self.max_bytes.unwrap_or(0)),
            max_age_seconds: Some(self.max_age_seconds.unwrap_or(0)),
            ..Default::default()
        }
    }
}

fn nonzero_i32(v: i32) -> Option<i32> {
    (v != 0).then_some(v)
}

fn nonzero_i64(v: i64) -> Option<i64> {
    (v != 0).then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_policy_is_empty() {
        assert!(StoragePolicy::default().is_empty());
        assert!(!StoragePolicy {
            max_segments: Some(5),
            ..Default::default()
        }
        .is_empty());
    }

    #[test]
    fn from_proto_zero_is_none_nonzero_is_some() {
        let proto = ProtoStoragePolicy {
            max_segments: Some(0),
            max_bytes: Some(100),
            max_age_seconds: Some(0),
            ..Default::default()
        };
        // We can't construct a StoragePolicyView here directly, so exercise the
        // owned-roundtrip path via the field-level nonzero helpers.
        let p = StoragePolicy {
            max_segments: nonzero_i32(proto.max_segments.unwrap()),
            max_bytes: nonzero_i64(proto.max_bytes.unwrap()),
            max_age_seconds: nonzero_i64(proto.max_age_seconds.unwrap()),
        };
        assert_eq!(p.max_segments, None);
        assert_eq!(p.max_bytes, Some(100));
        assert_eq!(p.max_age_seconds, None);
    }

    #[test]
    fn from_proto_view_none_is_empty() {
        assert_eq!(
            StoragePolicy::from_proto_view(None),
            StoragePolicy::default()
        );
    }

    #[test]
    fn to_proto_none_is_zero() {
        let proto = StoragePolicy {
            max_segments: None,
            max_bytes: Some(42),
            max_age_seconds: None,
        }
        .to_proto_owned();
        assert_eq!(proto.max_segments, Some(0));
        assert_eq!(proto.max_bytes, Some(42));
        assert_eq!(proto.max_age_seconds, Some(0));
    }
}
