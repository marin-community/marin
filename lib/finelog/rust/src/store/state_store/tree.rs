//! Pure catalog-tree delta and fold operations.

use std::collections::{BTreeMap, BTreeSet};

use buffa::MessageField;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    CatalogDelta, CatalogSegment, CatalogSegmentAddition, CatalogSegmentKey, NamespaceCatalog,
    ObjectRef, ReleasedObject, TableVersionSegments,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SegmentKey {
    version: u64,
    retired: bool,
    segment_id: String,
}

pub(crate) fn canonical_catalog(mut catalog: NamespaceCatalog) -> NamespaceCatalog {
    catalog
        .version_segments
        .sort_by_key(|segments| segments.table_spec_version.unwrap_or(0));
    for segments in &mut catalog.version_segments {
        segments.live_segments.sort_by_key(segment_id);
        segments.retired_segments.sort_by_key(segment_id);
    }
    catalog.direct_query_segments.sort_by_key(segment_id);
    catalog
}

pub(crate) fn catalogs_equal(left: &NamespaceCatalog, right: &NamespaceCatalog) -> bool {
    canonical_catalog(left.clone()) == canonical_catalog(right.clone())
}

pub(crate) fn delta(
    previous: &NamespaceCatalog,
    next: &NamespaceCatalog,
) -> Result<CatalogDelta, StatsError> {
    let previous_segments = version_segment_map(previous)?;
    let next_segments = version_segment_map(next)?;
    let mut additions = Vec::new();
    let mut removals = Vec::new();
    for (key, segment) in &next_segments {
        if previous_segments.get(key) != Some(segment) {
            additions.push(CatalogSegmentAddition {
                key: MessageField::some(proto_key(key)),
                segment: MessageField::some(segment.clone()),
                ..Default::default()
            });
        }
    }
    for key in previous_segments.keys() {
        if !next_segments.contains_key(key) {
            removals.push(proto_key(key));
        }
    }

    let previous_direct = direct_segment_map(previous)?;
    let next_direct = direct_segment_map(next)?;
    let direct_query_additions = next_direct
        .iter()
        .filter(|(id, segment)| previous_direct.get(*id) != Some(*segment))
        .map(|(_, segment)| segment.clone())
        .collect();
    let direct_query_removals = previous_direct
        .keys()
        .filter(|id| !next_direct.contains_key(*id))
        .cloned()
        .collect();

    Ok(CatalogDelta {
        metadata: MessageField::some(metadata(next)),
        segment_additions: additions,
        segment_removals: removals,
        direct_query_additions,
        direct_query_removals,
        ..Default::default()
    })
}

pub(crate) fn apply(
    previous: &NamespaceCatalog,
    delta: &CatalogDelta,
) -> Result<NamespaceCatalog, StatsError> {
    let metadata = delta
        .metadata
        .as_option()
        .ok_or_else(|| StatsError::Internal("catalog delta has no metadata state".to_string()))?;
    let mut segments = version_segment_map(previous)?;
    for removal in &delta.segment_removals {
        let key = internal_key(removal)?;
        if segments.remove(&key).is_none() {
            return Err(StatsError::Internal(format!(
                "catalog delta removes absent segment {:?}",
                key.segment_id
            )));
        }
    }
    for addition in &delta.segment_additions {
        let key = internal_key(addition.key.as_option().ok_or_else(|| {
            StatsError::Internal("catalog segment addition has no key".to_string())
        })?)?;
        let segment = addition.segment.as_option().ok_or_else(|| {
            StatsError::Internal("catalog segment addition has no segment".to_string())
        })?;
        if segment_id(segment) != key.segment_id {
            return Err(StatsError::Internal(format!(
                "catalog segment addition key {:?} does not match its segment",
                key.segment_id
            )));
        }
        segments.insert(key, segment.clone());
    }

    let mut direct = direct_segment_map(previous)?;
    for segment_id in &delta.direct_query_removals {
        if direct.remove(segment_id).is_none() {
            return Err(StatsError::Internal(format!(
                "catalog delta removes absent direct-query segment {segment_id:?}"
            )));
        }
    }
    for segment in &delta.direct_query_additions {
        direct.insert(segment_id(segment), segment.clone());
    }

    let mut result = metadata.clone();
    let mut versions: BTreeMap<u64, TableVersionSegments> = metadata
        .version_segments
        .iter()
        .map(|version| {
            let number = version.table_spec_version.unwrap_or(0);
            (
                number,
                TableVersionSegments {
                    table_spec_version: Some(number),
                    ..Default::default()
                },
            )
        })
        .collect();
    for (key, segment) in segments {
        let version = versions
            .entry(key.version)
            .or_insert_with(|| TableVersionSegments {
                table_spec_version: Some(key.version),
                ..Default::default()
            });
        if key.retired {
            version.retired_segments.push(segment);
        } else {
            version.live_segments.push(segment);
        }
    }
    result.version_segments = versions.into_values().collect();
    result.direct_query_segments = direct.into_values().collect();
    Ok(canonical_catalog(result))
}

pub(crate) fn released_objects(
    previous: &NamespaceCatalog,
    next: &NamespaceCatalog,
    delete_after_ms: i64,
) -> Vec<ReleasedObject> {
    let next_ids = referenced_objects(next)
        .into_iter()
        .map(|object| object.object_id.clone().unwrap_or_default())
        .collect::<BTreeSet<_>>();
    referenced_objects(previous)
        .into_iter()
        .filter(|object| {
            object
                .object_id
                .as_ref()
                .is_some_and(|id| !next_ids.contains(id))
        })
        .map(|object| ReleasedObject {
            object: MessageField::some(logical_reference(&object)),
            delete_after_ms: Some(delete_after_ms),
            ..Default::default()
        })
        .collect()
}

pub(crate) fn referenced_objects(catalog: &NamespaceCatalog) -> Vec<ObjectRef> {
    let mut objects = BTreeMap::new();
    for segment in catalog
        .version_segments
        .iter()
        .flat_map(|version| {
            version
                .live_segments
                .iter()
                .chain(&version.retired_segments)
        })
        .chain(&catalog.direct_query_segments)
    {
        for object in segment_objects(segment) {
            if let Some(id) = &object.object_id {
                objects.entry(id.clone()).or_insert_with(|| object.clone());
            }
        }
    }
    objects.into_values().collect()
}

pub(crate) fn logical_reference(reference: &ObjectRef) -> ObjectRef {
    ObjectRef {
        object_id: reference.object_id.clone(),
        byte_size: reference.byte_size,
        ..Default::default()
    }
}

fn metadata(catalog: &NamespaceCatalog) -> NamespaceCatalog {
    let mut metadata = catalog.clone();
    metadata.version_segments = catalog
        .version_segments
        .iter()
        .map(|version| TableVersionSegments {
            table_spec_version: version.table_spec_version,
            ..Default::default()
        })
        .collect();
    metadata.direct_query_segments.clear();
    metadata
}

fn version_segment_map(
    catalog: &NamespaceCatalog,
) -> Result<BTreeMap<SegmentKey, CatalogSegment>, StatsError> {
    let mut result = BTreeMap::new();
    for version in &catalog.version_segments {
        let version_number = version.table_spec_version.unwrap_or(0);
        for (retired, segment) in version
            .live_segments
            .iter()
            .map(|segment| (false, segment))
            .chain(
                version
                    .retired_segments
                    .iter()
                    .map(|segment| (true, segment)),
            )
        {
            let key = SegmentKey {
                version: version_number,
                retired,
                segment_id: segment_id(segment),
            };
            if key.segment_id.is_empty() || result.insert(key.clone(), segment.clone()).is_some() {
                return Err(StatsError::Internal(format!(
                    "catalog has duplicate or empty segment ID {:?}",
                    key.segment_id
                )));
            }
        }
    }
    Ok(result)
}

fn direct_segment_map(
    catalog: &NamespaceCatalog,
) -> Result<BTreeMap<String, CatalogSegment>, StatsError> {
    let mut result = BTreeMap::new();
    for segment in &catalog.direct_query_segments {
        let id = segment_id(segment);
        if id.is_empty() || result.insert(id.clone(), segment.clone()).is_some() {
            return Err(StatsError::Internal(format!(
                "catalog has duplicate or empty direct-query segment ID {id:?}"
            )));
        }
    }
    Ok(result)
}

fn segment_id(segment: &CatalogSegment) -> String {
    segment.segment_id.clone().unwrap_or_default()
}

fn proto_key(key: &SegmentKey) -> CatalogSegmentKey {
    CatalogSegmentKey {
        table_spec_version: Some(key.version),
        segment_id: Some(key.segment_id.clone()),
        retired: Some(key.retired),
        ..Default::default()
    }
}

fn internal_key(key: &CatalogSegmentKey) -> Result<SegmentKey, StatsError> {
    let segment_id = key.segment_id.clone().unwrap_or_default();
    if segment_id.is_empty() {
        return Err(StatsError::Internal(
            "catalog segment key has an empty segment ID".to_string(),
        ));
    }
    Ok(SegmentKey {
        version: key.table_spec_version.unwrap_or(0),
        retired: key.retired.unwrap_or(false),
        segment_id,
    })
}

fn segment_objects(segment: &CatalogSegment) -> Vec<&ObjectRef> {
    segment
        .source
        .as_option()
        .into_iter()
        .chain(segment.index_bundle.as_option())
        .chain(
            segment
                .projections
                .iter()
                .filter_map(|projection| projection.object.as_option()),
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(id: &str, object: &str) -> CatalogSegment {
        CatalogSegment {
            segment_id: Some(id.to_string()),
            source: MessageField::some(ObjectRef {
                object_id: Some(object.to_string()),
                byte_size: Some(17),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn catalog(generation: u64, segments: Vec<CatalogSegment>) -> NamespaceCatalog {
        NamespaceCatalog {
            format_version: Some(2),
            namespace: Some("logs".to_string()),
            catalog_generation: Some(generation),
            version_segments: vec![TableVersionSegments {
                table_spec_version: Some(1),
                live_segments: segments.clone(),
                ..Default::default()
            }],
            direct_query_segments: segments,
            ..Default::default()
        }
    }

    #[test]
    fn delta_round_trip_preserves_the_complete_catalog() {
        let previous = catalog(
            7,
            vec![segment("a", "a.parquet"), segment("b", "b.parquet")],
        );
        let next = catalog(
            8,
            vec![segment("b", "b2.parquet"), segment("c", "c.parquet")],
        );
        let patch = delta(&previous, &next).unwrap();

        assert_eq!(apply(&previous, &patch).unwrap(), canonical_catalog(next));
        assert_eq!(patch.segment_additions.len(), 2);
        assert_eq!(patch.segment_removals.len(), 1);
    }

    #[test]
    fn release_sidecar_contains_only_objects_that_left_the_state() {
        let previous = catalog(
            7,
            vec![segment("a", "a.parquet"), segment("b", "b.parquet")],
        );
        let next = catalog(8, vec![segment("b", "b.parquet")]);

        let released = released_objects(&previous, &next, 1234);
        assert_eq!(released.len(), 1);
        assert_eq!(
            released[0]
                .object
                .as_option()
                .and_then(|object| object.object_id.as_deref()),
            Some("a.parquet")
        );
        assert_eq!(released[0].delete_after_ms, Some(1234));
    }
}
