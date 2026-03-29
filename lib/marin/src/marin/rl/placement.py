# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence

from fray.current_client import current_client
from fray.iris_backend import FrayIrisClient
from fray.types import ResourceConfig, TpuConfig
from rigging.filesystem import data_config, marin_region

logger = logging.getLogger(__name__)


def _regions_for_tpu_variant_from_iris(variant: str) -> set[str] | None:
    try:
        client = current_client()
    except Exception:
        return None
    if not isinstance(client, FrayIrisClient):
        return None

    variant = variant.lower()
    try:
        # TODO: expose autoscaler status through a public Fray API.
        autoscaler_status = client._iris._cluster_client.get_autoscaler_status()
    except Exception:
        logger.warning("Could not query Iris autoscaler status for TPU region inference", exc_info=True)
        return None

    regions: set[str] = set()
    for group in autoscaler_status.status.groups:
        if group.device_type != "tpu":
            continue
        group_variant = group.device_variant.lower().strip()
        if group_variant and group_variant != variant:
            continue

        region = group.region.strip().lower()
        if region:
            regions.add(region)

    return regions or None


def _regions_for_tpu_variants_from_iris(
    variants: list[str],
    *,
    variant_region_cache: dict[str, set[str] | None],
) -> set[str] | None:
    inferred_regions: set[str] = set()
    for variant in variants:
        normalized_variant = variant.lower()
        if normalized_variant not in variant_region_cache:
            variant_region_cache[normalized_variant] = _regions_for_tpu_variant_from_iris(normalized_variant)
        cached = variant_region_cache[normalized_variant]
        if cached is None:
            return None
        inferred_regions |= cached
    return inferred_regions


def infer_tpu_variant_regions_from_iris(variants: Sequence[str]) -> list[str] | None:
    """Return sorted TPU-capable regions for the requested variants, if known."""
    inferred_regions = _regions_for_tpu_variants_from_iris(
        list(variants),
        variant_region_cache={},
    )
    if not inferred_regions:
        return None
    return sorted(inferred_regions)


def _region_from_zone(zone: str) -> str:
    if "-" not in zone:
        raise ValueError(f"Invalid zone {zone!r}; expected a value like 'us-east5-d'.")
    return zone.rsplit("-", 1)[0]


def _region_key(region: str) -> str:
    return region.lower()


def _deduplicate_regions(regions: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for region in regions:
        key = _region_key(region)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(region)
    return deduplicated


def _region_key_set(regions: list[str]) -> set[str]:
    return {_region_key(region) for region in regions}


def requested_tpu_variants(train_resources: ResourceConfig, rollout_resources: ResourceConfig) -> list[str]:
    """Return the deduplicated TPU variants requested by an RL run."""
    variants: list[str] = []
    for resource in (train_resources, rollout_resources):
        if not isinstance(resource.device, TpuConfig):
            continue
        for variant in (resource.device.variant, *(resource.device_alternatives or ())):
            normalized = variant.lower()
            if normalized not in variants:
                variants.append(normalized)
    return variants


def _normalized_resource_regions(resource: ResourceConfig) -> list[str] | None:
    normalized_regions = _deduplicate_regions(list(resource.regions or ()))

    if resource.zone is None:
        return normalized_regions or None

    zone_region = _region_from_zone(resource.zone)
    if not normalized_regions:
        return [zone_region]
    if _region_key(zone_region) not in _region_key_set(normalized_regions):
        raise ValueError(f"RL resource zone {resource.zone!r} conflicts with requested regions {normalized_regions}.")
    return [zone_region]


def _normalized_regions(resources: tuple[ResourceConfig, ResourceConfig]) -> list[str] | None:
    requested_regions: list[list[str]] = []
    for resource in resources:
        normalized = _normalized_resource_regions(resource)
        if normalized is None:
            continue
        requested_regions.append(normalized)

    if not requested_regions:
        return None

    shared_regions = requested_regions[0]
    for regions in requested_regions[1:]:
        region_keys = _region_key_set(regions)
        shared_regions = [region for region in shared_regions if _region_key(region) in region_keys]

    if not shared_regions:
        raise ValueError("RL trainer and rollout workers must share at least one compatible region")

    return shared_regions


def resolve_launcher_region(train_resources: ResourceConfig, rollout_resources: ResourceConfig) -> str:
    """Choose the concrete region for an RL run."""
    resources = (train_resources, rollout_resources)
    variants = requested_tpu_variants(train_resources, rollout_resources)
    current_region = marin_region()
    allowed_regions = infer_tpu_variant_regions_from_iris(variants) if variants else None
    requested_regions = _normalized_regions(resources)

    if current_region is not None:
        current_region = _region_key(current_region)

    if requested_regions is not None:
        if allowed_regions is not None:
            allowed_region_keys = _region_key_set(allowed_regions)
            requested_regions = [region for region in requested_regions if _region_key(region) in allowed_region_keys]
            if not requested_regions:
                raise ValueError(
                    f"RL run requests TPU variants {variants}, available in {allowed_regions}, "
                    "but the configured resource regions are incompatible."
                )
            if current_region is not None:
                if current_region not in _region_key_set(requested_regions):
                    raise ValueError(
                        f"RL run is pinned to regions {requested_regions}, but the current launcher region is "
                        f"{current_region!r}. Relaunch the root Iris job with --region/--zone in a compatible region."
                    )
                return current_region
            return requested_regions[0]

        if current_region is not None and current_region in _region_key_set(requested_regions):
            return current_region
        return requested_regions[0]

    if allowed_regions is not None:
        allowed_region_keys = _region_key_set(allowed_regions)
        if current_region is not None:
            if current_region not in allowed_region_keys:
                raise ValueError(
                    f"RL run requests TPU variants {variants}, which are available in {allowed_regions}, "
                    f"but the current launcher region is {current_region!r}. Relaunch the root Iris job "
                    "with --region/--zone in a compatible region."
                )
            return current_region
        return allowed_regions[0]

    if current_region is not None:
        if not variants:
            return current_region
        logger.warning(
            "Could not infer TPU-capable regions for %s from Iris autoscaler; defaulting to current launcher "
            "region %s.",
            variants,
            current_region,
        )
        return current_region

    raise ValueError(
        "Could not determine a launcher region for the RL run. "
        "Launch the root Iris job with --region/--zone, set MARIN_PREFIX to a regional bucket, "
        "or set regions on the RL resource configs."
    )


def marin_prefix_for_region(region: str) -> str:
    """Return the canonical Marin bucket prefix for a region."""
    bucket = data_config().region_buckets.get(region.lower())
    if bucket is None:
        raise ValueError(f"No Marin data bucket configured for region {region!r}.")
    return f"gs://{bucket}"


def singleton_region_list(region: str) -> list[str]:
    """Wrap a concrete region in the list form expected by Iris resources."""
    return [region]
