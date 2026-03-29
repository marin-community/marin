# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from fray.v2.types import ResourceConfig, TpuConfig
from marin.execution.executor import infer_tpu_variant_regions_from_iris
from rigging.filesystem import REGION_TO_DATA_BUCKET, marin_region

logger = logging.getLogger(__name__)


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


def _normalized_regions(resources: tuple[ResourceConfig, ResourceConfig]) -> list[str] | None:
    requested_regions: list[list[str]] = []
    for resource in resources:
        if resource.regions is None:
            continue
        normalized = []
        for region in resource.regions:
            lowered = region.lower()
            if lowered not in normalized:
                normalized.append(lowered)
        requested_regions.append(normalized)

    if not requested_regions:
        return None

    shared_regions = requested_regions[0]
    for regions in requested_regions[1:]:
        shared_regions = [region for region in shared_regions if region in regions]

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
        current_region = current_region.lower()

    if requested_regions is not None:
        if allowed_regions is not None:
            requested_regions = [region for region in requested_regions if region in allowed_regions]
            if not requested_regions:
                raise ValueError(
                    f"RL run requests TPU variants {variants}, available in {allowed_regions}, "
                    "but the configured resource regions are incompatible."
                )
        if current_region is not None:
            if current_region not in requested_regions:
                raise ValueError(
                    f"RL run is pinned to regions {requested_regions}, but the current launcher region is "
                    f"{current_region!r}. Relaunch the root Iris job with --region/--zone in a compatible region."
                )
            return current_region
        return requested_regions[0]

    if allowed_regions is not None:
        if current_region is not None:
            if current_region not in allowed_regions:
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
    bucket = REGION_TO_DATA_BUCKET.get(region.lower())
    if bucket is None:
        raise ValueError(f"No Marin data bucket configured for region {region!r}.")
    return f"gs://{bucket}"


def singleton_region_list(region: str) -> list[str]:
    """Wrap a concrete region in the list form expected by Iris resources."""
    return [region.lower()]
