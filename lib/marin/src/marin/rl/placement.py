# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence

from fray.current_client import current_client
from fray.iris_backend import FrayIrisClient
from iris.cluster.constraints import WellKnownAttribute
from iris.rpc import config_pb2
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
        resources = group.config.resources
        if resources.device_type != config_pb2.ACCELERATOR_TYPE_TPU:
            continue
        group_variant = resources.device_variant.lower().strip()
        if group_variant and group_variant != variant:
            continue

        attrs = group.config.worker.attributes
        region = attrs.get(WellKnownAttribute.REGION, "").strip().lower()
        if region:
            regions.add(region)
            continue

        zone = attrs.get(WellKnownAttribute.ZONE, "").strip().lower()
        if zone and "-" in zone:
            regions.add(zone.rsplit("-", 1)[0])

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


def requested_tpu_variants(train_tpu_type: str, inference_tpu_type: str | None) -> list[str]:
    """Return the deduplicated TPU variants requested by an RL run."""
    variants: list[str] = []
    for variant in (train_tpu_type, inference_tpu_type or train_tpu_type):
        normalized = variant.lower()
        if normalized not in variants:
            variants.append(normalized)
    return variants


def resolve_launcher_region(train_tpu_type: str, inference_tpu_type: str | None) -> str:
    """Choose the concrete region for an RL run.

    The root Iris job must already be launched in the desired region. This
    function validates that region against the requested TPU variants when
    autoscaler information is available.
    """
    variants = requested_tpu_variants(train_tpu_type, inference_tpu_type)
    current_region = marin_region()
    allowed_regions = infer_tpu_variant_regions_from_iris(variants)

    if current_region is not None:
        current_region = current_region.lower()

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
        logger.warning(
            "Could not infer TPU-capable regions for %s from Iris autoscaler; defaulting to current launcher "
            "region %s.",
            variants,
            current_region,
        )
        return current_region

    raise ValueError(
        f"Could not determine a launcher region for requested TPU variants {variants}. "
        "Launch the root Iris job with --region/--zone or set MARIN_PREFIX to a regional bucket."
    )


def marin_prefix_for_region(region: str) -> str:
    """Return the canonical Marin bucket prefix for a region."""
    bucket = data_config().region_buckets.get(region.lower())
    if bucket is None:
        raise ValueError(f"No Marin data bucket configured for region {region!r}.")
    return f"gs://{bucket}"


def singleton_region_list(region: str) -> list[str]:
    """Wrap a concrete region in the list form expected by Iris resources."""
    return [region.lower()]
