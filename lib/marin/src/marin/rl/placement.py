# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from marin.execution.executor import infer_tpu_variant_regions_from_iris
from rigging.filesystem import REGION_TO_DATA_BUCKET, marin_region

logger = logging.getLogger(__name__)


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
    bucket = REGION_TO_DATA_BUCKET.get(region.lower())
    if bucket is None:
        raise ValueError(f"No Marin data bucket configured for region {region!r}.")
    return f"gs://{bucket}"


def singleton_region_list(region: str) -> list[str]:
    """Wrap a concrete region in the list form expected by Iris resources."""
    return [region.lower()]
