# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a model path to a same-region GCS bucket based on the current worker's zone.

Usage:
    from experiments.inference.regional_model_path import resolve_regional_model_path

    model_path = resolve_regional_model_path("meta-llama--Llama-3-1-8B-Instruct--0e9e39f")
    # Returns e.g. "gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"
    # or falls back to gcsfuse_mount path if the model is there instead.

The zone is detected from the TPU_WORKER_HOSTNAMES or KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS
env vars, or by querying the GCE metadata server.
"""

import subprocess
import sys

# Map GCS region prefix → list of (bucket, path_template) to check, in order.
# First match wins.
_REGION_TO_BUCKETS: dict[str, list[str]] = {
    "us-east1": ["gs://marin-us-east1"],
    "us-east5": ["gs://marin-us-east5"],
    "us-central1": ["gs://marin-us-central1"],
    "us-central2": ["gs://marin-us-central2"],
    "europe-west4": ["gs://marin-eu-west4"],
}

# Subdirectories to check for models, in priority order.
_MODEL_SUBDIRS = ["models", "gcsfuse_mount/models"]


def _detect_region() -> str | None:
    """Detect the GCE region from metadata server."""
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-H",
                "Metadata-Flavor: Google",
                "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Returns e.g. "projects/123456/zones/us-east1-d"
            zone = result.stdout.strip().rsplit("/", 1)[-1]
            # Extract region from zone (e.g. "us-east1" from "us-east1-d")
            parts = zone.rsplit("-", 1)
            if len(parts) == 2:
                return parts[0]
    except Exception:
        pass
    return None


def resolve_regional_model_path(model_name: str) -> str:
    """Resolve a model name to a same-region GCS path.

    Args:
        model_name: The model directory name, e.g.
            "meta-llama--Llama-3-1-8B-Instruct--0e9e39f"

    Returns:
        Full GCS path like "gs://marin-us-east1/models/meta-llama--..."

    Raises:
        FileNotFoundError: If the model can't be found in any regional bucket.
    """
    region = _detect_region()
    if region is None:
        print("WARNING: Could not detect GCE region, falling back to us-central1", file=sys.stderr)
        region = "us-central1"

    print(f"Detected region: {region}", file=sys.stderr)

    buckets = _REGION_TO_BUCKETS.get(region)
    if not buckets:
        print(f"WARNING: No bucket mapping for region {region}, trying us-central1", file=sys.stderr)
        buckets = _REGION_TO_BUCKETS["us-central1"]

    # Check each bucket + subdir combination
    for bucket in buckets:
        for subdir in _MODEL_SUBDIRS:
            path = f"{bucket}/{subdir}/{model_name}"
            try:
                result = subprocess.run(
                    ["gcloud", "storage", "ls", f"{path}/config.json"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    print(f"Resolved model path: {path} (same-region ✅)", file=sys.stderr)
                    return path
            except Exception:
                continue

    raise FileNotFoundError(
        f"Model {model_name!r} not found in region {region}. "
        f"Checked buckets: {buckets}, subdirs: {_MODEL_SUBDIRS}. "
        f"Copy the model to the right region first."
    )
