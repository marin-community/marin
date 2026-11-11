# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze Marin's selected release or an explicit published release at submission."""

import dataclasses
import json
import tempfile
from pathlib import Path
from urllib.parse import quote

import draccus
import requests
from marin.external_dependencies import VLLM_GPU_RELEASE, VllmGpuRelease
from marin.inference.vllm_release import load_vllm_gpu_release


def resolve_release(tag: str = "") -> dict:
    """Resolve once on CPU; queued points and retries consume only the saved value."""
    release = VLLM_GPU_RELEASE
    if tag:
        url = (
            f"https://github.com/marin-community/vllm/releases/download/{quote(tag, safe='')}/"
            "marin-vllm-gpu-manifest.json"
        )
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with tempfile.TemporaryDirectory(prefix="agentx-release-") as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(response.content)
            release = load_vllm_gpu_release(path)
        if release is None or release.release_tag != tag:
            raise ValueError("Published manifest differs from the requested release")
    return json.loads(json.dumps(dataclasses.asdict(release)))


def gpu_release(metadata: dict) -> VllmGpuRelease:
    """Decode the frozen plan; never consult a mutable global pin on a GPU worker."""
    return draccus.decode(VllmGpuRelease, metadata)
