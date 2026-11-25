# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze Marin's serving runtime or an explicit build without adopting it."""

import dataclasses
import json
import platform
import re
import tempfile
from pathlib import Path
from urllib.parse import quote

import draccus
import requests
from marin.external_dependencies import CUDA_TOOLCHAIN_VERSION_BY_BACKEND, VLLM_GPU_RELEASE, VllmGpuRelease
from marin.inference.vllm_release import load_vllm_gpu_release
from marin.profiling.trace_summary import sha256_for_path

from experiments.evaluation.knotmeter.artifacts import copy_file, write_record
from experiments.evaluation.knotmeter.config import Runtime


def resolve_release(runtime: Runtime) -> dict:
    release = VLLM_GPU_RELEASE
    location = runtime.manifest
    if runtime.release:
        location = (
            f"https://github.com/marin-community/vllm/releases/download/{quote(runtime.release, safe='')}/"
            "marin-vllm-gpu-manifest.json"
        )
    if location:
        with tempfile.TemporaryDirectory(prefix="knotmeter-runtime-") as directory:
            path = Path(directory) / "manifest.json"
            if location.startswith(("https://", "http://")):
                response = requests.get(location, timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
            else:
                copy_file(location, str(path))
            if runtime.release:
                selected = load_vllm_gpu_release(path)
                assert selected is not None
                release = selected
            else:
                # Saved trial/build descriptors use Marin's existing release type.
                release = gpu_release(json.loads(path.read_text()))
    if re.fullmatch(r"[0-9a-f]{40}", release.source_commit) is None:
        raise ValueError("Runtime requires a full source commit")
    if any(re.fullmatch(r"[0-9a-f]{64}", wheel.sha256) is None for wheel in release.wheels):
        raise ValueError("Runtime wheels require SHA-256 identities")
    if release.torch_backend not in CUDA_TOOLCHAIN_VERSION_BY_BACKEND:
        raise ValueError(f"No registered CUDA toolchain for {release.torch_backend!r}")
    if runtime.release and release.release_tag != runtime.release:
        raise ValueError("Runtime manifest does not match the requested release")
    if runtime.revision and release.source_commit != runtime.revision:
        raise ValueError("Runtime manifest does not match the requested source revision")
    return json.loads(json.dumps(dataclasses.asdict(release)))


def gpu_release(metadata: dict) -> VllmGpuRelease:
    return draccus.decode(VllmGpuRelease, metadata)


def materialize_release(metadata: dict, output: Path) -> VllmGpuRelease:
    """Materialize this host's private wheel and verify its bytes."""
    release = gpu_release(metadata)
    wheels = []
    for wheel in release.wheels:
        if wheel.architecture != platform.machine() or not wheel.url.startswith(("s3://", "gs://")):
            wheels.append(wheel)
            continue
        directory = Path(tempfile.gettempdir()) / "knotmeter-wheels" / wheel.sha256
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / wheel.url.rsplit("/", 1)[-1]
        if not path.is_file():
            copy_file(wheel.url, str(path))
        actual = sha256_for_path(path)
        if actual != wheel.sha256:
            raise ValueError("Experimental wheel checksum differs from its immutable manifest")
        write_record(
            str(output / "runtime-wheel.json"),
            {"uri": wheel.url, "sha256": actual, "local_uri": path.as_uri(), "source_commit": release.source_commit},
        )
        wheels.append(dataclasses.replace(wheel, url=path.as_uri()))
    return dataclasses.replace(release, wheels=tuple(wheels))
