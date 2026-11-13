# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin the SkyRL serving release used by CPU launch validation."""

import json
import tomllib
from pathlib import Path
from urllib.parse import unquote

from marin.external_dependencies import VllmGpuRelease, VllmGpuWheel
from rigging.config_discovery import find_project_root


def serving_release() -> VllmGpuRelease:
    manifest = json.loads(Path(__file__).with_name("vllm-release.json").read_text())
    return VllmGpuRelease(
        release_tag=manifest["release"]["tag"],
        source_commit=manifest["source"]["fork_commit"],
        version=manifest["distribution"]["version"],
        torch_backend=manifest["abi"]["cuda_variant"],
        torch_version=manifest["abi"]["torch_version"],
        wheels=tuple(
            VllmGpuWheel(
                architecture=p["architecture"],
                sm_targets=tuple(p["sm_targets"]),
                url=p["wheel"]["url"],
                sha256=p["wheel"]["sha256"],
            )
            for p in manifest["platforms"]
        ),
    )


def check_serving_pin(release: VllmGpuRelease, skyrl_project: Path) -> None:
    project = tomllib.loads(skyrl_project.read_text())
    sources = project["tool"]["uv"]["sources"]["vllm"]
    urls = {unquote(source["url"]) for source in sources}
    requirements = project["project"]["optional-dependencies"]["vllm"]
    if not any(req.split(";")[0].strip() == f"vllm=={release.version}" for req in requirements):
        raise ValueError("Endpoint vLLM version differs from the RL stack")
    if {unquote(wheel.url) for wheel in release.wheels} != urls:
        raise ValueError("Endpoint vLLM wheels differ from the RL stack")


def checked_serving_release() -> VllmGpuRelease:
    root = find_project_root()
    assert root is not None
    release = serving_release()
    check_serving_pin(release, root / "vendor/skyrl/pyproject.toml")
    return release


# Pinned base config: open-athena/snowball-67b-a2b-base-262k-qk175-skew8
# revision 058ecaf27b9e4f37219df221a51e7d490d58ec3d. SFT packing does not change RoPE.
SNOWBALL_CONTEXT_LIMIT = 262144


def snowball_context_override(requested_window: int) -> dict[str, int]:
    check_context_limit("Snowball", SNOWBALL_CONTEXT_LIMIT, requested_window)
    return {"max_position_embeddings": SNOWBALL_CONTEXT_LIMIT, "max_seq_len": SNOWBALL_CONTEXT_LIMIT}


def check_context_limit(model: str, max_position_embeddings: int, max_model_len: int) -> None:
    if max_model_len > max_position_embeddings:
        raise ValueError(
            f"{model}: requested max_model_len={max_model_len} exceeds "
            f"max_position_embeddings={max_position_embeddings}"
        )
