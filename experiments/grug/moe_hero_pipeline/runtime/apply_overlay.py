# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the Hero startup overlay to a clean, pinned JAXPP installation."""

import gzip
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

JAXPP_COMMIT = "328f75a80cecf22c7cc030a82d8941d3c1e220b6"
OVERLAY_SHA256 = "d987c16ece399fe5eeb69d413703d549135635f19a57dc4b6c60cc0d1ea4b1a7"
RUNTIME_PACKAGES = (
    "jax",
    "jaxlib",
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
    "jaxpp",
    "torch",
    "cuda-bindings",
    "nvidia-nccl-cu13",
    "nvidia-cublas",
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "quack-kernels",
)


def main():
    distribution = importlib.metadata.distribution("jaxpp")
    direct_url = distribution.read_text("direct_url.json")
    assert direct_url is not None, "Install JAXPP from the pinned Git source before applying the overlay"
    source = json.loads(direct_url)
    assert source["vcs_info"]["commit_id"] == JAXPP_COMMIT, source
    overlay = Path(__file__).with_name("jaxpp_host_startup.patch.gz")
    patch = gzip.decompress(overlay.read_bytes())
    assert hashlib.sha256(patch).hexdigest() == OVERLAY_SHA256, overlay
    site_packages = Path(distribution.locate_file("jaxpp")).parent
    # Strip a/src so the patch paths address jaxpp inside site-packages.
    command = ["git", "apply", "-p2", "-"]
    subprocess.run([*command, "--check"], input=patch, cwd=site_packages, check=True)
    subprocess.run(command, input=patch, cwd=site_packages, check=True)
    print(
        json.dumps(
            {
                "event": "hero_runtime_install",
                "jaxpp_commit": JAXPP_COMMIT,
                "overlay_sha256": OVERLAY_SHA256,
                "versions": {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
