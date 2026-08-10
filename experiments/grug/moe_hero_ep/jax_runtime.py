# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned JAX nightly runtime selection for EP transport experiments."""

import re

JAX_NIGHTLY_INDEX = "https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/"

_JAX_NIGHTLY_VERSION = re.compile(r"\d+\.\d+\.\d+\.dev\d{8}")


def jax_nightly_pip_packages(version: str | None) -> tuple[str, ...]:
    """Return worker-side pip arguments for one exact CUDA 13 JAX nightly."""
    if version is None:
        return ()
    if _JAX_NIGHTLY_VERSION.fullmatch(version) is None:
        raise ValueError(f"jax_nightly_version must look like 0.11.1.dev20260808, got {version!r}")
    return (
        f"jax=={version}",
        f"jaxlib=={version}",
        f"jax-cuda13-plugin[with-cuda]=={version}",
        f"jax-cuda13-pjrt=={version}",
        "--index",
        JAX_NIGHTLY_INDEX,
    )
