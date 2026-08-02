# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install the fixed JAX build used by the MoonEP rack experiment."""

import sys
from dataclasses import dataclass
from enum import StrEnum

from fray.cluster import ResourceConfig
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from marin.training.run_environment import extras_for_resources


class MoonEPJaxWheelBuild(StrEnum):
    """Named JAX builds that are valid for the MoonEP rack experiment."""

    LSA_20260802 = "lsa-20260802"
    LSA_NCCL_2307_20260802 = "lsa-nccl-2307-20260802"


@dataclass(frozen=True)
class _WheelArtifact:
    filename: str
    sha256: str


@dataclass(frozen=True)
class _WheelSet:
    prefix: str
    wheels: tuple[_WheelArtifact, ...]


_LSA_20260802 = _WheelSet(
    prefix="s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="fd2724cd9f128ea1a0d1f74029ce6fcdaf7915db1a351b088316cc821ac2408d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_plugin-0.11.1.dev0+selfbuilt-cp312-cp312-manylinux_2_27_aarch64.whl",
            sha256="d04ee6bdc956979fa0c43ed95bfdba7bc4f665ceceb34531ef792cff742ddf95",
        ),
        _WheelArtifact(
            filename="jaxlib-0.11.1.dev0+selfbuilt-cp312-cp312-manylinux_2_27_aarch64.whl",
            sha256="03e838842547a66af13bc93a533ce1943dc0f2eb83026a94994eca7f47c072b4",
        ),
    ),
)

_LSA_NCCL_2307_20260802 = _WheelSet(
    prefix="s3://marin-us-east-02a/marin/research/moonep/jax-f9f6bbace-xla-5d53e1e-nccl2307-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="a1bb00b9ed594e7d1b85251bce63660bb85c5f7a661d618af677cee481a4572a",
        ),
        _WheelArtifact(
            filename="jax_cuda13_plugin-0.11.1.dev0+selfbuilt-cp312-cp312-manylinux_2_27_aarch64.whl",
            sha256="d04ee6bdc956979fa0c43ed95bfdba7bc4f665ceceb34531ef792cff742ddf95",
        ),
        _WheelArtifact(
            filename="jaxlib-0.11.1.dev0+selfbuilt-cp312-cp312-manylinux_2_27_aarch64.whl",
            sha256="03e838842547a66af13bc93a533ce1943dc0f2eb83026a94994eca7f47c072b4",
        ),
    ),
)


def _wheel_set(build: MoonEPJaxWheelBuild) -> _WheelSet:
    if build == MoonEPJaxWheelBuild.LSA_20260802:
        return _LSA_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_20260802:
        return _LSA_NCCL_2307_20260802
    raise ValueError(f"unknown MoonEP JAX wheel build: {build}")


def _wheel_install_script(build: MoonEPJaxWheelBuild) -> str:
    wheel_set = _wheel_set(build)
    wheel_records = tuple((wheel.filename, wheel.sha256) for wheel in wheel_set.wheels)
    wheel_paths = " ".join(f'"$wheel_dir/{wheel.filename}"' for wheel in wheel_set.wheels)
    return f"""set -e
: "${{IRIS_WORKDIR:?}}"
: "${{IRIS_VENV:?}}"
wheel_dir="$IRIS_WORKDIR/.moonep-jax/{build.value}"
rm -rf "$wheel_dir"
mkdir -p "$wheel_dir"
echo 'downloading fixed MoonEP JAX wheels'
"$IRIS_VENV/bin/python" - <<'PY'
import hashlib
import os
from pathlib import Path

import fsspec

prefix = {wheel_set.prefix!r}
wheels = {wheel_records!r}
wheel_dir = Path(os.environ["IRIS_WORKDIR"]) / ".moonep-jax" / {build.value!r}
filesystem, remote_root = fsspec.core.url_to_fs(prefix)
for filename, expected_sha256 in wheels:
    digest = hashlib.sha256()
    destination = wheel_dir / filename
    with filesystem.open(f"{{remote_root}}/{{filename}}", "rb") as source, destination.open("wb") as target:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
            target.write(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {{filename}}: {{actual_sha256}}")
PY
echo 'installing fixed MoonEP JAX wheels'
uv pip install --python "$IRIS_VENV/bin/python" --no-deps --reinstall {wheel_paths}
"$IRIS_VENV/bin/python" - <<'PY'
import jax
import jaxlib

if jax.__version__ != "0.11.1.dev20260802+f9f6bbace":
    raise ValueError(f"unexpected JAX version: {{jax.__version__}}")
if jaxlib.__version__ != "0.11.1.dev0+selfbuilt":
    raise ValueError(f"unexpected jaxlib version: {{jaxlib.__version__}}")
print(f"fixed MoonEP JAX runtime: jax={{jax.__version__}} jaxlib={{jaxlib.__version__}}")
PY
"""


def moonep_jax_setup_scripts(
    build: MoonEPJaxWheelBuild | None,
    resources: ResourceConfig,
) -> list[str] | None:
    """Return standard GPU setup plus the selected fixed JAX wheel build."""
    if build is None:
        return None

    extras = extras_for_resources(resources)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return [
        default_setup_script(extras=extras, python_version=python_version),
        _wheel_install_script(build),
        cuda_toolchain_setup_script(),
    ]
