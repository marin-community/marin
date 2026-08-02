# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install the fixed JAX build used by the MoonEP rack experiment."""

import sys
from dataclasses import dataclass
from enum import StrEnum

from fray.cluster import ResourceConfig
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from marin.training.run_environment import extras_for_resources

_WHEEL_ARTIFACT_ROOT = "s3://marin-us-east-02a/marin/research/moonep"


class MoonEPJaxWheelBuild(StrEnum):
    """Named JAX builds that are valid for the MoonEP rack experiment."""

    LSA_20260802 = "lsa-20260802"
    LSA_NCCL_2307_20260802 = "lsa-nccl-2307-20260802"
    LSA_NCCL_2307_HYBRID_20260802 = "lsa-nccl-2307-hybrid-resources-20260802"
    LSA_NCCL_2307_HYBRID_WEAK_20260802 = "lsa-nccl-2307-hybrid-weak-20260802"
    LSA_NCCL_2307_HYBRID_CTA16_20260802 = "lsa-nccl-2307-hybrid-cta16-20260802"
    LSA_NCCL_2307_WORLD_GIN_20260802 = "lsa-nccl-2307-world-gin-20260802"
    LSA_NCCL_2307_BARRIER_ONLY_20260802 = "lsa-nccl-2307-barrier-only-20260802"
    LSA_NCCL_2307_LSA_BARRIER_ONLY_20260802 = "lsa-nccl-2307-lsa-barrier-only-20260802"
    LSA_NCCL_2307_NOOP_KERNEL_20260802 = "lsa-nccl-2307-noop-kernel-20260802"
    LSA_NCCL_2307_NOOP_LSA_HOST_20260802 = "lsa-nccl-2307-noop-lsa-host-20260802"
    LSA_NCCL_2307_NOOP_RUN_COLLECTIVE_20260802 = "lsa-nccl-2307-noop-run-collective-20260802"
    LSA_NCCL_2307_NOOP_AFTER_DEVICE_COMM_20260802 = "lsa-nccl-2307-noop-after-device-comm-20260802"
    LSA_NCCL_2307_NOOP_BEFORE_DEVICE_COMM_20260802 = "lsa-nccl-2307-noop-before-device-comm-20260802"
    LSA_NCCL_2307_NOOP_AFTER_BUFFER_CONVERSION_20260802 = "lsa-nccl-2307-noop-after-buffer-conversion-20260802"
    LSA_NCCL_2307_FULL_MNNVL_20260802 = "lsa-nccl-2307-full-mnnvl-20260802"


@dataclass(frozen=True)
class _WheelArtifact:
    filename: str
    sha256: str


@dataclass(frozen=True)
class _RuntimeLibraryArtifact:
    prefix: str
    filename: str
    sha256: str
    site_packages_path: str


@dataclass(frozen=True)
class _WheelSet:
    prefix: str
    wheels: tuple[_WheelArtifact, ...]
    runtime_libraries: tuple[_RuntimeLibraryArtifact, ...] = ()


_LSA_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-20260802",
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
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-20260802",
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

_LSA_NCCL_2307_HYBRID_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-resources-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="ad8ee4dff204460f10bff5eb468957b332131203b628bf02ad2bcc0fdff73d0f",
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

_LSA_NCCL_2307_HYBRID_WEAK_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-weak-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="c71148f3901030525093480bbdf6582d255d7b34af5564a636ac409b24de1ffa",
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

_LSA_NCCL_2307_HYBRID_CTA16_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-hybrid-cta16-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="6a87208443b820f2e37c6e4517d22d7b9d1f143b224b1c6d91550d9cae604b2e",
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

_LSA_NCCL_2307_WORLD_GIN_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-world-gin-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="c2521e40e7cd87f445b42445ba5221771b89c754caf5fa81c92a7b47add6cb31",
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

_LSA_NCCL_2307_BARRIER_ONLY_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-barrier-only-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="d3e391455196736d8793e4a983c63ed1644fe90e8ce87e9f56635fa43c83196c",
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

_LSA_NCCL_2307_LSA_BARRIER_ONLY_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-lsa-barrier-only-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="2a9411320bbc9fc36ce21c60eb9a2825b3c54b2a6afcbb75cbfa0fb9ed3a1023",
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

_LSA_NCCL_2307_NOOP_KERNEL_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-kernel-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="458f3277349276d5a13a3c652d625339f34c8602ed5a230f9bea861e1005fa44",
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

_LSA_NCCL_2307_NOOP_LSA_HOST_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-lsa-host-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="c2b5461dfbfd53dfcebf76af16eb55736c02aa934020e81edc2477d59851f973",
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

_LSA_NCCL_2307_NOOP_RUN_COLLECTIVE_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-run-collective-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="ba8fb4ba686e18ec710f6495e7f4e8d407cadf5076e8786a06314d30443e6eb4",
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

_LSA_NCCL_2307_NOOP_AFTER_DEVICE_COMM_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-after-device-comm-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="4b1ddc5011aa44126ff711c4efe2fe889ef43d380e31609e560437cc6cbee0cd",
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

_LSA_NCCL_2307_NOOP_BEFORE_DEVICE_COMM_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-before-device-comm-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="b3c2d632c70628d026af8bc87f09d48c405b03a57dcba69880327b66ecd748d2",
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

_LSA_NCCL_2307_NOOP_AFTER_BUFFER_CONVERSION_20260802 = _WheelSet(
    prefix=f"{_WHEEL_ARTIFACT_ROOT}/jax-f9f6bbace-xla-5d53e1e-nccl2307-noop-after-buffer-conversion-20260802",
    wheels=(
        _WheelArtifact(
            filename="jax-0.11.1.dev20260802+f9f6bbace-py3-none-any.whl",
            sha256="40b447b71c8a45032abe9ebdbadfd9d0d434165500c27831a408a8ee053dac4d",
        ),
        _WheelArtifact(
            filename="jax_cuda13_pjrt-0.11.1.dev0+selfbuilt-py3-none-manylinux_2_27_aarch64.whl",
            sha256="3475c89c11683f1106290ab487f55087bd924b6de9c787153f83f9403ce9b2bf",
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

_LSA_NCCL_2307_FULL_MNNVL_20260802 = _WheelSet(
    prefix=_LSA_NCCL_2307_HYBRID_WEAK_20260802.prefix,
    wheels=_LSA_NCCL_2307_HYBRID_WEAK_20260802.wheels,
    runtime_libraries=(
        _RuntimeLibraryArtifact(
            prefix=f"{_WHEEL_ARTIFACT_ROOT}/nccl-2.30.7-full-mnnvl-lsa-20260802",
            filename="libnccl.so.2.30.7",
            sha256="e38471a61852b2ec56265a1d39b866a33d65b340498380c1ba2101c77e729b38",
            site_packages_path="nvidia/nccl/lib/libnccl.so.2",
        ),
    ),
)


def _wheel_set(build: MoonEPJaxWheelBuild) -> _WheelSet:
    if build == MoonEPJaxWheelBuild.LSA_20260802:
        return _LSA_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_20260802:
        return _LSA_NCCL_2307_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_HYBRID_20260802:
        return _LSA_NCCL_2307_HYBRID_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_HYBRID_WEAK_20260802:
        return _LSA_NCCL_2307_HYBRID_WEAK_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_HYBRID_CTA16_20260802:
        return _LSA_NCCL_2307_HYBRID_CTA16_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_WORLD_GIN_20260802:
        return _LSA_NCCL_2307_WORLD_GIN_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_BARRIER_ONLY_20260802:
        return _LSA_NCCL_2307_BARRIER_ONLY_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_LSA_BARRIER_ONLY_20260802:
        return _LSA_NCCL_2307_LSA_BARRIER_ONLY_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_KERNEL_20260802:
        return _LSA_NCCL_2307_NOOP_KERNEL_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_LSA_HOST_20260802:
        return _LSA_NCCL_2307_NOOP_LSA_HOST_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_RUN_COLLECTIVE_20260802:
        return _LSA_NCCL_2307_NOOP_RUN_COLLECTIVE_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_AFTER_DEVICE_COMM_20260802:
        return _LSA_NCCL_2307_NOOP_AFTER_DEVICE_COMM_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_BEFORE_DEVICE_COMM_20260802:
        return _LSA_NCCL_2307_NOOP_BEFORE_DEVICE_COMM_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_NOOP_AFTER_BUFFER_CONVERSION_20260802:
        return _LSA_NCCL_2307_NOOP_AFTER_BUFFER_CONVERSION_20260802
    if build == MoonEPJaxWheelBuild.LSA_NCCL_2307_FULL_MNNVL_20260802:
        return _LSA_NCCL_2307_FULL_MNNVL_20260802
    raise ValueError(f"unknown MoonEP JAX wheel build: {build}")


def _wheel_install_script(build: MoonEPJaxWheelBuild) -> str:
    wheel_set = _wheel_set(build)
    artifact_records = tuple((wheel_set.prefix, wheel.filename, wheel.sha256) for wheel in wheel_set.wheels) + tuple(
        (library.prefix, library.filename, library.sha256) for library in wheel_set.runtime_libraries
    )
    runtime_library_records = tuple(
        (library.filename, library.site_packages_path) for library in wheel_set.runtime_libraries
    )
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

artifacts = {artifact_records!r}
wheel_dir = Path(os.environ["IRIS_WORKDIR"]) / ".moonep-jax" / {build.value!r}
for prefix, filename, expected_sha256 in artifacts:
    filesystem, remote_root = fsspec.core.url_to_fs(prefix)
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
import os
import shutil
import sysconfig
from importlib.metadata import version
from pathlib import Path

wheel_dir = Path(os.environ["IRIS_WORKDIR"]) / ".moonep-jax" / {build.value!r}
site_packages = Path(sysconfig.get_path("purelib"))
for filename, site_packages_path in {runtime_library_records!r}:
    source = wheel_dir / filename
    target = site_packages / site_packages_path
    target.unlink(missing_ok=True)
    shutil.copy2(source, target)

jax_version = version("jax")
jaxlib_version = version("jaxlib")
if jax_version != "0.11.1.dev20260802+f9f6bbace":
    raise ValueError(f"unexpected JAX version: {{jax_version}}")
if jaxlib_version != "0.11.1.dev0+selfbuilt":
    raise ValueError(f"unexpected jaxlib version: {{jaxlib_version}}")
print(f"fixed MoonEP JAX runtime: jax={{jax_version}} jaxlib={{jaxlib_version}}")
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
        cuda_toolchain_setup_script(),
        _wheel_install_script(build),
    ]
