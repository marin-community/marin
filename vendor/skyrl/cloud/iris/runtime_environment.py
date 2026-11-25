"""Resolve and install the immutable MarinSkyRL task environment."""

from __future__ import annotations

import argparse
import shlex
from enum import StrEnum


MARINSKYRL_TASK_ROOT = "/app/vendor/skyrl"
CHECKPOINT_EXPORT_ENTRYPOINT = "skyrl_train.entrypoints.checkpoint_export"
# Ray requires the exact Python patch version on every node. Use one multi-arch
# task image digest so a gang cannot mix cached versions of the mutable tag.
IRIS_TASK_IMAGE = (
    "ghcr.io/marin-community/iris-task@sha256:13519c59442bd70ed5ec2902869f15094992d6849d7e34e353f1338dec7ff5f8"
)


class RuntimeProfile(StrEnum):
    """Locked dependency set installed for training or checkpoint conversion."""

    MEGATRON = "megatron"
    MEGATRON_EXPORT = "megatron-export"
    HUB_PUBLISH = "hub-publish"

    @property
    def extras(self) -> tuple[str, ...]:
        return _PROFILE_DEPENDENCIES[self][0]

    @property
    def checks(self) -> tuple[str, ...]:
        return _PROFILE_DEPENDENCIES[self][1]

    @property
    def includes_vllm(self) -> bool:
        return "vllm" in self.extras


_PROFILE_DEPENDENCIES = {
    RuntimeProfile.MEGATRON: (
        ("megatron", "vllm", "telemetry"),
        ("flash-attention", "megatron", "agents", "vllm"),
    ),
    RuntimeProfile.MEGATRON_EXPORT: (("megatron",), ("flash-attention", "megatron", "export")),
    RuntimeProfile.HUB_PUBLISH: (("cpu",), ("hub-publish",)),
}


class RuntimeMode(StrEnum):
    TRAINING = "training"
    CHECKPOINT_EXPORT = "checkpoint-export"


def runtime_profile_for_strategy(
    strategy: str | None,
    *,
    mode: RuntimeMode = RuntimeMode.TRAINING,
) -> RuntimeProfile:
    """Return the locked dependency profile for a trainer strategy."""
    checkpoint_export = mode is RuntimeMode.CHECKPOINT_EXPORT
    if strategy == "megatron":
        return RuntimeProfile.MEGATRON_EXPORT if checkpoint_export else RuntimeProfile.MEGATRON
    if strategy is None:
        return RuntimeProfile.MEGATRON_EXPORT if checkpoint_export else RuntimeProfile.MEGATRON
    raise ValueError(f"Unsupported training strategy: {strategy}")


def task_setup_command(profile: RuntimeProfile) -> str:
    """Return the Iris setup invocation for the selected frozen profile."""
    return shlex.join(("bash", f"{MARINSKYRL_TASK_ROOT}/cloud/iris/setup_task_runtime.sh", profile.value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=RuntimeProfile, choices=tuple(RuntimeProfile))
    profile = parser.parse_args().profile
    print("cpu" if "cpu" in profile.extras else "cuda")
    print(" ".join(profile.extras))
    print(" ".join(profile.checks))
