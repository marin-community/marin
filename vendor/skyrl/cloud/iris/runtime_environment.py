"""Resolve and install the immutable MarinSkyRL task environment."""

from __future__ import annotations

import shlex
from enum import StrEnum

from cloud.iris.paths import PROJECT_RELATIVE_PATH

MARINSKYRL_TASK_ROOT = f"/app/{PROJECT_RELATIVE_PATH.as_posix()}"
MARINSKYRL_ACTIVATION_FILE = f"{MARINSKYRL_TASK_ROOT}/.iris-runtime-env"
MARINSKYRL_BOOTSTRAP_SCRIPT = "cloud/iris/bootstrap_runtime.sh"
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


def task_setup_script(source_digest: str, profile: RuntimeProfile) -> str:
    """Verify bundled source and install its independently frozen dependency profile."""
    checkout = MARINSKYRL_TASK_ROOT
    activation_file = MARINSKYRL_ACTIVATION_FILE
    bootstrap_script = f"{checkout}/{MARINSKYRL_BOOTSTRAP_SCRIPT}"
    return f"""set -euo pipefail
checkout={shlex.quote(checkout)}
cd "$checkout"
python3.12 -c {shlex.quote("from cloud.iris.runtime_bundle import validate_bundled_runtime; assert validate_bundled_runtime() == " + repr(source_digest))}
bash {shlex.quote(bootstrap_script)} \
  "$checkout" \
  "$IRIS_VENV" \
  {shlex.quote(activation_file)} \
  {shlex.quote(profile.value)} \
  production
source {shlex.quote(activation_file)}
"""
