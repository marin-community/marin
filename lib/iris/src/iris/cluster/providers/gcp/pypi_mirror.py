# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PyPI mirror env-var construction for Iris workers.

Builds the ``UV_*`` env vars that point an Iris task at the Google Artifact
Registry remote-Python repos that proxy ``pypi.org`` and the PyTorch CPU
index. The worker callsite (``task_attempt._prepare_container_config``) is
responsible for region resolution and opt-out handling; this module is a
pure function over a multi-region string.
"""

from dataclasses import dataclass

from iris.cluster.providers.gcp.bootstrap import _ZONE_PREFIX_TO_MULTI_REGION

AR_PROJECT: str = "hai-gcp-models"
PYPI_MIRROR_REPO: str = "pypi-mirror"
PYTORCH_CPU_MIRROR_REPO: str = "pytorch-cpu-mirror"

# Env-var contract for the per-job opt-out. Hoisted to constants so the
# magic strings live in exactly one place (AGENTS.md § Naming).
IRIS_PYPI_MIRROR_ENV_VAR: str = "IRIS_PYPI_MIRROR"
IRIS_PYPI_MIRROR_OPT_OUT: str = "0"

# Single source of truth for which multi-regions have AR repos provisioned.
# Derived from ``bootstrap._ZONE_PREFIX_TO_MULTI_REGION`` so adding a third
# continent only requires one edit.
SUPPORTED_MULTI_REGIONS: frozenset[str] = frozenset(_ZONE_PREFIX_TO_MULTI_REGION.values())


@dataclass(frozen=True)
class PypiMirrorEnv:
    """Typed env-var bundle for AR-mirror uv configuration.

    Use ``as_env()`` to materialize as a ``dict[str, str]`` at the worker
    callsite. Field-level access keeps tests readable.
    """

    default_index: str
    pytorch_cpu_index: str
    keyring_provider: str = "subprocess"

    def as_env(self) -> dict[str, str]:
        """Return env vars as a flat ``dict[str, str]`` for ``env.update``."""
        return {
            "UV_DEFAULT_INDEX": self.default_index,
            "UV_INDEX_PYTORCH_CPU": self.pytorch_cpu_index,
            "UV_KEYRING_PROVIDER": self.keyring_provider,
        }


def build_pypi_mirror_env(multi_region: str, project: str = AR_PROJECT) -> PypiMirrorEnv:
    """Build uv env vars that point dependency resolution at AR remote PyPI repos.

    Caller responsibilities (not enforced here, kept out so the helper is
    trivially testable):

    1. Verify ``IRIS_WORKER_REGION`` is set and resolves via
       ``zone_to_multi_region`` to a non-None continent.
    2. Verify the user has not set ``IRIS_PYPI_MIRROR=0`` in
       ``EnvironmentConfig.env_vars``.

    Args:
        multi_region: AR multi-region location. Must be in
            ``SUPPORTED_MULTI_REGIONS``; other values raise. Pass the result
            of ``zone_to_multi_region``.
        project: GCP project hosting the AR repos. Defaults to ``AR_PROJECT``.

    Returns:
        ``PypiMirrorEnv`` with three populated fields. URL form:
        ``https://{multi_region}-python.pkg.dev/{project}/<repo>/simple/``.

    Raises:
        ValueError: if ``multi_region`` is not in ``SUPPORTED_MULTI_REGIONS``.
    """
    if multi_region not in SUPPORTED_MULTI_REGIONS:
        raise ValueError(
            f"multi_region {multi_region!r} is not in SUPPORTED_MULTI_REGIONS "
            f"({sorted(SUPPORTED_MULTI_REGIONS)}). Pass the result of "
            f"zone_to_multi_region()."
        )
    base = f"https://{multi_region}-python.pkg.dev/{project}"
    return PypiMirrorEnv(
        default_index=f"{base}/{PYPI_MIRROR_REPO}/simple/",
        pytorch_cpu_index=f"{base}/{PYTORCH_CPU_MIRROR_REPO}/simple/",
    )
