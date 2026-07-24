# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned configuration artifacts consumed by Marin's evaluation launchers.

The lightweight resolver wheels validate mechanism-specific configuration in Marin's normal
environment. The Harbor runtime remains isolated, but is pinned to the resolver's parent commit so
the job that executes a resolved configuration cannot drift from the job that validated it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigArtifact:
    """An immutable resolver wheel and the upstream source revision that produced it."""

    package: str
    repository: str
    revision: str
    release_tag: str
    wheel_url: str
    wheel_sha256: str
    schema_fingerprint: str
    resolver_fingerprint: str | None = None

    @property
    def wheel_requirement(self) -> str:
        """Return the hash-pinned PEP 508 dependency used by Marin's lockfile."""
        return f"{self.package} @ {self.wheel_url}#sha256={self.wheel_sha256}"

    @property
    def source_requirement(self) -> str:
        """Return the matching source requirement for an isolated upstream runtime."""
        return f"{self.package.removesuffix('-config')} @ git+https://github.com/{self.repository}.git@{self.revision}"


# BEGIN GENERATED CONFIG ARTIFACT PINS
HARBOR_CONFIG = ConfigArtifact(
    package="harbor-config",
    repository="marin-community/harbor",
    revision="95dcdfd3f1f2fdc745b15f9ca411755abc6734c5",
    release_tag="harbor-config-95dcdfd3f1f2fdc745b15f9ca411755abc6734c5",
    wheel_url="https://github.com/marin-community/harbor/releases/download/harbor-config-95dcdfd3f1f2fdc745b15f9ca411755abc6734c5/harbor_config-0.1.0-py3-none-any.whl",
    wheel_sha256="eebfe87f7a0408c0540f0e633822158b45a2f95933fa1e19cf13b41c498777f9",
    schema_fingerprint="22029cc681d1ce38c9b647ee088a28f2f9c8a1425d1ba6b52f4f0912cfd9fb83",
    resolver_fingerprint="714196edd02467f2e53c6b3b12c6ce05cf4c77e4db5fa856faa61e614c491fc6",
)

EVALCHEMY_CONFIG = ConfigArtifact(
    package="evalchemy-config",
    repository="marin-community/evalchemy",
    revision="3aac962b462751ea6097cde7b9c6a61663c21029",
    release_tag="evalchemy-config-12a73251f540428e8ea01cdeefa2611d2344a8f266b4c2c38a181de5106c3b24",
    wheel_url="https://github.com/marin-community/evalchemy/releases/download/evalchemy-config-12a73251f540428e8ea01cdeefa2611d2344a8f266b4c2c38a181de5106c3b24/evalchemy_config-0.1.0-py3-none-any.whl",
    wheel_sha256="ef95d211e9f4bf5a6b8dec7bd3436b3cda88e685096587b9ae7cd06e59dfae00",
    schema_fingerprint="12a73251f540428e8ea01cdeefa2611d2344a8f266b4c2c38a181de5106c3b24",
)
# END GENERATED CONFIG ARTIFACT PINS
