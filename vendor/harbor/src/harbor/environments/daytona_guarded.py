"""Daytona sandboxes whose only package-download route is a policy gateway."""

import re
from pathlib import Path
from urllib.parse import urlsplit

from daytona import Image

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.daytona.environment import DaytonaEnvironment
from harbor.models.environment_type import EnvironmentType
from harbor.security.origin import public_origin

# Allow-listing these hosts would let clients bypass the package gateway.
PACKAGE_HOSTS = frozenset(
    {
        "pypi.org",
        "pythonhosted.org",
        "npmjs.org",
        "github.com",
        "githubusercontent.com",
        "nodejs.org",
        "astral.sh",
        "bun.sh",
    }
)
DOMAIN = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\Z")
IMAGE_DIGEST = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")


def gateway_domains(gateway: str, service_domains: list[str]) -> str:
    """Validate exact operator-owned endpoints for Daytona's network firewall."""
    host = urlsplit(public_origin(gateway)).hostname
    assert host is not None
    domains = sorted({host, *service_domains})
    for domain in domains:
        if not DOMAIN.fullmatch(domain) or "." not in domain or domain.rsplit(".", 1)[-1].isdigit():
            raise ValueError(f"An exact DNS service hostname is required: {domain!r}")
        if any(domain == blocked or domain.endswith("." + blocked) for blocked in PACKAGE_HOSTS):
            raise ValueError(f"Direct package/source host bypasses the gateway: {domain}")
    return ",".join(domains)


class GuardedDaytonaEnvironment(DaytonaEnvironment):
    """Use a reviewed immutable image and provider-enforced destination allow list.

    Model/service endpoints are trusted exceptions supplied by the operator.
    Do not allow a general proxy, file host, or attacker-controlled service.
    """

    def __init__(
        self,
        *,
        environment_dir: Path,
        package_gateway: str,
        service_domains: list[str],
        approved_image: str,
        **kwargs,
    ):
        self.package_gateway = public_origin(package_gateway)
        if not IMAGE_DIGEST.fullmatch(approved_image):
            raise ValueError("Guarded Daytona requires an approved image pinned by SHA-256 digest")
        self.approved_image = approved_image
        if (environment_dir / "docker-compose.yaml").exists():
            raise ValueError("Guarded Daytona does not support Docker-in-Docker or compose builds")
        if kwargs.pop("extra_docker_compose", None):
            raise ValueError("Guarded Daytona does not support compose overlays")
        forbidden = {"network_policy", "snapshot_template_name", "auto_snapshot", "assume_global_snapshot"}
        if forbidden.intersection(kwargs):
            raise ValueError(f"Guarded Daytona owns these settings: {sorted(forbidden.intersection(kwargs))}")
        super().__init__(
            environment_dir=environment_dir,
            network_policy={
                "mode": "domain_allow_list",
                "value": gateway_domains(self.package_gateway, service_domains),
            },
            **kwargs,
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DAYTONA_GUARDED

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        # Task Dockerfiles are not built by an uncontrolled remote builder.
        if not self.environment_dir.is_dir():
            raise ValueError(f"Missing task environment directory: {self.environment_dir}")

    async def _resolve_start_sandbox_params(self, daytona, resources, *, force_build):
        if force_build:
            raise ValueError("Guarded Daytona cannot rebuild images; publish a reviewed digest first")
        return self._image_sandbox_params(image=Image.base(self.approved_image), resources=resources)

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str]:
        merged = super()._merge_env(env) or {}
        # Routing convenience. The provider's network firewall is the enforcement
        # boundary when a subprocess clears or replaces these variables.
        merged.update(
            {
                "PIP_INDEX_URL": f"{self.package_gateway}/pypi/simple/",
                "PIP_EXTRA_INDEX_URL": "",
                "UV_DEFAULT_INDEX": f"{self.package_gateway}/pypi/simple/",
                "UV_INDEX_URL": f"{self.package_gateway}/pypi/simple/",
                "UV_EXTRA_INDEX_URL": "",
                "UV_PYTHON_DOWNLOADS": "never",
                "NPM_CONFIG_REGISTRY": f"{self.package_gateway}/npm/",
                "npm_config_registry": f"{self.package_gateway}/npm/",
            }
        )
        return merged
