# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog deployment config schema, loader, and endpoint derivation.

A finelog config is a yaml file describing a single logical log server: its
name, port, image, optional remote-archive directory, and a deployment
backend (exactly one of `gcp` or `k8s`).  The schema is intentionally small
and explicit; finelog owns its deployment knobs so iris's cluster yaml only
has to reference the config by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml

USER_CONFIG_DIR = Path.home() / ".config" / "marin" / "finelog"


def _bundled_config_dir() -> Path:
    """Locate the `config/` directory adjacent to the `finelog` package source.

    Works for both editable repo checkouts (`lib/finelog/config/`) and wheel
    installs (`<site-packages>/config/` if shipped at the wheel root).
    """
    pkg_root = Path(str(files("finelog")))
    # `src/finelog/` in the repo → walk up to `lib/finelog/` and find `config/`.
    for candidate in (pkg_root.parent.parent / "config", pkg_root.parent / "config"):
        if candidate.is_dir():
            return candidate
    # Fall back to the repo layout even if missing — callers will see a clear
    # FileNotFoundError listing the searched paths.
    return pkg_root.parent.parent / "config"


@dataclass(frozen=True)
class GcpDeployment:
    """GCE VM deployment knobs."""

    project: str
    zone: str
    machine_type: str = "n2-standard-4"
    boot_disk_size_gb: int = 200
    service_account: str | None = None
    network_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class K8sDeployment:
    """Kubernetes deployment knobs."""

    namespace: str
    storage_class: str | None = None
    storage_gb: int = 200


@dataclass(frozen=True)
class Deployment:
    """Backend selector. Exactly one of `gcp` or `k8s` must be set."""

    gcp: GcpDeployment | None = None
    k8s: K8sDeployment | None = None

    def __post_init__(self) -> None:
        configured = [name for name, val in (("gcp", self.gcp), ("k8s", self.k8s)) if val is not None]
        if len(configured) == 0:
            raise ValueError("deployment must set exactly one of {gcp, k8s}; got none")
        if len(configured) > 1:
            raise ValueError(f"deployment must set exactly one of {{gcp, k8s}}; got {configured}")


@dataclass(frozen=True)
class FinelogConfig:
    """Parsed finelog deployment config."""

    name: str
    port: int
    image: str
    remote_log_dir: str
    deployment: Deployment


def _config_search_paths(name_or_path: str) -> list[Path]:
    """Return the list of paths searched for `name_or_path`, in order."""
    direct = Path(name_or_path)
    if direct.suffix in (".yaml", ".yml") or direct.exists():
        return [direct]
    return [
        USER_CONFIG_DIR / f"{name_or_path}.yaml",
        _bundled_config_dir() / f"{name_or_path}.yaml",
    ]


def _build_gcp(raw: dict) -> GcpDeployment:
    tags = raw.get("network_tags") or ()
    return GcpDeployment(
        project=raw["project"],
        zone=raw["zone"],
        machine_type=raw.get("machine_type", "n2-standard-4"),
        boot_disk_size_gb=int(raw.get("boot_disk_size_gb", 200)),
        service_account=raw.get("service_account"),
        network_tags=tuple(tags),
    )


def _build_k8s(raw: dict) -> K8sDeployment:
    return K8sDeployment(
        namespace=raw["namespace"],
        storage_class=raw.get("storage_class"),
        storage_gb=int(raw.get("storage_gb", 200)),
    )


def load_finelog_config(name_or_path: str) -> FinelogConfig:
    """Load a finelog config by name or path.

    Search order:
      1. `name_or_path` as a literal path (absolute or relative).
      2. `~/.config/marin/finelog/<name>.yaml`.
      3. Repo-bundled `lib/finelog/config/<name>.yaml`.
    """
    candidates = _config_search_paths(name_or_path)
    for path in candidates:
        if path.is_file():
            return _load_from_path(path)
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"finelog config '{name_or_path}' not found; searched:\n  {searched}")


def _load_from_path(path: Path) -> FinelogConfig:
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a yaml mapping at top level")

    deploy_raw = raw.get("deployment")
    if not isinstance(deploy_raw, dict):
        raise ValueError(f"{path}: missing or invalid `deployment` block")

    gcp = _build_gcp(deploy_raw["gcp"]) if "gcp" in deploy_raw else None
    k8s = _build_k8s(deploy_raw["k8s"]) if "k8s" in deploy_raw else None
    deployment = Deployment(gcp=gcp, k8s=k8s)

    return FinelogConfig(
        name=raw["name"],
        port=int(raw["port"]),
        image=raw["image"],
        remote_log_dir=raw.get("remote_log_dir", ""),
        deployment=deployment,
    )


def derive_endpoint_uri(cfg: FinelogConfig) -> tuple[str, dict[str, str]]:
    """Map a finelog config onto an iris endpoint (uri, metadata) pair.

    Compatible with `iris.cluster.endpoints.resolve_endpoint_uri`.
    """
    if cfg.deployment.gcp is not None:
        gcp = cfg.deployment.gcp
        return (
            f"gcp://{cfg.name}",
            {"project": gcp.project, "zone": gcp.zone, "port": str(cfg.port)},
        )
    assert cfg.deployment.k8s is not None  # guaranteed by Deployment.__post_init__
    k8s = cfg.deployment.k8s
    return (
        f"k8s://{cfg.name}.{k8s.namespace}",
        {"port": str(cfg.port)},
    )
