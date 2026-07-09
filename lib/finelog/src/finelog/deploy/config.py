# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog deployment config schema, loader, and endpoint derivation.

A finelog config is a yaml file describing a single logical log server: its
name, port, image, optional remote-archive directory, and a deployment
backend (exactly one of `gcp` or `k8s`).  The schema is intentionally small
and explicit; finelog owns its deployment knobs so iris's cluster yaml only
has to reference the config by name.
"""

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml
from rigging.tunnel import GcpSshForwardTarget, K8sPortForwardTarget, TunnelTarget

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
    # Kubeconfig file and context every kubectl operation binds to. Unset falls
    # back to kubectl's own resolution (KUBECONFIG env var or ~/.kube/config);
    # setting both makes deploys independent of the operator's environment and
    # of the file's current-context.
    kubeconfig: str | None = None
    kube_context: str | None = None
    storage_class: str | None = None
    storage_gb: int = 200
    # S3-compatible endpoint (e.g. Cloudflare R2) for an `s3://` remote_log_dir.
    # Required there: `finelog deploy up` mints a Secret holding this endpoint
    # plus the operator's R2 creds, projected into the pod via envFrom so the
    # server can authenticate. Unused for `gs://` (GCS uses workload identity).
    object_storage_endpoint: str | None = None
    # PriorityClass stamped on the finelog pod. When finelog is the log backend
    # for an Iris control plane, set this to `iris-system` so a user job cannot
    # preempt it off the shared control node. `deploy up` creates the class
    # (idempotently) from name+value before applying the Deployment, so finelog
    # can still be brought up first on a fresh cluster. Iris is the canonical
    # owner of the iris-* bands (see IRIS_PRIORITY_CLASSES); keep priority_class_value
    # in sync with it — value/preemptionPolicy are immutable, so a mismatch makes
    # one side's apply fail loudly rather than silently disagree.
    priority_class_name: str | None = None
    priority_class_value: int | None = None

    def __post_init__(self) -> None:
        if (self.priority_class_name is None) != (self.priority_class_value is None):
            raise ValueError("priority_class_name and priority_class_value must be set together")


@dataclass(frozen=True)
class CidrAuthLayer:
    """Admit a request whose transport peer is in one of ``cidrs``.

    Matches the transport peer only, never a spoofable forwarded header. See
    ``INTRA_CLUSTER_CIDRS`` for the ranges an in-cluster finelog trusts.
    """

    cidrs: tuple[str, ...]

    def to_policy_dict(self) -> dict:
        return {"type": "cidr", "cidrs": list(self.cidrs)}


# The private-network + loopback ranges an in-cluster finelog trusts by cidr: its
# own cluster's pods (CoreWeave is 10.x; the rest of RFC 1918 covers other
# platforms) plus loopback (a port-forward), never the public internet. The
# bundled deploy configs (`lib/finelog/config/*.yaml`) spell the same set into
# their `cidr` layer; the controller's embedded fallback server uses it directly
# (see iris `build_log_stack`).
INTRA_CLUSTER_CIDRS: tuple[str, ...] = (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "127.0.0.0/8",
    "::1/128",
)


@dataclass(frozen=True)
class JwtKeyEntry:
    """A trusted cluster and its Ed25519 delegation public keys (PEM).

    ``public_keys`` is a list so a key rotation can carry the old and new keys
    together during the overlap window; a token signed by either verifies.
    """

    cluster: str
    public_keys: tuple[str, ...]


@dataclass(frozen=True)
class JwtAuthLayer:
    """Admit a bearer JWT whose EdDSA signature verifies against one of ``keys`` and
    whose audience is ``finelog``.

    Each key is a trusted cluster's Ed25519 public key(s); every configured key
    admits equally. Public keys are not secret material, so a jwt layer inlines
    safely into a plaintext deploy artifact.
    """

    keys: tuple[JwtKeyEntry, ...]

    def to_policy_dict(self) -> dict:
        return {
            "type": "jwt",
            "keys": [{"cluster": k.cluster, "public_keys": list(k.public_keys)} for k in self.keys],
        }


# One entry in the ordered auth stack. Evaluation order == list order (first
# Allow admits, first Reject denies, none → deny; see the Rust `AuthPolicy`).
AuthLayer = CidrAuthLayer | JwtAuthLayer


def auth_policy_json(layers: tuple[AuthLayer, ...]) -> str:
    """Serialize an ordered auth-layer stack to the `FINELOG_AUTH_POLICY` JSON the
    finelog server parses."""
    return json.dumps([layer.to_policy_dict() for layer in layers], separators=(",", ":"))


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
    # Rigging transport URL clients use to reach this server through the controller proxy
    # (e.g. `iap+https://iris.oa.dev/proxy/system.log-server`); unset = fall back to SSH/k8s tunnel.
    client_url: str | None = None
    # Ordered authenticated-ingress layer stack. Empty leaves the server on its
    # allow-localhost default (loopback only, never open).
    auth: tuple[AuthLayer, ...] = ()


def _config_search_paths(name_or_path: str) -> list[Path]:
    """Return the list of paths searched for `name_or_path`, in order."""
    direct = Path(name_or_path)
    if direct.suffix in (".yaml", ".yml") or direct.exists():
        return [direct]
    return [
        USER_CONFIG_DIR / f"{name_or_path}.yaml",
        _bundled_config_dir() / f"{name_or_path}.yaml",
    ]


def find_finelog_config(name_or_path: str) -> Path | None:
    """Return the path `name_or_path` resolves to, or None when no such config exists."""
    return next((path for path in _config_search_paths(name_or_path) if path.is_file()), None)


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


def _build_auth_layers(raw: list, path: Path) -> tuple[AuthLayer, ...]:
    """Parse the `auth:` YAML list into an ordered layer stack. List order is
    evaluation order (see the Rust `AuthPolicy`)."""
    layers: list[AuthLayer] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "type" not in item:
            raise ValueError(f"{path}: auth[{i}] must be a mapping with a `type` key")
        layer_type = item["type"]
        if layer_type == "cidr":
            layers.append(CidrAuthLayer(cidrs=tuple(item.get("cidrs") or ())))
        elif layer_type == "jwt":
            keys = tuple(
                JwtKeyEntry(cluster=k["cluster"], public_keys=tuple(k["public_keys"])) for k in item.get("keys") or ()
            )
            layers.append(JwtAuthLayer(keys=keys))
        else:
            raise ValueError(f"{path}: auth[{i}] has unknown type {layer_type!r} (expected cidr|jwt)")
    return tuple(layers)


def _build_k8s(raw: dict) -> K8sDeployment:
    priority_class_value = raw.get("priority_class_value")
    return K8sDeployment(
        namespace=raw["namespace"],
        kubeconfig=raw.get("kubeconfig"),
        kube_context=raw.get("kube_context"),
        storage_class=raw.get("storage_class"),
        storage_gb=int(raw.get("storage_gb", 200)),
        object_storage_endpoint=raw.get("object_storage_endpoint"),
        priority_class_name=raw.get("priority_class_name"),
        priority_class_value=None if priority_class_value is None else int(priority_class_value),
    )


def load_finelog_config(name_or_path: str) -> FinelogConfig:
    """Load a finelog config by name or path.

    Search order:
      1. `name_or_path` as a literal path (absolute or relative).
      2. `~/.config/marin/finelog/<name>.yaml`.
      3. Repo-bundled `lib/finelog/config/<name>.yaml`.
    """
    path = find_finelog_config(name_or_path)
    if path is not None:
        return _load_from_path(path)
    searched = "\n  ".join(str(p) for p in _config_search_paths(name_or_path))
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

    auth_raw = raw.get("auth")
    if auth_raw is not None and not isinstance(auth_raw, list):
        raise ValueError(f"{path}: `auth` must be a list of layers")
    auth = _build_auth_layers(auth_raw, path) if auth_raw else ()

    return FinelogConfig(
        name=raw["name"],
        port=int(raw["port"]),
        image=raw["image"],
        remote_log_dir=raw.get("remote_log_dir", ""),
        deployment=deployment,
        client_url=raw.get("client_url"),
        auth=auth,
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


def tunnel_target_for(cfg: FinelogConfig) -> TunnelTarget:
    """Build a rigging tunnel target from a finelog deployment block.

    The GCP path forwards ``deployment.gcp.service_account`` as the SSH
    impersonation principal, matching the deploy CLI's own SSH calls, so this
    target works wherever ``finelog deploy status`` does.
    """
    if cfg.deployment.gcp is not None:
        gcp = cfg.deployment.gcp
        return GcpSshForwardTarget(
            project=gcp.project,
            zone=gcp.zone,
            instance=cfg.name,
            port=cfg.port,
            impersonate_service_account=gcp.service_account,
        )
    assert cfg.deployment.k8s is not None  # guaranteed by Deployment.__post_init__
    k8s = cfg.deployment.k8s
    return K8sPortForwardTarget(
        namespace=k8s.namespace,
        service=cfg.name,
        port=cfg.port,
        kubeconfig=str(Path(k8s.kubeconfig).expanduser()) if k8s.kubeconfig else None,
        context=k8s.kube_context,
    )
