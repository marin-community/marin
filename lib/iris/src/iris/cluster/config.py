# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris cluster configuration schema, loading, and serialization.

This module is *pure configuration*: it parses YAML into validated pydantic
models, applies defaults, and serializes models back to YAML/JSON-friendly
dicts. It deliberately imports nothing from the backends, controller, or
autoscaler — turning a parsed config into live objects (providers, backends,
controllers) is the job of :mod:`iris.cluster.composer`.

The model field names are the canonical wire names. Human-authored YAML may use
a few friendly spellings (``cpu``/``ram``/``disk`` under ``resources``,
``on-demand`` for capacity type, ``platform: {local:}`` for oneof selection);
these are normalized at the parse boundary.
"""

from __future__ import annotations

import copy
import ipaddress
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import fsspec
import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, field_validator, model_validator
from rigging.timing import Duration

from iris.cluster.tpu_topology import TPU_FAMILY_VARIANT_PREFIX, get_tpu_topology, tpu_variant_name
from iris.cluster.types import (
    DEFAULT_BACKEND_ID,
    LOCAL_CLUSTER,
    AcceleratorType,
    CapacityType,
    GcpSliceMode,
    WellKnownAttribute,
    parse_memory_string,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

DEFAULT_SSH_PORT = 22
DEFAULT_SSH_CONNECT_TIMEOUT = Duration.from_seconds(30)
DEFAULT_PRIORITY = 100

_COREWEAVE_TOPOLOGY_LABEL_PREFIXES = (
    "backend.coreweave.cloud/",
    "ib.coreweave.cloud/",
    "node.coreweave.cloud/",
)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _coerce_duration(value: Any) -> Duration:
    if isinstance(value, Duration):
        return value
    if isinstance(value, Mapping):
        if "milliseconds" in value:
            return Duration(int(value["milliseconds"]))
        if "seconds" in value:
            return Duration.from_seconds(float(value["seconds"]))
        raise ValueError(f"duration mapping must set 'milliseconds' or 'seconds', got {dict(value)!r}")
    if isinstance(value, bool):
        raise ValueError("duration cannot be a bool")
    if isinstance(value, (int, float)):
        return Duration(int(value))
    raise ValueError(f"cannot parse duration from {value!r}")


def _dump_duration(value: Duration) -> dict[str, int]:
    return {"milliseconds": value.to_ms()}


DurationField = Annotated[
    Duration,
    BeforeValidator(_coerce_duration),
    PlainSerializer(_dump_duration, return_type=dict, when_used="always"),
]


def _coerce_capacity_type(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_")
    return value


def _coerce_priority_band(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("max_band cannot be a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        name = value if value.startswith("PRIORITY_BAND_") else f"PRIORITY_BAND_{value.upper()}"
        return job_pb2.PriorityBand.Value(name)
    raise ValueError(f"cannot parse priority band from {value!r}")


CapacityTypeField = Annotated[CapacityType, BeforeValidator(_coerce_capacity_type)]
PriorityBandField = Annotated[int, BeforeValidator(_coerce_priority_band)]


def _normalize_oneof(data: Any, keys: tuple[str, ...]) -> Any:
    """Normalize a former-protobuf ``oneof`` group on raw input.

    Two responsibilities that together reproduce protobuf ``oneof`` semantics:

    - PyYAML parses ``platform:\n  local:`` as ``{"platform": {"local": None}}``;
      a selected empty arm's ``None`` becomes ``{}`` so the sub-model is built.
    - At most one arm may be present. Setting two (e.g. ``gcp:`` and ``manual:``)
      is rejected rather than silently resolving to the first present arm.

    Selection is by key *presence*: an unselected arm is absent, not ``None``.
    Config serialization uses ``exclude_none`` (see :func:`config_to_dict`), so a
    ``model_dump`` round-trip never reintroduces the unselected arms. Zero arms
    is permitted; downstream validators require a selection where one is needed.
    """
    if not isinstance(data, Mapping):
        return data
    out = dict(data)
    selected = [key for key in keys if key in out]
    if len(selected) > 1:
        raise ValueError(f"at most one of {{{', '.join(keys)}}} may be set, got {', '.join(selected)}")
    for key in selected:
        if out[key] is None:
            out[key] = {}
    return out


class _Config(BaseModel):
    """Base for all config models: reject unknown keys, allow Duration."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class _OneofConfig(_Config):
    """Base for configs translating a protobuf ``oneof``: at most one arm field is set.

    Subclasses list their mutually-exclusive arm fields in ``_ONEOF_ARMS``. A shared
    before-validator enforces at-most-one selection (and expands a selected null arm to
    ``{}``); ``_selected_arm`` reports which arm is set. Subclasses expose it under a
    domain name (``platform_kind``/``controller_kind``/``provider_kind``).
    """

    _ONEOF_ARMS: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _normalize_arms(cls, data: Any) -> Any:
        return _normalize_oneof(data, cls._ONEOF_ARMS)

    def _selected_arm(self) -> str | None:
        for arm in self._ONEOF_ARMS:
            if getattr(self, arm) is not None:
                return arm
        return None


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------


class GcpPlatformConfig(_Config):
    project_id: str = ""
    zones: list[str] = Field(default_factory=list)  # all zones, for list_all_slices


class ManualPlatformConfig(_Config):
    pass


class LocalPlatformConfig(_Config):
    pass


class CoreweavePlatformConfig(_Config):
    region: str = ""
    namespace: str = ""  # default: "iris"
    kubeconfig_path: str = ""  # optional; in-cluster auth if empty
    object_storage_endpoint: str = ""  # S3 base endpoint, not bucket-specific


class PlatformConfig(_OneofConfig):
    _ONEOF_ARMS = ("gcp", "manual", "local", "coreweave")
    label_prefix: str = "iris"
    gcp: GcpPlatformConfig | None = None
    manual: ManualPlatformConfig | None = None
    local: LocalPlatformConfig | None = None
    coreweave: CoreweavePlatformConfig | None = None

    def platform_kind(self) -> str | None:
        return self._selected_arm()


# ---------------------------------------------------------------------------
# VM (controller VM provisioning)
# ---------------------------------------------------------------------------


class GcpVmConfig(_Config):
    zone: str = ""
    machine_type: str = ""  # default: "n2-standard-4"
    boot_disk_size_gb: int = 0  # default: 50
    service_account: str = ""


class ManualVmConfig(_Config):
    host: str = ""
    ssh_user: str = "root"
    ssh_key_file: str = ""


class VmConfig(_OneofConfig):
    _ONEOF_ARMS = ("gcp", "manual")
    name: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    gcp: GcpVmConfig | None = None
    manual: ManualVmConfig | None = None


# ---------------------------------------------------------------------------
# Slice (per-scale-group provisioning template)
# ---------------------------------------------------------------------------


class GcpSliceConfig(_Config):
    mode: GcpSliceMode = GcpSliceMode.TPU
    zone: str = ""
    runtime_version: str = ""
    topology: str = ""
    machine_type: str = ""  # required for mode=vm
    service_account: str = ""
    # None means "default" (True): provision with an external IP. Set False to
    # provision without one (honored only for TPU-mode slices).
    enable_external_ip: bool | None = None


class CoreweaveSliceConfig(_Config):
    region: str = ""
    instance_type: str = ""  # e.g. "gd-8xh100ib-i128"
    gpu_class: str = ""  # e.g. "H100"
    infiniband: bool = False


class ManualSliceConfig(_Config):
    hosts: list[str] = Field(default_factory=list)
    ssh_user: str = "root"
    ssh_key_file: str = ""


class LocalSliceConfig(_Config):
    pass


class SliceConfig(_OneofConfig):
    _ONEOF_ARMS = ("gcp", "coreweave", "manual", "local")
    name_prefix: str = ""
    num_vms: int = 0
    # Derived from ScaleGroupResources by ScaleGroupConfig; None until derived.
    accelerator_type: AcceleratorType | None = None
    accelerator_variant: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    gpu_count: int = 0
    disk_size_gb: int = 0
    capacity_type: CapacityTypeField | None = None
    gcp: GcpSliceConfig | None = None
    coreweave: CoreweaveSliceConfig | None = None
    manual: ManualSliceConfig | None = None
    local: LocalSliceConfig | None = None

    def platform_kind(self) -> str | None:
        return self._selected_arm()


# ---------------------------------------------------------------------------
# Scale group
# ---------------------------------------------------------------------------


class ScaleGroupResources(_Config):
    """Resources available per VM in a scale group (the canonical declaration).

    Accepts friendly YAML keys: ``cpu`` (cores), ``ram`` and ``disk`` (size
    strings like ``64GB``). These normalize to ``cpu_millicores`` /
    ``memory_bytes`` / ``disk_bytes``.
    """

    cpu_millicores: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0
    device_type: AcceleratorType | None = None
    device_variant: str = ""
    device_count: int = 0
    capacity_type: CapacityTypeField | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        cpu = out.pop("cpu", None)
        if cpu is not None:
            out["cpu_millicores"] = int(float(cpu) * 1000)
        ram = out.pop("ram", None)
        if ram is not None:
            out["memory_bytes"] = _parse_memory_value(ram, "resources.ram")
        disk = out.pop("disk", None)
        if disk is not None:
            out["disk_bytes"] = _parse_memory_value(disk, "resources.disk")
        return out


class WorkerSettings(_Config):
    attributes: dict[str, str] = Field(default_factory=dict)
    # Overrides DefaultsConfig.worker.cache_dir for this group (e.g. a real disk
    # for CPU workers instead of the default /dev/shm tmpfs).
    cache_dir: str = ""


class ScaleGroupConfig(_Config):
    name: str = ""
    # Extra slices to keep warm beyond demand: target = min(demand + buffer, max).
    buffer_slices: int = 0
    max_slices: int = 0
    resources: ScaleGroupResources | None = None
    num_vms: int | None = None
    # Priority for waterfall routing (lower = higher priority).
    priority: int = DEFAULT_PRIORITY
    # Max scale-up requests / scale-down terminations per minute (0 = default 5/min).
    scale_up_rate_limit: int = 0
    scale_down_rate_limit: int = 0
    slice_template: SliceConfig | None = None
    # Always present (empty unless overridden) so consumers read it without a None check.
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    # Allocation pool grouping; groups sharing a quota_pool propagate quota state.
    quota_pool: str = ""
    # Tier within the quota pool (1-based, lower preferred; 0 = unset).
    allocation_tier: int = 0

    @model_validator(mode="after")
    def _derive_slice_template(self) -> ScaleGroupConfig:
        """Populate slice_template scalars from resources (used by providers)."""
        if self.resources is None or self.slice_template is None:
            return self
        res = self.resources
        tmpl = self.slice_template
        tmpl.accelerator_type = res.device_type
        if res.device_variant:
            tmpl.accelerator_variant = res.device_variant
        tmpl.capacity_type = res.capacity_type
        if res.device_type == AcceleratorType.GPU and res.device_count > 0:
            tmpl.gpu_count = res.device_count
        if res.disk_bytes:
            tmpl.disk_size_gb = res.disk_bytes // (1024**3)
        return self


# ---------------------------------------------------------------------------
# Worker (bootstrap wire config)
# ---------------------------------------------------------------------------


class WorkerConfig(_Config):
    """Everything the worker process needs.

    Built by the controller from ``defaults.worker`` + per-scale-group
    overrides, serialized to JSON, and shipped to the worker via the bootstrap.
    """

    docker_image: str = ""
    host: str = "0.0.0.0"
    port: int = 10001
    port_range: str = "30000-40000"
    worker_id: str = ""  # auto-generated if empty
    controller_address: str = ""
    cache_dir: str = "/dev/shm/iris"
    slice_id: str = ""
    default_task_image: str = ""
    task_env: dict[str, str] = Field(default_factory=dict)
    runtime: str = ""  # "docker" or "kubernetes"
    accelerator_type: AcceleratorType | None = None
    accelerator_variant: str = ""
    gpu_count: int = 0
    capacity_type: CapacityTypeField | None = None
    cpu_millicores: int = 0  # advertised scheduling CPU capacity (0 = probe host)
    worker_attributes: dict[str, str] = Field(default_factory=dict)
    poll_interval: DurationField | None = None
    heartbeat_timeout: DurationField | None = None
    platform: PlatformConfig | None = None
    storage_prefix: str = ""  # task-artifact prefix; empty disables profile upload
    auth_token: str = ""  # worker→controller bearer token; empty when auth disabled


# ---------------------------------------------------------------------------
# SSH / storage / controller
# ---------------------------------------------------------------------------


class SshConfig(_Config):
    user: str = "root"
    key_file: str = ""
    port: int | None = None  # default: 22
    connect_timeout: DurationField | None = None  # default: 30s
    impersonate_service_account: str = ""


class StorageConfig(_Config):
    local_state_dir: str = ""  # controller DB/logs/bundle cache; survives restarts
    remote_state_dir: str = ""  # remote URI for checkpoints + worker profiles


class GcpControllerConfig(_Config):
    zone: str = ""
    machine_type: str = ""  # default: "n2-standard-4"
    boot_disk_size_gb: int = 0  # default: 50
    port: int = 0  # default: 10000
    service_account: str = ""


class ManualControllerConfig(_Config):
    host: str = ""
    port: int = 0  # default: 10000


class LocalControllerConfig(_Config):
    port: int = 0  # 0 = auto-assign


class CoreweaveControllerConfig(_Config):
    port: int = 0  # default: 10000
    service_name: str = ""  # K8s Service name for discovery
    scale_group: str = ""  # scale group whose NodePool runs the controller
    # When set, start_controller creates an Ingress publishing ONLY /proxy
    # off-cluster; the controller's per-endpoint auth is the sole gate, so keep
    # auth.provider set. See docs/coreweave.md.
    public_proxy_host: str = ""
    ingress_class: str = "traefik"  # ingressClassName for the /proxy ingress
    tls_secret: str = ""  # secret holding the TLS cert for public_proxy_host
    # cert-manager ClusterIssuer. When set, the Ingress is annotated
    # cert-manager.io/cluster-issuer=<name> so cert-manager auto-issues the cert
    # into tls_secret. Empty = bring your own cert in tls_secret (or no TLS).
    cluster_issuer: str = ""


class ControllerVmConfig(_OneofConfig):
    _ONEOF_ARMS = ("gcp", "manual", "local", "coreweave")
    image: str = ""  # controller docker image (shared by all controller types)
    # Periodic checkpoint interval (seconds); 0 = controller default (hourly).
    checkpoint_interval_seconds: float = 0
    gcp: GcpControllerConfig | None = None
    manual: ManualControllerConfig | None = None
    local: LocalControllerConfig | None = None
    coreweave: CoreweaveControllerConfig | None = None

    def controller_kind(self) -> str | None:
        return self._selected_arm()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class AutoscalerConfig(_Config):
    evaluation_interval: DurationField = Field(default_factory=lambda: Duration.from_seconds(10))
    scale_up_delay: DurationField = Field(default_factory=lambda: Duration.from_seconds(60))
    scale_down_delay: DurationField = Field(default_factory=lambda: Duration.from_seconds(600))


class DefaultsConfig(_Config):
    ssh: SshConfig = Field(default_factory=SshConfig)
    autoscaler: AutoscalerConfig = Field(default_factory=AutoscalerConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    task_env: dict[str, str] = Field(default_factory=dict)  # injected into every task
    inject_env: list[str] = Field(default_factory=list)  # operator-shell var names

    @model_validator(mode="after")
    def _propagate_task_env(self) -> DefaultsConfig:
        # Workers receive the cluster task_env via the wire WorkerConfig.
        for key, value in self.task_env.items():
            self.worker.task_env.setdefault(key, value)
        return self


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class GcpAuthConfig(_Config):
    project_id: str = ""


class StaticAuthConfig(_Config):
    tokens: dict[str, str] = Field(default_factory=dict)  # token -> username


class IapAuthConfig(_Config):
    url: str = ""
    oauth_client_id: str = ""
    oauth_client_secret: str = ""
    # OIDC ID-token audiences the controller accepts on interactive-login tokens
    # (the `iris login` user flow); typically just the desktop client id.
    audiences: list[str] = Field(default_factory=list)
    # Audiences a service-account (CI / in-cluster) caller mints its IAP *edge*
    # token for -- kept separate from the login `audiences` above. Empty falls
    # back to the desktop client id, which IAP registers as a programmatic client
    # (sufficient for the common single-client setup); set this only to give
    # machine callers an `aud` distinct from the interactive-login client.
    programmatic_audiences: list[str] = Field(default_factory=list)
    signed_header_audience: str = ""
    # Role granted to an IAP-verified email with no row in the user store; a
    # provisioned user always resolves to their stored role. "admin" makes
    # IAP's own allowlist the sole gate.
    unprovisioned_role: Literal["admin", "user", "dashboard"] = "dashboard"


class AuthConfig(_OneofConfig):
    _ONEOF_ARMS = ("gcp", "static", "iap")
    gcp: GcpAuthConfig | None = None
    static: StaticAuthConfig | None = None
    iap: IapAuthConfig | None = None
    # Network-location trust, orthogonal to the login-provider arm: a tokenless
    # request whose *direct transport peer* is inside one of these CIDRs
    # authenticates as the anonymous admin identity (like loopback). Requests
    # forwarded through a proxy hop (X-Forwarded-For present) never match, so
    # an in-network ingress cannot lend its address to external traffic. With
    # no arm selected, a non-empty list alone enables auth (provider "cidr").
    trusted_cidrs: list[str] = Field(default_factory=list)
    admin_users: list[str] = Field(default_factory=list)
    # Authenticate-but-not-require: valid tokens get their identity; tokenless
    # requests fall through as anonymous admin; invalid tokens still rejected.
    optional: bool = False

    @field_validator("trusted_cidrs")
    @classmethod
    def _parse_cidrs(cls, cidrs: list[str]) -> list[str]:
        for cidr in cidrs:
            ipaddress.ip_network(cidr)  # ValueError on a malformed CIDR — fail at load
        return cidrs

    def provider_kind(self) -> str | None:
        return self._selected_arm()


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class WorkerProviderConfig(_Config):
    pass


class KueueTopology(_Config):
    node_label: str = ""
    required: bool = False  # True => required-topology (hard); False => preferred


class KueueConfig(_Config):
    cluster_queue: str = ""  # setting this ENABLES Kueue gang admission
    priority_classes: dict[str, str] = Field(default_factory=dict)  # band -> class
    topologies: dict[str, KueueTopology] = Field(default_factory=dict)  # group_by -> topo


class KubernetesProviderConfig(_Config):
    namespace: str = ""  # default: "iris"
    kubeconfig: str = ""  # empty = in-cluster auth
    default_image: str = ""
    service_account: str = ""
    host_network: bool = False
    cache_dir: str = ""  # hostPath base for cache mounts (default: "/cache")
    controller_address: str = ""  # injected into task pods
    kueue: KueueConfig = Field(default_factory=KueueConfig)
    preempt_namespaces: list[str] = Field(default_factory=list)
    priority_classes: dict[str, str] = Field(default_factory=dict)  # band -> PriorityClass


# ---------------------------------------------------------------------------
# Budgets / endpoints
# ---------------------------------------------------------------------------


class UserBudgetTier(_Config):
    user_ids: list[str] = Field(default_factory=list)
    budget_limit: int = 0  # resource-value spend before downgrade to BATCH (0 = unlimited)
    max_band: PriorityBandField = 0  # highest band these users may submit to


class EndpointSpec(_Config):
    uri: str = ""  # http(s)://host:port, gcp://<service>, k8s://<service>[.<ns>]
    metadata: dict[str, str] = Field(default_factory=dict)  # resolver hints


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class AllowPolicy(_Config):
    """Which users may route tasks to a backend. ``"*"`` matches all users."""

    users: list[str] = Field(default_factory=lambda: ["*"])


class BackendConfig(_Config):
    """One task backend in a multi-backend cluster.

    ``kind`` selects the provider arm: ``worker_daemon`` drives the worker-daemon
    provider (``worker_provider``), ``k8s`` drives the Kubernetes provider
    (``kubernetes_provider``). ``transport`` must be ``in_process``; ``remote`` is
    rejected by :func:`validate_config`.

    ``attributes`` values are comma-split into sets by
    :func:`backend_attribute_sets` for the task→backend meta-scheduler (for
    example ``device-variant: "v5e-4,v5p-8"``).
    """

    kind: Literal["worker_daemon", "k8s"]
    transport: Literal["in_process", "remote"] = "in_process"
    attributes: dict[str, str] = Field(default_factory=dict)
    allow_policy: AllowPolicy = Field(default_factory=AllowPolicy)
    worker_provider: WorkerProviderConfig | None = None
    kubernetes_provider: KubernetesProviderConfig | None = None
    scale_groups: dict[str, ScaleGroupConfig] = Field(default_factory=dict)
    platform: PlatformConfig | None = None


# ---------------------------------------------------------------------------
# Federation peers
# ---------------------------------------------------------------------------


class PeerConfig(_Config):
    """One federation peer: a remote Iris controller this cluster may delegate to.

    A peer is *not* a backend. A backend is a local execution substrate this
    controller drives and folds into its own job DAG; a peer owns its own DAG and
    is handed whole jobs. A peer declares identity, reachability, and trust — never
    capabilities, which the peer advertises live at runtime (the capability
    heartbeat), so a peer that loses a pool stops advertising it without a config
    edit here.

    ``cluster`` names the peer's cluster manifest, from which the client
    credentials this controller presents to the peer are resolved (the same
    ``credentials_for`` path every cross-cluster client uses). Empty ``cluster``
    means loopback / no-auth — a local or same-VPC peer that trusts the connection.
    """

    controller_address: str  # peer controller RPC address (reachability)
    dashboard_url: str = ""  # peer public dashboard origin, for deep links
    cluster: str = ""  # peer cluster manifest name; resolves the presented credentials
    static_token: str = ""  # client token for a peer whose manifest uses static auth


class ClusterFinelogConfig(_Config):
    """This cluster's finelog: the local log store, and optionally a relay to a shared one.

    ``config`` names the finelog deployment that serves this cluster (iris derives
    ``/system/log-server`` from it). Set ``relay_address`` to also run a durable relay
    that forwards this cluster's logs to a shared global finelog; leave it empty and the
    log plane stays single-cluster (no relay, local reads, byte-identical behavior).

    The relay authenticates its pushes with a delegation credential the global finelog
    verifies on ingress:

    - ``delegation_key`` — a shared HS256 secret. When set, the relay mints a short-lived
      JWT signed with it, and the global finelog carries a matching ``jwt`` layer keyed on
      ``{cluster: <this cluster's name>, secret: <delegation_key>}``. Dedicated (not the
      control-plane signing key), so the shared store can *verify* this cluster's tokens
      without gaining the power to *mint* control-plane tokens. HS256-only: an IAP/GCP
      token would not verify.
    - ``static_token`` — a pre-minted bearer used verbatim, for a simple/local deployment.
      Lower-preference than ``delegation_key``.
    - Neither — no bearer; the global finelog must admit this controller by a ``cidr``
      layer (a same-VPC/loopback dev store).
    """

    config: str = ""  # finelog deploy-config name; iris derives /system/log-server from it
    relay_address: str = ""  # shared global finelog to forward logs to (empty = single-cluster)
    delegation_key: str = ""  # shared HS256 secret; the relay mints short-lived JWTs with it
    static_token: str = ""  # pre-minted static bearer (alternative to delegation_key)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class IrisClusterConfig(_OneofConfig):
    """Full cluster configuration: platform, defaults, storage, controller, scale groups, auth."""

    _ONEOF_ARMS = ("worker_provider", "kubernetes_provider")
    name: str = ""
    # Always present (empty unless configured); the selected sub-platform is an
    # at-most-one arm. validate_config requires exactly one arm for a real cluster.
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    controller: ControllerVmConfig = Field(default_factory=ControllerVmConfig)
    scale_groups: dict[str, ScaleGroupConfig] = Field(default_factory=dict)
    auth: AuthConfig | None = None
    kubernetes_provider: KubernetesProviderConfig | None = None
    worker_provider: WorkerProviderConfig | None = None
    # Explicit multi-backend map. Absent (None) = the implicit single-backend form,
    # synthesized from the top-level platform/scale_groups/provider fields by
    # resolve_backends. Mixing the two is rejected by validate_config.
    backends: dict[str, BackendConfig] | None = None
    user_budgets: list[UserBudgetTier] = Field(default_factory=list)
    endpoints: dict[str, EndpointSpec] = Field(default_factory=dict)
    # Federation peers (peer id -> declaration): remote Iris controllers this
    # cluster may delegate whole jobs to. Absent/empty leaves federation inert —
    # a single-cluster deployment is unchanged.
    peers: dict[str, PeerConfig] = Field(default_factory=dict)
    # This cluster's finelog: the local log store (`finelog.config` names the deploy
    # config iris derives /system/log-server from) and an optional `finelog.relay_address`
    # for a shared global finelog. Setting `finelog.relay_address` turns on the forwarder.
    finelog: ClusterFinelogConfig = Field(default_factory=ClusterFinelogConfig)
    # Public dashboard origin (e.g. "https://iris.oa.dev"); enables clickable job URLs.
    dashboard_url: str = ""

    def provider_kind(self) -> str | None:
        return self._selected_arm()

    def controller_address(self) -> str:
        """Controller address from defaults.worker, or empty if unset."""
        return self.defaults.worker.controller_address


# ===========================================================================
# Validation
# ===========================================================================


def _validate_provider_platform_compat(config: IrisClusterConfig) -> None:
    is_coreweave = config.platform.platform_kind() == "coreweave"
    if is_coreweave and config.provider_kind() == "worker_provider":
        raise ValueError(
            "CoreWeave platform does not support worker_provider (CoreweavePlatform no longer "
            "manages slices). Use kubernetes_provider instead."
        )
    if config.provider_kind() == "kubernetes_provider" and not config.kubernetes_provider.controller_address:
        raise ValueError(
            "kubernetes_provider.controller_address is required. Task pods need it to fetch "
            "bundles from the controller. Set it to the in-cluster service URL, e.g. "
            "http://iris-controller-svc.<namespace>.svc.cluster.local:<port>"
        )


def _validate_accelerator_types(config: IrisClusterConfig) -> None:
    for name, sg in config.scale_groups.items():
        if sg.resources is None:
            continue
        if sg.resources.device_type is None:
            raise ValueError(f"Scale group '{name}' must set resources.device_type to cpu, gpu, or tpu.")


def validate_scale_group_resources(scale_groups: dict[str, ScaleGroupConfig]) -> None:
    """Validate that scale groups define per-VM resources and num_vms."""
    for name, sg in scale_groups.items():
        if sg.resources is None:
            raise ValueError(f"Scale group '{name}' must set resources.")
        if sg.num_vms is None:
            raise ValueError(f"Scale group '{name}' must set num_vms.")
        if sg.num_vms <= 0:
            raise ValueError(f"Scale group '{name}' has invalid num_vms={sg.num_vms}.")

        res = sg.resources
        if res.cpu_millicores < 0:
            raise ValueError(f"Scale group '{name}' has invalid cpu_millicores={res.cpu_millicores}.")
        if res.memory_bytes < 0:
            raise ValueError(f"Scale group '{name}' has invalid memory_bytes={res.memory_bytes}.")
        if res.disk_bytes < 0:
            raise ValueError(f"Scale group '{name}' has invalid disk_bytes={res.disk_bytes}.")
        if res.device_count < 0:
            raise ValueError(f"Scale group '{name}' has invalid device_count={res.device_count}.")
        if res.capacity_type is None:
            raise ValueError(
                f"Scale group '{name}': resources.capacity_type is required "
                "(one of: preemptible, on-demand, reserved)."
            )


def _validate_slice_templates(config: IrisClusterConfig) -> None:
    for name, sg in config.scale_groups.items():
        if sg.slice_template is None:
            raise ValueError(f"Scale group '{name}': slice_template is required.")

        tmpl = sg.slice_template
        platform = tmpl.platform_kind()
        if platform is None:
            raise ValueError(
                f"Scale group '{name}': slice_template must have a platform (gcp, manual, coreweave, local)."
            )

        if platform == "gcp":
            if not tmpl.gcp.zone:
                raise ValueError(f"Scale group '{name}': slice_template.gcp.zone must be non-empty.")
            res = sg.resources
            if tmpl.gcp.mode == GcpSliceMode.VM:
                if res.capacity_type != CapacityType.ON_DEMAND:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices only support capacity_type on-demand.")
                if sg.num_vms != 1:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices require num_vms=1.")
                if res.device_type != AcceleratorType.CPU:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices currently require device_type=cpu.")
                if res.device_variant:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices do not support device_variant.")
                if not tmpl.gcp.machine_type:
                    raise ValueError(
                        f"Scale group '{name}': slice_template.gcp.machine_type must be non-empty for VM mode."
                    )
                if res.device_count > 0:
                    raise ValueError(f"Scale group '{name}': VM-backed GCP slices currently support CPU-only resources.")
            elif not tmpl.gcp.runtime_version:
                raise ValueError(f"Scale group '{name}': slice_template.gcp.runtime_version must be non-empty.")
        elif platform == "manual":
            if not tmpl.manual.hosts:
                raise ValueError(f"Scale group '{name}': slice_template.manual.hosts must be non-empty.")
        elif platform == "coreweave":
            if not tmpl.coreweave.region:
                raise ValueError(f"Scale group '{name}': slice_template.coreweave.region must be non-empty.")

    if config.platform.gcp is not None and config.platform.gcp.zones:
        platform_zones = set(config.platform.gcp.zones)
        for name, sg in config.scale_groups.items():
            tmpl = sg.slice_template
            if tmpl is not None and tmpl.platform_kind() == "gcp" and tmpl.gcp.zone:
                if tmpl.gcp.zone not in platform_zones:
                    raise ValueError(
                        f"Scale group '{name}': zone '{tmpl.gcp.zone}' is not in "
                        f"platform.gcp.zones {sorted(platform_zones)}. Add it to platform.gcp.zones."
                    )


_WELL_KNOWN_RESOURCE_ATTRS = frozenset(
    {
        WellKnownAttribute.PREEMPTIBLE,
        WellKnownAttribute.DEVICE_TYPE,
        WellKnownAttribute.DEVICE_VARIANT,
        WellKnownAttribute.REGION,
        WellKnownAttribute.ZONE,
    }
)


def _validate_worker_settings(config: IrisClusterConfig) -> None:
    """Reject worker.attributes that are auto-derived from resources/slice_template."""
    for name, sg in config.scale_groups.items():
        attributes = sg.worker.attributes

        for attr_key in _WELL_KNOWN_RESOURCE_ATTRS:
            if attr_key in attributes:
                raise ValueError(
                    f"Scale group '{name}': worker.attributes.{attr_key} is derived automatically "
                    f"(from resources or slice_template) and must not be set explicitly. "
                    f"Remove it from worker.attributes."
                )

        tmpl = sg.slice_template
        if (
            tmpl is not None
            and tmpl.coreweave is not None
            and sg.resources is not None
            and sg.resources.device_type == AcceleratorType.GPU
            and (sg.num_vms or 0) > 1
        ):
            topology_attrs = {
                key: value
                for key, value in attributes.items()
                if any(key.startswith(prefix) for prefix in _COREWEAVE_TOPOLOGY_LABEL_PREFIXES)
            }
            if not topology_attrs:
                raise ValueError(
                    f"Scale group '{name}': CoreWeave GPU groups with num_vms>1 must set at least one "
                    "topology label in worker.attributes (for example "
                    "'backend.coreweave.cloud/superpod: same-slice')."
                )


def _validate_worker_defaults(config: IrisClusterConfig) -> None:
    """Require a worker image for worker-based (non-local) platforms."""
    platform_kind = config.platform.platform_kind()
    if platform_kind in (None, "local"):
        return

    docker_image = config.defaults.worker.docker_image.strip()
    if not docker_image:
        raise ValueError("defaults.worker.docker_image is required for non-local platforms (gcp/manual/coreweave).")

    runtime = config.defaults.worker.runtime.strip()
    if runtime and runtime not in ("docker", "kubernetes"):
        raise ValueError(f"defaults.worker.runtime must be 'docker' or 'kubernetes', got {runtime!r}.")


def _validate_gcp_service_accounts(config: IrisClusterConfig) -> None:
    """Require explicit service accounts on every GCP VM (OS Login needs them)."""
    if config.controller.controller_kind() == "gcp" and not config.controller.gcp.service_account:
        raise ValueError("controller.gcp.service_account is required for GCP controllers.")

    for name, sg in config.scale_groups.items():
        tmpl = sg.slice_template
        if tmpl is None or tmpl.platform_kind() != "gcp":
            continue
        if not tmpl.gcp.service_account:
            raise ValueError(f"Scale group '{name}': slice_template.gcp.service_account is required.")


def validate_autoscaler_config(config: AutoscalerConfig, context: str = "autoscaler") -> None:
    """Validate autoscaler timing values (defaults already resolved)."""
    interval_ms = config.evaluation_interval.to_ms()
    if interval_ms <= 0:
        raise ValueError(
            f"{context}: evaluation_interval must be positive, got {interval_ms}ms. "
            f"This controls how often the autoscaler evaluates scaling decisions."
        )
    if config.scale_up_delay.to_ms() < 0:
        raise ValueError(
            f"{context}: scale_up_delay must be non-negative, got {config.scale_up_delay.to_ms()}ms. "
            f"Use 0 for no cooldown after scaling up."
        )
    if config.scale_down_delay.to_ms() < 0:
        raise ValueError(
            f"{context}: scale_down_delay must be non-negative, got {config.scale_down_delay.to_ms()}ms. "
            f"Use 0 for no cooldown after scaling down."
        )


def _validate_backends(config: IrisClusterConfig) -> None:
    """Validate an explicit ``backends:`` map.

    The implicit single-backend form (no ``backends:``) is covered by the
    top-level validators and synthesized by :func:`resolve_backends`.
    """
    if config.backends is None:
        return

    conflicting = []
    if config.scale_groups:
        conflicting.append("scale_groups")
    if config.worker_provider is not None:
        conflicting.append("worker_provider")
    if config.kubernetes_provider is not None:
        conflicting.append("kubernetes_provider")
    if config.platform.platform_kind() is not None:
        conflicting.append("platform")
    if conflicting:
        raise ValueError(
            f"backends: cannot be combined with top-level {', '.join(conflicting)}. "
            "The top-level platform/scale_groups/provider fields are the implicit single-backend "
            "form; move them under a backends: entry instead."
        )

    for backend_id, backend in config.backends.items():
        if backend.transport == "remote":
            raise ValueError(
                f"backend '{backend_id}': transport 'remote' is not supported yet — "
                "remote transport lands in a later PR."
            )
        if backend.kind == "worker_daemon":
            if backend.worker_provider is None:
                raise ValueError(f"backend '{backend_id}': kind 'worker_daemon' requires worker_provider.")
            if backend.kubernetes_provider is not None:
                raise ValueError(f"backend '{backend_id}': kind 'worker_daemon' must not set kubernetes_provider.")
        elif backend.kind == "k8s":
            if backend.kubernetes_provider is None:
                raise ValueError(f"backend '{backend_id}': kind 'k8s' requires kubernetes_provider.")
            if backend.worker_provider is not None:
                raise ValueError(f"backend '{backend_id}': kind 'k8s' must not set worker_provider.")


def _validate_peers(config: IrisClusterConfig) -> None:
    """Validate the ``peers:`` federation registry.

    ``'local'`` is reserved as the federation sentinel for "this controller"; the
    cluster's own ``name`` and every peer id must stay disjoint from it so the
    sentinel and the real cluster-id namespace never collide.
    """
    if config.name == LOCAL_CLUSTER:
        raise ValueError(
            f"cluster name may not be {LOCAL_CLUSTER!r}: it is reserved as the federation "
            "sentinel for this controller. Choose a distinct cluster name."
        )
    for peer_id, peer in config.peers.items():
        if not peer_id.strip():
            raise ValueError("peers: peer id must be a non-empty string.")
        if peer_id == LOCAL_CLUSTER:
            raise ValueError(
                f"peer id may not be {LOCAL_CLUSTER!r}: it is reserved as the federation "
                "sentinel. A peer must have a distinct cluster id."
            )
        if not peer.controller_address.strip():
            raise ValueError(f"peer '{peer_id}': controller_address is required.")
        if peer.static_token and not peer.cluster:
            raise ValueError(
                f"peer '{peer_id}': static_token requires cluster to be set (the manifest "
                "whose static auth the token authenticates against)."
            )


def _validate_finelog_relay(config: IrisClusterConfig) -> None:
    """Validate the finelog relay credential (only when ``finelog.relay_address`` is set)."""
    finelog = config.finelog
    if not finelog.relay_address.strip():
        return
    # The finelog HS256 verifier rejects a delegation secret shorter than 16 bytes
    # (MIN_SECRET_BYTES); fail fast here rather than at the store on first push.
    if finelog.delegation_key and len(finelog.delegation_key.encode()) < 16:
        raise ValueError("finelog.delegation_key must be at least 16 bytes.")


def validate_config(config: IrisClusterConfig) -> None:
    """Validate cluster config; raises ValueError on the first violation."""
    _validate_backends(config)
    _validate_peers(config)
    _validate_finelog_relay(config)
    _validate_provider_platform_compat(config)
    _validate_accelerator_types(config)
    validate_scale_group_resources(config.scale_groups)
    _validate_slice_templates(config)
    _validate_worker_settings(config)
    _validate_worker_defaults(config)
    _validate_gcp_service_accounts(config)
    validate_autoscaler_config(config.defaults.autoscaler, context="config.defaults.autoscaler")


# ===========================================================================
# Backend resolution
# ===========================================================================


def backend_attribute_sets(backend: BackendConfig) -> dict[str, set[str]]:
    """Comma-split each ``attributes`` value into a normalized set.

    ``device-variant: "v5e-4, v5p-8"`` becomes ``{"device-variant": {"v5e-4", "v5p-8"}}``.
    Empty and whitespace-only entries are dropped. The task→backend meta-scheduler
    uses this to expand set-valued attributes into posting lists.
    """
    return {key: {part.strip() for part in raw.split(",") if part.strip()} for key, raw in backend.attributes.items()}


def resolve_backends(config: IrisClusterConfig) -> dict[str, BackendConfig]:
    """Resolve the cluster's backends as a ``{backend_id: BackendConfig}`` map.

    An explicit ``backends:`` map is returned as-is. A single-cluster config (no
    ``backends:``) synthesizes exactly one entry keyed :data:`DEFAULT_BACKEND_ID`
    from the top-level platform/scale_groups/provider fields.
    """
    if config.backends is not None:
        return dict(config.backends)

    kind = "k8s" if config.provider_kind() == "kubernetes_provider" else "worker_daemon"
    backend = BackendConfig(
        kind=kind,
        worker_provider=config.worker_provider,
        kubernetes_provider=config.kubernetes_provider,
        scale_groups=dict(config.scale_groups),
        platform=config.platform,
    )
    return {DEFAULT_BACKEND_ID: backend}


# ===========================================================================
# Loading / preprocessing
# ===========================================================================


def _parse_memory_value(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an int or size string (got bool)")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        return int(parse_memory_string(value))
    raise ValueError(f"{field_name} must be an int or size string (got {type(value).__name__})")


def _validate_zone_list(zones: object, context: str) -> list[str]:
    if not isinstance(zones, list) or not zones:
        raise ValueError(f"{context}: zones must be a non-empty list")
    for zone in zones:
        if not isinstance(zone, str) or not zone.strip():
            raise ValueError(f"{context}: each zone must be a non-empty string, got {zone!r}")
    if len(zones) != len(set(zones)):
        raise ValueError(f"{context}: zones list contains duplicates: {zones}")
    return zones


def _merge_zones_into_platform_gcp(data: dict, zones: set[str]) -> None:
    if not zones:
        return
    platform = data.setdefault("platform", {})
    platform_gcp = platform.get("gcp")
    if isinstance(platform_gcp, dict):
        existing = set(platform_gcp.get("zones", []))
        existing.update(zones)
        platform_gcp["zones"] = sorted(existing)


def _expand_tpu_pools(data: dict) -> None:
    """Expand ``tpu_pools`` into per-(size, zone) scale groups.

    Each pool defines shared properties for a TPU family. The ``sizes`` map
    lists per-size overrides; for each pool x size x zone this emits a fully
    specified scale group with topology-derived fields and allocation metadata.
    """
    tpu_pools = data.pop("tpu_pools", None)
    if not tpu_pools:
        return
    if not isinstance(tpu_pools, dict):
        raise ValueError("tpu_pools must be a mapping")

    scale_groups = data.setdefault("scale_groups", {})
    all_zones: set[str] = set()

    for pool_name, pool in tpu_pools.items():
        if not isinstance(pool, dict):
            raise ValueError(f"tpu_pools.{pool_name} must be a mapping")

        family = pool.get("family")
        if not family or family not in TPU_FAMILY_VARIANT_PREFIX:
            raise ValueError(
                f"tpu_pools.{pool_name}: 'family' must be one of {sorted(TPU_FAMILY_VARIANT_PREFIX)}, got {family!r}"
            )

        zones = _validate_zone_list(pool.get("zones"), f"tpu_pools.{pool_name}")
        all_zones.update(zones)

        sizes = pool.get("sizes")
        if not isinstance(sizes, dict) or not sizes:
            raise ValueError(f"tpu_pools.{pool_name}: 'sizes' must be a non-empty mapping")

        base_priority = pool.get("base_priority", 10)
        base_resources = pool.get("resources", {})
        base_slice_template = pool.get("slice_template", {})

        sorted_sizes = sorted(sizes.keys(), key=lambda s: int(s))
        for size in sorted_sizes:
            variant = tpu_variant_name(family, int(size))
            try:
                get_tpu_topology(variant)
            except ValueError:
                raise ValueError(f"tpu_pools.{pool_name}.sizes.{size}: unknown TPU topology '{variant}'") from None

        for tier_index, size in enumerate(sorted_sizes):
            size_int = int(size)
            size_overrides = sizes[size] or {}
            variant = tpu_variant_name(family, size_int)
            topo = get_tpu_topology(variant)

            for zone in zones:
                sg_name = f"tpu_{pool_name}_{size_int}-{zone}"
                if sg_name in scale_groups:
                    raise ValueError(
                        f"tpu_pools.{pool_name}: expanded name '{sg_name}' collides with an existing scale group"
                    )

                resources = copy.deepcopy(base_resources)
                resources["device_type"] = "tpu"
                resources["device_variant"] = variant
                resources["device_count"] = topo.chips_per_vm

                st = copy.deepcopy(base_slice_template)
                gcp = st.setdefault("gcp", {})
                gcp["zone"] = zone

                scale_groups[sg_name] = {
                    "name": sg_name,
                    "num_vms": topo.vm_count,
                    "priority": size_overrides.get("priority", base_priority + tier_index * 10),
                    "quota_pool": f"{pool_name}/{zone}",
                    "allocation_tier": tier_index + 1,
                    "resources": resources,
                    "buffer_slices": size_overrides.get("buffer_slices", 0),
                    "max_slices": size_overrides["max_slices"],
                    "slice_template": st,
                }

    _merge_zones_into_platform_gcp(data, all_zones)


def _expand_multi_zone_groups(data: dict) -> None:
    """Expand scale groups with a YAML-only ``zones`` key into one group per zone."""
    scale_groups = data.get("scale_groups")
    if not isinstance(scale_groups, dict):
        return

    all_expanded_zones: set[str] = set()
    expanded: dict[str, dict] = {}
    to_remove: list[str] = []

    for name, sg in list(scale_groups.items()):
        if not isinstance(sg, dict) or "zones" not in sg:
            continue

        zones = _validate_zone_list(sg.pop("zones"), f"Scale group '{name}'")
        to_remove.append(name)

        st = sg.get("slice_template") or {}
        non_gcp_platforms = {"manual", "local", "coreweave"}
        specified_platforms = non_gcp_platforms & st.keys()
        if specified_platforms:
            raise ValueError(
                f"Scale group '{name}': 'zones' expansion is only supported for GCP slice templates, "
                f"but slice_template specifies {', '.join(sorted(specified_platforms))}."
            )

        existing_gcp_zone = (sg.get("slice_template") or {}).get("gcp", {}).get("zone")
        if existing_gcp_zone:
            raise ValueError(
                f"Scale group '{name}': cannot set both 'zones' and 'slice_template.gcp.zone'. "
                f"Remove slice_template.gcp.zone — it is set automatically by zone expansion."
            )

        for zone in zones:
            expanded_name = f"{name}-{zone}"
            if expanded_name in scale_groups:
                raise ValueError(
                    f"Scale group '{name}': expanded name '{expanded_name}' collides with an existing scale group."
                )
            if expanded_name in expanded:
                raise ValueError(
                    f"Scale group '{name}': expanded name '{expanded_name}' collides with another expanded group."
                )

            expanded_sg = copy.deepcopy(sg)
            expanded_sg["name"] = expanded_name
            st = expanded_sg.setdefault("slice_template", {})
            gcp = st.setdefault("gcp", {})
            gcp["zone"] = zone
            if "buffer_slices" not in expanded_sg:
                expanded_sg["buffer_slices"] = 0

            expanded[expanded_name] = expanded_sg
            all_expanded_zones.add(zone)

    for name in to_remove:
        del scale_groups[name]
    scale_groups.update(expanded)

    _merge_zones_into_platform_gcp(data, all_expanded_zones)


def _inject_scale_group_names_into(scale_groups: object) -> None:
    if not isinstance(scale_groups, dict):
        return
    for name, sg in scale_groups.items():
        if sg is None:
            scale_groups[name] = {"name": name}
        elif isinstance(sg, dict) and "name" not in sg:
            sg["name"] = name


def _inject_scale_group_names(data: dict) -> None:
    """Stamp each scale group's map key onto its ``name`` field.

    Applies to the top-level ``scale_groups`` and to every backend's
    ``scale_groups`` so a backend's workers register under the same scale-group
    name the backend's routing is keyed by.
    """
    _inject_scale_group_names_into(data.get("scale_groups"))
    backends = data.get("backends")
    if isinstance(backends, dict):
        for backend in backends.values():
            if isinstance(backend, dict):
                _inject_scale_group_names_into(backend.get("scale_groups"))


def parse_config(data: dict) -> IrisClusterConfig:
    """Parse and validate a raw config dict (post-expansion) into the model."""
    config = IrisClusterConfig.model_validate(data)
    validate_config(config)
    return config


def load_config(config_path: Path | str) -> IrisClusterConfig:
    """Load, expand, and validate a cluster config from a YAML file."""
    config_path = Path(config_path)
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    # Expand $VARS in controller_address only — it often needs
    # $IRIS_CONTROLLER_ADDRESS for dynamic discovery. Other fields are literal.
    worker_defaults = data.get("defaults", {}).get("worker") if isinstance(data.get("defaults"), dict) else None
    if isinstance(worker_defaults, dict) and "controller_address" in worker_defaults:
        worker_defaults["controller_address"] = os.path.expandvars(worker_defaults["controller_address"])

    _expand_tpu_pools(data)
    _expand_multi_zone_groups(data)
    _inject_scale_group_names(data)

    config = parse_config(data)

    platform_kind = config.platform.platform_kind() or "unspecified"
    logger.debug(
        "Config loaded: platform=%s, scale_groups=%s",
        platform_kind,
        list(config.scale_groups.keys()) or "(none)",
    )
    return config


# ===========================================================================
# Serialization / transformation
# ===========================================================================


def config_to_dict(config: IrisClusterConfig) -> dict:
    """Serialize config to a JSON-friendly dict (for ConfigMap / VM bootstrap).

    The output round-trips through :func:`load_config` / :func:`parse_config`.
    """
    return config.model_dump(mode="json", exclude_none=True)


def make_local_config(base_config: IrisClusterConfig) -> IrisClusterConfig:
    """Transform any config to run locally (in-process controller + workers)."""
    config = base_config.model_copy(deep=True)
    config.platform = PlatformConfig(label_prefix=config.platform.label_prefix)
    config.platform.local = LocalPlatformConfig()
    config.controller = ControllerVmConfig(image=config.controller.image, local=LocalControllerConfig(port=0))
    config.storage.remote_state_dir = ""  # LocalCluster sets a temp path
    # Fast timings for local dev (override any production timings).
    config.defaults.autoscaler = AutoscalerConfig(
        evaluation_interval=Duration.from_seconds(0.5),
        scale_up_delay=Duration.from_seconds(1),
        scale_down_delay=Duration.from_seconds(1),
    )
    return config


def build_ssh_command_config(config: IrisClusterConfig, group_name: str | None = None) -> SshConfig:
    """Resolve SSH config: cluster defaults merged with per-group manual overrides."""
    ssh = config.defaults.ssh
    user = ssh.user or "root"
    key_file = ssh.key_file or ""
    port = ssh.port if ssh.port and ssh.port > 0 else DEFAULT_SSH_PORT
    impersonate = ssh.impersonate_service_account or ""
    connect_timeout = ssh.connect_timeout if ssh.connect_timeout is not None else DEFAULT_SSH_CONNECT_TIMEOUT

    if group_name and group_name in config.scale_groups:
        group = config.scale_groups[group_name]
        if group.slice_template is not None and group.slice_template.manual is not None:
            manual = group.slice_template.manual
            if manual.ssh_user:
                user = manual.ssh_user
            if manual.ssh_key_file:
                key_file = manual.ssh_key_file

    return SshConfig(
        user=user,
        key_file=key_file,
        port=port,
        impersonate_service_account=impersonate,
        connect_timeout=connect_timeout,
    )


def clear_remote_state(remote_state_dir: str) -> None:
    """Remove all files under the remote state dir so the controller starts fresh."""
    fs, path = fsspec.core.url_to_fs(remote_state_dir)
    if fs.exists(path):
        fs.rm(path, recursive=True)
