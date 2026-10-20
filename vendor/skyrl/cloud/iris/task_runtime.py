#!/usr/bin/env python3
"""Supervise one node of a multi-node MarinSkyRL Iris task.

iris gang-schedules a multi-node job as N coscheduled tasks (one per node) and
runs THIS SAME entrypoint on every node, injecting ``IRIS_TASK_ID`` /
``IRIS_NUM_TASKS``. This script bootstraps one cross-node Ray cluster, then
runs the driver on rank 0:

- **rank 0 (head):** ``ray start --head``; publish the head IP to a shared
  rendezvous file; wait until ``ray.nodes()`` shows all ``IRIS_NUM_TASKS``
  nodes joined; then ``exec`` the MarinSkyRL training command with
  ``RAY_ADDRESS`` pointing at the head so skyrl-train's bare ``ray.init()``
  attaches to the existing cluster.
- **ranks 1..N-1 (workers):** read the head IP from the rendezvous, run
  ``ray start --address=<head_ip>:<port>``, then BLOCK until the current head
  attempt publishes success or the task is terminated. Only the success result
  exits 0. They contribute their 8 H100s to the Ray cluster; they do NOT run the
  training driver.

Head-IP discovery: rank 0 uses iris's ``IRIS_ADVERTISE_HOST`` (routable IP
under ``host_network: true``) and publishes it to a rendezvous file on a shared
object store (``--rendezvous-dir``). The URI may be ``gs://``, ``s3://``, or a
local path — opened via ``fsspec``.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
import glob
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf

from cloud.iris.artifacts import ArtifactSource, atomic_directory_update, file_inventory, fs_and_path, materialize
from cloud.iris.hf_model_cache import (
    download_hugging_face_snapshot,
    ensure_hugging_face_model_cache,
    load_model_manifest,
    stage_artifact_model,
    stage_artifact_model_metadata,
    stage_model_metadata,
)
from marinskyrl.environment_contract import (
    DEBUG_ARTIFACT_DIR_ENV,
    DebugMode,
    EnvVarManager,
    EnvVarScope,
    FR_DUMP_TEMP_FILE_ENV,
    NCCL_DEBUG_INFO_TEMP_FILE_ENV,
    RUN_ID_ENV,
    TELEMETRY_ENDPOINT_ENV,
    TRAINING_TYPE_ENV,
    TrainingType,
    ensure_debug_artifact_directories,
    ray_cluster_owner_environment,
)
from marinskyrl.hf_model import immutable_model_cache_key, validate_hf_model_weights
from marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME
from cloud.iris.telemetry_env import telemetry_environment
from marinskyrl.resource_locator import RolloutModelLoading, is_cloud_uri, is_hugging_face_repo_id, join_resource_path
from marinskyrl.speculative_decoding import SpeculatorModelConfig, SpeculatorModelSourceKind
from marinskyrl.distillation import TeacherModelSpec, TeacherSource, compile_distillation_plan_from_config
from marinskyrl.process_diagnostics import (
    ProcessOutcomeKind,
    initialize_process_diagnostics,
    write_process_outcome,
)
from cloud.iris.paths import resolve_repo_path
from cloud.iris.ray_ports import RayPort, allocated_ray_port, resolve_ray_ports
from cloud.iris.ray_storage import (
    RaySpillBackend,
    RaySpillTarget,
    resolve_ray_spill_target,
    validate_ray_spill_dir,
)
from cloud.iris.launch_config import RunMode, load_launch_config
from cloud.iris.rl_config_translation import materialize_launch_config
from cloud.iris.rl_config_translation import TaskLocalSkyRLValues, apply_task_local_values
from cloud.iris.runtime_bundle import validate_bundled_runtime
from cloud.iris.runtime_environment import RuntimeProfile

try:
    from skyrl_train.ray_metrics import ray_metrics_telemetry
    from skyrl_train.telemetry import CONTROLLER_ROLE, WORKER_ROLE
except ImportError as error:
    # An installed rigging without the telemetry submodule raises ImportError, not
    # ModuleNotFoundError; the name check still keeps a failure inside these packages visible.
    if error.name not in {"prometheus_client", "rigging", "skyrl_train"}:
        raise
    _RAY_METRICS_UNAVAILABLE_REASON = str(error)
    CONTROLLER_ROLE = "controller"
    WORKER_ROLE = "worker"

    def ray_metrics_telemetry(_node_ip: str, _metrics_port: int, _role: str):
        _log(f"Ray metric forwarding is unavailable: {_RAY_METRICS_UNAVAILABLE_REASON}")
        return contextlib.nullcontext()


RENDEZVOUS_FILENAME = "ray_head.json"
DONE_FILENAME = "ray_head.done"
HEAD_RESULT_SUCCEEDED = "succeeded"


@dataclass(frozen=True)
class HeadResult:
    gang_epoch: str
    outcome: str
    written_at: float

    @classmethod
    def from_dict(cls, value: object) -> "HeadResult":
        if not isinstance(value, dict):
            raise RuntimeError("Ray head result must be a JSON object")
        try:
            result = cls(
                gang_epoch=str(value["gang_epoch"]),
                outcome=str(value["outcome"]),
                written_at=float(value["written_at"]),
            )
            if result.outcome != HEAD_RESULT_SUCCEEDED:
                raise ValueError(f"unknown outcome {result.outcome!r}")
            return result
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Ray head result contains invalid field values") from error


@dataclass(frozen=True)
class RendezvousPayload:
    gang_epoch: str
    head_ip: str
    head_node: str
    port: int
    num_tasks: int
    python_version: str
    ray_version: str
    written_at: float

    @classmethod
    def from_dict(cls, value: object) -> "RendezvousPayload":
        if not isinstance(value, dict):
            raise RuntimeError("Ray head rendezvous must be a JSON object")
        required_fields = tuple(field.name for field in fields(cls))
        missing_fields = [field for field in required_fields if value.get(field) in (None, "")]
        if missing_fields:
            raise RuntimeError(f"Ray head rendezvous is missing fields: {', '.join(missing_fields)}")
        try:
            return cls(
                gang_epoch=str(value["gang_epoch"]),
                head_ip=str(value["head_ip"]),
                head_node=str(value["head_node"]),
                port=int(value["port"]),
                num_tasks=int(value["num_tasks"]),
                python_version=str(value["python_version"]),
                ray_version=str(value["ray_version"]),
                written_at=float(value["written_at"]),
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("Ray head rendezvous contains invalid field values") from error


def _ray_bin() -> str:
    """Resolve the ``ray`` CLI from the SAME venv as the running interpreter.

    iris's uv-sync activates /app/.venv on $PATH, which has no ``ray``; use the
    binary next to this interpreter, falling back to PATH.
    """
    import shutil

    candidate = os.path.join(os.path.dirname(sys.executable), "ray")
    if os.path.exists(candidate):
        return candidate
    found = shutil.which("ray")
    return found or "ray"


# Generous: cold GPU nodes (image pull + setup) take minutes to reach rendezvous.
DEFAULT_RENDEZVOUS_TIMEOUT = 1800
DEFAULT_CLUSTER_JOIN_TIMEOUT = 1800
POLL_INTERVAL = 5
# Tolerates clock skew between nodes and the time rank-0 needs to start Ray.
RENDEZVOUS_FRESHNESS_SLACK = 60
# Bound the rank-0 rendezvous PutObject: unbounded fsspec/s3fs put can wedge the
# head forever. Hard timeout per attempt + bounded retries; final failure raises.
RENDEZVOUS_WRITE_ATTEMPTS = 5
RENDEZVOUS_WRITE_TIMEOUT = 30  # seconds per attempt
# Bound `ray start --head` so a hung Ray CLI fails loud (TimeoutExpired).
RAY_START_HEAD_TIMEOUT = 300  # seconds
# Failure reporting must never keep an Iris task alive after its driver exits.
FAILURE_ARTIFACT_TIMEOUT = 30
# Allow healthy policy phases longer than one hour while bounding a live but silent driver.
DEFAULT_DRIVER_LIVENESS_TIMEOUT = 9000
DRIVER_WATCHDOG_POLL_INTERVAL = 1.0
DRIVER_KILL_TIMEOUT = 30
DRIVER_STALLED_EXIT_CODE = 124
IGNORED_DRIVER_OUTPUT_MARKERS = (b"[fd-monitor]",)


def _log(msg: str) -> None:
    print(f"[task-runtime] {msg}", flush=True)


def stage_task_data(data_json: str, *, role: str) -> None:
    """Extract HF task datasets to this node's local task directory on every node.

    The controller runs on every node before Ray bootstrap; pods don't share a
    filesystem, so each pod fetches+extracts the parquet repo to the identical
    node-local path the rollout workers read. Idempotent (``on_exist=skip``).
    """
    try:
        data = json.loads(data_json)
    except (ValueError, TypeError):
        data = [data_json] if data_json else []
    if not data:
        return

    # PYTHONPATH includes /app (launcher bootstrap), so cloud.iris is importable.
    from cloud.iris.rl_data import resolve_rl_train_data

    _log(f"Staging {role} data on this node (rank {_rank()}/{_num_tasks()}): {data}")
    # This HF *parquet* dataset is NOT pre-cached, so force-clear HF_HUB_OFFLINE /
    # TRANSFORMERS_OFFLINE for the extraction only (the ranks' env is untouched).
    # Without this the snapshot_download inside resolve_rl_train_data dies
    # OfflineModeIsEnabled.
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    for k in saved:
        os.environ.pop(k, None)
    try:
        resolved = resolve_rl_train_data(data, on_exist="skip", verbose=True)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    _log(f"{role.capitalize()} data staged to node-local paths: {resolved}")


def stage_model(model_path: str, revision: str | None = None) -> None:
    """Pre-download one model revision into this NODE's local HF cache on EVERY node.

    The controller runs on every node before Ray bootstrap, so pre-download the
    weights ONCE PER NODE here (N pulls, not N*8) into the node-local HF cache.
    The download runs in the controller's pre-Ray phase where no collective
    barrier is active. With ``HF_HUB_OFFLINE=1`` set for the training ranks
    (config extra_env), all ranks on a node read the pre-populated cache.
    Idempotent: ``snapshot_download`` skips already-complete cached files.
    """
    if is_cloud_uri(model_path):
        raise ValueError(f"stage_model accepts a Hugging Face repo ID, not object-store model URI {model_path!r}")
    if not model_path or os.path.isdir(model_path):
        _log(f"stage_model: skip (model_path={model_path!r} is empty or a local directory)")
        return

    # Weights + config + tokenizer + repository-provided chat templates and any
    # trust_remote_code modeling files needed by from_pretrained in offline ranks.
    allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.model", "*.py", "*.jinja"]

    _log(f"Pre-staging model on this node (rank {_rank()}/{_num_tasks()}): {model_path}")
    local_dir = download_hugging_face_snapshot(
        model_path,
        revision=revision,
        allow_patterns=tuple(allow_patterns),
    )
    _log(f"model pre-staged to node-local HF cache: {local_dir}")


def _metadata_path(source_uri: str, identity: str) -> str:
    return os.path.join(
        tempfile.gettempdir(), "marinskyrl", "model_metadata", immutable_model_cache_key(source_uri, identity)
    )


@dataclass(frozen=True)
class PreparedPolicyModel:
    source_uri: str
    source_identity: str
    local_path: str


@dataclass(frozen=True)
class PreparedPolicyTokenizer:
    local_path: str


TOKENIZER_METADATA_PATTERNS = (
    "added_tokens.json",
    "chat_template*.jinja",
    "*.model",
    "merges.txt",
    "special_tokens_map.json",
    "tokenization_*.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
)

# AutoTokenizer resolves the tokenizer class from config.json's model_type when
# tokenizer_config.json carries no tokenizer_class (common for older exports). A
# staged tokenizer directory without it falls back to PreTrainedTokenizerFast with
# no serialised backend and fails to load at all.
TOKENIZER_CLASS_RESOLUTION_FILES = ("config.json",)


def _stage_tokenizer_class_resolution(source: Path, destination: Path) -> None:
    for name in TOKENIZER_CLASS_RESOLUTION_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)


def prepare_policy_model(args: argparse.Namespace) -> PreparedPolicyModel | None:
    """Resolve one immutable policy source and stage what its runtime needs."""
    source_uri = args.model_source_uri
    source_identity = args.model_source_identity
    manifest = None
    if source_uri:
        if not source_identity:
            raise ValueError(f"Policy source identity is required for {source_uri}")
        if source_identity.startswith("sha256:"):
            manifest = load_model_manifest(source_uri)
            if source_identity != manifest.identity:
                raise ValueError(
                    f"Policy manifest identity mismatch: requested {source_identity}, "
                    f"found {manifest.identity} at {source_uri}"
                )
            source_identity = manifest.identity
    else:
        return None

    assert source_uri and source_identity
    RuntimeProfile(args.runtime_profile)
    local_path = _metadata_path(source_uri, source_identity)
    if args.rollout_model_loading is RolloutModelLoading.STAGE_LOCAL:
        staged_bytes = stage_artifact_model(source_uri, source_identity, local_path)
        validate_hf_model_weights({path.name for path in Path(local_path).iterdir()}, source_uri)
        _log(
            f"Policy checkpoint staged on rank {_rank()}/{_num_tasks()}: {source_uri} -> {local_path} "
            f"(identity={source_identity}; local_disk_high_water_bytes={staged_bytes})"
        )
        return PreparedPolicyModel(source_uri, source_identity, local_path)
    if manifest is None:
        metadata_bytes = stage_artifact_model_metadata(source_uri, source_identity, local_path)
    else:
        stage_model_metadata(source_uri, manifest, local_path)
        metadata_bytes = sum(path.stat().st_size for path in Path(local_path).rglob("*") if path.is_file())
    _log(
        f"Policy metadata ready on rank {_rank()}/{_num_tasks()}: {source_uri} -> {local_path} "
        f"(identity={source_identity}; local_disk_high_water_bytes={metadata_bytes}; weight shards remain remote)"
    )
    return PreparedPolicyModel(source_uri, source_identity, local_path)


def _tokenizer_metadata_files(path: Path) -> tuple[Path, ...]:
    return tuple(
        sorted({metadata_file for pattern in TOKENIZER_METADATA_PATTERNS for metadata_file in path.glob(pattern)})
    )


def _require_tokenizer_metadata(path: Path, source: str) -> tuple[Path, ...]:
    metadata_files = _tokenizer_metadata_files(path)
    if not metadata_files:
        raise ValueError(f"Tokenizer source contains no tokenizer metadata: {source}")
    return metadata_files


def _copy_tokenizer_metadata(source: Path, destination: Path) -> None:
    for metadata_file in _require_tokenizer_metadata(source, str(source)):
        shutil.copy2(metadata_file, destination / metadata_file.name)


def prepare_policy_tokenizer(args: argparse.Namespace) -> PreparedPolicyTokenizer | None:
    """Stage the requested tokenizer independently from the policy weights."""
    tokenizer_path = args.policy_tokenizer
    if not tokenizer_path:
        return None

    revision = args.policy_tokenizer_revision
    local_path = _metadata_path(tokenizer_path, revision or "main")
    target = Path(local_path)
    if is_cloud_uri(tokenizer_path):
        manifest = load_model_manifest(tokenizer_path)
        stage_model_metadata(tokenizer_path, manifest, local_path)
        _require_tokenizer_metadata(target, tokenizer_path)
        (target / MODEL_MANIFEST_FILENAME).unlink(missing_ok=True)
    else:
        with atomic_directory_update(target, staging_prefix=f".{target.name}.tokenizer-") as staging:
            staging.mkdir()
            if os.path.isdir(tokenizer_path):
                _copy_tokenizer_metadata(Path(tokenizer_path), staging)
                _stage_tokenizer_class_resolution(Path(tokenizer_path), staging)
            elif is_hugging_face_repo_id(tokenizer_path):
                download_hugging_face_snapshot(
                    tokenizer_path,
                    revision=revision or None,
                    destination=staging,
                    allow_patterns=TOKENIZER_METADATA_PATTERNS + TOKENIZER_CLASS_RESOLUTION_FILES,
                )
                _require_tokenizer_metadata(staging, tokenizer_path)
            else:
                raise ValueError(f"Unsupported policy tokenizer source: {tokenizer_path!r}")

    return PreparedPolicyTokenizer(local_path)


def prepare_draft_model(
    model: SpeculatorModelConfig,
    *,
    cache_ttl_days: int,
    cache_source_prefix: str,
) -> SpeculatorModelConfig:
    """Resolve a draft locator, mirroring a Hub revision once when needed."""
    if model.source_kind is SpeculatorModelSourceKind.LOCAL:
        assert model.local_source_path is not None
        filesystem, root = fs_and_path(model.local_source_path)
        names = {entry.path for _, entry in file_inventory(filesystem, root)}
        validate_hf_model_weights(names, model.local_source_path)
        return model
    if model.source_kind is SpeculatorModelSourceKind.HUGGING_FACE:
        if cache_ttl_days <= 0:
            raise ValueError("--draft-model-cache-ttl-days must be positive for a Hugging Face draft model")
        if not cache_source_prefix:
            raise ValueError("--draft-model-cache-source-prefix is required for a Hugging Face draft model")
        assert model.hugging_face_repo_id is not None
        source_uri, manifest = ensure_hugging_face_model_cache(
            model.hugging_face_repo_id,
            model.source_identity,
            ttl_days=cache_ttl_days,
            source_prefix=cache_source_prefix,
            tokenizer_mode="policy",
        )
    else:
        source_uri = model.source_uri
        manifest = load_model_manifest(source_uri)
        if model.source_identity.startswith("sha256:") and model.source_identity != manifest.identity:
            raise ValueError(
                f"Draft manifest identity mismatch: requested {model.source_identity}, "
                f"found {manifest.identity} at {source_uri}"
            )
    return SpeculatorModelConfig(source_uri=source_uri, source_identity=manifest.identity)


def materialize_data_sources(data_sources_json: str) -> None:
    """Copy immutable train and validation data locators onto this allocated node."""
    sources = json.loads(data_sources_json)
    if not isinstance(sources, list):
        raise ValueError("--data-sources-json must contain a JSON list")
    for value in sources:
        source = ArtifactSource(uri=value["uri"], local_path=value["local_path"], identity=value["identity"])
        artifact = materialize(source)
        _log(
            f"Dataset staged on rank {_rank()}/{_num_tasks()}: {source.uri} -> {source.local_path} "
            f"({len(artifact.files)} files, identity={source.identity})"
        )


# Special tokens the delphi_v0 reasoning protocol depends on; asserted present in the
# tokenizer's vocab after the override so a lossy SFT export (fragmented-to-bytes tokens)
# fails loud here instead of silently collapsing reward at the first rollout.
_REQUIRED_CHAT_TEMPLATE_TOKENS = ("<|start_think|>", "<|end_think|>")


def apply_policy_chat_template(model_path: str, template_repo_rel: str) -> None:
    """Force the policy tokenizer's cached chat template to ``template_repo_rel`` on THIS node.

    The delphi SFT repos ship no chat_template, so the policy tokenizer must be forced to the
    delphi_v0 template the rollouts + held-out eval use, else reward silently collapses on a
    train/rollout/eval mismatch. Must run on EVERY node (the ``skyrl_entrypoint`` actor may
    load its tokenizer on any node), after the model is staged into the node-local cache and
    before Ray bootstrap.

    Args:
        model_path: HF repo ID staged in the node-local cache or an object-store model's
            materialized local directory.
        template_repo_rel: chat-template jinja path, resolved against the in-pod repo root.

    Raises:
        RuntimeError: if the override does not take, or a required think-protocol token
            (``<|start_think|>`` / ``<|end_think|>``) is not a single registered token.
    """
    template_path = resolve_repo_path(template_repo_rel)
    delphi = template_path.read_text()

    # Import transformers lazily (matches wait_for_nodes' local `import ray`): the
    # controller bootstraps Ray on every node and must not pull this heavy dependency
    # into the fast bootstrap path for configs that set no chat-template.
    from transformers import AutoTokenizer

    snap = model_path if os.path.isdir(model_path) else download_hugging_face_snapshot(model_path, revision=None)
    tc_path = os.path.join(snap, "tokenizer_config.json")
    jinja_path = os.path.join(snap, "chat_template.jinja")

    # Back up originals once (copy resolved content, not the cache symlink).
    for p in (tc_path, jinja_path):
        if os.path.exists(p):
            bak = p + ".plainbak"
            if not os.path.exists(bak):
                with open(p, "rb") as s, open(bak, "wb") as d:
                    d.write(s.read())

    # Write delphi_v0 as a real chat_template.jinja (break the cache symlink; keep the blob).
    if os.path.islink(jinja_path):
        os.remove(jinja_path)
    with open(jinja_path, "w") as f:
        f.write(delphi)

    # And as the chat_template key in tokenizer_config.json (break symlink).
    tc = {}
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
    tc["chat_template"] = delphi
    if os.path.islink(tc_path):
        os.remove(tc_path)
    with open(tc_path, "w") as f:
        json.dump(tc, f, ensure_ascii=False, indent=2)

    # Verify the loaded tokenizer now renders delphi_v0 and keeps the think protocol tokens.
    try:
        tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    except Exception:
        listing = sorted(os.listdir(snap)) if os.path.isdir(snap) else None
        _log(
            f"apply_policy_chat_template: fast tokenizer load failed for {model_path} "
            f"(snapshot files: {listing}); retrying with use_fast=False"
        )
        tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True, use_fast=False)
    ct = tok.chat_template
    if not ct or ct.strip() != delphi.strip():
        raise RuntimeError(
            f"chat_template override did NOT take for {model_path} "
            f"(loaded len={len(ct) if ct else None}, expected {len(delphi)})"
        )
    vocab = tok.get_vocab()
    # The think-protocol gate applies only when the template itself carries the tokens;
    # completion-style templates for base models (e.g. GPT-2) legitimately omit them.
    missing = [t for t in _REQUIRED_CHAT_TEMPLATE_TOKENS if t in delphi and t not in vocab]
    if missing:
        raise RuntimeError(
            f"chat-template protocol tokens {missing} are NOT single registered tokens in "
            f"{model_path}'s tokenizer (lossy SFT export?) — they would fragment to bytes and "
            f"break the reward/parse contract. Aborting before a silent reward-zero run."
        )
    _log(
        f"apply_policy_chat_template: template applied + verified for {model_path} "
        f"(chat_template len={len(ct)}, protocol tokens OK) on rank {_rank()}/{_num_tasks()} (snapshot={snap})"
    )


def policy_chat_template_model(prestage_model: str, model_local_path: str) -> str:
    """Return the model directory or Hub ID whose tokenizer should be rewritten."""
    model_path = prestage_model or model_local_path
    if not model_path:
        raise ValueError(
            "--policy-chat-template requires --prestage-model or --model-local-path "
            "(the template override rewrites the materialized model tokenizer)"
        )
    return model_path


def _rank() -> int:
    # IRIS_TASK_ID is the full task path (e.g. "/user/job/0"); on retried tasks
    # iris appends a ":N" retry suffix. The rank is the trailing path segment
    # with any retry suffix stripped.
    return int(os.environ.get("IRIS_TASK_ID", "0").rsplit("/", 1)[-1].split(":", 1)[0])


def _num_tasks() -> int:
    return int(os.environ.get("IRIS_NUM_TASKS", "1"))


def _own_ip() -> str:
    """Routable IP of this node.

    Prefers iris's ``IRIS_ADVERTISE_HOST`` (the routable IP iris computed for
    this task under ``host_network: true``); falls back to a UDP-socket probe.
    """
    advertised = os.environ.get("IRIS_ADVERTISE_HOST")
    if advertised and advertised != "127.0.0.1":
        return advertised
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def _iface_for_ip(ip: str) -> str | None:
    """Name of the interface holding ``ip``, or None if it cannot be determined.

    Reads /proc/net/fib_trie-free: uses the socket interface table directly so no
    external binary (`ip`, `ifconfig`) and no third-party import is required.
    """
    try:
        import fcntl
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _, name in socket.if_nameindex():
                try:
                    packed = fcntl.ioctl(
                        sock.fileno(),
                        0x8915,  # SIOCGIFADDR
                        struct.pack("256s", name[:15].encode()),
                    )
                    if socket.inet_ntoa(packed[20:24]) == ip:
                        return name
                except OSError:
                    continue  # interface has no IPv4 address
        finally:
            sock.close()
    except Exception as exc:  # noqa: BLE001
        _log(f"[fabric] interface lookup failed: {exc}")
    return None


def pin_socket_ifname() -> str | None:
    """Derive GLOO_SOCKET_IFNAME from the address iris advertises for this task.

    Returns the interface name derived on THIS node, or None when nothing was
    derived (an operator already set the variable, or no interface holds the
    node's IP).

    Gloo carries the DP-rank-0 optimizer gather at checkpoint save (the megatron
    strategy picks ``fully_reshardable`` + mem_efficient precisely so that gather
    runs on CPU and needs no GPU memory). With no interface pin gloo applies its
    own heuristic; on this cluster it picked loopback, every rank advertised
    127.0.0.1, and the mesh never formed.

    NCCL's ``^a,b,c`` exclude syntax cannot be reused here — gloo reads the value
    as a literal interface name and dies with ``Unable to find address for: ^ibs``.
    A static name cannot be used either: the PF differs per node and region.

    So derive it. ``IRIS_ADVERTISE_HOST`` is the pod IP, injected per pod by iris
    through the Kubernetes downward API (``status.podIP``), which is exactly the
    address whose interface the collectives must bind. Whatever interface holds
    it is correct on this node by construction.

    iris runs this same entrypoint on every node of the gang, so every node
    derives its own name before its own ``ray start`` and Ray's workers inherit
    it. The name is only ever correct for the node that derived it.

    Fail-safe by design: an existing value is never overridden, and any failure to
    resolve leaves the environment untouched and the previous behaviour intact.
    """
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        _log("[fabric] GLOO_SOCKET_IFNAME already set; leaving it alone")
        return None
    ip = _own_ip()
    if not ip or ip == "127.0.0.1":
        _log("[fabric] no routable IP; not pinning GLOO_SOCKET_IFNAME")
        return None
    iface = _iface_for_ip(ip)
    if not iface:
        _log(f"[fabric] no interface holds {ip}; not pinning GLOO_SOCKET_IFNAME")
        return None
    os.environ["GLOO_SOCKET_IFNAME"] = iface
    _log(f"[fabric] GLOO_SOCKET_IFNAME={iface} (derived from {ip})")
    return iface


def export_telemetry_environment(run_id: str | None, training_type: TrainingType | None) -> None:
    resolved = telemetry_environment(run_id=run_id, training_type=training_type)
    if not resolved:
        _log("[telemetry] no telemetry environment resolved; MarinSkyRL telemetry stays inert")
        return
    os.environ.update(resolved)
    stamped = f" training_type={resolved[TRAINING_TYPE_ENV]}" if TRAINING_TYPE_ENV in resolved else ""
    _log(f"[telemetry] {resolved[TELEMETRY_ENDPOINT_ENV]} run_id={resolved[RUN_ID_ENV]}{stamped}")


def training_driver_env(derived_gloo_ifname: str | None) -> dict[str, str]:
    """Environment for the rank-0 training driver subprocess.

    Strips a GLOO_SOCKET_IFNAME that ``pin_socket_ifname`` derived on this node.
    skyrl-train forwards GLOO_SOCKET_IFNAME from the driver's environment into
    ``ray.init``'s job-level ``runtime_env``, which pushes ONE value to the actors
    on EVERY node. NIC names are not uniform across this gang: job
    20260729-102429-52af30 had the head derive ``enp90s0np0`` while 10.168.206.93
    names its NIC ``enp90s0f0np0``, so megatron's gloo group creation on that node
    died with ``Unable to find address for: enp90s0np0``. Each node's own
    controller already derived the right name before its ``ray start``.

    A value an operator set explicitly (via the launcher or the config's
    ``extra_env``) reaches every pod already and is passed through untouched.
    """
    env = os.environ.copy()
    env.update(ray_cluster_owner_environment())
    if derived_gloo_ifname is not None and env.get("GLOO_SOCKET_IFNAME") == derived_gloo_ifname:
        del env["GLOO_SOCKET_IFNAME"]
        _log(f"[fabric] withholding node-derived GLOO_SOCKET_IFNAME={derived_gloo_ifname} from the training driver")
    return env


# ---------------------------------------------------------------------------
# Rendezvous — head publishes its IP, workers poll for it. Backend-agnostic via
# fsspec so the URI scheme (gs://, s3://, file://, plain path) selects storage.
# ---------------------------------------------------------------------------


def _pin_boto3_s3_addressing_style() -> None:
    """Pin virtual-hosted (hostname-based) S3 addressing for the boto3 code path,
    cluster-wide, via an AWS config file + ``AWS_CONFIG_FILE``.

    Companion to the fsspec/s3fs pin in ``fs_and_path``: the CoreWeave object
    store (marin-us-east-02a / R2) REJECTS path-style S3 requests with
    ``PathStyleRequestNotAllowed``. Ray's object-spill IO workers call a bare
    ``boto3.resource("s3")`` with NO botocore ``Config`` in child processes we
    never construct, and botocore has NO env var for ``addressing_style`` (it is
    resolved ONLY from the shared-config file's ``s3`` section). So we WRITE a
    minimal AWS config file that sets ``s3.addressing_style`` and point
    ``AWS_CONFIG_FILE`` at it. This runs before ``ray start`` (head + every
    worker) so every Ray subprocess / raylet IO worker / the training driver
    inherits it.

    R2 credentials/region/endpoint come from env vars independently of the config
    file. Env-overridable via ``OT_AGENT_S3_ADDRESSING_STYLE`` (default
    ``virtual``; set ``path``/``auto`` for a path-style store). If an operator
    already exported ``AWS_CONFIG_FILE``, we RESPECT it and skip.
    """
    style = os.environ.get("OT_AGENT_S3_ADDRESSING_STYLE", "virtual")
    existing = os.environ.get("AWS_CONFIG_FILE")
    if existing and os.path.exists(existing):
        _log(
            f"AWS_CONFIG_FILE already set ({existing}); NOT overriding for S3 "
            f"addressing_style. If Ray spill hits PathStyleRequestNotAllowed, add "
            f"'s3 =\\n    addressing_style = {style}' to that file's [default] profile."
        )
        return
    cfg_path = os.path.join(tempfile.gettempdir(), "ot_agent_aws_config")
    try:
        with open(cfg_path, "w") as f:
            f.write(f"[default]\ns3 =\n    addressing_style = {style}\n")
        os.environ["AWS_CONFIG_FILE"] = cfg_path
        _log(
            f"S3 addressing_style={style} pinned via AWS_CONFIG_FILE={cfg_path} "
            f"(boto3/Ray-object-spill virtual-hosted addressing for CoreWeave R2)."
        )
    except OSError as exc:  # noqa: BLE001 - best-effort; falls back to boto3 default
        _log(
            f"WARNING: could not write AWS config for S3 addressing_style ({exc}); "
            f"Ray object-spill may hit PathStyleRequestNotAllowed on CoreWeave R2."
        )


def ensure_fr_dump_dir() -> None:
    """``mkdir -p`` the NCCL flight-recorder dump directory on THIS node before torch init.

    The FR dump path is set via ``TORCH_NCCL_DUMP_ON_TIMEOUT=1`` +
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` / ``TORCH_FR_DUMP_TEMP_FILE`` (a per-rank
    FILENAME PREFIX; torch appends the global rank). torch's ``DebugInfoWriter``
    is a plain ``std::ofstream`` that does NOT ``mkdir -p`` its parent, so on a
    collective timeout every rank's dump silently fails to write.

    This controller runs on EVERY node in the pre-Ray phase (before any
    torch/NCCL init), so creating the dir here once per node guarantees it exists.
    The dir is DERIVED from the cvar (``dirname`` of the dump-path prefix), so it
    is robust to the path changing per-job. Best-effort: a failure here must never
    block bring-up.
    """
    # TORCH_FR_DUMP_TEMP_FILE is the newer alias torch checks first;
    # TORCH_NCCL_DEBUG_INFO_TEMP_FILE is the canonical name. The value is a per-rank
    # FILENAME PREFIX (torch appends the global rank), so the dir to create is its dirname.
    dump_prefix = os.environ.get(FR_DUMP_TEMP_FILE_ENV) or os.environ.get(NCCL_DEBUG_INFO_TEMP_FILE_ENV)
    if not dump_prefix:
        _log("FR dump dir: no TORCH_(FR_DUMP|NCCL_DEBUG_INFO)_TEMP_FILE set; nothing to create.")
        return
    dump_dir = os.path.dirname(dump_prefix)
    if not dump_dir:
        # Bare filename with no directory component (e.g. the generic "/tmp/nccl_fr_rank"
        # has dirname "/tmp", which exists; a relative bare name has no dir to make).
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        _log(f"FR dump dir ensured (mkdir -p): {dump_dir} (from cvar prefix {dump_prefix!r})")
    except OSError as exc:  # noqa: BLE001 - best-effort; never block bring-up on instrumentation
        _log(
            f"WARNING: could not create FR dump dir {dump_dir} ({exc}); "
            f"NCCL flight-recorder dumps may fail to write on a collective timeout."
        )


def ensure_task_debug_artifact_directory(debug_mode: DebugMode = DebugMode.LIGHT) -> None:
    """Give the task uploader and its Ray workers the same artifact root."""
    if debug_mode is DebugMode.OFF:
        return
    environment = EnvVarManager.for_debug_launch(
        job_name=os.environ.get(RUN_ID_ENV, "run"),
        artifact_root=os.environ.get(DEBUG_ARTIFACT_DIR_ENV),
    ).environment_for(EnvVarScope.TASK_RUNTIME)
    # The trainer resolves its diagnostic preset; the supervisor only owns the root.
    EnvVarManager({DEBUG_ARTIFACT_DIR_ENV: environment[DEBUG_ARTIFACT_DIR_ENV]}).apply_to_process(
        EnvVarScope.TASK_RUNTIME
    )


def _rendezvous_uri(rendezvous_dir: str) -> str:
    return f"{rendezvous_dir.rstrip('/')}/{RENDEZVOUS_FILENAME}"


def _done_uri(rendezvous_dir: str) -> str:
    return f"{rendezvous_dir.rstrip('/')}/{DONE_FILENAME}"


def _runtime_versions() -> tuple[str, str]:
    # Preserve this module's bootstrap fast path; Ray is only needed once the
    # head or worker reaches cluster rendezvous.
    import ray  # noqa: PLC0415

    python_version = ".".join(str(component) for component in sys.version_info[:3])
    return python_version, ray.__version__


def validate_rendezvous_runtime(
    payload: RendezvousPayload,
    *,
    worker_node: str,
    python_version: str,
    ray_version: str,
) -> RendezvousPayload:
    """Reject a worker whose Python or Ray version differs from the head."""
    head_python_version = payload.python_version
    head_ray_version = payload.ray_version
    if head_python_version == python_version and head_ray_version == ray_version:
        return payload

    raise RuntimeError(
        "Iris runtime version mismatch before Ray join: "
        f"head {payload.head_node} uses Python {head_python_version} and Ray {head_ray_version}; "
        f"worker {worker_node} uses Python {python_version} and Ray {ray_version}"
    )


def _write_rendezvous_once(fs, path: str, payload: dict[str, object]) -> None:
    with fs.open(path, "w") as f:
        json.dump(payload, f)


def write_rendezvous(rendezvous_dir: str, head_ip: str, ray_port: int, gang_epoch: str) -> None:
    uri = _rendezvous_uri(rendezvous_dir)
    python_version, ray_version = _runtime_versions()
    payload = RendezvousPayload(
        gang_epoch=gang_epoch,
        head_ip=head_ip,
        head_node=socket.gethostname(),
        port=ray_port,
        num_tasks=_num_tasks(),
        python_version=python_version,
        ray_version=ray_version,
        written_at=time.time(),
    )
    serialized_payload = asdict(payload)
    fs, path = fs_and_path(uri)
    # Bound the object-store PutObject with a hard per-attempt timeout via a DAEMON
    # thread + join(timeout) + bounded retries/backoff. An unbounded s3fs/fsspec put
    # has no connect/read timeout and hangs the head invisibly. A daemon thread (NOT a
    # ThreadPoolExecutor) is used because a non-daemon executor worker thread is JOINED
    # by Python's atexit `_python_exit` forever at interpreter shutdown if abandoned,
    # leaving a zombie process; a daemon thread is not joined, so a wedged write cannot
    # block process exit.
    last_exc: BaseException | None = None
    for attempt in range(1, RENDEZVOUS_WRITE_ATTEMPTS + 1):
        t0 = time.time()
        _log(
            f"Writing rendezvous {uri} (attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}, "
            f"per-attempt timeout {RENDEZVOUS_WRITE_TIMEOUT}s)..."
        )
        result_box: dict = {}

        def _target() -> None:
            try:
                _write_rendezvous_once(fs, path, serialized_payload)
                result_box["ok"] = True
            except BaseException as exc:  # noqa: BLE001 - surface to the joiner
                result_box["exc"] = exc

        writer = threading.Thread(target=_target, name=f"rendezvous-write-{attempt}", daemon=True)
        writer.start()
        writer.join(timeout=RENDEZVOUS_WRITE_TIMEOUT)
        if writer.is_alive():
            # STALLED: the put is still running past the timeout. Abandon the daemon
            # thread (never joined at exit, so it cannot wedge process teardown) and
            # fall through to retry / fail-fast.
            last_exc = TimeoutError(f"rendezvous PutObject did not complete within {RENDEZVOUS_WRITE_TIMEOUT}s")
            _log(
                f"Rendezvous write STALLED — timed out after {time.time() - t0:.1f}s "
                f"(attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}); object-store PutObject "
                f"to {uri} is not completing."
            )
        elif "exc" in result_box:
            last_exc = result_box["exc"]
            _log(
                f"Rendezvous write FAILED after {time.time() - t0:.1f}s "
                f"(attempt {attempt}/{RENDEZVOUS_WRITE_ATTEMPTS}): {last_exc!r}"
            )
        else:
            _log(
                f"Wrote rendezvous {uri}: head_ip={head_ip} port={ray_port} "
                f"(attempt {attempt}, {time.time() - t0:.1f}s)"
            )
            return
        if attempt < RENDEZVOUS_WRITE_ATTEMPTS:
            backoff = min(2 ** (attempt - 1), 10)
            _log(f"Retrying rendezvous write in {backoff}s...")
            time.sleep(backoff)
    raise RuntimeError(
        f"Rank-0 failed to publish the rendezvous to {uri} after "
        f"{RENDEZVOUS_WRITE_ATTEMPTS} attempts (last error: {last_exc!r}). Failing "
        f"fast so the gang aborts with a clear cause instead of hanging to the worker "
        f"rendezvous deadline."
    ) from last_exc


def poll_rendezvous(
    rendezvous_dir: str,
    timeout: int,
    min_written_at: float | None = None,
) -> RendezvousPayload:
    """Poll for the head's rendezvous file. Returns its parsed payload.

    Payloads with ``written_at`` older than ``min_written_at`` (minus slack) are
    treated as stale (from a prior iris task attempt) and ignored.
    """
    uri = _rendezvous_uri(rendezvous_dir)
    fs, path = fs_and_path(uri)
    deadline = time.time() + timeout
    threshold = (min_written_at - RENDEZVOUS_FRESHNESS_SLACK) if min_written_at else None
    _log(f"Polling for rendezvous {uri} (timeout {timeout}s)...")
    while time.time() < deadline:
        try:
            if fs.exists(path):
                with fs.open(path, "r") as f:
                    payload = RendezvousPayload.from_dict(json.load(f))
                if threshold is not None and payload.written_at < threshold:
                    _log(
                        f"Ignoring stale rendezvous (written_at={payload.written_at:.0f} "
                        f"< threshold={threshold:.0f}); waiting for rank-0 rewrite."
                    )
                else:
                    _log(f"Found rendezvous: {asdict(payload)}")
                    return payload
        except RuntimeError:
            raise
        except Exception as exc:  # transient object-store hiccup
            _log(f"rendezvous poll error (will retry): {exc}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Worker rank {_rank()} timed out after {timeout}s waiting for "
        f"rank-0 rendezvous at {uri}. Did the head task fail to start?"
    )


def write_head_result(rendezvous_dir: str, gang_epoch: str) -> None:
    """Publish successful completion for exactly one Ray gang attempt."""
    uri = _done_uri(rendezvous_dir)
    fs, path = fs_and_path(uri)
    result = HeadResult(gang_epoch=gang_epoch, outcome=HEAD_RESULT_SUCCEEDED, written_at=time.time())
    with fs.open(path, "w") as destination:
        json.dump(asdict(result), destination)
    _log(f"Published successful head result for gang epoch {gang_epoch}: {uri}")


def head_succeeded(rendezvous_dir: str, gang_epoch: str) -> bool:
    """Whether the head published success for ``gang_epoch``."""
    uri = _done_uri(rendezvous_dir)
    fs, path = fs_and_path(uri)
    try:
        if not fs.exists(path):
            return False
        with fs.open(path, "r") as source:
            result = HeadResult.from_dict(json.load(source))
    except OSError as error:
        _log(f"head-result poll error (will retry): {error}")
        return False
    return result.gang_epoch == gang_epoch


def clear_rendezvous(rendezvous_dir: str) -> None:
    """Best-effort delete of rendezvous state before a new head attempt starts."""
    for name in (RENDEZVOUS_FILENAME, DONE_FILENAME):
        uri = f"{rendezvous_dir.rstrip('/')}/{name}"
        try:
            fs, path = fs_and_path(uri)
            if fs.exists(path):
                fs.rm(path)
                _log(f"Removed {uri}")
        except Exception as exc:
            _log(f"Warning: could not remove {uri}: {exc}")


# ---------------------------------------------------------------------------
# Ray cluster bootstrap (mirrors start_vllm_iris_controller).
# ---------------------------------------------------------------------------


def _ray_port_flags() -> list[str]:
    # Iris tasks share host networking. Named endpoints must use scheduler allocations;
    # worker processes ask the kernel for ports outside Iris's reserved allocation range.
    return [
        f"--metrics-export-port={allocated_ray_port(RayPort.METRICS)}",
        f"--runtime-env-agent-port={allocated_ray_port(RayPort.RUNTIME_ENV_AGENT)}",
        f"--dashboard-agent-grpc-port={allocated_ray_port(RayPort.DASHBOARD_AGENT_GRPC)}",
        f"--dashboard-agent-listen-port={allocated_ray_port(RayPort.DASHBOARD_AGENT_LISTEN)}",
        f"--node-manager-port={allocated_ray_port(RayPort.NODE_MANAGER)}",
        f"--object-manager-port={allocated_ray_port(RayPort.OBJECT_MANAGER)}",
        "--min-worker-port=0",
        "--max-worker-port=0",
    ]


# --- Ray cgroup-aware memory ----------------------------------------------------
# In a memory-cgroup-limited pod, Ray can read the HOST's physical RAM (~2 TB via
# /proc/meminfo) instead of the --memory cgroup limit, and size its plasma object
# store at the default ~30% of that (~600 GB). That can overrun the container
# limit once training also allocates memory. Read the cgroup limit and pass it
# to `ray start` as --memory, and bound the plasma store.
RAY_OBJECT_STORE_CAP_GIB = 96  # bounded plasma; default can be ~30% of *detected* RAM (huge if host is misread)
# Per-job override (env) for the plasma cap, in GiB. Default = RAY_OBJECT_STORE_CAP_GIB
# (96). The min(., cgroup//8) OOM guard below still applies as the HARD ceiling.
_RAY_STORE_CAP_ENV = "OT_AGENT_RAY_OBJECT_STORE_CAP_GIB"


def _cgroup_mem_limit_bytes() -> int | None:
    """The container's memory cgroup limit in bytes (cgroup v2 then v1), or None if
    unreadable / unlimited (so callers fall back to Ray's own detection for --memory)."""
    for path in (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):  # cgroup v1
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw in ("max", ""):  # v2 unlimited
            return None
        try:
            v = int(raw)
        except ValueError:
            continue
        if v <= 0 or v > (1 << 62):  # v1 "unlimited" sentinel is a near-2^63 value
            return None
        return v
    return None


def _ray_mem_flags() -> list[str]:
    """Make `ray start` cgroup-aware (see the block comment above). Always bounds the
    plasma object store; additionally pins --memory to the cgroup limit when readable."""
    try:
        cap_gib = float(os.environ.get(_RAY_STORE_CAP_ENV, RAY_OBJECT_STORE_CAP_GIB))
    except ValueError:
        _log(
            f"Ray cgroup-aware: {_RAY_STORE_CAP_ENV}={os.environ.get(_RAY_STORE_CAP_ENV)!r} "
            f"not a number; falling back to default {RAY_OBJECT_STORE_CAP_GIB}GiB"
        )
        cap_gib = float(RAY_OBJECT_STORE_CAP_GIB)
    store_cap = int(cap_gib * (1 << 30))
    if cap_gib != RAY_OBJECT_STORE_CAP_GIB:
        _log(f"Ray cgroup-aware: plasma cap overridden via {_RAY_STORE_CAP_ENV} -> ~{cap_gib:.0f}GiB (pre-guard)")
    limit = _cgroup_mem_limit_bytes()
    flags: list[str] = []
    if limit:
        store_cap = min(store_cap, limit // 8)  # never let the store exceed ~1/8 of the container
        flags.append(f"--memory={limit}")
        _log(
            f"Ray cgroup-aware: --memory={limit} (~{limit / (1 << 30):.0f}GiB cgroup limit), "
            f"--object-store-memory={store_cap} (~{store_cap / (1 << 30):.0f}GiB plasma cap)"
        )
    else:
        _log(
            f"Ray cgroup-aware: no cgroup mem limit readable; bounding "
            f"--object-store-memory={store_cap} (~{store_cap / (1 << 30):.0f}GiB) only"
        )
    flags.append(f"--object-store-memory={store_cap}")
    return flags


def ray_start_head(
    head_ip: str,
    spill_target: RaySpillTarget,
) -> None:
    spill_target.prepare_node()
    cmd = [
        _ray_bin(),
        "start",
        "--head",
        f"--node-ip-address={head_ip}",
        f"--port={allocated_ray_port(RayPort.HEAD)}",
        f"--dashboard-port={allocated_ray_port(RayPort.DASHBOARD)}",
        f"--ray-client-server-port={allocated_ray_port(RayPort.CLIENT)}",
        "--dashboard-host=0.0.0.0",
        *_ray_port_flags(),
        *_ray_mem_flags(),
        *spill_target.head_flags(),
    ]
    _log(spill_target.description())
    _log(f"Starting Ray HEAD: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True, timeout=RAY_START_HEAD_TIMEOUT)
    _log(f"Ray HEAD subprocess returned (exit 0) in {time.time() - t0:.1f}s")


def ray_start_worker(
    head_ip: str,
    ray_port: int,
    node_ip: str,
    spill_target: RaySpillTarget,
) -> None:
    spill_target.prepare_node()
    cmd = [
        _ray_bin(),
        "start",
        f"--address={head_ip}:{ray_port}",
        f"--node-ip-address={node_ip}",
        *_ray_port_flags(),
        *_ray_mem_flags(),
        *spill_target.worker_flags(),
    ]
    _log(spill_target.description())
    _log(f"Starting Ray WORKER: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ray_stop() -> None:
    try:
        completed = subprocess.run(
            [_ray_bin(), "stop", "--force"],
            check=False,
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.TimeoutExpired:
        _log("Warning: 'ray stop' timed out")
        return

    if completed.returncode != 0:
        detail = next((line.strip() for line in reversed(completed.stdout.splitlines()) if line.strip()), "")
        suffix = f": {detail[:500]}" if detail else ""
        _log(f"Warning: 'ray stop' exited {completed.returncode}{suffix}")
        return
    _log(f"Ray stop completed (exit {completed.returncode})")


def wait_for_nodes(ray_address: str, expected_nodes: int, timeout: int, rewrite_cb=None) -> None:
    """Block until the Ray cluster reports ``expected_nodes`` alive nodes.

    ``rewrite_cb`` (head only): a no-arg callable invoked on every poll to RE-PUBLISH
    the rendezvous so its ``written_at`` stays fresh. A worker pod on a cold node can
    start >RENDEZVOUS_FRESHNESS_SLACK (60s) after the head wrote the rendezvous; its
    ``poll_rendezvous(min_written_at=worker_start)`` would then reject the head's
    one-shot rendezvous as "stale" and wait forever while the head waits for that node —
    a mutual deadlock. Rewriting each poll keeps the timestamp ahead of any late worker's
    freshness threshold without weakening the prior-ATTEMPT protection (a stale file from
    a dead PRIOR attempt is still never refreshed).
    """
    import ray

    deadline = time.time() + timeout
    _log(f"Waiting for {expected_nodes} Ray node(s) at {ray_address} (timeout {timeout}s)...")
    ray.init(address=ray_address, ignore_reinit_error=True)
    try:
        last_count = -1
        while time.time() < deadline:
            if rewrite_cb is not None:
                try:
                    rewrite_cb()
                except Exception as exc:
                    _log(f"Warning: rendezvous rewrite failed (will retry): {exc}")
            alive = [n for n in ray.nodes() if n.get("Alive")]
            count = len(alive)
            if count != last_count:
                _log(f"Ray nodes alive: {count}/{expected_nodes}")
                last_count = count
            if count >= expected_nodes:
                _log(f"All {expected_nodes} Ray node(s) joined. Resources: {ray.cluster_resources()}")
                return
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"Only {last_count}/{expected_nodes} Ray nodes joined within {timeout}s.")
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Roles.
# ---------------------------------------------------------------------------


def capture_termination_artifacts(rendezvous_dir: str | None, reason: str) -> None:
    """On teardown, snapshot a FAST diagnostic summary to the rendezvous store
    BEFORE the pod is reaped. Best-effort; never raises; bounded to finish inside
    the k8s grace period.

    An iris/k8s-level termination (ephemeral-storage EVICTION, cgroup OOM, VRAM
    OOM) sends the controller a plain SIGTERM and leaves NOTHING in the iris
    finelog — and the per-node Ray logs are deleted with the pod, so the real
    cause is unrecoverable post-mortem. This persists disk-hogs + GPU mem + df +
    dmesg-OOM, keyed by task id, to ``<rendezvous_dir>/term_artifacts/`` so the
    next probe reads the true cause (disk vs VRAM-OOM vs RAM-OOM).
    """
    if not rendezvous_dir:
        return
    import subprocess as _sp

    task_id = os.environ.get("IRIS_TASK_ID", "unknown").replace("/", "_")
    ts = int(time.time())

    def _run(cmd: str, timeout: int = 7) -> str:
        try:
            return _sp.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
        except Exception as exc:  # noqa: BLE001 - best-effort
            return f"<{cmd!r} failed: {exc}>"

    summary = "\n".join(
        [
            f"=== TERMINATION ARTIFACT task={task_id} ts={ts} reason={reason} ===",
            "--- df -h /tmp /dev/shm ---",
            _run("df -h /tmp /dev/shm 2>&1"),
            "--- top /tmp disk hogs (ephemeral-storage eviction cause) ---",
            _run("du -sh /tmp/* /tmp/ray/session*/logs /tmp/ray/session*/*spill* 2>/dev/null | sort -rh | head -25", 9),
            "--- nvidia-smi (VRAM OOM cause) ---",
            _run("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>&1"),
            "--- top RSS procs (host-RAM OOM cause) ---",
            _run("ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -12"),
            "--- dmesg OOM/kill tail ---",
            _run("dmesg 2>/dev/null | grep -iE 'oom|killed process|out of memory|Xid' | tail -10"),
        ]
    )
    try:
        uri = f"{rendezvous_dir.rstrip('/')}/term_artifacts/{task_id}_{ts}.txt"
        fs, path = fs_and_path(uri)
        with fs.open(path, "w") as f:
            f.write(summary)
        _log(f"[term-capture] wrote termination artifact -> {uri}")
    except Exception as exc:  # noqa: BLE001 - still emit to finelog as fallback
        _log(f"[term-capture] upload FAILED ({exc}); emitting inline:\n{summary}")


# --- Ray session-log -> object-store sync -----------------------------------------
# The per-actor Ray WORKER logs (/tmp/ray/session_*/logs/worker-*.{out,err},
# raylet.out, ...) are the only place the policy / rollout actor stdout+tracebacks
# land — the iris finelog aggregates only what reaches the head, and a pod GC / eviction
# DELETES these node-local logs with the pod. This periodically (+ on SIGTERM) uploads
# THIS node's session logs to the job's durable Ray-log root, keyed by node id, reusing
# the SAME fsspec/boto3 + AWS_ENDPOINT_URL credentials as the rendezvous / spill /
# term-artifact writers. Per-node: each pod writes its own logs under
# <ray_log_dir>/<node_id>/. Gate:
# OT_AGENT_RAY_LOG_SYNC (default "1"); interval OT_AGENT_RAY_LOG_SYNC_INTERVAL_S (300s).
RAY_LOG_SYNC_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024  # skip a single >2 GiB log (pathological)
# Ray emits many small log files, so upload time is dominated by per-object latency.
# Keep enough concurrency for a cold short-lived job to finish within the five-minute
# teardown budget while staying far below object-store request-rate limits.
RAY_LOG_SYNC_MAX_WORKERS = 16
RAY_LOG_SYNC_MAX_FAILURE_LOGS = 3
# A periodic pass must not consume the entire teardown window on a long-lived Ray
# session. Deferred files remain eligible for the next pass; diagnostic logs sort first.
RAY_LOG_SYNC_MAX_FILES_PER_PASS = 1024
RAY_LOG_SYNC_MAX_BYTES_PER_PASS = 1024 * 1024 * 1024
DEBUG_SYNC_MAX_FILE_BYTES = 512 * 1024 * 1024
DEBUG_SYNC_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class RayLogSyncResult:
    uploaded_files: int = 0
    uploaded_bytes: int = 0
    unchanged_files: int = 0
    failed_files: int = 0
    deferred_files: int = 0


class RayLogSyncWaitStatus(StrEnum):
    DISABLED = "disabled"
    SKIPPED = "skipped"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"


class _RayLogFilesystem(Protocol):
    def put(self, local_path: str, remote_path: str) -> object: ...


@dataclass(frozen=True)
class _RayLogVersion:
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True)
class _RayLogUpload:
    local_path: str
    remote_path: str
    version: _RayLogVersion


@dataclass(frozen=True)
class _RayLogUploadResult:
    upload: _RayLogUpload
    current_version: _RayLogVersion | None
    error: Exception | None


@dataclass(frozen=True)
class _RayLogScanFailure:
    local_path: str
    error: OSError


@dataclass(frozen=True)
class _RayLogUploadPlan:
    uploads: tuple[_RayLogUpload, ...]
    active_remote_paths: frozenset[str]
    unchanged_files: int
    scan_failures: tuple[_RayLogScanFailure, ...]
    deferred_files: int


@dataclass(frozen=True)
class _RayLogSyncSettings:
    enabled: bool
    interval_seconds: int
    final_timeout_seconds: float

    @classmethod
    def from_environment(cls) -> "_RayLogSyncSettings":
        return cls(
            enabled=os.environ.get("OT_AGENT_RAY_LOG_SYNC", "1") == "1",
            interval_seconds=int(os.environ.get("OT_AGENT_RAY_LOG_SYNC_INTERVAL_S", "300")),
            final_timeout_seconds=float(os.environ.get("OT_AGENT_RAY_LOG_FINAL_SYNC_TIMEOUT_S", "300")),
        )


def _ray_log_version(path: str) -> _RayLogVersion:
    """Return the byte size and nanosecond modification time for one local log."""
    stat = os.stat(path)
    return _RayLogVersion(size_bytes=stat.st_size, modified_ns=stat.st_mtime_ns)


def _upload_ray_logs(
    filesystem: _RayLogFilesystem,
    uploads: tuple[_RayLogUpload, ...],
) -> tuple[_RayLogUploadResult, ...]:
    """Return one success or captured exception for each requested upload."""
    if not uploads:
        return ()
    pending: queue.Queue[_RayLogUpload] = queue.Queue()
    completed: queue.Queue[_RayLogUploadResult] = queue.Queue()
    for upload in uploads:
        pending.put(upload)

    def _worker() -> None:
        while True:
            try:
                upload = pending.get_nowait()
            except queue.Empty:
                return
            try:
                filesystem.put(upload.local_path, upload.remote_path)
                current_version = _ray_log_version(upload.local_path)
            except Exception as error:  # noqa: BLE001 - capture per-file I/O failures for the result and log
                completed.put(_RayLogUploadResult(upload=upload, current_version=None, error=error))
            else:
                completed.put(_RayLogUploadResult(upload=upload, current_version=current_version, error=None))

    workers = [
        threading.Thread(target=_worker, daemon=True, name=f"ray-log-upload-{index}")
        for index in range(min(RAY_LOG_SYNC_MAX_WORKERS, len(uploads)))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return tuple(completed.get_nowait() for _ in uploads)


def _plan_ray_log_uploads(
    log_dirs: list[str],
    destination_path: str,
    uploaded_versions: dict[str, _RayLogVersion],
) -> _RayLogUploadPlan:
    uploads: list[_RayLogUpload] = []
    active_remote_paths: set[str] = set()
    scan_failures: list[_RayLogScanFailure] = []
    unchanged_files = 0
    for log_dir in log_dirs:
        session = os.path.basename(os.path.dirname(log_dir))
        for root, directories, files in os.walk(log_dir):
            directories.sort()
            for filename in sorted(files):
                local_path = os.path.join(root, filename)
                try:
                    version = _ray_log_version(local_path)
                except OSError as error:
                    scan_failures.append(_RayLogScanFailure(local_path=local_path, error=error))
                    continue
                if version.size_bytes > RAY_LOG_SYNC_MAX_FILE_BYTES:
                    continue
                relative_path = os.path.relpath(local_path, log_dir)
                remote_path = f"{destination_path}/{session}/{relative_path}"
                active_remote_paths.add(remote_path)
                if uploaded_versions.get(remote_path) == version:
                    unchanged_files += 1
                    continue
                uploads.append(_RayLogUpload(local_path=local_path, remote_path=remote_path, version=version))

    def priority(upload: _RayLogUpload) -> tuple[bool, int, str]:
        filename = os.path.basename(upload.local_path)
        diagnostic = (
            filename.endswith(".err")
            or filename in {"raylet.out", "gcs_server.out"}
            or filename.startswith("python-core-driver")
        )
        return diagnostic, upload.version.modified_ns, upload.local_path

    selected: list[_RayLogUpload] = []
    selected_bytes = 0
    for upload in sorted(uploads, key=priority, reverse=True):
        if len(selected) >= RAY_LOG_SYNC_MAX_FILES_PER_PASS:
            break
        if selected_bytes + upload.version.size_bytes > RAY_LOG_SYNC_MAX_BYTES_PER_PASS:
            continue
        selected.append(upload)
        selected_bytes += upload.version.size_bytes
    return _RayLogUploadPlan(
        uploads=tuple(selected),
        active_remote_paths=frozenset(active_remote_paths),
        unchanged_files=unchanged_files,
        scan_failures=tuple(scan_failures),
        deferred_files=len(uploads) - len(selected),
    )


def _apply_ray_log_upload_results(
    uploaded_versions: dict[str, _RayLogVersion],
    plan: _RayLogUploadPlan,
    upload_results: tuple[_RayLogUploadResult, ...],
) -> tuple[RayLogSyncResult, tuple[_RayLogUploadResult, ...]]:
    """Prune stale versions, record stable uploads, and return the pass result and failures."""
    for remote_path in tuple(uploaded_versions):
        if remote_path not in plan.active_remote_paths:
            del uploaded_versions[remote_path]
    uploaded_files = uploaded_bytes = 0
    failures: list[_RayLogUploadResult] = []
    for upload_result in upload_results:
        if upload_result.error is not None or upload_result.current_version is None:
            failures.append(upload_result)
            continue
        upload = upload_result.upload
        uploaded_files += 1
        uploaded_bytes += upload.version.size_bytes
        if upload_result.current_version == upload.version:
            uploaded_versions[upload.remote_path] = upload.version
    return (
        RayLogSyncResult(
            uploaded_files=uploaded_files,
            uploaded_bytes=uploaded_bytes,
            unchanged_files=plan.unchanged_files,
            failed_files=len(plan.scan_failures) + len(failures),
            deferred_files=plan.deferred_files,
        ),
        tuple(failures),
    )


def sync_debug_artifacts(rendezvous_dir: str | None, node_id: str, reason: str) -> None:
    """Boundedly persist this node's managed debug directory beside rendezvous data."""
    source_root = os.environ.get(DEBUG_ARTIFACT_DIR_ENV)
    if not rendezvous_dir or not source_root or not os.path.isdir(source_root):
        return
    destination = f"{rendezvous_dir.rstrip('/')}/debug_artifacts/{node_id}"
    try:
        filesystem, destination_path = fs_and_path(destination)
    except Exception as exc:  # noqa: BLE001 - teardown evidence is best-effort
        _log(f"[debug-sync] cannot resolve {destination} ({exc}) [{reason}]")
        return

    copied: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    total_bytes = 0
    for root, directories, files in os.walk(source_root):
        directories.sort()
        for filename in sorted(files):
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, source_root)
            try:
                size = os.path.getsize(local_path)
                if size > DEBUG_SYNC_MAX_FILE_BYTES or total_bytes + size > DEBUG_SYNC_MAX_TOTAL_BYTES:
                    skipped.append({"path": relative_path, "bytes": size, "reason": "budget"})
                    continue
                filesystem.put(local_path, f"{destination_path}/{relative_path}")
                copied.append({"path": relative_path, "bytes": size})
                total_bytes += size
            except Exception as exc:  # noqa: BLE001 - retain a complete sync receipt
                skipped.append({"path": relative_path, "reason": str(exc)})

    receipt = {
        "schema_version": 1,
        "node_id": node_id,
        "reason": reason,
        "source_root": source_root,
        "copied": copied,
        "skipped": skipped,
        "copied_bytes": total_bytes,
    }
    try:
        with filesystem.open(f"{destination_path}/sync-manifest.json", "w") as manifest:
            json.dump(receipt, manifest, sort_keys=True)
        _log(f"[debug-sync] uploaded {len(copied)} file(s) / {total_bytes} bytes -> {destination} [{reason}]")
    except Exception as exc:  # noqa: BLE001 - teardown evidence is best-effort
        _log(f"[debug-sync] manifest upload failed ({exc}) [{reason}]")


@dataclass
class RayLogSyncSession:
    """Incrementally upload one node's Ray logs, with a preemptive final evidence pass."""

    ray_log_dir: str | None
    node_id: str
    _settings: _RayLogSyncSettings = field(default_factory=_RayLogSyncSettings.from_environment, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _uploaded_versions: dict[str, _RayLogVersion] = field(default_factory=dict, repr=False)

    @property
    def destination(self) -> str | None:
        if not self.ray_log_dir:
            return None
        return join_resource_path(self.ray_log_dir, self.node_id)

    def _sync_pass(
        self,
        reason: str,
        uploaded_versions: dict[str, _RayLogVersion],
    ) -> RayLogSyncResult:
        if not self.destination or not self._settings.enabled:
            return RayLogSyncResult()

        log_dirs = sorted(glob.glob("/tmp/ray/session_*/logs"))
        if not log_dirs:
            return RayLogSyncResult()
        try:
            filesystem, destination_path = fs_and_path(self.destination)
        except Exception as error:  # noqa: BLE001 - best-effort teardown evidence
            _log(f"[ray-log-sync] cannot resolve dest {self.destination} ({error}) [{reason}]")
            return RayLogSyncResult()

        plan = _plan_ray_log_uploads(log_dirs, destination_path, uploaded_versions)
        upload_results = _upload_ray_logs(filesystem, plan.uploads)
        result, upload_failures = _apply_ray_log_upload_results(uploaded_versions, plan, upload_results)

        failure_messages = [
            f"stat failed for {failure.local_path}: {type(failure.error).__name__}: {failure.error}"
            for failure in plan.scan_failures
        ]
        failure_messages.extend(
            f"upload failed for {failure.upload.local_path}: {type(failure.error).__name__}: {failure.error}"
            for failure in upload_failures
        )
        for failure_message in failure_messages[:RAY_LOG_SYNC_MAX_FAILURE_LOGS]:
            _log(f"[ray-log-sync] {failure_message} [{reason}]")
        if len(failure_messages) > RAY_LOG_SYNC_MAX_FAILURE_LOGS:
            additional_failures = len(failure_messages) - RAY_LOG_SYNC_MAX_FAILURE_LOGS
            _log(f"[ray-log-sync] {additional_failures} additional sync failure(s) [{reason}]")
        _log(
            f"[ray-log-sync] uploaded {result.uploaded_files} file(s) / "
            f"{result.uploaded_bytes / 1073741824.0:.2f} GiB "
            f"(unchanged={result.unchanged_files}, failed={result.failed_files}, deferred={result.deferred_files}) "
            f"-> {self.destination} [{reason}]"
        )
        return result

    def sync(self, reason: str) -> RayLogSyncResult:
        """Upload a bounded selection of changed files and return per-pass counts."""
        with self._lock:
            return self._sync_pass(reason, self._uploaded_versions)

    def _sync_final(self, reason: str) -> RayLogSyncResult:
        """Run a final pass even when a periodic pass is stuck in remote I/O."""
        if self._lock.acquire(blocking=False):
            try:
                return self._sync_pass(reason, self._uploaded_versions)
            finally:
                self._lock.release()
        _log(f"[ray-log-sync] periodic upload still active; starting independent final pass [{reason}]")
        # Do not mutate shared version state concurrently. Re-uploading bounded diagnostic
        # files is preferable to losing the exception that caused process teardown.
        return self._sync_pass(reason, {})

    def sync_bounded(self, reason: str) -> RayLogSyncWaitStatus:
        """Wait at most the configured teardown budget and report the terminal wait status."""
        if not self.destination or not self._settings.enabled:
            return RayLogSyncWaitStatus.DISABLED
        timeout = self._settings.final_timeout_seconds
        if timeout <= 0:
            _log(f"[ray-log-sync] skipping final upload because timeout is {timeout}s [{reason}]")
            return RayLogSyncWaitStatus.SKIPPED

        sync_thread = threading.Thread(
            target=self._sync_final,
            args=(reason,),
            daemon=True,
            name="ray-log-final-sync",
        )
        sync_thread.start()
        sync_thread.join(timeout)
        if sync_thread.is_alive():
            _log(
                f"[ray-log-sync] final upload exceeded {timeout}s; continuing teardown with a partial upload [{reason}]"
            )
            return RayLogSyncWaitStatus.TIMED_OUT
        return RayLogSyncWaitStatus.COMPLETED

    def start_periodic(self, rendezvous_dir: str | None) -> threading.Event:
        """Start periodic Ray-log and debug-artifact uploads."""
        stop = threading.Event()
        if (not self.destination and not rendezvous_dir) or not self._settings.enabled:
            return stop
        interval = self._settings.interval_seconds
        if interval <= 0:
            return stop

        def _loop() -> None:
            if stop.wait(min(60, interval)):
                return
            self.sync("periodic")
            sync_debug_artifacts(rendezvous_dir, self.node_id, "periodic")
            while not stop.wait(interval):
                self.sync("periodic")
                sync_debug_artifacts(rendezvous_dir, self.node_id, "periodic")

        threading.Thread(target=_loop, daemon=True, name="ray-log-sync").start()
        destinations: list[str] = []
        if self.destination:
            destinations.append(self.destination)
        if rendezvous_dir:
            destinations.append(join_resource_path(rendezvous_dir, "debug_artifacts", self.node_id))
        _log(f"[artifact-sync] started (every {interval}s, first ~{min(60, interval)}s) -> {', '.join(destinations)}")
        return stop


@dataclass
class DriverOutputActivity:
    """Track child output that provides evidence of training progress."""

    last_progress_at: float = field(default_factory=time.monotonic)
    condition: threading.Condition = field(default_factory=threading.Condition)

    def record(self, line: bytes) -> None:
        if not line.strip() or any(marker in line for marker in IGNORED_DRIVER_OUTPUT_MARKERS):
            return
        with self.condition:
            self.last_progress_at = time.monotonic()
            self.condition.notify_all()

    def stream_closed(self) -> None:
        with self.condition:
            self.condition.notify_all()


@dataclass(frozen=True)
class DriverExited:
    exit_code: int


@dataclass(frozen=True)
class DriverStalled:
    silence: float


def training_driver_command(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "cloud.iris.training_driver", "--config", str(config_path)]


def launch_training_driver(config_path: Path, env: dict[str, str]) -> subprocess.Popen:
    """Start the driver from the immutable runtime checkout, not the bootstrap bundle."""
    runtime_checkout = env.get("SKYRL_HOME")
    if not runtime_checkout:
        raise RuntimeError("SKYRL_HOME must identify the immutable MarinSkyRL runtime checkout")
    return subprocess.Popen(
        training_driver_command(config_path),
        env=env,
        cwd=runtime_checkout,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def start_driver_output_tee(
    process: subprocess.Popen,
    activity: DriverOutputActivity,
) -> threading.Thread:
    """Forward child output to the task log and update its liveness signal."""
    if process.stdout is None:
        raise RuntimeError("Training driver stdout must be piped for liveness supervision")

    def forward_output() -> None:
        try:
            for line in iter(process.stdout.readline, b""):
                sys.stdout.write(line.decode(errors="replace"))
                sys.stdout.flush()
                activity.record(line)
        finally:
            activity.stream_closed()

    output_thread = threading.Thread(target=forward_output, daemon=True, name="driver-output-tee")
    output_thread.start()
    return output_thread


def wait_for_driver(
    process: subprocess.Popen,
    activity: DriverOutputActivity,
    timeout: float,
) -> DriverExited | DriverStalled:
    """Return the named child exit or output-stall outcome."""
    if timeout <= 0:
        return DriverExited(process.wait())

    while True:
        exit_code = process.poll()
        if exit_code is not None:
            return DriverExited(exit_code)
        with activity.condition:
            silence = time.monotonic() - activity.last_progress_at
            if silence >= timeout:
                return DriverStalled(silence)
            activity.condition.wait(timeout=min(timeout - silence, DRIVER_WATCHDOG_POLL_INTERVAL))


def kill_driver_process_group(process: subprocess.Popen) -> None:
    """Kill a stalled driver and its descendants before the Iris task exits."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        process.poll()
        return
    try:
        process.wait(timeout=DRIVER_KILL_TIMEOUT)
    except subprocess.TimeoutExpired:
        _log(f"Training driver did not become waitable within {DRIVER_KILL_TIMEOUT}s after SIGKILL")


def _persist_failure_artifacts_bounded(action, timeout: float) -> None:
    """Give failure evidence a bounded upload window without delaying task failure indefinitely."""
    artifact_thread = threading.Thread(target=action, daemon=True, name="failure-artifact-sync")
    artifact_thread.start()
    artifact_thread.join(timeout)
    if artifact_thread.is_alive():
        _log(f"Failure artifact upload exceeded {timeout}s; continuing task teardown")


def run_head(
    args: argparse.Namespace,
    config_path: Path,
    derived_gloo_ifname: str | None = None,
    *,
    debug_mode: DebugMode = DebugMode.LIGHT,
) -> int:
    ensure_task_debug_artifact_directory(debug_mode)
    num_tasks = _num_tasks()
    gang_epoch = uuid.uuid4().hex
    head_ip = _own_ip()
    ray_port = allocated_ray_port(RayPort.HEAD)
    ray_address = f"{head_ip}:{ray_port}"
    node_id = f"rank0-{socket.gethostname()}"
    _log(f"ROLE=head rank=0/{num_tasks} head_ip={head_ip} ray_port={ray_port}")
    ray_log_sync_stop: threading.Event | None = None
    ray_log_sync = RayLogSyncSession(args.ray_log_dir, node_id)

    def persist_runtime_artifacts(reason: str) -> None:
        if ray_log_sync_stop is not None:
            ray_log_sync_stop.set()
        ray_log_sync.sync_bounded(reason)
        sync_debug_artifacts(args.rendezvous_dir, node_id, reason)

    # Install the SIGTERM/SIGINT handler + termination-artifact capture at the TOP of
    # bring-up (BEFORE clear_rendezvous / ray_start_head / rendezvous write), so a reap
    # ANYWHERE in bring-up produces the term artifact instead of dying silently.
    # `process` is None during bring-up; the handler skips the driver kill until the
    # training driver is launched below (closure reads the current value).
    process = None

    def _shutdown(signum, _frame) -> None:
        _log(f"Received signal {signum}; terminating training driver and stopping Ray...")
        # Capture FIRST (before teardown mutates disk/GPU state) — a SIGTERM here is
        # often a k8s eviction / OOM whose cause survives nowhere else.
        capture_termination_artifacts(args.rendezvous_dir, f"signal {signum} (head rank 0)")
        # Flush this node's Ray session logs (per-actor worker stdout/tracebacks) before
        # the pod is reaped — Ray's node-local logs are deleted with the pod.
        ray_log_sync.sync_bounded(f"signal {signum} (head)")
        sync_debug_artifacts(args.rendezvous_dir, node_id, f"signal {signum} (head)")
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=60)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        ray_stop()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # On iris task retry, a rendezvous file from a previous attempt still points
    # at a now-dead head. Purge before starting the new head.
    if num_tasks > 1 and args.rendezvous_dir:
        clear_rendezvous(args.rendezvous_dir)

    ray_start_head(
        head_ip,
        resolve_ray_spill_target(args.rendezvous_dir, args.ray_spill_backend, args.ray_spill_dir),
    )
    _log("Ray head bootstrap complete; entering rendezvous / cluster-join phase.")

    # Start periodic uploads now that Ray has created its session directory.
    ray_log_sync_stop = ray_log_sync.start_periodic(args.rendezvous_dir)

    if num_tasks > 1:
        if not args.rendezvous_dir:
            raise ValueError(
                "Multi-node iris slice (IRIS_NUM_TASKS>1) requires --rendezvous-dir "
                "(or OT_AGENT_IRIS_RENDEZVOUS_DIR) so worker ranks can find the head IP."
            )
        _log(
            f"[task-runtime] Ray head subprocess returned; writing rendezvous -> {_rendezvous_uri(args.rendezvous_dir)}"
        )
        write_rendezvous(args.rendezvous_dir, head_ip, ray_port, gang_epoch)
        # Re-publish the rendezvous each poll so a late cold-node worker never sees it
        # as "stale" (see wait_for_nodes docstring — prevents the freshness deadlock).
        wait_for_nodes(
            ray_address,
            num_tasks,
            args.cluster_join_timeout,
            rewrite_cb=lambda: write_rendezvous(args.rendezvous_dir, head_ip, ray_port, gang_epoch),
        )
    else:
        _log("Single-node slice: skipping rendezvous and multi-node wait.")

    with ray_metrics_telemetry(head_ip, allocated_ray_port(RayPort.METRICS), CONTROLLER_ROLE):
        env = training_driver_env(derived_gloo_ifname)
        env["RAY_ADDRESS"] = ray_address  # skyrl-train's bare ray.init() attaches here
        env["PYTHONUNBUFFERED"] = "1"

        _log("Launching MarinSkyRL training driver:")
        _log("  " + " ".join(training_driver_command(config_path)))
        sys.stdout.flush()
        sys.stderr.flush()

        # The SIGTERM/SIGINT handler is already installed at the top of run_head; assigning
        # `process` here arms its driver-teardown path (the closure reads this value).
        process = launch_training_driver(config_path, env)
        driver_activity = DriverOutputActivity()
        output_thread = start_driver_output_tee(process, driver_activity)
        if args.driver_liveness_timeout > 0:
            _log(f"Driver liveness watchdog armed: {args.driver_liveness_timeout:g}s maximum stdout/stderr silence")
        wait_result = wait_for_driver(process, driver_activity, args.driver_liveness_timeout)
        if isinstance(wait_result, DriverStalled):
            reason = f"driver stalled after {wait_result.silence:.0f}s without stdout/stderr progress (head rank 0)"
            _log(reason)

            def persist_stall_artifacts() -> None:
                capture_termination_artifacts(args.rendezvous_dir, reason)
                persist_runtime_artifacts(reason)

            _persist_failure_artifacts_bounded(persist_stall_artifacts, timeout=FAILURE_ARTIFACT_TIMEOUT)
            kill_driver_process_group(process)
            output_thread.join(timeout=DRIVER_KILL_TIMEOUT)
            ray_stop()
            return DRIVER_STALLED_EXIT_CODE
        output_thread.join(timeout=DRIVER_KILL_TIMEOUT)
        driver_outcome, outcome_receipt = write_process_outcome(
            "training-driver",
            wait_result.exit_code,
            pid=process.pid,
        )
        exit_code = driver_outcome.public_exit_code
        if driver_outcome.kind is ProcessOutcomeKind.SIGNAL:
            _log(
                f"Training driver terminated by {driver_outcome.signal_name} "
                f"(raw_returncode={driver_outcome.raw_returncode}, exit_code={exit_code}, "
                f"receipt={outcome_receipt})"
            )
        if exit_code != 0:

            def persist_failure_artifacts() -> None:
                reason = f"driver exit_code={exit_code} (head rank 0)"
                capture_termination_artifacts(args.rendezvous_dir, reason)
                persist_runtime_artifacts(reason)

            _persist_failure_artifacts_bounded(
                persist_failure_artifacts,
                timeout=FAILURE_ARTIFACT_TIMEOUT,
            )
            ray_stop()
            return exit_code
        # Final flush of this node's Ray session logs before teardown reaps them.
        persist_runtime_artifacts(f"driver exit_code={exit_code} (head)")
    # Signal workers to unpark, then tear down.
    if args.rendezvous_dir and num_tasks > 1:
        write_head_result(args.rendezvous_dir, gang_epoch)
    ray_stop()
    return exit_code


def run_worker(args: argparse.Namespace, *, debug_mode: DebugMode = DebugMode.LIGHT) -> int:
    ensure_task_debug_artifact_directory(debug_mode)
    worker_start = time.time()
    rank = _rank()
    num_tasks = _num_tasks()
    node_ip = _own_ip()
    node_id = f"rank{rank}-{socket.gethostname()}"
    _log(f"ROLE=worker rank={rank}/{num_tasks} node_ip={node_ip}")
    ray_log_sync = RayLogSyncSession(args.ray_log_dir, node_id)

    if not args.rendezvous_dir:
        raise ValueError(
            "Worker rank requires --rendezvous-dir (or OT_AGENT_IRIS_RENDEZVOUS_DIR) to discover the head IP."
        )

    payload = poll_rendezvous(args.rendezvous_dir, args.rendezvous_timeout, min_written_at=worker_start)
    python_version, ray_version = _runtime_versions()
    payload = validate_rendezvous_runtime(
        payload,
        worker_node=node_id,
        python_version=python_version,
        ray_version=ray_version,
    )
    head_ip = payload.head_ip
    ray_port = payload.port
    ray_address = f"{head_ip}:{ray_port}"

    ray_start_worker(
        head_ip,
        ray_port,
        node_ip,
        resolve_ray_spill_target(args.rendezvous_dir, args.ray_spill_backend, args.ray_spill_dir),
    )
    wait_for_nodes(ray_address, num_tasks, args.cluster_join_timeout)
    _log(f"Worker rank {rank} joined Ray cluster at {ray_address}; parking until the head finishes.")

    # Periodic Ray session-log -> object-store sync for THIS worker node (the policy/rollout
    # actors on this node log to its local /tmp/ray session, deleted with the pod on GC).
    ray_log_sync_stop = ray_log_sync.start_periodic(args.rendezvous_dir)

    stop = threading.Event()
    termination_signal: int | None = None

    def _shutdown(signum, _frame) -> None:
        nonlocal termination_signal
        termination_signal = termination_signal or signum
        _log(f"Worker rank {rank} received signal {signum}; stopping Ray.")
        # A SIGTERM on a worker node is often a k8s eviction/OOM of that node (it
        # hosts the training actors' GPUs); capture its disk/GPU state before reap.
        capture_termination_artifacts(args.rendezvous_dir, f"signal {signum} (worker rank {rank})")
        # Flush this node's per-actor Ray worker logs before the pod is reaped.
        ray_log_sync.sync_bounded(f"signal {signum} (worker rank {rank})")
        sync_debug_artifacts(args.rendezvous_dir, node_id, f"signal {signum} (worker rank {rank})")
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Exit successfully only after the current head attempt publishes success.
    # A signal without that result tears down Ray but remains a non-successful
    # task outcome so Iris cannot mistake a bounced gang peer for completed work.
    head_completed = False
    with ray_metrics_telemetry(node_ip, allocated_ray_port(RayPort.METRICS), WORKER_ROLE):
        while True:
            if head_succeeded(args.rendezvous_dir, payload.gang_epoch):
                head_completed = True
                _log(f"Worker rank {rank} saw successful head result; shutting down.")
                break
            if stop.is_set():
                break
            time.sleep(POLL_INTERVAL)
        # Final flush of this worker node's Ray session logs before Ray teardown.
        ray_log_sync_stop.set()
        ray_log_sync.sync_bounded(f"worker rank {rank} teardown")
        sync_debug_artifacts(args.rendezvous_dir, node_id, f"worker rank {rank} teardown")
    ray_stop()
    if head_completed:
        return 0
    assert termination_signal is not None
    return 128 + termination_signal


def _container(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _json_list(value: Any) -> str:
    values = _container(value)
    if not isinstance(values, list):
        raise ValueError("launch config data fields must be lists")
    return json.dumps(values, sort_keys=True)


def _local_teacher_models(skyrl: DictConfig) -> tuple[TeacherModelSpec, ...]:
    plan = compile_distillation_plan_from_config(skyrl)
    if plan is None:
        return ()
    return tuple(teacher.model for teacher in plan.teachers if teacher.source is TeacherSource.LOCAL_INFERENCE)


def _runtime_namespace(config: DictConfig) -> argparse.Namespace:
    """Project the typed launch document into the controller's private runtime state."""
    model = config.inputs.model
    skyrl = config.skyrl
    checkpoint_export = config.run.mode == RunMode.CHECKPOINT_EXPORT
    task_env = {str(key): str(value) for key, value in config.runtime.task_env.items()}
    model_uri = str(model.uri)
    model_identity = str(model.identity)
    train_data = [] if checkpoint_export else _container(config.inputs.train_data)
    validation_data = [] if checkpoint_export else _container(config.inputs.validation_data)
    data_sources = train_data + validation_data
    generator = skyrl.get("generator", {})
    speculative = generator.get("speculative_decoding", {}) if isinstance(generator, DictConfig) else {}
    draft_mapping = speculative.get("model") if isinstance(speculative, DictConfig) and not checkpoint_export else None
    draft_model = None
    if draft_mapping:
        draft_model = SpeculatorModelConfig.from_mapping(
            _container(draft_mapping), context="generator.speculative_decoding.model"
        )
    offline = task_env.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    prestage_model = model_uri if not is_cloud_uri(model_uri) and (model.chat_template or offline) else ""
    return argparse.Namespace(
        ray_spill_dir=validate_ray_spill_dir(str(config.ray.spill_dir)),
        ray_spill_backend=RaySpillBackend(str(config.ray.spill_backend)),
        rendezvous_dir=str(config.ray.rendezvous_dir),
        ray_log_dir=str(config.ray.log_dir),
        rendezvous_timeout=int(config.ray.rendezvous_timeout),
        cluster_join_timeout=int(config.ray.cluster_join_timeout),
        driver_liveness_timeout=int(config.ray.driver_liveness_timeout),
        run_id=str(config.run.id),
        training_type=None if config.runtime.training_type is None else TrainingType(config.runtime.training_type),
        task_env=task_env,
        terminal_bench_data=_json_list(skyrl.get("data", {}).get("terminal_bench_data", [])),
        data_sources_json=_json_list(data_sources) if data_sources else "",
        prestage_model=prestage_model,
        model_revision=str(skyrl.get("trainer", {}).get("policy", {}).get("model", {}).get("revision") or ""),
        prestage_teacher_models=() if checkpoint_export else _local_teacher_models(skyrl),
        model_source_uri=model_uri if is_cloud_uri(model_uri) else "",
        model_local_path=str(model.local_path),
        model_source_identity=model_identity if is_cloud_uri(model_uri) else "",
        runtime_profile=str(config.runtime.profile),
        rollout_model_loading=RolloutModelLoading(generator.get("model_loading", RolloutModelLoading.STREAM)),
        policy_tokenizer=str(model.tokenizer_uri),
        policy_tokenizer_revision=str(model.tokenizer_revision),
        policy_chat_template=str(model.chat_template or ""),
        draft_model=draft_model,
        draft_model_cache_ttl_days=int(config.artifacts.draft_model_cache_ttl_days),
        draft_model_cache_source_prefix=str(config.artifacts.checkpoint_root),
    )


def _final_config_path(config: DictConfig) -> Path:
    def safe(value: str) -> str:
        return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)

    return (
        Path(tempfile.gettempdir())
        / "marinskyrl"
        / f"{safe(str(config.run.id))}-{safe(str(config.run.attempt_id))}.yaml"
    )


def _write_final_config(
    config: DictConfig,
    *,
    policy_model: PreparedPolicyModel | None,
    policy_tokenizer: PreparedPolicyTokenizer | None,
    draft_model: SpeculatorModelConfig | None,
) -> Path:
    """Persist the post-staging root document consumed by the training subprocess."""
    root = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    original_model_path = str(root.skyrl.get("trainer", {}).get("policy", {}).get("model", {}).get("path", ""))
    if not original_model_path:
        original_model_path = str(root.inputs.model.uri)
    values = TaskLocalSkyRLValues(
        train_data=(),
        validation_data=(),
        terminal_bench_data=(),
        policy_model_path=policy_model.local_path if policy_model else None,
        draft_model_uri=draft_model.source_uri if draft_model else None,
    )
    skyrl = apply_task_local_values(root.skyrl, values)
    if policy_model is not None:
        OmegaConf.update(skyrl, "trainer.policy.model.source_uri", policy_model.source_uri, force_add=True)
        OmegaConf.update(skyrl, "trainer.policy.model.source_identity", policy_model.source_identity, force_add=True)
        OmegaConf.update(
            skyrl,
            "generator.engine_init_kwargs.served_model_name",
            original_model_path.rstrip("/").rsplit("/", 1)[-1],
            force_add=True,
        )
    tokenizer_path = (
        policy_tokenizer.local_path
        if policy_tokenizer is not None
        else (policy_model.local_path if policy_model is not None else None)
    )
    if tokenizer_path is not None:
        for role in ("policy", "ref"):
            OmegaConf.update(
                skyrl,
                f"trainer.{role}.model.tokenizer_path",
                tokenizer_path,
                force_add=True,
            )
            OmegaConf.update(skyrl, f"trainer.{role}.model.tokenizer_revision", None, force_add=True)
    if draft_model is not None:
        OmegaConf.update(
            skyrl,
            "generator.speculative_decoding.model.source_identity",
            draft_model.source_identity,
            force_add=True,
        )
    root.skyrl = skyrl
    destination = _final_config_path(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(root, destination, resolve=True)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Ray and run one typed MarinSkyRL launch config.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def _print_env_snapshot() -> None:
    _log("environment snapshot:")
    for key in (
        "IRIS_TASK_ID",
        "IRIS_NUM_TASKS",
        "IRIS_ADVERTISE_HOST",
        "RAY_ADDRESS",
        "SKYRL_HOME",
        "PYTHONPATH",
        "HF_HOME",
        "NUM_INFERENCE_ENGINES",
        "POLICY_NUM_NODES",
        "TENSOR_PARALLEL_SIZE",
    ):
        print(f"  {key}={os.environ.get(key, '<unset>')}", flush=True)


def main() -> None:
    validate_bundled_runtime()
    cli_args = parse_args()
    config_path = Path(materialize_launch_config(str(cli_args.config)))
    launch_config = load_launch_config(config_path)
    args = _runtime_namespace(launch_config)
    os.environ.update(args.task_env)
    _print_env_snapshot()
    # Pin virtual-hosted S3 addressing for the boto3 path (Ray object-spill IO workers)
    # BEFORE any `ray start`, on head + every worker — CoreWeave R2 rejects path-style.
    _pin_boto3_s3_addressing_style()
    # Derive GLOO_SOCKET_IFNAME from the pod IP BEFORE any torch/gloo init, on head
    # and every worker. Must precede `ray start`: Ray actors inherit this env, and
    # the gloo mesh is built long after, at the first checkpoint save. The derived
    # name is node-local; training_driver_env keeps it out of the driver, which
    # would otherwise broadcast the head's name to every node.
    derived_gloo_ifname = pin_socket_ifname()
    # Resolve before Ray starts so its actors inherit this task's telemetry settings.
    export_telemetry_environment(args.run_id, args.training_type)
    # Ensure the NCCL flight-recorder dump dir exists on THIS node BEFORE any torch/NCCL
    # init, so a collective-timeout FR dump actually writes. See ensure_fr_dump_dir.
    ensure_fr_dump_dir()
    # The launcher projects this manager-owned path before task runtime starts.
    debug_artifact_root = os.environ.get(DEBUG_ARTIFACT_DIR_ENV)
    if debug_artifact_root:
        ensure_debug_artifact_directories(debug_artifact_root)
        initialize_process_diagnostics(f"task-runtime-rank{_rank()}")
    # Stage task datasets on THIS node before Ray bootstrap (head + every worker).
    # Without this, only rank-0 has the extracted tasks and the rollout workers die
    # with FileNotFoundError on task.toml. See stage_task_data docstring.
    if args.data_sources_json:
        materialize_data_sources(args.data_sources_json)
    if args.terminal_bench_data:
        stage_task_data(args.terminal_bench_data, role="terminal-bench sidechannel")
    policy_model = prepare_policy_model(args)
    policy_local_path = policy_model.local_path if policy_model is not None else None
    policy_tokenizer = prepare_policy_tokenizer(args)
    if args.prestage_model:
        stage_model(
            args.prestage_model,
            revision=(args.model_revision or None),
        )
    for teacher_model in args.prestage_teacher_models:
        stage_model(teacher_model.path, revision=teacher_model.revision)
    draft_model = args.draft_model
    if draft_model is not None:
        draft_model = prepare_draft_model(
            draft_model,
            cache_ttl_days=args.draft_model_cache_ttl_days,
            cache_source_prefix=args.draft_model_cache_source_prefix,
        )
    # Force the policy chat template onto staged metadata or a local model on every
    # node before Ray; the training driver's tokenizer may load anywhere.
    if args.policy_chat_template:
        model_path = (
            policy_tokenizer.local_path
            if policy_tokenizer is not None
            else policy_local_path or policy_chat_template_model(args.prestage_model, args.model_local_path)
        )
        apply_policy_chat_template(model_path, args.policy_chat_template)
    resolve_ray_ports()
    rank = _rank()
    debug_mode = DebugMode(str(launch_config.skyrl.get("trainer", {}).get("debug_mode", DebugMode.LIGHT.value)))
    if rank == 0:
        final_config_path = _write_final_config(
            launch_config,
            policy_model=policy_model,
            policy_tokenizer=policy_tokenizer,
            draft_model=draft_model,
        )
        exit_code = run_head(args, final_config_path, derived_gloo_ifname, debug_mode=debug_mode)
    else:
        exit_code = run_worker(args, debug_mode=debug_mode)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
