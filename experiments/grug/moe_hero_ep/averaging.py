# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory, same-region Hero checkpoint preparation.

Run ``uv run python -m experiments.grug.moe_hero_ep.averaging --config request.json``.
The JSON is an AverageConfig encoded with draccus. No job is launched by this CLI.
A single worker streams native checkpoint arrays; its host buffers and TensorStore
cache are bounded independently of model size. Source checkpoints are never changed.
Source chunk grids must agree; the budget must hold one native FP32 chunk.
"""

import argparse
import base64
import dataclasses
import datetime
import hashlib
import ipaddress
import itertools
import json
import math
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import draccus
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
import yaml
from iris.cluster.platforms.k8s.service import ApiException, CloudK8sService
from iris.cluster.platforms.k8s.types import K8sResource, KubectlError
from levanter.checkpoint_manifest import CheckpointArray, CheckpointManifest, build_manifest
from levanter.tensorstore_serialization import build_kvstore_spec
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import data_buckets, get_bucket_location, region_from_metadata
from rigging.filesystem.s3_compat import configure_fsspec_s3, s3_credentials, s3_endpoint
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig

_ROUTER = "stacked_blocks/stacked/mlp/router_bias"
_PENDING = "pending_qb_betas"
_DONORS = (
    ("<|start_header_id|>", 128006, " role", 3560),
    ("<|end_header_id|>", 128007, " message", 1984),
    ("<|start_think|>", 128002, " think", 1781),
    ("<|end_think|>", 128003, " answer", 4320),
    ("<|eot_id|>", 128009, "<|end_of_text|>", 128001),
)


@dataclasses.dataclass(frozen=True)
class CheckpointPin:
    path: str
    metadata_digest: str
    config_path: str
    config_digest: str
    tokenizer_revision: str
    manifest_digest: str
    tokenizer_revision_evidence: str
    """Durable revision evidence or explicit accepted assumption when the saved config pins only its name."""


@dataclasses.dataclass(frozen=True)
class LrSample:
    # Telemetry is indexed by the completed update, as opposed to scheduler index.
    completed_update: int
    muon_lr: float
    adam_lr: float


@dataclasses.dataclass(frozen=True)
class TokenizerPin:
    name: str
    revision: str


@dataclasses.dataclass(frozen=True)
class AverageConfig:
    checkpoints: tuple[CheckpointPin, ...]
    output: str
    temporary_storage: str
    tokenizer: TokenizerPin
    telemetry: tuple[LrSample, ...]
    source_revision: str
    forward_context_length: int
    # The current Hero optimizer starts count=0: completed update k uses index k-1.
    scheduler_index_offset: int = -1
    max_chunk_bytes: int = 512 * 1024 * 1024
    initialize_donors: bool = False


@dataclasses.dataclass(frozen=True)
class PublishedNodeLocation:
    region: str
    topology_region: str
    node_cidr: str
    cluster_ca_sha256: str
    source_url: str
    verified_on: str


# Provider-published exclusive AZ node range, checked against the live CKS CA on
# 2026-10-19. CA binding prevents another provider's reused private CIDR from
# masquerading as this location. Unknown clusters/ranges require new evidence.
_COREWEAVE_NODE_LOCATIONS = (
    PublishedNodeLocation(
        region="us-east-02a",
        topology_region="us-east-02",
        node_cidr="10.184.192.0/18",
        cluster_ca_sha256="5ddf0baf2d77aaeb92effe3b40439cdcabd2b8194ff4f25f27700af16ef70259",
        source_url="https://docs.coreweave.com/platform/regions/us-east/us-east-02#network-information",
        verified_on="2026-10-19",
    ),
)


class RegionProbe(Protocol):
    worker_identity: dict[str, str]

    def worker_region(self) -> str: ...

    def storage_region(self, path: str) -> str: ...


class ProviderRegionProbe:
    """Resolve placement from provider metadata, never from bucket/cluster names."""

    def __init__(self, service_account_directory: Path = Path("/var/run/secrets/kubernetes.io/serviceaccount")):
        self.service_account_directory = service_account_directory
        self.worker_identity: dict[str, str] = {}

    def worker_region(self) -> str:
        if os.environ.get("KUBERNETES_SERVICE_HOST") or self.service_account_directory.exists():
            return self._kubernetes_region()
        region = region_from_metadata()
        if not region:
            raise ValueError("Physical averaging worker region is unavailable; refusing model I/O")
        self.worker_identity = {"provider": "gcp-instance-metadata", "hostname": socket.gethostname()}
        return region.lower()

    def _kubernetes_region(self) -> str:
        # The API authenticates this mounted token. Its bound pod identity avoids
        # trusting task-supplied region/node environment variables or cluster names.
        account = self.service_account_directory
        token = (account / "token").read_text().strip()
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        bound = claims["kubernetes.io"]
        namespace = (account / "namespace").read_text().strip()
        if bound["namespace"] != namespace:
            raise ValueError("Service-account token namespace differs from its mounted namespace")
        identity = bound["pod"]
        service = CloudK8sService(namespace=namespace, timeout=15)
        pod = service.get_json(K8sResource.PODS, identity["name"])
        if pod is None or pod["metadata"]["uid"] != identity["uid"]:
            raise ValueError("Authenticated averaging pod identity could not be verified")
        node_name = pod["spec"]["nodeName"]
        cluster_ca = hashlib.sha256((account / "ca.crt").read_bytes()).hexdigest()
        self.worker_identity = {
            "namespace": namespace,
            "pod": identity["name"],
            "pod_uid": identity["uid"],
            "node": node_name,
            "cluster_ca_sha256": cluster_ca,
        }
        try:
            node = service.get_json(K8sResource.NODES, node_name)
        except KubectlError as error:
            # Default task accounts can GET their pod but cannot GET nodes. A
            # transport/authentication failure is not evidence of that capability.
            cause = error.__cause__
            if not isinstance(cause, ApiException) or cause.status != 403:
                raise
            node = None
        labels = node["metadata"]["labels"] if node is not None else {}
        node_region = labels.get("topology.kubernetes.io/region") or labels.get(
            "failure-domain.beta.kubernetes.io/region"
        )
        locations = tuple(location for location in _COREWEAVE_NODE_LOCATIONS if location.cluster_ca_sha256 == cluster_ca)
        if node_region and (not locations or node_region.lower() in {location.region for location in locations}):
            self.worker_identity.update(provider="kubernetes-node-topology", region=node_region.lower())
            return node_region.lower()
        host_ip = ipaddress.ip_address(pod["status"]["hostIP"])
        matches = tuple(
            location
            for location in locations
            if host_ip in ipaddress.ip_network(location.node_cidr)
            and (not node_region or node_region.lower() == location.topology_region)
        )
        if len(matches) != 1:
            raise ValueError(
                "No unambiguous provider-published AZ mapping for the authenticated pod hostIP and cluster CA"
            )
        location = matches[0]
        self.worker_identity.update(
            provider="coreweave-published-node-cidr",
            region=location.region,
            host_ip=str(host_ip),
            node_cidr=location.node_cidr,
            location_source=location.source_url,
            location_verified_on=location.verified_on,
        )
        return location.region

    def storage_region(self, path: str) -> str:
        parsed = StoragePath(path)
        if parsed.scheme == "gs":
            return get_bucket_location(path).lower()
        if parsed.scheme == "s3":
            fs, _ = filesystem_for(path)
            response = fs.call_s3("head_bucket", Bucket=parsed.bucket)
            region = response.get("ResponseMetadata", {}).get("HTTPHeaders", {}).get("x-amz-bucket-region")
            if not isinstance(region, str) or not region:
                raise ValueError(f"Storage provider did not report a physical region for {path}")
            return region.lower()
        raise ValueError(f"Remote averaging requires region-verifiable object storage: {path}")


class Coefficients(BaseModel):
    model_config = ConfigDict(frozen=True)
    completed_updates: tuple[int, ...]
    scheduler_indices: tuple[int, ...]
    actual_muon_lr: tuple[float, ...]
    actual_adam_lr: tuple[float, ...]
    target_muon_lr: tuple[float, ...]
    interval_actual_lr_sums: tuple[float, ...]
    interval_target_lr_sums: tuple[float, ...]
    increment_ratios: tuple[float, ...]
    snapshots: tuple[float, ...]


def cooldown_coefficients(
    steps: tuple[int, ...],
    optimizer: GrugMoeMuonHConfig,
    num_train_steps: int,
    telemetry: tuple[LrSample, ...],
    *,
    scheduler_index_offset: int = -1,
) -> Coefficients:
    """Integrate the saved scheduler over completed-update intervals, checking telemetry."""
    if len(steps) < 2 or any(b <= a for a, b in itertools.pairwise(steps)):
        raise ValueError("Need at least two distinct checkpoints ordered by completed update")
    if scheduler_index_offset != -1:
        raise ValueError("Current Hero uses optimizer count k-1 for completed update k")
    updates = np.arange(steps[0], steps[-1] + 1, dtype=np.int64)
    indices = updates + scheduler_index_offset
    if indices[0] < 0:
        raise ValueError("The cooldown window must start after completed update zero")
    muon_fn = optimizer.lr_scheduler(num_train_steps)
    adam_fn = optimizer.lr_scheduler(num_train_steps, override_lr=optimizer.adam_lr)
    # This is scalar scheduler work, with no checkpoint reads and no model-sized allocation.
    muon_values, adam_values = jax.jit(jax.vmap(lambda index: (muon_fn(index), adam_fn(index))))(
        jnp.asarray(indices, dtype=jnp.int32)
    )
    muon, adam = np.asarray(muon_values, dtype=np.float64), np.asarray(adam_values, dtype=np.float64)
    if np.any(muon <= 0) or np.any(adam <= 0) or not np.all(np.isfinite(muon + adam)):
        raise ValueError("Nonpositive or nonfinite actual LR in selected window")
    if not np.allclose(muon / optimizer.learning_rate, adam / optimizer.adam_lr, rtol=1e-6, atol=1e-8):
        raise ValueError("Optimizer groups have different normalized schedules")
    in_window = {
        sample.completed_update: sample for sample in telemetry if steps[0] <= sample.completed_update <= steps[-1]
    }
    if len(in_window) < 3:
        raise ValueError("Need at least three distinct logged LR samples inside the selected window")
    for update, sample in in_window.items():
        index = update - steps[0]
        if not np.allclose([muon[index], adam[index]], [sample.muon_lr, sample.adam_lr], rtol=1e-5, atol=1e-10):
            raise ValueError(f"Reconstructed LR disagrees with telemetry at completed update {update}")
    target = muon[0] * (1 - (updates - steps[0]) / (steps[-1] - steps[0]))
    actual_sums, target_sums = [], []
    for start, end in itertools.pairwise(steps):
        interval = slice(start + 1 - steps[0], end + 1 - steps[0])
        actual_sums.append(float(muon[interval].sum()))
        target_sums.append(float(target[interval].sum()))
    ratios = np.asarray(target_sums) / np.asarray(actual_sums)
    weights = np.concatenate(([1 - ratios[0]], ratios[:-1] - ratios[1:], [ratios[-1]]))
    if np.any(weights < 0) or np.any(ratios > 1) or not math.isclose(float(weights.sum()), 1):
        raise ValueError("Virtual cooldown does not define a convex checkpoint average")
    return Coefficients(
        completed_updates=tuple(updates.tolist()),
        scheduler_indices=tuple(indices.tolist()),
        actual_muon_lr=tuple(muon.tolist()),
        actual_adam_lr=tuple(adam.tolist()),
        target_muon_lr=tuple(target.tolist()),
        interval_actual_lr_sums=tuple(actual_sums),
        interval_target_lr_sums=tuple(target_sums),
        increment_ratios=tuple(ratios.tolist()),
        snapshots=tuple(weights.tolist()),
    )


def _read_json(path: str) -> dict:
    fs, plain = filesystem_for(path)
    with fs.open(plain, "r") as source:
        return json.load(source)


def read_prepared_provenance(checkpoint: str, metadata_digest: str) -> dict:
    """Validate permanent prepared weights and return metadata-bound model provenance.

    This reads only small JSON records, never model arrays. Consumers additionally
    check candidate-specific requirements, including donor initialization for C.
    """
    metadata = _read_json(f"{checkpoint}/metadata.json")
    if metadata.get("is_temporary") is not False or digest(metadata) != metadata_digest:
        raise ValueError("Prepared checkpoint metadata changed or is not permanent")
    provenance = _read_json(f"{checkpoint}/averaging.json")
    if digest(provenance) != metadata["averaging_digest"]:
        raise ValueError("Prepared checkpoint provenance does not match its completion metadata")
    if digest(provenance["model"]) != provenance["model_digest"]:
        raise ValueError("Prepared checkpoint forward configuration digest does not match")
    return provenance


def _write_json(path: str, value: object) -> None:
    fs, plain = filesystem_for(path)
    fs.makedirs(os.path.dirname(plain), exist_ok=True)
    with fs.open(plain, "w") as output:
        json.dump(value, output, indent=2)


def _load_config(pin: CheckpointPin) -> dict:
    fs, plain = filesystem_for(pin.config_path)
    with fs.open(plain, "r") as source:
        config = json.load(source) if pin.config_path.endswith(".json") else yaml.safe_load(source)
    if digest(config) != pin.config_digest:
        raise ValueError(f"Saved run configuration changed: {pin.config_path}")
    return config


def _configure_tensorstore_credentials(paths: list[str]) -> None:
    # TensorStore accepts AWS environment credentials, unlike bucket-routed fsspec.
    # This worker serves one region/backend; reject ambient credentials for another
    # backend rather than silently sign requests with the wrong identity.
    registry = data_buckets()
    stores = {registry[StoragePath(path).bucket].store for path in paths if StoragePath(path).scheme == "s3"}
    if not stores:
        return
    if len(stores) != 1:
        raise ValueError("One streamed worker cannot mix S3 credential backends")
    store = stores.pop()
    credentials = s3_credentials(store)
    if credentials is None:
        raise ValueError("No credentials for the selected object-store backend")
    key, secret = credentials
    if os.environ.get("AWS_ACCESS_KEY_ID", key) != key or os.environ.get("AWS_SECRET_ACCESS_KEY", secret) != secret:
        raise ValueError("Ambient AWS credentials differ from the selected storage backend")
    configure_fsspec_s3(s3_endpoint(store), key=key, secret=secret)


def _kvstore(path: str) -> dict:
    spec = build_kvstore_spec(path)
    parsed = StoragePath(path)
    if parsed.scheme == "s3":
        bucket = data_buckets().get(parsed.bucket)
        if bucket is None:
            raise ValueError(f"S3 bucket is not registered with rigging: {parsed.bucket}")
        endpoint = s3_endpoint(bucket.store)
        scheme, _, host = endpoint.partition("://")
        spec.pop("bucket", None)
        spec["endpoint"] = f"{scheme}://{parsed.bucket}.{host}"
        spec["aws_region"] = bucket.signing_region
    return {"driver": "ocdbt", "base": spec}


def _open_array(root: str, entry: CheckpointArray, context: ts.Context, *, create: bool = False) -> ts.TensorStore:
    spec = {"driver": "zarr3", "kvstore": {**_kvstore(root), "path": entry.path}}
    if create:
        spec["metadata"] = {
            "shape": list(entry.shape),
            "data_type": entry.dtype,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(entry.chunk_shape)}},
        }
    return ts.open(spec, open=not create, create=create, context=context).result()


def _chunk_shape(shape: tuple[int, ...], maximum_elements: int) -> tuple[int, ...]:
    chunks = list(shape)
    while math.prod(chunks) > maximum_elements:
        axis = max(range(len(chunks)), key=chunks.__getitem__)
        chunks[axis] = (chunks[axis] + 1) // 2
    return tuple(chunks)


def _slices(shape: tuple[int, ...], chunks: tuple[int, ...]) -> Iterator[tuple[slice, ...]]:
    for starts in itertools.product(*(range(0, dim, chunk) for dim, chunk in zip(shape, chunks, strict=True))):
        yield tuple(
            slice(start, min(dim, start + chunk)) for start, dim, chunk in zip(starts, shape, chunks, strict=True)
        )


def _parameter_entries(manifest: CheckpointManifest) -> dict[str, CheckpointArray]:
    # Hero and the pipeline checkpoint writer both publish canonical global model paths.
    for prefix in ("master_params/", "train_state/master_params/", "params/", "train_state/params/"):
        entries = {array.path.removeprefix(prefix): array for array in manifest.arrays if array.path.startswith(prefix)}
        if entries:
            return entries
    raise ValueError("No authoritative canonical parameter tree in checkpoint manifest")


def _validate_donors(pin: TokenizerPin) -> None:
    if len(pin.revision) != 40 or any(char not in "0123456789abcdef" for char in pin.revision):
        raise ValueError("Tokenizer revision must be an immutable git SHA")
    tokenizer = AutoTokenizer.from_pretrained(pin.name, revision=pin.revision)
    for text, destination, donor_text, donor in _DONORS:
        if tokenizer.encode(text, add_special_tokens=False) != [destination]:
            raise ValueError(f"Destination token does not match tokenizer: {text}")
        if tokenizer.encode(donor_text, add_special_tokens=False) != [donor]:
            raise ValueError(f"Donor is not the expected single token: {donor_text}")


def average_checkpoints(config: AverageConfig, *, region_probe: RegionProbe | None = None) -> dict:
    """Write a new native FP32 checkpoint, publishing metadata only after all writes finish.

    ``region_probe`` is the provider I/O boundary for local roundtrip tests. Production
    CLI always uses ProviderRegionProbe. Multi-process execution is deliberately rejected:
    this streamed implementation has exactly one physical worker to verify.
    """
    if len(config.source_revision) != 40 or any(char not in "0123456789abcdef" for char in config.source_revision):
        raise ValueError("Averaging implementation source revision must be an immutable Git SHA")
    if jax.process_count() != 1 or int(os.environ.get("IRIS_NUM_TASKS", "1")) != 1:
        raise ValueError("Streamed averaging requires exactly one worker and one Iris task")
    if config.max_chunk_bytes < 4:
        raise ValueError("max_chunk_bytes must hold at least one FP32 element")
    sources = [pin.path.rstrip("/") for pin in config.checkpoints]
    if not sources or len(set(sources)) != len(sources) or (len(sources) == 1 and not config.initialize_donors):
        raise ValueError("Need distinct averaging sources, or one donor-initialization source")
    if config.initialize_donors and len(sources) != 1:
        raise ValueError("Prepare donor candidate C from exactly one selected A or B parent")
    output = config.output.rstrip("/")
    if any(output == source or output.startswith(source + "/") or source.startswith(output + "/") for source in sources):
        raise ValueError("Output must be separate from all source checkpoints")
    probe = region_probe or ProviderRegionProbe()
    worker_region = probe.worker_region().lower()
    regions = {path: probe.storage_region(path).lower() for path in [*sources, config.temporary_storage, output]}
    if not worker_region or any(region != worker_region for region in regions.values()):
        raise ValueError(f"All source/temp/output regions must match worker {worker_region}: {regions}")
    output_fs, output_plain = filesystem_for(output)
    if output_fs.exists(output_plain):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    metadata, run_configs, manifests, parameter_maps = [], [], [], []
    for pin in config.checkpoints:
        meta = _read_json(f"{pin.path}/metadata.json")
        if meta.get("is_temporary") is not False or digest(meta) != pin.metadata_digest:
            raise ValueError(f"Checkpoint is not the pinned permanent snapshot: {pin.path}")
        metadata.append(meta)
        run_configs.append(_load_config(pin))
        raw_manifest = _read_json(f"{pin.path}/manifest.json")
        if digest(raw_manifest) != pin.manifest_digest:
            raise ValueError(f"Checkpoint manifest changed: {pin.path}")
        if not pin.tokenizer_revision_evidence:
            raise ValueError("Each source needs a durable tokenizer revision evidence reference")
        manifest = CheckpointManifest.model_validate(raw_manifest)
        if manifest.format_version != 1 or manifest.array_driver != "zarr3" or manifest.kvstore_driver != "ocdbt":
            raise ValueError("Averaging requires native canonical OCDBT/zarr3 checkpoints")
        manifests.append(manifest)
        parameter_maps.append(_parameter_entries(manifest))
    reference_config = run_configs[0]
    if reference_config["model"]["max_seq_len"] != config.forward_context_length:
        raise ValueError("Source checkpoint is outside the explicitly selected context regime")
    # Compare forward semantics and complete optimizer history settings, not just tensor shapes.
    for run in run_configs:
        if (
            run["model"] != reference_config["model"]
            or run["optimizer"] != reference_config["optimizer"]
            or run["trainer"]["trainer"]["num_train_steps"] != reference_config["trainer"]["trainer"]["num_train_steps"]
        ):
            raise ValueError("Source forward configuration or optimizer/schedule history is incompatible")
        tokenizer = run["data"]
        if tokenizer["tokenizer"] != config.tokenizer.name:
            raise ValueError("Source tokenizer name does not match tokenizer pin")
    if any(pin.tokenizer_revision != config.tokenizer.revision for pin in config.checkpoints):
        raise ValueError("Source tokenizer revision attestations disagree")
    reference = parameter_maps[0]
    for entries in parameter_maps:
        if entries.keys() != reference.keys() or any(
            entries[name].shape != ref.shape for name, ref in reference.items()
        ):
            raise ValueError("Canonical source parameter layouts differ")
        if any(entries[name].chunk_shape != ref.chunk_shape for name, ref in reference.items()):
            raise ValueError("Native source chunk grids differ; rechunk in the source region before averaging")
        for entry in entries.values():
            if entry.dtype not in ("float32", "bfloat16", "float16"):
                raise ValueError("Authoritative parameters must be floating point")
            if "master_params/" in entry.path and entry.dtype != "float32":
                raise ValueError("Master parameters must be FP32")
    largest_chunk_bytes = max(math.prod(entry.chunk_shape) * 4 for entry in reference.values())
    if largest_chunk_bytes > config.max_chunk_bytes:
        raise ValueError(
            f"max_chunk_bytes={config.max_chunk_bytes} cannot hold the largest native FP32 chunk "
            f"({largest_chunk_bytes} bytes); raise the explicit worker memory budget"
        )
    if _ROUTER not in reference or "output_proj" not in reference or "token_embed" not in reference:
        raise ValueError("Sources do not expose the current Hero canonical parameter tree")
    optimizer_values = dict(reference_config["optimizer"])
    optimizer_type = optimizer_values.pop("type", "grug_moe_hero_ep_muonh_v1")
    if optimizer_type != "grug_moe_hero_ep_muonh_v1":
        raise ValueError("This implementation supports only the current Hero MuonH optimizer history")
    optimizer = draccus.decode(GrugMoeMuonHConfig, optimizer_values)
    coefficients = (
        cooldown_coefficients(
            tuple(int(meta["step"]) for meta in metadata),
            optimizer,
            reference_config["trainer"]["trainer"]["num_train_steps"],
            config.telemetry,
            scheduler_index_offset=config.scheduler_index_offset,
        )
        if len(sources) > 1
        else None
    )
    snapshot_weights = coefficients.snapshots if coefficients else (1.0,)
    if config.initialize_donors:
        _validate_donors(config.tokenizer)
        if reference["output_proj"].shape[-1] <= max(row[1] for row in _DONORS):
            raise ValueError("Output vocabulary does not contain destination IDs")

    # No model arrays are opened before region, permanent-pin, semantics and LR gates.
    _configure_tensorstore_credentials([*sources, output])
    context = ts.Context({"cache_pool": {"total_bytes_limit": config.max_chunk_bytes}})
    latest_pending_entry = next(
        array for array in manifests[-1].arrays if array.path in (_PENDING, f"train_state/{_PENDING}")
    )
    pending_store = _open_array(sources[-1], latest_pending_entry, context)
    output_entries, norms = [], {}
    for name, ref in reference.items():
        # Read a complete native source chunk once. Slicing a global-shaped grid
        # can intersect dozens of native chunks and repeatedly decompress each.
        chunks = ref.chunk_shape
        target = CheckpointArray(path=f"master_params/{name}", shape=ref.shape, dtype="float32", chunk_shape=chunks)
        output_entries.append(target)
        destination = _open_array(output, target, context, create=True)
        stores = [
            _open_array(source, entries[name], context) for source, entries in zip(sources, parameter_maps, strict=True)
        ]
        source_norms = np.zeros(len(sources), dtype=np.float64)
        output_norm = 0.0
        for index in _slices(ref.shape, chunks):
            accumulator = np.zeros(tuple(s.stop - s.start for s in index), dtype=np.float32)
            if name == _ROUTER:
                # apply_qb_betas sets bias = -pending - mean(-pending); read complete
                # expert rows for centering even when output chunks divide that axis.
                beta = np.asarray(pending_store[(*index[:-1], slice(None))].read().result(), dtype=np.float32)
                effective = -beta + beta.mean(axis=-1, keepdims=True)
                accumulator[...] = effective[(slice(None),) * (beta.ndim - 1) + (index[-1],)]
            else:
                for position, (store, weight) in enumerate(zip(stores, snapshot_weights, strict=True)):
                    incoming = np.asarray(store[index].read().result(), dtype=np.float32)
                    if not np.all(np.isfinite(incoming)):
                        raise ValueError(f"Nonfinite source parameter: {name}")
                    source_norms[position] += float(
                        np.einsum("i,i->", incoming.ravel(), incoming.ravel(), dtype=np.float64)
                    )
                    np.multiply(incoming, np.float32(weight), out=incoming)
                    accumulator += incoming
                    del incoming
                if config.initialize_donors and name == "output_proj":
                    for _, token, _, donor in _DONORS:
                        if index[-1].start <= token < index[-1].stop:
                            donor_index = (*index[:-1], slice(donor, donor + 1))
                            column = np.zeros((*accumulator.shape[:-1], 1), dtype=np.float32)
                            for store, weight in zip(stores, snapshot_weights, strict=True):
                                column += np.asarray(store[donor_index].read().result(), dtype=np.float32) * np.float32(
                                    weight
                                )
                            accumulator[..., token - index[-1].start] = column[..., 0]
            if not np.all(np.isfinite(accumulator)):
                raise ValueError(f"Nonfinite merged parameter: {name}")
            output_norm += float(np.einsum("i,i->", accumulator.ravel(), accumulator.ravel(), dtype=np.float64))
            destination[index].write(accumulator).result()
            del accumulator
        norms[name] = {
            "source_l2": None if name == _ROUTER else np.sqrt(source_norms).tolist(),
            "output_l2": math.sqrt(output_norm),
        }
    # Preserve latest pending control state: restore_weights reapplies the same effective bias.
    pending_target = latest_pending_entry.model_copy(
        update={"path": _PENDING, "chunk_shape": _chunk_shape(latest_pending_entry.shape, config.max_chunk_bytes // 4)}
    )
    output_entries.append(pending_target)
    pending_output = _open_array(output, pending_target, context, create=True)
    for index in _slices(pending_target.shape, pending_target.chunk_shape):
        pending_output[index].write(pending_store[index].read().result()).result()
    output_manifest = build_manifest(output_entries, array_driver="zarr3", kvstore_driver="ocdbt")
    provenance = {
        "request": dataclasses.asdict(config),
        "candidate": "C" if config.initialize_donors else "B",
        "selected_parent": dataclasses.asdict(config.checkpoints[0]) if config.initialize_donors else None,
        "source_pt_step": int(metadata[-1]["step"]),
        "model": reference_config["model"],
        "model_digest": digest(reference_config["model"]),
        "tokenizer_pin": dataclasses.asdict(config.tokenizer),
        "tokenizer_revision_evidence": [pin.tokenizer_revision_evidence for pin in config.checkpoints],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coefficients": coefficients.model_dump() if coefficients else {"snapshots": [1.0]},
        "source_metadata": metadata,
        "worker_region": worker_region,
        "worker_identity": probe.worker_identity,
        "storage_regions": regions,
        "router_bias_policy": "latest effective pending QB bias; pending retained for idempotent weights restore",
        "norms": norms,
        "donors": list(_DONORS) if config.initialize_donors else [],
        "max_chunk_bytes": config.max_chunk_bytes,
        "largest_source_storage_chunk_bytes": largest_chunk_bytes,
        "estimated_array_buffer_peak_bytes": config.max_chunk_bytes + 4 * largest_chunk_bytes,
        "worker_memory_requirement": "Reserve at least 8 GiB including Python/JAX/TensorStore runtime and I/O overhead",
        "memory_contract": (
            "native-aligned reads; one accumulator and one incoming FP32 chunk; cache bounded by max_chunk_bytes; "
            "peak estimate additionally reserves one decompression chunk and one write staging chunk; "
            "runtime/allocator overhead is additional; router rows span all experts"
        ),
    }
    _write_json(f"{output}/manifest.json", output_manifest.model_dump())
    _write_json(f"{output}/averaging.json", provenance)
    result = {
        "step": metadata[-1]["step"],
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "is_temporary": False,
        "averaging_digest": digest(provenance),
        "weights_only": True,
    }
    _write_json(f"{output}/metadata.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as source:
        config = draccus.decode(AverageConfig, json.load(source))
    print(json.dumps(average_checkpoints(config), indent=2))


if __name__ == "__main__":
    main()
