# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicitly bounded exploratory SFT, separate from formal pilot qualification.

Prepare with ``python -m ...sft_campaign request.json --review review.json``.
No training or tokenization is submitted by preparation. Completed chat caches
must already exist. Native retries restore their latest completed compact state;
a new request may extend a pinned parent's schedule without resetting Adam.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import draccus
import jax
import numpy as np
from fray.types import ResourceConfig
from haliax.jax_utils import broadcast_one_to_all
from jax.experimental.multihost_utils import sync_global_devices
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.distributed import DistributedConfig
from marin.execution.lazy import StepContext, lower
from marin.execution.step_runner import StepRunner
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.format_eval import AcquisitionDiagnostic, AcquisitionPanel
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.sft import BaseReference, SftConfig, sft_manifest, validate_prepared_base
from experiments.grug.moe_hero_ep.sft_launch import (
    HeroSftModel,
    HeroSftRunConfig,
    SftInputs,
    _run_sft_local,
    build_chat_sft_inputs,
)
from experiments.grug.moe_hero_ep.train import _apply_hero_ep_runtime_defaults
from experiments.sft.launcher import _prebuilt_chat_data_config

_MAX_ATTEMPTS = 64
_MAX_ANCESTRY = 8
_BROADCAST_BYTES = 65536


@dataclass(frozen=True)
class RecoveryParent:
    request_uri: str
    request_sha256: str
    request_digest: str
    path: str
    step: int
    metadata_digest: str
    counters: dict[str, int]


@dataclass(frozen=True)
class CompletedCheckpoint:
    path: str
    step: int
    metadata_digest: str
    counters: dict[str, int]
    sft_config: SftConfig


@dataclass(frozen=True)
class SftCampaign:
    """Immutable inputs for one exploratory segment; limits are cumulative."""

    name: str
    version: str
    source_revision: str
    reference: BaseReference
    model: GrugModelConfig
    sft: SftConfig
    data_root: str
    data_manifest: str
    data_manifest_digest: str
    acquisition_panel: str
    acquisition_panel_digest: str
    cache_prefix: str
    output_root: str
    resources: ResourceConfig
    batch_size: int
    expert_axis_size: int
    context_axis_size: int
    replica_axis_size: int
    max_processed_tokens: int
    maximum_metrics: tuple[tuple[str, float], ...]
    recovery_parent: RecoveryParent | None = None

    def __post_init__(self):
        if self.model.max_seq_len != 16384 or self.model.yarn is not None:
            raise ValueError("Exploratory format acquisition retains the source 16K configuration")
        if self.max_processed_tokens < 1 or self.batch_size < 1 or not self.maximum_metrics:
            raise ValueError("Declare cumulative token/update limits and numerical abort bounds")
        if self.max_processed_tokens > self.sft.num_steps * self.batch_size * self.model.max_seq_len:
            raise ValueError("Token cap exceeds the declared update and batch scope")


def read_campaign(path: str) -> tuple[SftCampaign, bytes]:
    raw = StoragePath(path).read_bytes()
    return draccus.decode(SftCampaign, json.loads(raw)), raw


def _json(path: StoragePath) -> dict:
    return json.loads(path.read_text())


def validate_continuation(previous: SftCampaign, current: SftCampaign) -> None:
    """Permit only cumulative horizon/cap extension and a new execution identity."""
    if previous.output_root == current.output_root:
        raise ValueError("A continuation request needs a fresh output root")
    previous_values = draccus.encode(previous)
    current_values = draccus.encode(current)
    for field in ("name", "version", "source_revision", "output_root", "recovery_parent", "max_processed_tokens"):
        previous_values.pop(field)
        current_values.pop(field)
    previous_values["sft"]["num_steps"] = current.sft.num_steps
    if previous_values != current_values:
        raise ValueError("Continuation changed weights, data, packing, optimizer or peak/warmup settings")
    if current.sft.num_steps < previous.sft.num_steps or current.max_processed_tokens < previous.max_processed_tokens:
        raise ValueError("Continuation cannot shrink its cumulative schedule or token cap")


def latest_completed_checkpoint(campaign: SftCampaign, depth: int = 0) -> CompletedCheckpoint | None:
    """Prefer own completed checkpoints, otherwise resolve the exact pinned parent.

    Only permanent metadata is read here. restore_sft_checkpoint subsequently
    validates and restores compact parameters, full Adam, step and token counters.
    """
    root = StoragePath(campaign.output_root)
    records = (root / "attempt-records" / "*.json").glob()
    if len(records) > _MAX_ATTEMPTS:
        raise ValueError("Attempt discovery exceeded the bounded campaign limit")
    identity = digest(draccus.encode(campaign))
    manifest = json.loads(json.dumps(sft_manifest(campaign.reference, campaign.sft)))
    candidates: list[CompletedCheckpoint] = []
    for path in records:
        record = _json(path)
        uid = record["leader_attempt_uid"]
        if not uid.isalnum() or len(uid) > 64:
            raise ValueError("Invalid attempt UID")
        destination = str(root / "attempts" / uid)
        if record["request_digest"] != identity or record["destination"] != destination:
            raise ValueError("Attempt belongs to another request or output root")
        for step in range(1, campaign.sft.num_steps + 1):
            checkpoint = StoragePath(destination) / "adapters" / f"step-{step}"
            metadata_path = checkpoint / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = _json(metadata_path)
            if metadata.get("is_temporary") is not False:
                continue
            counts = metadata.get("hero_sft_counters", {})
            if metadata.get("hero_sft") != manifest or metadata.get("step") != step or counts.get("sft_step") != step:
                raise ValueError("Permanent compact checkpoint identity/counters changed")
            if (
                not 0
                < counts["supervised_tokens"]
                <= counts["processed_tokens"]
                <= min(campaign.max_processed_tokens, step * campaign.batch_size * campaign.model.max_seq_len)
            ):
                raise ValueError("Compact counters exceed the cumulative campaign scope")
            candidates.append(
                CompletedCheckpoint(
                    path=str(checkpoint),
                    step=step,
                    metadata_digest=digest(metadata),
                    counters=counts,
                    sft_config=campaign.sft,
                )
            )
    if candidates:
        latest = max(candidates, key=lambda row: (row.step, row.path))
        if len({digest(row.counters) for row in candidates if row.step == latest.step}) != 1:
            raise ValueError("Same-step checkpoints disagree on the deterministic data cursor")
        return latest
    parent = campaign.recovery_parent
    if parent is None:
        return None
    if depth >= _MAX_ANCESTRY:
        raise ValueError("Recovery ancestry exceeds the bounded campaign limit")
    previous, raw = read_campaign(parent.request_uri)
    if hashlib.sha256(raw).hexdigest() != parent.request_sha256 or digest(json.loads(raw)) != parent.request_digest:
        raise ValueError("Recovery parent request bytes changed")
    validate_continuation(previous, campaign)
    selected = latest_completed_checkpoint(previous, depth + 1)
    if (
        selected is None
        or selected.path != parent.path
        or selected.step != parent.step
        or selected.metadata_digest != parent.metadata_digest
        or selected.counters != parent.counters
    ):
        raise ValueError("Latest parent checkpoint differs from the reviewed full-state boundary")
    return selected


def campaign_inputs(campaign: SftCampaign) -> SftInputs:
    """Validate C/data and construct the shared CPU chat-tokenization dependencies."""
    reference = campaign.reference
    validate_prepared_base(reference, campaign.model, campaign.sft)
    manifest = _json(StoragePath(campaign.data_manifest))
    if (
        manifest.get("manifest_digest") != campaign.data_manifest_digest
        or digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
        != campaign.data_manifest_digest
    ):
        raise ValueError("Prepared data manifest changed")
    if (
        manifest["tokenizer_revision"] != reference.tokenizer_revision
        or manifest["recipe"]["tokenizer_path"] != reference.tokenizer
    ):
        raise ValueError("Prepared training tokenizer differs from C")
    panel = AcquisitionPanel.model_validate_json(StoragePath(campaign.acquisition_panel).read_text())
    if (panel.panel_digest, panel.source_artifact_digest, panel.tokenizer_revision) != (
        campaign.acquisition_panel_digest,
        campaign.data_manifest_digest,
        reference.tokenizer_revision,
    ):
        raise ValueError("Acquisition panel differs from the training-source/tokenizer pins")
    for slug, source in manifest["source_outputs"].items():
        if "train" not in source:
            continue
        path = StoragePath(campaign.data_root) / "train" / f"{slug}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["train"]["sha256"]:
            raise ValueError(f"Prepared training bytes changed for {slug}")
    backend = HeroSftModel(
        reference,
        campaign.model,
        campaign.sft,
        campaign.expert_axis_size,
        campaign.context_axis_size,
        campaign.replica_axis_size,
        campaign.max_processed_tokens,
        1,
        campaign.maximum_metrics,
        AcquisitionDiagnostic.from_panel(panel),
        campaign.data_manifest_digest,
        campaign.acquisition_panel_digest,
    )
    return build_chat_sft_inputs(
        name=campaign.name,
        version=campaign.version,
        backend=backend,
        batch_size=campaign.batch_size,
        data_root=campaign.data_root,
        data_manifest_digest=campaign.data_manifest_digest,
        manifest=manifest,
    )


def build_caches(campaign: SftCampaign) -> None:
    """Materialize chat caches on CPU into an explicitly local artifact prefix.

    Publish completed files separately with fsutil; change cache_prefix to that
    verified published prefix before request review. GPU workers never tokenize.
    """
    prefix = StoragePath(campaign.cache_prefix)
    if "://" in campaign.cache_prefix:
        raise ValueError("Build caches under a local prefix, then publish and verify completed files")
    inputs = campaign_inputs(campaign)
    previous = os.environ.get("MARIN_PREFIX")
    os.environ["MARIN_PREFIX"] = str(prefix)
    try:
        StepRunner().run([lower(cache) for cache in inputs.caches], max_concurrent=1)
    finally:
        if previous is None:
            os.environ.pop("MARIN_PREFIX", None)
        else:
            os.environ["MARIN_PREFIX"] = previous


def build_campaign(campaign: SftCampaign) -> tuple[HeroSftRunConfig, list[dict]]:
    """Resolve only completed, exact-fingerprint caches for a GPU configuration."""
    inputs = campaign_inputs(campaign)
    backend, spec, caches = inputs.backend, inputs.spec, inputs.caches
    paths = [cache.path(campaign.cache_prefix) for cache in caches]
    records = []
    for cache, path in zip(caches, paths, strict=True):
        if StatusFile(path, worker_id="cache-check").status != STATUS_SUCCESS:
            raise ValueError("Build the shared chat caches on CPU before training")
        actual = TokenizedCache.raw_load(path)
        if actual.record is None or actual.record.fingerprint != cache.fingerprint():
            raise ValueError("A completed shared chat cache with the exact recipe is required")
        if actual.tokenizer != backend.tokenizer_cache_key() or actual.num_train_tokens <= 0:
            raise ValueError("Completed cache tokenizer/counts differ")
        records.append({"path": path, "fingerprint": cache.fingerprint(), "num_train_tokens": actual.num_train_tokens})
    data = _prebuilt_chat_data_config(spec, paths, backend.tokenizer_cache_key())
    ctx = StepContext.for_run(output_path=campaign.output_root, prefix=campaign.cache_prefix, deps=())
    return backend.build_train_config(ctx, spec, data, campaign.resources, campaign.sft.num_steps), records


def campaign_review(campaign: SftCampaign, raw: bytes) -> dict:
    config, caches = build_campaign(campaign)
    resume = latest_completed_checkpoint(campaign)
    return {
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "request_digest": digest(draccus.encode(campaign)),
        "resume": dataclasses.asdict(resume) if resume is not None else None,
        "cache_records": caches,
        "runtime_config": draccus.encode(config),
        "submission_requested": False,
        "scope": "Exploratory learning only; no heldout-quality, serving or formal pilot qualification",
    }


def run_attempt(campaign: SftCampaign, raw: bytes) -> None:
    """Run one native attempt, restoring full state before any remaining gradients."""
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        processes_per_task=campaign.resources.device.count,
        moe_implementation=campaign.model.moe_implementation,
        remat_mode=campaign.model.remat_mode,
    )
    DistributedConfig().initialize()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    if (
        jax.default_backend() != "gpu"
        or jax.device_count() != campaign.resources.chip_count()
        or len(jax.local_devices()) != 1
    ):
        raise ValueError("Actual GPU ranks differ from the requested multiprocess topology")
    install(cutlass_kernel_cache())
    os.environ["MARIN_PREFIX"] = campaign.cache_prefix
    config, _ = build_campaign(campaign)
    selected = None
    if jax.process_index() == 0:
        resume = latest_completed_checkpoint(campaign)
        selected = {
            "leader_attempt_uid": os.environ["IRIS_ATTEMPT_UID"],
            "resume": dataclasses.asdict(resume) if resume is not None else None,
        }
    encoded = json.dumps(selected).encode()
    if len(encoded) >= _BROADCAST_BYTES:
        raise ValueError("Attempt record exceeds bounded broadcast")
    buffer = np.zeros(_BROADCAST_BYTES, dtype=np.uint8)
    buffer[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    selected = json.loads(np.asarray(broadcast_one_to_all(buffer)).tobytes().rstrip(b"\0"))
    uid = selected["leader_attempt_uid"]
    if not uid.isalnum() or len(uid) > 64:
        raise ValueError("Invalid Iris attempt UID")
    root = StoragePath(campaign.output_root)
    destination = str(root / "attempts" / uid)
    resume = selected["resume"]
    config = dataclasses.replace(
        config,
        output_path=destination,
        trainer=dataclasses.replace(config.trainer, id=f"{campaign.name.split('/')[-1]}-{uid}"),
        resume_checkpoint=resume["path"] if resume else None,
        resume_sft_config=draccus.decode(SftConfig, resume["sft_config"]) if resume else None,
    )
    record = {
        "leader_attempt_uid": uid,
        "destination": destination,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "request_digest": digest(draccus.encode(campaign)),
        "source_revision": campaign.source_revision,
        "resume": resume,
    }
    if jax.process_index() == 0:
        conditional_object(str(root / "attempt-records" / f"{uid}.json")).write(
            (json.dumps(record, indent=2) + "\n").encode(), expected_version=None
        )
    sync_global_devices("sft-campaign-attempt-recorded")
    _run_sft_local(config)
    sync_global_devices("sft-campaign-attempt-complete")
    if jax.process_index() == 0:
        handoff = _json(StoragePath(destination) / "handoff.json")
        if handoff["step"] > campaign.sft.num_steps or handoff["processed_tokens"] > campaign.max_processed_tokens:
            raise ValueError("Handoff exceeds cumulative campaign limits")
        conditional_object(str(root / "completions" / f"{uid}.json")).write(
            (json.dumps({**record, "handoff": handoff}, indent=2) + "\n").encode(), expected_version=None
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--worker", action="store_true", help="Internal distributed worker entrypoint")
    parser.add_argument("--build-caches", action="store_true", help="Build CPU chat caches under a local prefix")
    args = parser.parse_args()
    configure_coreweave_s3()
    campaign, raw = read_campaign(args.request)
    if args.worker:
        if args.build_caches:
            parser.error("GPU workers cannot build caches")
        run_attempt(campaign, raw)
        return
    if args.build_caches:
        build_caches(campaign)
        return
    if args.review is None:
        parser.error("Preparation requires --review; training is never submitted by this command")
    review = campaign_review(campaign, raw)
    with args.review.open("x") as stream:
        stream.write(json.dumps(review, indent=2) + "\n")


if __name__ == "__main__":
    main()
