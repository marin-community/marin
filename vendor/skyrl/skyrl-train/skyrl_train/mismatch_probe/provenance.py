"""Immutable software, trainer-step and archive provenance for mismatch probes."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import asdict

import torch
from finestore.rl import mismatch_probe as mismatch
from omegaconf import OmegaConf

from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.mismatch_probe.protocol import request_seed


def software_provenance(probe, trainer) -> dict[str, str | None]:
    def version(package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    if probe.tokenizer_fingerprint is None:
        probe.tokenizer_fingerprint = tokenizer_vocabulary_fingerprint(trainer.inference_engine_client.tokenizer)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "marinskyrl_commit": probe.cfg.get("runtime", {}).get("launcher_commit"),
        "marinskyrl_version": version("marinskyrl"),
        "marin_finestore_version": version("marin-finestore"),
        "vllm_version": version("vllm"),
    }


def manifest(probe, trainer, *, status: mismatch.ArchiveStatus):
    software = software_provenance(probe, trainer)
    schema = mismatch
    return schema.ManifestRow(
        archive=probe.archive_uri,
        status=status,
        probe_hash=probe.probe_hash,
        checkpoint_path=str(
            trainer.loaded_checkpoint_path
            or probe.cfg.get("runtime", {}).get("checkpoint_path")
            or trainer.cfg.trainer.resume_path
            or trainer.cfg.trainer.policy.model.path
        ),
        runtime_commit=software["marinskyrl_commit"],
        source_probe_archive=probe.spec.reuse_probe,
        tokenizer_fingerprint=probe.tokenizer_fingerprint,
        starting_global_step=probe.starting_global_step,
        scored_updates=sorted(probe.scored_global_steps),
        scored_global_steps=[probe.scored_global_steps[key] for key in sorted(probe.scored_global_steps)],
        architecture=str(trainer.cfg.trainer.policy.model.path),
        vllm_enforce_eager=bool(trainer.cfg.generator.enforce_eager),
        optimizer_steps_per_update=int(trainer.all_metrics.get("policy/policy_update_steps", 0)),
        seed=int(probe.spec.seed),
        bootstrap_seed=request_seed(int(probe.spec.seed), "bootstrap", 0),
        created_at_utc=probe.created_at,
        config_json=json.dumps(OmegaConf.to_container(probe.cfg, resolve=True), sort_keys=True, default=str),
        software_json=json.dumps(software, sort_keys=True),
        hardware_json=json.dumps({"placement": OmegaConf.to_container(trainer.cfg.trainer.placement)}, sort_keys=True),
        batch_layout_json=json.dumps(asdict(probe.batch_layout), sort_keys=True),
        timing_json=json.dumps(probe.timing, sort_keys=True),
        step_metrics_json=json.dumps(probe.metrics, sort_keys=True),
    )
