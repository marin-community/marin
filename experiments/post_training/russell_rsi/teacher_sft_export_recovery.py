# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a failed SFT export under the explicit counter-based evidence amendment."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax import Axis
from levanter.compat import MODEL_MANIFEST_FILENAME
from levanter.compat.hf_checkpoints import (
    DEFAULT_MAX_SHARD_SIZE,
    _save_tokenizer_pretrained,
    _shard_hf_checkpoint,
    _to_state_dict_with_dtype,
    build_generation_config,
)
from levanter.grug.attention import RotaryConfig
from levanter.models.snowball import SnowballConfig
from levanter.tokenizers import load_tokenizer
from levanter.utils.jax_utils import local_cpu_mesh
from marin.execution.lazy import ArtifactStep, artifact_identity
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from s3fs import S3FileSystem

from experiments.evaluation.pipeline import eval_step
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.launch import CLUSTER, evaluation_model
from experiments.post_training.russell_rsi.launch_teacher_sft import SFT_LEARNING_RATE
from experiments.post_training.russell_rsi.teacher_sft_export_qualification import (
    FLOAT32_LEARNING_RATE,
    PROTOCOL,
    validate_numeric_proof,
)

COUNTERS = (
    "step",
    "opt_state/count",
    "opt_state/inner_state/1/count",
    "opt_state/hyperparams_states/learning_rate/count",
)


@dataclass(frozen=True)
class NamedEvidence:
    name: str
    file: PinnedFile


@dataclass(frozen=True)
class ExportRecoveryConfig:
    amendment: PinnedFile
    producer_config: PinnedFile
    evidence: tuple[NamedEvidence, ...]
    numeric_manifest: PinnedFile
    numeric_proof: PinnedFile
    parent_files: tuple[NamedEvidence, ...]
    output_path: str


@dataclass(frozen=True)
class RecoveryInputs:
    amendment: dict
    training_config: dict
    numeric_proof: dict


def recovery_inputs(config: ExportRecoveryConfig) -> RecoveryInputs:
    """Validate immutable producer and numerical evidence before writing an export."""
    amendment = config.amendment.read_json()
    if amendment["protocol"] != PROTOCOL or amendment["original_telemetry_gate"] != "UNMET":
        raise ValueError("Recovery requires the explicit unmet-telemetry amendment")
    if amendment["missing_metric_steps"] != [0, 1, 2] or amendment["producer"]["status"] != "FAILED":
        raise ValueError("Recovery differs from the failed four-update producer")
    # Validate the original config bytes, not only the hash printed in the amendment.
    config.producer_config.read_bytes()
    if config.producer_config.sha256 != amendment["producer"]["config_sha256"]:
        raise ValueError("Recovery producer config differs from the amendment")
    expected = {item["relative_path"]: item["sha256"] for item in amendment["evidence_pins"]}
    supplied = {item.name: item.file for item in config.evidence}
    if len(supplied) != len(config.evidence) or set(supplied) != set(expected):
        raise ValueError("Recovery evidence inventory differs from the amendment")
    raw = {}
    for name, file in supplied.items():
        if file.sha256 != expected[name]:
            raise ValueError(f"Recovery evidence pin differs: {name}")
        raw[name] = file.read_bytes()
    if raw["terminal-executor-status.txt"].decode().strip() != "FAILED":
        raise ValueError("Original training must remain FAILED")
    info = json.loads(raw["terminal-executor-info.json"])
    producer = amendment["producer"]
    if info["output_path"] != producer["root"] or info["name"] != producer["identity"].split("@")[0]:
        raise ValueError("Original executor metadata identifies a different producer")
    counters = json.loads(raw["native-counter-proof.json"])["scalars"]
    hparams = json.loads(raw["actual-train-hparams.json"])
    if (
        any(counters[key] != 4 for key in COUNTERS)
        or counters["opt_state/hyperparams/learning_rate"] != FLOAT32_LEARNING_RATE
        or hparams["optimizer"]["skip_bad_steps"] is not False
        or hparams["optimizer"]["learning_rate"] != SFT_LEARNING_RATE
        or hparams["optimizer"]["lr_schedule"] != "constant"
        or hparams["optimizer"]["warmup"] != 0
        or hparams["trainer"]["num_train_steps"] != 4
        or hparams["hf_save_steps"] != 4
        or hparams["hf_save_dtype"] != "bfloat16"
        or hparams["initialize_from_checkpoint_path"] is not None
    ):
        raise ValueError("Recovery requires the saved four-update fixed-rate training recipe")
    manifest = config.numeric_manifest.read_json()
    proof = config.numeric_proof.read_json()
    if proof["input_manifest_sha256"] != config.numeric_manifest.sha256:
        raise ValueError("Numerical proof identifies another input manifest")
    validate_numeric_proof(proof, amendment)
    if proof["parent_source"] != hparams["initialize_from_hf"]:
        raise ValueError("Numerical comparison uses another parent")
    manifest_pins = manifest["inputs"]
    numeric_names = {
        "saved-inventory.json": "saved-hf-headers/inventory.json",
        "expected-mapping-proof.json": "saved-hf-headers/expected-mapping-proof.json",
        "expected-model.safetensors.index.json": "saved-hf-headers/model.safetensors.index.json",
        "actual-train-hparams.json": "actual-train-hparams.json",
        "native-counter-proof.json": "native-counter-proof.json",
    }
    if (
        proof["evidence_pins"] != manifest_pins
        or manifest_pins["export-recovery-amendment.json"]["sha256"] != config.amendment.sha256
        or any(manifest_pins[name]["sha256"] != expected[source] for name, source in numeric_names.items())
    ):
        raise ValueError("Numerical manifest differs from the amended producer evidence")
    return RecoveryInputs(amendment=amendment, training_config=hparams, numeric_proof=proof)


def save_recovery_metadata(config: ExportRecoveryConfig, hparams: dict, destination: Path) -> dict:
    """Use the original converter on abstract shapes and pinned small parent files."""
    with TemporaryDirectory() as parent_directory:
        parent = Path(parent_directory)
        if not config.parent_files or len({item.name for item in config.parent_files}) != len(config.parent_files):
            raise ValueError("Recovery needs a unique pinned parent metadata inventory")
        if not {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} <= {
            item.name for item in config.parent_files
        }:
            raise ValueError("Pinned parent metadata must include model config and the complete tokenizer")
        if hparams["data"]["tokenizer"] != hparams["initialize_from_hf"]:
            raise ValueError("Recovery requires the original parent tokenizer")
        for item in config.parent_files:
            relative = Path(item.name)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix in {".safetensors", ".bin", ".pt"}:
                raise ValueError("Parent metadata inventory cannot contain model weights or path escapes")
            if item.file.uri != prefix_join(hparams["initialize_from_hf"], item.name):
                raise ValueError("Parent metadata file belongs to another model")
            target = parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.file.read_bytes())
        tokenizer = load_tokenizer(str(parent))
        values = dict(hparams["model"])
        values["rope"] = RotaryConfig(**values["rope"])
        model_config = SnowballConfig(**values)
        converter = model_config.hf_checkpoint_converter().replaced(
            reference_checkpoint=str(parent), tokenizer=tokenizer
        )
        if hparams["pad_tokenizer_to_match_model"]:
            converter = converter.with_tokenizer_padded_to_match_model()
            tokenizer = converter.tokenizer
        with local_cpu_mesh(jax.sharding.AxisType.Explicit):
            if hparams["use_hf_model_config"]:
                model_config = converter.config_from_hf_config(converter.default_hf_config)
            model = eqx.filter_eval_shape(
                model_config.build, Axis("vocab", model_config.vocab_size), key=jax.random.PRNGKey(0)
            )
            state = eqx.filter_eval_shape(_to_state_dict_with_dtype, model, jnp.bfloat16, None)
        _, index = _shard_hf_checkpoint(state, DEFAULT_MAX_SHARD_SIZE, "model.safetensors")
        if index is None:
            raise ValueError("Recovery requires the original sharded export index")
        config_dict = converter._build_hf_config_dict(model)
        if converter._resolve_save_reference_code(None):
            converter._save_code_local(str(destination))
        # The parent's publication manifest describes its weights, not this saved SFT export.
        parent_manifest = destination / MODEL_MANIFEST_FILENAME
        if parent_manifest.exists():
            parent_manifest.unlink()
        _save_tokenizer_pretrained(tokenizer, str(destination))
        generation = build_generation_config(tokenizer, hparams["hf_generation_eos_token_ids"])
        for name, value in (
            ("config.json", config_dict),
            ("model.safetensors.index.json", index),
            ("generation_config.json", generation),
        ):
            if value is not None:
                (destination / name).write_text(json.dumps(value, indent=2) + "\n")
        return index


class RecoveryStorage(Protocol):
    def split_path(self, path: str) -> tuple[str, str, str | None]: ...

    def call_s3(self, method: str, **kwargs) -> dict: ...


def copy_saved_shard(fs: RecoveryStorage, source: str, destination: str, shard: dict) -> dict:
    source_bucket, source_key, source_version = fs.split_path(source)
    target_bucket, target_key, target_version = fs.split_path(destination)
    if (
        source_bucket != target_bucket
        or source_version is not None
        or target_version is not None
        or source == destination
    ):
        raise ValueError("Recovery copies must use distinct unversioned keys in the same regional bucket")
    before = fs.call_s3("head_object", Bucket=source_bucket, Key=source_key)
    if before["ETag"] != shard["etag"] or before["ContentLength"] != shard["bytes"]:
        raise ValueError("Saved shard changed before recovery")
    result = fs.call_s3(
        "copy_object",
        Bucket=target_bucket,
        Key=target_key,
        CopySource={"Bucket": source_bucket, "Key": source_key},
        CopySourceIfMatch=shard["etag"],
    )
    after = fs.call_s3("head_object", Bucket=source_bucket, Key=source_key)
    copied = fs.call_s3("head_object", Bucket=target_bucket, Key=target_key)
    if (
        after["ETag"] != before["ETag"]
        or after["ContentLength"] != before["ContentLength"]
        or copied["ContentLength"] != before["ContentLength"]
        or copied["ETag"] != result["CopyObjectResult"]["ETag"]
    ):
        raise ValueError("Shard identity or copied size changed during recovery")
    return {
        "path": shard["name"],
        "size": copied["ContentLength"],
        "sha256": shard["sha256"],
        "source_etag": before["ETag"],
        "copy_etag": result["CopyObjectResult"]["ETag"],
    }


def run_export_recovery(config: ExportRecoveryConfig) -> None:
    inputs = recovery_inputs(config)
    amendment = inputs.amendment
    hparams = inputs.training_config
    proof = inputs.numeric_proof
    export = prefix_join(config.output_path, "hf/step-3")
    try:
        StoragePath(config.output_path).relative_to(StoragePath(amendment["producer"]["root"]))
    except ValueError:
        pass
    else:
        raise ValueError("Recovery cannot overwrite the failed producer")
    with TemporaryDirectory() as directory:
        local = Path(directory)
        index = save_recovery_metadata(config, hparams, local)
        saved_index = next(
            item.file.read_json()
            for item in config.evidence
            if item.name == "saved-hf-headers/model.safetensors.index.json"
        )
        if index != saved_index or set(index["weight_map"].values()) != {item["name"] for item in proof["shards"]}:
            raise ValueError("Reconstructed index differs from saved shards")
        fs, _ = filesystem_for(export)
        if not isinstance(fs, S3FileSystem):
            raise ValueError("Recovery requires regional S3 server-side copies")
        copies = [
            copy_saved_shard(
                cast(RecoveryStorage, fs),
                prefix_join(proof["source"], item["name"]),
                prefix_join(export, item["name"]),
                item,
            )
            for item in proof["shards"]
        ]
        metadata = []
        for path in sorted(local.rglob("*")):
            if path.is_file():
                raw = path.read_bytes()
                name = str(path.relative_to(local))
                target = StoragePath(prefix_join(export, name))
                target.write_bytes(raw)
                if target.read_bytes() != raw:
                    raise ValueError(f"Recovered metadata readback differs: {name}")
                metadata.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})
    write_once(
        StoragePath(prefix_join(config.output_path, "recovery.json")),
        {
            "protocol": PROTOCOL,
            "producer": amendment["producer"],
            "amendment": amendment,
            "amendment_sha256": config.amendment.sha256,
            "original_telemetry_gate": "UNMET",
            "missing_metric_steps": [0, 1, 2],
            "training_evidence": amendment["training_evidence"],
            "numeric_proof_sha256": config.numeric_proof.sha256,
            "numeric_manifest_sha256": config.numeric_manifest.sha256,
            "numeric_proof": proof,
            "hf_export_uri": export,
            "hf_shards": copies,
            "hf_files": metadata,
            "hf_weight_map": index["weight_map"],
        },
    )


def export_recovery_workflow(config: dict, producer: ArtifactStep[LevanterCheckpoint]) -> dict[str, ArtifactStep]:
    def evidence(items: list[dict]) -> tuple[NamedEvidence, ...]:
        return tuple(NamedEvidence(item["name"], PinnedFile(**item["file"])) for item in items)

    amendment = PinnedFile(**config["amendment"])
    if artifact_identity(producer) != amendment.read_json()["producer"]["identity"]:
        raise ValueError("Reconstructed producer identity differs from amendment")
    base = ExportRecoveryConfig(
        amendment,
        PinnedFile(**config["producer_config"]),
        evidence(config["evidence"]),
        PinnedFile(**config["numeric_manifest"]),
        PinnedFile(**config["numeric_proof"]),
        evidence(config["parent_files"]),
        "",
    )
    recovered = ArtifactStep(
        name="checkpoints/russell-rsi-teacher-sft-export-recovery",
        version=config["version"],
        artifact_type=LevanterCheckpoint,
        deps=(),
        build_config=lambda ctx: replace(base, output_path=ctx.output_path),
        run=run_export_recovery,
    )
    model = evaluation_model("russell-rsi-teacher-sft-export-recovery-reload", "<recovered-export>", None)
    reload = eval_step(
        model,
        "mmlu-smoke",
        version=config["version"],
        deps=(recovered,),
        resolve_model=lambda ctx: replace(
            model, location=prefix_join(ctx.artifact_path(recovered), "hf/step-3"), identity=artifact_identity(recovered)
        ),
        limit=1,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return {"recover": recovered, "reload": reload}
