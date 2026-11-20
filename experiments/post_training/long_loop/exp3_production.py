# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the Exp3 long loop from one sealed data release; dry-run by default."""

import copy
import hashlib
import io
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import click
import draccus
import jmp
import yaml
from fray.types import ResourceConfig
from iris.cluster.client.job_info import get_job_info
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.eval_audit import InitialEvalMatchConfig
from levanter.models.snowball import SnowballConfig
from levanter.trainer import TrainerConfig
from levanter.training_budget import TrainingBudgetConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.execution_policy import SFT_CHECKPOINT_INTERVAL
from experiments.post_training.long_loop.exp3_recipe import TOKEN_IDS, verified_initializer_model
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph

REFERENCE_ROOT = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/evidence/20261113-snowball-recipe-v1"
REFERENCE_BUNDLE_SHA256 = "16d487dd31e9af86033aaf55fefe3e994f2631beba03cb4b7836f88972475612"
REFERENCE_MANIFEST_SHA256 = "66cb9020ee2507ae9d1ecbe44b17e77955e08d956611ccd08133ea2129572098"
INITIALIZER_RECEIPT = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/evidence/initializer-hf/2026.11.15.32.source.json"
)
INITIALIZER_SHA256 = "ecff173c3caf2439b166a008f016397d8fc0cb0b7e0bd5e8b0de59624de02d1c"
REFERENCE_INPUTS = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3/long-loop-exp3/initializer-proof-inputs/2026.11.15.47"
)
OUTPUT_PREFIX = "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp3"
TEMPORARY_PREFIX = "s3://marin-us-east-08a/tmp"
COMPILATION_CACHE = prefix_join(TEMPORARY_PREFIX, "ttl=30d/compilation-cache")
ARTIFACT_NAMES = (
    "cache_catalog.json",
    "source_loss.json",
    "data.json",
    "train_weights.json",
    "retention-by-source.csv",
    "retention-by-transform.csv",
    "mixture-rationale.json",
)
EVIDENCE_NAMES = (
    "publisher_manifest",
    "inventory",
    "native_consumer_receipt",
    "native_measurement_job",
    "static_evidence",
    "atlas_report",
    "atlas_population_pins",
    "atlas_bloom_pin",
)


@dataclass(frozen=True)
class StoragePin:
    uri: str
    sha256: str

    def read(self) -> bytes:
        content = StoragePath(self.uri).read_bytes()
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise ValueError(f"Data release member changed: {self.uri}")
        return content


@dataclass(frozen=True)
class ReleaseGeometry:
    updates: int
    global_batch: int
    sequence_length: int
    seed: int
    packing_cap: int
    lookahead: int


GEOMETRY = ReleaseGeometry(746, 256, 262144, 0, 256, 128)
SFT_DURATION = timedelta(hours=20)
SFT_H100_HOUR_LIMIT = 5120


@dataclass(frozen=True)
class DataRelease:
    pin: StoragePin
    training_ready: bool
    artifacts: dict[str, StoragePin]
    evidence: dict[str, StoragePin]
    data: dict
    document: dict


def read_data_release(uri: str, *, require_ready: bool) -> DataRelease:
    """Verify metadata pins without fetching cached token bodies or weight payloads."""
    raw = StoragePath(uri).read_bytes()
    document = json.loads(raw)
    if document["schema_version"] != 1 or draccus.decode(ReleaseGeometry, document["geometry"]) != GEOMETRY:
        raise ValueError("Exp3 release changed the controlled data geometry")
    ready = draccus.decode(bool, document["training_ready"])
    if require_ready and not ready:
        raise ValueError("CPU-only data release is not admitted for GPU training")
    artifacts = {name: draccus.decode(StoragePin, document["artifacts"][name]) for name in ARTIFACT_NAMES}
    contents = {name: pin.read() for name, pin in artifacts.items()}
    evidence = {}
    for name in EVIDENCE_NAMES:
        if ready or document.get(name) is not None:
            pin = draccus.decode(StoragePin, document[name])
            pin.read()
            evidence[name] = pin
    data = json.loads(contents["data.json"])
    if data["cache_catalog"] != artifacts["cache_catalog.json"].uri:
        raise ValueError("Data configuration names another cache catalog")
    if data["train_weights"] != json.loads(contents["train_weights.json"]):
        raise ValueError("Data configuration differs from the frozen natural mixture weights")
    if data["num_validation_sequences"] != document["num_validation_sequences"]:
        raise ValueError("Data configuration changed the frozen IID validation reservations")
    return DataRelease(StoragePin(uri, hashlib.sha256(raw).hexdigest()), ready, artifacts, evidence, data, document)


def baseline_files() -> dict[str, bytes]:
    """Read and verify the small frozen Exp1 recipe bundle, including every member."""
    raw = StoragePath(prefix_join(REFERENCE_ROOT, "recipe.tar.gz")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_BUNDLE_SHA256:
        raise ValueError("Frozen Exp1 recipe bundle changed")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = {}
        for member in archive:
            if member.isdir():
                continue
            name = member.name.removeprefix("snowball-recipe-bundle/")
            if not member.isfile() or name == member.name or name in members:
                raise ValueError("Unexpected frozen recipe archive member")
            stream = archive.extractfile(member)
            assert stream is not None
            members[name] = stream.read()
    manifest = members.pop("manifest.json")
    if hashlib.sha256(manifest).hexdigest() != REFERENCE_MANIFEST_SHA256:
        raise ValueError("Frozen Exp1 recipe manifest changed")
    expected = {entry["path"]: entry for entry in json.loads(manifest)["files"]}
    if members.keys() != expected.keys():
        raise ValueError("Frozen Exp1 recipe member set changed")
    for name, content in members.items():
        entry = expected[name]
        if (len(content), hashlib.sha256(content).hexdigest()) != (entry["size"], entry["sha256"]):
            raise ValueError(f"Frozen Exp1 recipe member changed: {name}")
    return members


def production_sft_recipe(release: DataRelease) -> dict:
    """Bind the approved full-context SFT controls independently of smoke scale."""
    mesh = MeshConfig(
        axes={"data": 1, "expert": 8, "model": 1, "replica": 1},
        dcn_axes={"context": 32, "replica_dcn": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
    )
    trainer = TrainerConfig(
        train_batch_size=GEOMETRY.global_batch,
        num_train_steps=GEOMETRY.updates,
        per_device_parallelism=1,
        per_device_eval_parallelism=1,
        seed=GEOMETRY.seed,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        mesh=mesh,
        use_explicit_mesh_axes=True,
        offload_opt_state=True,
        loader_fetch_batch_size=4,
        loader_max_buffered_batches=512,
        max_eval_batches=None,
        steps_per_eval=1000,
        jax_compilation_cache_dir=COMPILATION_CACHE,
        log_jaxprs=False,
        log_xla_hlo=False,
        watch=WatchConfig(watch_targets=["params", "updates", "grads"], interval=1, include_per_parameter_norms=False),
        checkpointer=CheckpointerConfig(
            save_interval=SFT_CHECKPOINT_INTERVAL,
            keep=None,
            keep_last_temporary_checkpoints=3,
        ),
        progress_watchdog=ProgressWatchdogConfig(
            startup_timeout=timedelta(minutes=90),
            step_timeout=timedelta(minutes=10),
            process_timeout=timedelta(minutes=30),
            startup_grace_period=timedelta(0),
        ),
        training_budget=TrainingBudgetConfig(SFT_DURATION, timedelta(minutes=40), timedelta(minutes=10)),
    )
    reference_data = draccus.decode(
        LmDataConfig, json.loads((StoragePath(REFERENCE_INPUTS) / "c4-data.json").read_text())
    )
    gate = InitialEvalMatchConfig(
        reference_data,
        prefix_join(REFERENCE_INPUTS, "c4-reference.json"),
        262144,
        1,
        TOKEN_IDS,
        0.5,
        prefix_join(REFERENCE_INPUTS, "c4-reference.json"),
    )
    model = draccus.encode(
        SnowballConfig(
            max_seq_len=GEOMETRY.sequence_length,
            moe_implementation="ring_gather_combine",
            moe_capacity_factor=1,
        )
    )
    model.pop("type", None)
    return {
        "format": "sft_spec",
        "profile": "snowball_exp3",
        "data": release.data,
        "evaluation_data": release.data,
        "trainer": draccus.encode(trainer),
        "model": model,
        "initializer_receipt": INITIALIZER_RECEIPT,
        "initializer_receipt_sha256": INITIALIZER_SHA256,
        "source_loss_metadata": release.artifacts["source_loss.json"].uri,
        "source_loss_metadata_sha256": release.artifacts["source_loss.json"].sha256,
        "packing_cap": GEOMETRY.packing_cap,
        "pack_lookahead_window": GEOMETRY.lookahead,
        "weight_decay": 0.1,
        "data_seed": GEOMETRY.seed,
        "tensorstore_cache_bytes": 100_000_000,
        "mesh": draccus.encode(mesh),
        "resources": {
            key: value
            for key, value in draccus.encode(
                ResourceConfig.with_gpu(
                    "H100",
                    count=8,
                    replicas=32,
                    cpu=64,
                    ram="1500GB",
                    disk="512GB",
                    preemptible=False,
                )
            ).items()
            if value is not None
        },
        "seq_len": GEOMETRY.sequence_length,
        "batch_size": GEOMETRY.global_batch,
        "per_device_parallelism": 1,
        "stop_after_steps": GEOMETRY.updates,
        "task_timeout_seconds": int(SFT_DURATION.total_seconds()),
        "target_positions": GEOMETRY.updates * GEOMETRY.global_batch * GEOMETRY.sequence_length,
        "xla_flags": ["--xla_gpu_shard_autotuning=false"],
        "wandb_entity": "marin-community",
        "wandb_project": "marin-long-loop",
        "stage_guard": {"h100_hour_limit": SFT_H100_HOUR_LIMIT, "poll_interval": 30, "stall_retries": 2},
        "initial_eval_match": draccus.encode(gate),
    }


def production_plan(release: DataRelease, source_revision: str, files: dict[str, bytes]) -> dict:
    """Overlay the candidate SFT on pinned Exp1 RL/data and current runtime controls."""
    documents = {name: yaml.safe_load(content) for name, content in files.items() if name.startswith("rl/")}
    spec = copy.deepcopy(documents["rl/round1.yaml"])
    for name in ("adopted_sft", "adopted_handoff", "adopted_endpoint"):
        spec.pop(name, None)
    spec.update(
        name="long-loop-exp3/round3",
        marin_revision=source_revision,
        storage={"output_prefix": OUTPUT_PREFIX, "temporary_prefix": TEMPORARY_PREFIX},
    )
    current = Path(__file__).with_name("specs")
    documents["rl/execution.yaml"] = yaml.safe_load((current / "execution.yaml").read_text())
    documents["rl/execution.yaml"]["coordinator"]["coordinator_name"] = "long-loop-exp3"
    # Exp1 retained a null Evalchemy option that the native judge no longer accepts.
    documents["rl/evaluation.yaml"]["judge"].pop("evalchemy_num_concurrent")
    documents["rl/sft.yaml"] = production_sft_recipe(release)
    documents["rl/execution.yaml"]["task_timeouts"] = {"rl": 3 * 24 * 3600}
    documents["rl/rl.yaml"]["execution"]["job_timeout_seconds"] = 3 * 24 * 3600
    documents["rl/sft-data.yaml"] = {
        "data": release.data,
        "sections": {},
        "files": [vars(release.pin), *map(vars, release.artifacts.values()), *map(vars, release.evidence.values())],
        "admission_pending": [],
    }
    model_ref = verified_initializer_model(INITIALIZER_RECEIPT, INITIALIZER_SHA256)
    documents["rl/initializer.yaml"] = {
        **documents["rl/initializer.yaml"],
        "uri": model_ref,
        "files": [{"uri": INITIALIZER_RECEIPT, "sha256": INITIALIZER_SHA256}],
    }
    for stage in ("rl", "evaluation"):
        config = documents[f"rl/{stage}.yaml"]
        operational = yaml.safe_load((current / f"{stage}.yaml").read_text())
        config["ray_timeouts"] = operational["ray_timeouts"]
        config["execution"]["wandb_entity"] = "marin-community"
        config["config"]["trainer"]["project_name"] = "marin-long-loop"
        harbor = config["config"]["terminal_bench"]["harbor"]
        current_harbor = operational["config"]["terminal_bench"]["harbor"]
        for field in (
            "environment_type",
            "daytona_image_build",
            "eval_daytona_image_build",
            "daytona_api_key_env",
            "eval_daytona_api_key_env",
            "verifier_override_timeout_sec",
        ):
            harbor[field] = current_harbor[field]
        if stage == "evaluation":
            harbor["trial_attempt_timeout_sec"] = current_harbor["trial_attempt_timeout_sec"]
        trainer = config["config"]["trainer"]
        if trainer.pop("seed") != config["seed"]:
            raise ValueError("Frozen recipe and launch seeds differ")
        if stage == "rl" and trainer.pop("max_ckpts_to_keep") != config["retained"]:
            raise ValueError("Frozen recipe and launch checkpoint retention differ")
        # The current launch owns the temporary attempt trace destination.
        config["config"]["generator"]["trajectory_retention"].pop("output_path")
        config["config"]["generator"]["model_loading"] = "stage_local"
    archive = documents["rl/archive.yaml"]
    archive["hf_upload"]["repo_prefix"] = "open-athena/Snowball-LongLoop-Round3"
    archive["evaldash"]["run_prefix"] = "long-loop-round3"
    archive["additional_evidence"] = []
    archive["sft_data_accounting"] = {
        "source_loss_metadata": {
            "source": release.artifacts["source_loss.json"].uri,
            "sha256": release.artifacts["source_loss.json"].sha256,
        },
        "pinned_evidence": {name: {"source": pin.uri, "sha256": pin.sha256} for name, pin in release.artifacts.items()},
    }
    with tempfile.TemporaryDirectory(prefix="exp3-spec-") as temporary:
        directory = Path(temporary)
        for section in (
            *spec["datasets"].values(),
            *(
                spec[name]
                for name in (
                    "sft",
                    "rl",
                    "evaluation",
                    "initializer",
                    "tokenizer",
                    "endpoint",
                    "execution",
                    "archive",
                )
            ),
        ):
            path = section["path"]
            content = yaml.safe_dump(documents[f"rl/{path}"], sort_keys=False).encode()
            (directory / path).write_bytes(content)
            section["sha256"] = hashlib.sha256(content).hexdigest()
        path = directory / "round3.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        resolved = resolve_spec(
            path,
            smoke=False,
            acknowledge_experimental_controls=True,
            experimental_controls_reason=(
                "Restore the standing 3-day outer RL dispatch timeout as an operational control; "
                "Exp1 learning 43200s and drain 1800s remain unchanged. "
                "SFT 20h/746 and all data controls remain unchanged."
            ),
        )
    resolved["data_release"] = {**vars(release.pin), "training_ready": release.training_ready}
    resolved["baseline_bundle"] = {
        "uri": prefix_join(REFERENCE_ROOT, "recipe.tar.gz"),
        "sha256": REFERENCE_BUNDLE_SHA256,
    }
    resolved.pop("resolved_sha256")
    resolved["resolved_sha256"] = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    return resolved


@click.command(help=__doc__)
@click.option("--data-release", required=True)
@click.option("--source-revision", required=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@click.option("--through", type=click.Choice(["sft", "archive"]), default="archive", show_default=True)
@rl_build_options
def main(data_release: str, source_revision: str, validation_output: Path, through: str) -> ArtifactStep:
    configure_coreweave_s3()
    release = read_data_release(data_release, require_ready=click.get_current_context().params["do_run"])
    workspace = Path(__file__).resolve().parents[3]
    if get_job_info() is None:
        require_clean_revision(workspace, source_revision)
    resolved = production_plan(release, source_revision, baseline_files())
    resolved["runtime_source"] = runtime_source_manifest(workspace, source_revision)
    os.environ.update(MARIN_PREFIX=OUTPUT_PREFIX, MARIN_TEMP_PREFIX=TEMPORARY_PREFIX)
    validation_output.mkdir(parents=True, exist_ok=True)
    (validation_output / "resolved-plan.json").write_text(json.dumps(resolved, indent=2))
    terminal = build_spec_graph(resolved, resolve_version(resolved["name"], None), through=through)
    model_ref = verified_initializer_model(INITIALIZER_RECEIPT, INITIALIZER_SHA256)
    validate_graph(
        terminal, validation_output, OUTPUT_PREFIX, model_config=(StoragePath(model_ref) / "config.json").read_bytes()
    )
    click.echo(
        json.dumps(
            {
                "data_release": vars(release.pin),
                "training_ready": release.training_ready,
                "gpu_peaks": resolved["gpu_peaks"],
                "sft_hours": 20,
                "sft_h100_hour_ceiling": SFT_H100_HOUR_LIMIT,
                "maximum_updates": 746,
            },
            indent=2,
        )
    )
    return terminal


if __name__ == "__main__":
    main()
