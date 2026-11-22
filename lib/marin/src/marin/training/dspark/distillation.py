# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a teacher and Speculators trainer together on one Iris GPU worker."""

import json
import logging
import math
import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
from levanter.compat.hf_export import write_export_json
from rigging.filesystem.storage_path import StoragePath
from safetensors import safe_open

from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.vllm_server import IsolatedCudaVllm, VllmType
from marin.training.dspark.data_preparation import (
    ARROW_DATASET_DIR,
    DRAFT_CONFIG_FILE,
    LM_HEAD,
    PREPARED_DIR,
    SAFETENSORS_INDEX,
    SPLITS_FILE,
    TOKEN_FREQUENCIES_FILE,
    DraftRecipe,
    file_sha256,
    inspect_teacher,
    stage_teacher,
    tensor_files,
    verify_sha256,
)

logger = logging.getLogger(__name__)

SPECULATORS_REPOSITORY = "https://github.com/atutej/speculators.git"
SPECULATORS_REVISION = "77eaa2390cab54ff7e5315345137394ee44b7a98"
VLLM_SOURCE_COMMIT = "47a8e415bff17b9376f02d9b92edcc67e962eaee"
_LOOPBACK_HOST = "127.0.0.1"
_TEACHER_PORT = 18000
_EXPORT_RECEIPT = "dspark-export.json"
_EPOCHS = 1
_MARKOV_HEAD = "vanilla"
_D2T_FILE = "d2t.npy"
_T2D_FILE = "t2d.npy"
_BUILD_VOCAB_MAPPINGS = """
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

data, d2t_path, t2d_path, draft_size, target_size, ranks = sys.argv[1:]
if torch.cuda.device_count() != int(ranks):
    raise RuntimeError("Trainer CUDA devices disagree with the recipe")
freq = {int(k): v for k, v in json.loads(Path(data).read_text()).items()}
d2t, t2d = build_vocab_mappings_from_distribution(freq, int(draft_size), int(target_size))
np.save(d2t_path, d2t.numpy())
np.save(t2d_path, t2d.numpy())
print(json.dumps({name: version(name) for name in ("torch", "transformers", "datasets")}))
"""


@dataclass(frozen=True)
class TrainingProvenance:
    """Immutable identities of the teacher export and the selected data mix."""

    teacher_identity: str
    teacher_manifest_sha256: str
    mix_manifest_sha256: str


def gpu_device_split(visible_devices: str, recipe: DraftRecipe) -> tuple[str, str]:
    """Partition the Iris task's visible GPU IDs into teacher and trainer masks."""

    devices = [device.strip() for device in visible_devices.split(",")]
    total = recipe.teacher_replicas + recipe.trainer_ranks
    if not all(devices) or len(set(devices)) != total or len(devices) != total:
        raise ValueError(f"DSpark needs exactly {total} distinct visible GPUs, got {visible_devices!r}")
    return ",".join(devices[: recipe.teacher_replicas]), ",".join(devices[recipe.teacher_replicas :])


def _worker_gpu_devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return visible
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
    return ",".join(output.splitlines())


def _install_speculators(scratch: Path, env: dict[str, str]) -> str:
    source = scratch / "speculators"
    trainer_env = scratch / "trainer-env"
    subprocess.run(["git", "init", "--quiet", str(source)], env=env, check=True)
    git = ["git", "-C", str(source)]
    subprocess.run([*git, "fetch", "--depth", "1", SPECULATORS_REPOSITORY, SPECULATORS_REVISION], env=env, check=True)
    subprocess.run([*git, "checkout", "--detach", SPECULATORS_REVISION], env=env, check=True)
    subprocess.run(["uv", "venv", "--python", "3.12", str(trainer_env)], env=env, check=True)
    python = str(trainer_env / "bin" / "python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "--torch-backend",
            "cu130",
            "-e",
            str(source),
            "torch==2.11.0+cu130",
            "torchaudio==2.11.0+cu130",
            "torchvision==0.26.0+cu130",
            "transformers==5.12.1",
            "datasets==5.0.1",
        ],
        env=env,
        check=True,
    )
    return python


@contextmanager
def _teacher_server(command: list[str], env: dict[str, str], port: int) -> Iterator[None]:
    with subprocess.Popen(command, env=env, start_new_session=True) as server:
        try:
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM hidden-state server exited with status {server.returncode}")
                try:
                    with urlopen(f"http://{_LOOPBACK_HOST}:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except HTTPError:
                    raise
                except URLError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("vLLM hidden-state server failed to become ready")
            yield
        finally:
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(server.pid, signal.SIGKILL)
                server.wait()


def extraction_config(recipe: DraftRecipe, hidden_states: Path) -> tuple[dict, dict]:
    """Return vLLM speculative-decoding and KV-transfer configs for teacher prefill."""

    speculative = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": list(recipe.extraction_taps)}},
    }
    transfer = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": str(hidden_states)},
    }
    return speculative, transfer


def validate_draft(checkpoint: Path, recipe: DraftRecipe, expected_mapping: np.ndarray) -> dict:
    """Check the Qwen3 DSpark serving contract and return its checkpoint config."""

    config = json.loads((checkpoint / "config.json").read_text())
    required = {
        "speculators_model_type": "dspark",
        "layer_type": "qwen3",
        "block_size": recipe.block_size,
        "sample_from_anchor": True,
        "aux_hidden_state_layer_ids": list(recipe.feature_taps),
        "mask_token_id": recipe.mask_token_id,
        "draft_vocab_size": recipe.draft_vocab_size,
        "markov_rank": recipe.markov_rank,
        "markov_head_type": _MARKOV_HEAD,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    for name, value in required.items():
        if config.get(name) != value:
            raise ValueError(f"DSpark checkpoint has unexpected {name}: {config.get(name)!r}")
    decoder = config["transformer_layer_config"]
    if (
        decoder["model_type"] != "qwen3"
        or decoder["hidden_size"] != recipe.shape.hidden_size
        or decoder["vocab_size"] != recipe.shape.vocab_size
        or decoder["num_hidden_layers"] != recipe.draft_layers
        or config["speculators_config"]["proposal_methods"][0]["speculative_tokens"] != recipe.block_size
    ):
        raise ValueError("DSpark decoder or proposal dimensions disagree with the recipe")

    files = tensor_files(checkpoint)
    expected_shapes = {
        "fc.weight": (recipe.shape.hidden_size, len(recipe.feature_taps) * recipe.shape.hidden_size),
        LM_HEAD: (recipe.draft_vocab_size, recipe.shape.hidden_size),
        "markov_head.markov_w1.weight": (recipe.shape.vocab_size, recipe.markov_rank),
        "markov_head.markov_w2.weight": (recipe.draft_vocab_size, recipe.markov_rank),
        "confidence_head.proj.weight": (1, recipe.shape.hidden_size + recipe.markov_rank),
    }
    for name, shape in expected_shapes.items():
        with safe_open(checkpoint / files[name], framework="np") as weights:
            if tuple(weights.get_slice(name).get_shape()) != shape:
                raise ValueError(f"DSpark checkpoint has invalid shape for {name}")
    with safe_open(checkpoint / files["d2t"], framework="np") as weights:
        mapping = weights.get_tensor("d2t")
    target_ids = mapping + np.arange(recipe.draft_vocab_size)
    if (
        mapping.dtype != np.int64
        or not np.array_equal(mapping, expected_mapping)
        or target_ids[0] < 0
        or target_ids[-1] >= recipe.shape.vocab_size
        or not (np.diff(target_ids) > 0).all()
    ):
        raise ValueError("DSpark checkpoint has an invalid draft-to-target vocabulary mapping")
    return config


def _publish_checkpoint(
    *,
    output: StoragePath,
    training: Path,
    provenance: TrainingProvenance,
    recipe: DraftRecipe,
    mapping: np.ndarray,
    packages: dict,
) -> None:
    checkpoint = training / str(_EPOCHS - 1)
    state = json.loads((checkpoint / "training_state.json").read_text())
    metrics = json.loads((checkpoint / "val_metrics.json").read_text())
    if (
        state["global_step"] != recipe.max_steps
        or not all(math.isfinite(value) for value in metrics.values())
        or not math.isfinite(metrics["loss_epoch"])
    ):
        raise ValueError("DSpark training finished without the requested steps and finite validation metrics")
    config = validate_draft(checkpoint, recipe, mapping)
    # The training-time staging path is not a portable verifier identity.
    config["speculators_config"]["verifier"]["name_or_path"] = provenance.teacher_identity
    names = sorted(set(tensor_files(checkpoint).values()))
    index = checkpoint / SAFETENSORS_INDEX
    if index.is_file():
        names.append(index.name)

    output.mkdirs()
    for name in names:
        (output / name).upload_from(str(checkpoint / name))
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    receipt = {
        **asdict(provenance),
        "speculators_revision": SPECULATORS_REVISION,
        "vllm_source_commit": VLLM_SOURCE_COMMIT,
        "trainer_packages": packages,
        "steps": state["global_step"],
        "validation": metrics,
        "files": {
            "config.json": file_sha256(config_path),
            **{name: file_sha256(checkpoint / name) for name in names},
        },
    }
    # Readers use the receipt to distinguish partial uploads from a completed export.
    write_export_json(output / _EXPORT_RECEIPT, receipt)


def _teacher_command(teacher: Path, recipe: DraftRecipe, hidden_states: Path) -> list[str]:
    if VLLM_GPU_RELEASE.source_commit != VLLM_SOURCE_COMMIT:
        raise ValueError("Requalify DSpark against the newly promoted vLLM wheel before changing the pin")
    speculative, transfer = extraction_config(recipe, hidden_states)
    server_options = {
        "host": _LOOPBACK_HOST,
        "port": str(_TEACHER_PORT),
        "dtype": "bfloat16",
        "tensor-parallel-size": "1",
        "data-parallel-size": str(recipe.teacher_replicas),
        "max-model-len": str(recipe.max_sequence_length + 1),
        "speculative-config": json.dumps(speculative),
        "kv-transfer-config": json.dumps(transfer),
    }
    if recipe.shape.sconv:
        server_options["compilation-config"] = json.dumps(
            {
                "mode": 3,
                "cudagraph_mode": "PIECEWISE",
                "splitting_ops": [
                    "vllm::unified_attention_with_output",
                    "vllm::unified_kv_cache_update",
                    "vllm::grug_moe_short_conv",
                ],
            }
        )
    return [
        *IsolatedCudaVllm(source=VllmType.MARIN_FORK).command(),
        "serve",
        str(teacher),
        "--enable-expert-parallel",
        *(f"--{key}={value}" for key, value in server_options.items()),
    ]


def _trainer_command(
    teacher: Path,
    recipe: DraftRecipe,
    python: str,
    data: Path,
    scratch: Path,
    training: Path,
    hidden_states: Path,
) -> list[str]:
    split = json.loads((data / SPLITS_FILE).read_text())
    train_config = {
        "speculator_type": "dspark",
        "seed": recipe.seed,
        "verifier": {"verifier_name_or_path": str(teacher)},
        "draft": {
            "draft_config": str(data / DRAFT_CONFIG_FILE),
            "draft_vocab_size": recipe.draft_vocab_size,
            "d2t_path": str(scratch / _D2T_FILE),
            "t2d_path": str(scratch / _T2D_FILE),
            "mask_token_id": recipe.mask_token_id,
            "target_layer_ids": list(recipe.feature_taps),
        },
        "data": {
            "data_path": str(data / ARROW_DATASET_DIR),
            "train_data_ratio": split["train_ratio"],
            "hidden_states_backend": "file",
            "total_seq_len": recipe.max_sequence_length,
            "max_anchors": recipe.max_anchors,
            "num_workers": 2,
            "prefetch_factor": 2,
        },
        "backend": {"hidden_states_path": str(hidden_states)},
        "generation": {
            "vllm_endpoint": f"http://{_LOOPBACK_HOST}:{_TEACHER_PORT}/v1",
            "on_missing": "generate",
            "on_generate": "delete",
        },
        "loss": {
            "loss_fn": json.dumps({"ce": recipe.cross_entropy_weight, "tv": recipe.total_variation_weight}),
        },
        "optimizer": {"optimizer": recipe.optimizer, "lr": recipe.learning_rate},
        "scheduler": {"scheduler_type": recipe.lr_schedule},
        "trainer": {
            "save_path": str(training),
            "no_resume_from_checkpoint": True,
            "epochs": _EPOCHS,
            "max_steps": recipe.max_steps,
            "gradient_checkpointing": True,
        },
        "dflash": {"block_size": recipe.block_size, "sample_from_anchor": True},
        "dspark": {
            "draft_layer_type": "qwen3",
            "markov_rank": recipe.markov_rank,
            "markov_head_type": _MARKOV_HEAD,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "confidence_head_alpha": 1.0,
        },
    }
    config_path = scratch / "train.json"
    config_path.write_text(json.dumps(train_config))
    train = [python]
    if recipe.trainer_ranks > 1:
        train += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={recipe.trainer_ranks}"]
    train += ["-m", "speculators.train", "--config", str(config_path)]
    return train


def _build_vocab_mappings(
    python: str, data: Path, scratch: Path, recipe: DraftRecipe, env: dict[str, str]
) -> dict[str, str]:
    return json.loads(
        subprocess.check_output(
            [
                python,
                "-c",
                _BUILD_VOCAB_MAPPINGS,
                str(data / TOKEN_FREQUENCIES_FILE),
                str(scratch / _D2T_FILE),
                str(scratch / _T2D_FILE),
                str(recipe.draft_vocab_size),
                str(recipe.shape.vocab_size),
                str(recipe.trainer_ranks),
            ],
            env=env,
            text=True,
        )
    )


def _verify_completed_export(export: StoragePath, provenance: TrainingProvenance) -> bool:
    receipt_path = export / _EXPORT_RECEIPT
    if not receipt_path.isfile():
        return False
    receipt = json.loads(receipt_path.read_text())
    if any(receipt[name] != value for name, value in asdict(provenance).items()):
        raise ValueError(f"Existing DSpark export at {export} has different inputs")
    for name, digest in receipt["files"].items():
        verify_sha256(export / name, digest)
    return True


def train_and_export(
    *,
    output: str,
    teacher: str,
    teacher_manifest: str,
    provenance: TrainingProvenance,
    prepared: str,
    recipe: DraftRecipe,
) -> None:
    """Stage the pinned inputs, train on an Iris worker, and publish the draft receipt last."""

    export = StoragePath(output) / "hf"
    if _verify_completed_export(export, provenance):
        return

    teacher_devices, trainer_devices = gpu_device_split(_worker_gpu_devices(), recipe)
    source = StoragePath(teacher)
    manifest = StoragePath(teacher_manifest)
    decoder = inspect_teacher(source, manifest, provenance.teacher_manifest_sha256, recipe)
    prepared_source = StoragePath(prepared) / PREPARED_DIR
    if json.loads((prepared_source / DRAFT_CONFIG_FILE).read_text()) != decoder:
        raise ValueError("Prepared draft config disagrees with the teacher")

    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    env.pop("PYTHONPATH", None)
    vllm_env = {**env, "CUDA_VISIBLE_DEVICES": teacher_devices}
    trainer_env = {**env, "CUDA_VISIBLE_DEVICES": trainer_devices}
    with TemporaryDirectory(prefix="marin-dspark-") as directory:
        scratch = Path(directory)
        training = scratch / "checkpoints"
        hidden_states = scratch / "hidden_states"
        hidden_states.mkdir()
        teacher_path = scratch / "teacher"
        data = scratch / "data"
        python = _install_speculators(scratch, trainer_env)

        logger.info("Staging verified DSpark inputs for %s", provenance.teacher_identity)
        stage_teacher(source, manifest, provenance.teacher_manifest_sha256, teacher_path)
        prepared_source.download_to(str(data), recursive=True, batch_size=8)
        packages = _build_vocab_mappings(python, data, scratch, recipe, trainer_env)
        server = _teacher_command(teacher_path, recipe, hidden_states)
        train = _trainer_command(teacher_path, recipe, python, data, scratch, training, hidden_states)
        logger.info("Training DSpark for %s at %s", provenance.teacher_identity, output)
        with _teacher_server(server, vllm_env, _TEACHER_PORT):
            subprocess.run(train, env=trainer_env, check=True)
        _publish_checkpoint(
            output=export,
            training=training,
            provenance=provenance,
            recipe=recipe,
            mapping=np.load(scratch / _D2T_FILE, allow_pickle=False),
            packages=packages,
        )
