# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and export DSpark with a live teacher or cached teacher activations."""

import json
import logging
import math
import os
import subprocess
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import rigging
from levanter.compat.hf_checkpoints import SAFE_TENSORS_INDEX_NAME
from rigging.filesystem.storage_path import StoragePath
from safetensors import safe_open
from transformers.utils import CONFIG_NAME

from marin.execution.artifact import Artifact
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import InferenceModelConfig
from marin.inference.vllm_server import IsolatedCudaVllm, VllmEnvironment, VllmLauncherWithEnvironment, VllmType
from marin.training.drafting.activation_store import OBJECT_STORE_BACKEND
from marin.training.drafting.dspark_data import (
    ARROW_DATASET_DIR,
    DRAFT_CONFIG_FILE,
    DRAFT_LAYER_TYPE,
    EMBEDDING,
    FINAL_NORM,
    GATE_DOWN,
    GATE_UP,
    LM_HEAD,
    PREPARED_DIR,
    SPLITS_FILE,
    TOKEN_FREQUENCIES_FILE,
    DraftDataConfig,
    PreparedSplits,
    tensor_files,
)

logger = logging.getLogger(__name__)

SPECULATORS_REPOSITORY = "https://github.com/atutej/speculators.git"
SPECULATORS_REVISION = "77eaa2390cab54ff7e5315345137394ee44b7a98"
VLLM_SOURCE_COMMIT = "47a8e415bff17b9376f02d9b92edcc67e962eaee"
_TEACHER_PORT = 18000
HF_DRAFT_SUBDIRECTORY = "hf"
HIDDEN_STATES_DIR = "hidden_states"
DEFAULT_DRAFT_SEED = 42
_EPOCHS = 1
_MARKOV_HEAD = "vanilla"
_LOSS_IMPLEMENTATION = "eager"
_D2T_FILE = "d2t.npy"
_T2D_FILE = "t2d.npy"
_D2T_TENSOR = "d2t"
_SPECULATOR_TYPE = "dspark"
_SAMPLE_FROM_ANCHOR = True
_ENABLE_CONFIDENCE_HEAD = True
_CONFIDENCE_HEAD_WITH_MARKOV = True
_PREPARE_TRAINER_INPUTS = """
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2
from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

(
    dataset_path, train_rows, seq_len, step_budget, freq_path,
    d2t_path, t2d_path, draft_size, target_size, ranks,
) = sys.argv[1:]
if torch.cuda.device_count() != int(ranks):
    raise RuntimeError("Trainer CUDA devices disagree with the recipe")
# Count epoch-zero batches exactly as Speculators does, without starting the teacher.
lengths = load_from_disk(dataset_path).select(range(int(train_rows))).with_format(None)["seq_len"]
sampler = MultipackDistributedBatchSamplerV2(
    batch_max_length=int(seq_len), lengths=list(lengths), num_replicas=int(ranks), rank=0,
)
steps = min(len(sampler), int(step_budget))
if steps == 0:
    raise RuntimeError("The prepared mix has no packed training batches")
freq = {int(k): v for k, v in json.loads(Path(freq_path).read_text()).items()}
d2t, t2d = build_vocab_mappings_from_distribution(freq, int(draft_size), int(target_size))
np.save(d2t_path, d2t.numpy())
np.save(t2d_path, t2d.numpy())
print(json.dumps({
    "steps": steps,
    "packages": {name: version(name) for name in ("torch", "transformers", "datasets")},
}))
"""


@dataclass(frozen=True)
class DraftRecipe:
    """DSpark model and optimizer settings."""

    data: DraftDataConfig
    feature_taps: tuple[int, ...]
    max_anchors: int
    max_steps: int
    trainer_ranks: int
    teacher_replicas: int
    cross_entropy_weight: float
    total_variation_weight: float
    optimizer: Literal["muon", "adamw"]
    lr_schedule: Literal["linear", "cosine", "none"]
    draft_vocab_size: int = 32000
    block_size: int = 8
    markov_rank: int = 256
    learning_rate: float = 3e-4
    seed: int = DEFAULT_DRAFT_SEED

    def __post_init__(self) -> None:
        if not self.feature_taps or tuple(sorted(set(self.feature_taps))) != self.feature_taps:
            raise ValueError("Feature taps must be strictly increasing")
        if self.feature_taps[0] < 0 or self.feature_taps[-1] >= self.data.shape.num_hidden_layers:
            raise ValueError("Feature taps must precede the final pre-norm distillation tap")
        if not 0 < self.draft_vocab_size < self.data.shape.vocab_size:
            raise ValueError("The draft vocabulary must be a strict subset of the teacher vocabulary")
        max_length = self.data.max_sequence_length
        if not max_length > self.block_size or not 0 < self.max_anchors <= max_length:
            raise ValueError("The sequence length and anchor budget must fit the draft block")
        if min(self.max_steps, self.trainer_ranks, self.teacher_replicas) <= 0:
            raise ValueError("Step budget and GPU counts must be positive")

    @property
    def extraction_taps(self) -> tuple[int, ...]:
        return (*self.feature_taps, self.data.shape.num_hidden_layers)


class DSparkExport(Artifact):
    """Training results stored in the draft artifact's provenance record."""

    hf_subdirectory: str = HF_DRAFT_SUBDIRECTORY
    steps: int
    validation: dict[str, float]
    trainer_packages: dict[str, str]
    speculators_revision: str
    vllm_source_commit: str
    loss_implementation: str = _LOSS_IMPLEMENTATION


def _gpu_device_split(visible_devices: str, recipe: DraftRecipe) -> tuple[str, str]:
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


def _install_speculators(scratch: Path, env: dict[str, str], extra_packages: tuple[str, ...]) -> str:
    """Install pinned Speculators and return the isolated Python executable."""
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
            *extra_packages,
        ],
        env=env,
        check=True,
    )
    return python


def require_supported_vllm_release() -> None:
    """Require the source revision used to qualify teacher extraction."""
    if VLLM_GPU_RELEASE.source_commit != VLLM_SOURCE_COMMIT:
        raise ValueError("Requalify DSpark against the newly promoted vLLM wheel before changing the pin")


def _require_single_worker_teacher(recipe: DraftRecipe) -> None:
    """Keep ShortConv teachers out of the single-worker training path."""
    if recipe.data.shape.sconv:
        raise ValueError("ShortConv DSpark training requires the distributed teacher and cached states")


def _require_training_rows(split: PreparedSplits, ranks: int) -> None:
    if min(split.train, split.validation) < ranks:
        raise ValueError("Each prepared split needs at least one usable row per trainer rank")


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

    config = json.loads((checkpoint / CONFIG_NAME).read_text())
    required = {
        "speculators_model_type": _SPECULATOR_TYPE,
        "layer_type": DRAFT_LAYER_TYPE,
        "block_size": recipe.block_size,
        "sample_from_anchor": _SAMPLE_FROM_ANCHOR,
        "aux_hidden_state_layer_ids": list(recipe.feature_taps),
        "mask_token_id": recipe.data.mask_token_id,
        "draft_vocab_size": recipe.draft_vocab_size,
        "markov_rank": recipe.markov_rank,
        "markov_head_type": _MARKOV_HEAD,
        "enable_confidence_head": _ENABLE_CONFIDENCE_HEAD,
        "confidence_head_with_markov": _CONFIDENCE_HEAD_WITH_MARKOV,
    }
    for name, value in required.items():
        if config.get(name) != value:
            raise ValueError(f"DSpark checkpoint has unexpected {name}: {config.get(name)!r}")
    decoder = config["transformer_layer_config"]
    if (
        decoder["model_type"] != DRAFT_LAYER_TYPE
        or decoder["hidden_size"] != recipe.data.shape.hidden_size
        or decoder["vocab_size"] != recipe.data.shape.vocab_size
        or decoder["num_hidden_layers"] != recipe.data.draft_layers
        or decoder["intermediate_size"] != recipe.data.intermediate_size
        or config["speculators_config"]["proposal_methods"][0]["speculative_tokens"] != recipe.block_size
    ):
        raise ValueError("DSpark decoder or proposal dimensions disagree with the recipe")

    files = tensor_files(checkpoint)
    expected_shapes = {
        "fc.weight": (recipe.data.shape.hidden_size, len(recipe.feature_taps) * recipe.data.shape.hidden_size),
        LM_HEAD: (recipe.draft_vocab_size, recipe.data.shape.hidden_size),
        "markov_head.markov_w1.weight": (recipe.data.shape.vocab_size, recipe.markov_rank),
        "markov_head.markov_w2.weight": (recipe.draft_vocab_size, recipe.markov_rank),
        "confidence_head.proj.weight": (1, recipe.data.shape.hidden_size + recipe.markov_rank),
    }
    for name, shape in expected_shapes.items():
        with safe_open(checkpoint / files[name], framework="np") as weights:
            if tuple(weights.get_slice(name).get_shape()) != shape:
                raise ValueError(f"DSpark checkpoint has invalid shape for {name}")
    with safe_open(checkpoint / files[_D2T_TENSOR], framework="np") as weights:
        mapping = weights.get_tensor(_D2T_TENSOR)
    target_ids = mapping + np.arange(recipe.draft_vocab_size)
    if (
        mapping.dtype != np.int64
        or not np.array_equal(mapping, expected_mapping)
        or target_ids[0] < 0
        or target_ids[-1] >= recipe.data.shape.vocab_size
        or not (np.diff(target_ids) > 0).all()
    ):
        raise ValueError("DSpark checkpoint has an invalid draft-to-target vocabulary mapping")
    return config


def _publish_checkpoint(
    *,
    output: StoragePath,
    training: Path,
    teacher_identity: str,
    recipe: DraftRecipe,
    expected_steps: int,
    mapping: np.ndarray,
    packages: dict,
) -> DSparkExport:
    checkpoint = training / str(_EPOCHS - 1)
    state = json.loads((checkpoint / "training_state.json").read_text())
    metrics = json.loads((checkpoint / "val_metrics.json").read_text())
    if (
        state["global_step"] != expected_steps
        or not all(math.isfinite(value) for value in metrics.values())
        or "loss_epoch" not in metrics
    ):
        raise ValueError("DSpark training finished without the requested steps and finite validation metrics")
    config = validate_draft(checkpoint, recipe, mapping)
    # The training-time staging path is not a portable verifier identity.
    config["speculators_config"]["verifier"]["name_or_path"] = teacher_identity
    names = sorted(set(tensor_files(checkpoint).values()))
    index = checkpoint / SAFE_TENSORS_INDEX_NAME
    if index.is_file():
        names.append(index.name)

    output.mkdirs()
    for name in names:
        (output / name).upload_from(str(checkpoint / name))
    (output / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    return DSparkExport(
        steps=state["global_step"],
        validation=metrics,
        trainer_packages=packages,
        speculators_revision=SPECULATORS_REVISION,
        vllm_source_commit=VLLM_SOURCE_COMMIT,
    )


def _teacher_environment(teacher: Path, recipe: DraftRecipe, hidden_states: Path, devices: str) -> VllmEnvironment:
    speculative, transfer = extraction_config(recipe, hidden_states)
    launcher = VllmLauncherWithEnvironment(
        IsolatedCudaVllm(source=VllmType.MARIN_FORK),
        {
            "CUDA_VISIBLE_DEVICES": devices,
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": "",
            # FlashInfer's Ninja builds can otherwise exhaust the Iris task's memory.
            "MAX_JOBS": "2",
        },
    )
    return VllmEnvironment(
        InferenceModelConfig(
            name=teacher.name,
            path=str(teacher),
            engine_kwargs={"dtype": "bfloat16", "max_model_len": recipe.data.max_sequence_length + 1},
        ),
        port=_TEACHER_PORT,
        timeout_seconds=1800,
        launcher=launcher,
        extra_args=[
            "--enable-expert-parallel",
            "--tensor-parallel-size=1",
            f"--data-parallel-size={recipe.teacher_replicas}",
            "--attention-backend=FLASH_ATTN",
            "--moe-backend=triton",
            f"--speculative-config={json.dumps(speculative)}",
            f"--kv-transfer-config={json.dumps(transfer)}",
        ],
    )


def _write_config_and_build_command(
    teacher: Path,
    recipe: DraftRecipe,
    python: str,
    data: Path,
    scratch: Path,
    training: Path,
    hidden_states: StoragePath,
    *,
    train_ratio: float,
    steps: int,
    endpoint: str | None,
) -> list[str]:
    object_store = hidden_states.scheme in ("s3", "gs")
    train_config = {
        "speculator_type": _SPECULATOR_TYPE,
        "seed": recipe.seed,
        "verifier": {"verifier_name_or_path": str(teacher)},
        "draft": {
            "draft_config": str(data / DRAFT_CONFIG_FILE),
            "draft_vocab_size": recipe.draft_vocab_size,
            "d2t_path": str(scratch / _D2T_FILE),
            "t2d_path": str(scratch / _T2D_FILE),
            "mask_token_id": recipe.data.mask_token_id,
            "target_layer_ids": list(recipe.feature_taps),
        },
        "data": {
            "data_path": str(data / ARROW_DATASET_DIR),
            "train_data_ratio": train_ratio,
            "hidden_states_backend": OBJECT_STORE_BACKEND if object_store else "file",
            "total_seq_len": recipe.data.max_sequence_length,
            "max_anchors": recipe.max_anchors,
            "num_workers": 2,
            "prefetch_factor": 2,
        },
        "backend": {"hidden_states_path": str(hidden_states)},
        "generation": (
            {"vllm_endpoint": endpoint, "on_missing": "generate", "on_generate": "delete"}
            if endpoint is not None
            else {"on_missing": "raise"}
        ),
        "loss": {
            # Dynamo cannot trace repeated calls to the pinned fused loss autotuner.
            "loss_implementation": _LOSS_IMPLEMENTATION,
            "loss_fn": json.dumps({"ce": recipe.cross_entropy_weight, "tv": recipe.total_variation_weight}),
        },
        "optimizer": {"optimizer": recipe.optimizer, "lr": recipe.learning_rate},
        "scheduler": {"scheduler_type": recipe.lr_schedule},
        "trainer": {
            "save_path": str(training),
            "no_resume_from_checkpoint": True,
            "epochs": _EPOCHS,
            "max_steps": steps,
            "gradient_checkpointing": True,
        },
        "dflash": {"block_size": recipe.block_size, "sample_from_anchor": _SAMPLE_FROM_ANCHOR},
        _SPECULATOR_TYPE: {
            "draft_layer_type": DRAFT_LAYER_TYPE,
            "markov_rank": recipe.markov_rank,
            "markov_head_type": _MARKOV_HEAD,
            "enable_confidence_head": _ENABLE_CONFIDENCE_HEAD,
            "confidence_head_with_markov": _CONFIDENCE_HEAD_WITH_MARKOV,
            "confidence_head_alpha": 1.0,
        },
    }
    config_path = scratch / "train.json"
    config_path.write_text(json.dumps(train_config))
    train = [python]
    if recipe.trainer_ranks > 1:
        train += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={recipe.trainer_ranks}"]
    if object_store:
        train.append(str(Path(__file__).with_name("speculators_storage.py")))
    else:
        train.extend(["-m", "speculators.train"])
    train.extend(["--config", str(config_path)])
    return train


def _prepare_trainer_inputs(
    python: str, data: Path, scratch: Path, recipe: DraftRecipe, train_rows: int, env: dict[str, str]
) -> tuple[int, dict[str, str]]:
    """Write vocabulary maps and return the capped step count and installed package versions."""
    result = json.loads(
        subprocess.check_output(
            [
                python,
                "-c",
                _PREPARE_TRAINER_INPUTS,
                str(data / ARROW_DATASET_DIR),
                str(train_rows),
                str(recipe.data.max_sequence_length),
                str(recipe.max_steps),
                str(data / TOKEN_FREQUENCIES_FILE),
                str(scratch / _D2T_FILE),
                str(scratch / _T2D_FILE),
                str(recipe.draft_vocab_size),
                str(recipe.data.shape.vocab_size),
                str(recipe.trainer_ranks),
            ],
            env=env,
            text=True,
        )
    )
    return result["steps"], result["packages"]


def _speculators_environment(devices: str) -> dict[str, str]:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": devices}
    env.pop("PYTHONPATH", None)
    return env


def _fit_and_publish(
    *,
    output: str,
    teacher_path: Path,
    teacher_identity: str,
    recipe: DraftRecipe,
    scratch: Path,
    hidden_states: StoragePath,
    split: PreparedSplits,
    devices: str,
    server_context: AbstractContextManager[VllmEnvironment | None],
) -> DSparkExport:
    env = _speculators_environment(devices)
    data = scratch / "data"
    training = scratch / "checkpoints"
    # Install the checkout's bucket routing in the isolated trainer environment.
    rigging_source = Path(rigging.__path__[0]).resolve().parents[1]
    extra_packages = (str(rigging_source),) if hidden_states.scheme in ("s3", "gs") else ()
    python = _install_speculators(scratch, env, extra_packages)
    steps, packages = _prepare_trainer_inputs(python, data, scratch, recipe, split.train, env)
    logger.info("Training DSpark for %s at %s for %d optimizer steps", teacher_identity, output, steps)
    with server_context as server:
        train = _write_config_and_build_command(
            teacher_path,
            recipe,
            python,
            data,
            scratch,
            training,
            hidden_states,
            train_ratio=split.train_ratio,
            steps=steps,
            endpoint=server.server_url if server is not None else None,
        )
        subprocess.run(train, env=env, check=True)
    return _publish_checkpoint(
        output=StoragePath(output) / HF_DRAFT_SUBDIRECTORY,
        training=training,
        teacher_identity=teacher_identity,
        recipe=recipe,
        expected_steps=steps,
        mapping=np.load(scratch / _D2T_FILE, allow_pickle=False),
        packages=packages,
    )


def _stage_frozen_weights(source: StoragePath, teacher_path: Path) -> None:
    teacher_path.mkdir()
    index = json.loads((source / SAFE_TENSORS_INDEX_NAME).read_text())
    tensors = (EMBEDDING, LM_HEAD, FINAL_NORM, GATE_DOWN, GATE_UP)
    weights = {name: index["weight_map"][name] for name in tensors}
    for file in (CONFIG_NAME, *sorted(set(weights.values()))):
        (source / file).download_to(str(teacher_path / file))
    (teacher_path / SAFE_TENSORS_INDEX_NAME).write_text(json.dumps({"weight_map": weights}))


def train_cached_and_export(
    *,
    output: str,
    teacher: str,
    teacher_identity: str,
    recipe: DraftRecipe,
    scratch: Path,
    hidden_states: StoragePath,
    split: PreparedSplits,
) -> DSparkExport:
    """Train on prepared Arrow data and indexed hidden states in object storage."""

    require_supported_vllm_release()
    _require_training_rows(split, recipe.trainer_ranks)
    teacher_path = scratch / "teacher"
    _stage_frozen_weights(StoragePath(teacher), teacher_path)
    devices = ",".join(_worker_gpu_devices().split(",")[: recipe.trainer_ranks])
    return _fit_and_publish(
        output=output,
        teacher_path=teacher_path,
        teacher_identity=teacher_identity,
        recipe=recipe,
        scratch=scratch,
        hidden_states=hidden_states,
        split=split,
        devices=devices,
        server_context=nullcontext(),
    )


def train_and_export(
    *,
    output: str,
    teacher: str,
    teacher_subdirectory: str,
    teacher_identity: str,
    prepared: str,
    recipe: DraftRecipe,
    tokenizer_source: str | None = None,
) -> DSparkExport:
    """Stage the pinned inputs, train on an Iris worker, and publish the HF draft."""

    require_supported_vllm_release()
    _require_single_worker_teacher(recipe)
    source = StoragePath(teacher) / teacher_subdirectory
    prepared_source = StoragePath(prepared) / PREPARED_DIR
    split = PreparedSplits(**json.loads((prepared_source / SPLITS_FILE).read_text()))
    _require_training_rows(split, recipe.trainer_ranks)
    teacher_devices, trainer_devices = _gpu_device_split(_worker_gpu_devices(), recipe)
    with TemporaryDirectory(prefix="marin-dspark-") as directory:
        scratch = Path(directory)
        hidden_states = scratch / HIDDEN_STATES_DIR
        hidden_states.mkdir()
        teacher_path = scratch / "teacher"
        prepared_source.download_to(str(scratch / "data"), recursive=True, batch_size=8)
        logger.info("Staging DSpark teacher %s", teacher_identity)
        source.download_to(str(teacher_path), recursive=True, batch_size=8)
        if tokenizer_source is not None:
            tokenizer_path = StoragePath(tokenizer_source)
            for pattern in ("tokenizer*", "chat_template*", "special_tokens_map.json", "added_tokens.json"):
                for file in (tokenizer_path / pattern).glob():
                    file.download_to(str(teacher_path / file.name))
        return _fit_and_publish(
            output=output,
            teacher_path=teacher_path,
            teacher_identity=teacher_identity,
            recipe=recipe,
            scratch=scratch,
            hidden_states=StoragePath(str(hidden_states)),
            split=split,
            devices=trainer_devices,
            server_context=_teacher_environment(teacher_path, recipe, hidden_states, teacher_devices),
        )
