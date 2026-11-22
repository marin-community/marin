# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run offline DSpark distillation on one allocated GPU host."""

import json
import logging
import math
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from inspect import getfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
from safetensors import safe_open

from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference import vllm_wheel_entrypoint
from marin.inference.vllm_release import current_vllm_gpu_wheel, vllm_gpu_wheel_provenance
from marin.training.dspark_data import (
    DRAFT_CONFIG_FILE,
    LM_HEAD,
    DraftRecipe,
    file_sha256,
    inspect_teacher,
    tensor_files,
    verify_sha256,
)

logger = logging.getLogger(__name__)

SPECULATORS_REVISION = "77eaa2390cab54ff7e5315345137394ee44b7a98"
VLLM_SOURCE_COMMIT = "47a8e415bff17b9376f02d9b92edcc67e962eaee"
_LOOPBACK_HOST = "127.0.0.1"
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


@dataclass(frozen=True)
class LocalGpuRuntime:
    """Executables and devices on a single already allocated host."""

    speculators_checkout: str
    trainer_python: str
    vllm_python: str
    teacher_devices: str
    trainer_devices: str
    port: int
    scratch_dir: str

    def validate(self, recipe: DraftRecipe) -> None:
        teacher = self.teacher_devices.split(",")
        trainer = self.trainer_devices.split(",")
        if (
            len(teacher) != recipe.teacher_replicas
            or len(trainer) != recipe.trainer_ranks
            or len(set(teacher + trainer)) != len(teacher) + len(trainer)
            or not all(device and device.strip() == device for device in teacher + trainer)
        ):
            raise ValueError("CUDA device lists must be disjoint and match the recipe's GPU counts")
        for path in (self.speculators_checkout, self.trainer_python, self.vllm_python, self.scratch_dir):
            if not Path(path).is_absolute():
                raise ValueError(f"Runtime paths must be absolute: {path}")


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
    """Build the published vLLM wheel's prefill-only hidden-state configuration."""

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
    output: Path,
    training: Path,
    provenance: TrainingProvenance,
    recipe: DraftRecipe,
    mapping: np.ndarray,
    packages: dict,
) -> None:
    checkpoint = training / "0"  # Speculators' terminal checkpoint for a one-epoch run.
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
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        names.append(index.name)

    with TemporaryDirectory(dir=output, prefix=".export-") as directory:
        export = Path(directory) / "hf"
        export.mkdir()
        (export / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in names:
            shutil.copyfile(checkpoint / name, export / name)
        receipt = {
            **asdict(provenance),
            "speculators_revision": SPECULATORS_REVISION,
            "vllm_source_commit": VLLM_SOURCE_COMMIT,
            "trainer_packages": packages,
            "steps": state["global_step"],
            "validation": metrics,
            "files": {name: file_sha256(export / name) for name in ["config.json", *names]},
        }
        (export / "dspark-export.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        export.rename(output / "hf")


def _teacher_command(teacher: str, recipe: DraftRecipe, runtime: LocalGpuRuntime, hidden_states: Path) -> list[str]:
    if VLLM_GPU_RELEASE.source_commit != VLLM_SOURCE_COMMIT:
        raise ValueError("Requalify DSpark against the newly promoted vLLM wheel before changing the pin")
    wheel = current_vllm_gpu_wheel(VLLM_GPU_RELEASE)
    wheel_provenance = vllm_gpu_wheel_provenance(VLLM_GPU_RELEASE, wheel)
    speculative, transfer = extraction_config(recipe, hidden_states)
    server_options = {
        "host": _LOOPBACK_HOST,
        "port": str(runtime.port),
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
        runtime.vllm_python,
        getfile(vllm_wheel_entrypoint),
        json.dumps(asdict(wheel_provenance)),
        "serve",
        teacher,
        "--enable-expert-parallel",
        *(f"--{key}={value}" for key, value in server_options.items()),
    ]


def _trainer_command(
    teacher: str,
    recipe: DraftRecipe,
    runtime: LocalGpuRuntime,
    data: Path,
    scratch: Path,
    training: Path,
    hidden_states: Path,
) -> list[str]:
    split = json.loads((data / "splits.json").read_text())
    train_config = {
        "speculator_type": "dspark",
        "seed": recipe.seed,
        "verifier": {"verifier_name_or_path": teacher},
        "draft": {
            "draft_config": str(data / DRAFT_CONFIG_FILE),
            "draft_vocab_size": recipe.draft_vocab_size,
            "d2t_path": str(scratch / _D2T_FILE),
            "t2d_path": str(scratch / _T2D_FILE),
            "mask_token_id": recipe.mask_token_id,
            "target_layer_ids": list(recipe.feature_taps),
        },
        "data": {
            "data_path": str(data / "dataset"),
            "train_data_ratio": split["train_ratio"],
            "hidden_states_backend": "file",
            "total_seq_len": recipe.max_sequence_length,
            "max_anchors": recipe.max_anchors,
            "num_workers": 2,
            "prefetch_factor": 2,
        },
        "backend": {"hidden_states_path": str(hidden_states)},
        "generation": {
            "vllm_endpoint": f"http://{_LOOPBACK_HOST}:{runtime.port}/v1",
            "on_missing": "generate",
            "on_generate": "delete",
        },
        "loss": {
            "loss_fn": json.dumps({"ce": recipe.cross_entropy_weight, "tv": recipe.total_variation_weight}),
        },
        "optimizer": {"optimizer": "muon", "lr": recipe.learning_rate},
        "scheduler": {"scheduler_type": "linear"},
        "trainer": {
            "save_path": str(training),
            "no_resume_from_checkpoint": True,
            "epochs": 1,
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
    train = [runtime.trainer_python]
    if recipe.trainer_ranks > 1:
        train += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={recipe.trainer_ranks}"]
    train += ["-m", "speculators.train", "--config", str(config_path)]
    return train


def train_and_export(
    *,
    output: str,
    teacher: str,
    provenance: TrainingProvenance,
    prepared: str,
    recipe: DraftRecipe,
    runtime: LocalGpuRuntime | None,
) -> None:
    """Train on fixed assistant responses and publish the validated draft atomically."""

    if runtime is None:
        raise ValueError("An allocated local GPU runtime is required for DSpark training")
    destination = Path(output)
    if (export := destination / "hf").exists():
        # The atomic rename can succeed before the artifact runner records success.
        receipt = json.loads((export / "dspark-export.json").read_text())
        if any(receipt[name] != value for name, value in asdict(provenance).items()):
            raise ValueError(f"Existing DSpark export at {export} has different inputs")
        for name, digest in receipt["files"].items():
            verify_sha256(export / name, digest)
        return

    runtime.validate(recipe)
    source = Path(runtime.speculators_checkout)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if head != SPECULATORS_REVISION or dirty:
        raise ValueError(f"Use a clean Speculators checkout at {SPECULATORS_REVISION}")
    decoder = inspect_teacher(Path(teacher), provenance.teacher_manifest_sha256, recipe)
    data = Path(prepared) / "prepared"
    if json.loads((data / DRAFT_CONFIG_FILE).read_text()) != decoder:
        raise ValueError("Prepared draft config disagrees with the teacher")

    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    env.pop("PYTHONPATH", None)
    vllm_env = {**env, "CUDA_VISIBLE_DEVICES": runtime.teacher_devices}
    trainer_env = {
        **env,
        "CUDA_VISIBLE_DEVICES": runtime.trainer_devices,
        "PYTHONPATH": os.pathsep.join(map(str, (source / "src", source / "hs_connectors" / "src"))),
    }
    with TemporaryDirectory(dir=runtime.scratch_dir, prefix="marin-dspark-") as directory:
        scratch = Path(directory)
        training = scratch / "checkpoints"
        hidden_states = scratch / "hidden_states"
        hidden_states.mkdir()
        packages = json.loads(
            subprocess.check_output(
                [
                    runtime.trainer_python,
                    "-c",
                    _BUILD_VOCAB_MAPPINGS,
                    str(data / "token_frequencies.json"),
                    str(scratch / _D2T_FILE),
                    str(scratch / _T2D_FILE),
                    str(recipe.draft_vocab_size),
                    str(recipe.shape.vocab_size),
                    str(recipe.trainer_ranks),
                ],
                env=trainer_env,
                text=True,
            )
        )
        inspect_teacher(Path(teacher), provenance.teacher_manifest_sha256, recipe, verify_weights=True)
        server = _teacher_command(teacher, recipe, runtime, hidden_states)
        train = _trainer_command(teacher, recipe, runtime, data, scratch, training, hidden_states)
        logger.info("Training DSpark for %s at %s", provenance.teacher_identity, output)
        with _teacher_server(server, vllm_env, runtime.port):
            subprocess.run(train, env=trainer_env, check=True)
        _publish_checkpoint(
            output=destination,
            training=training,
            provenance=provenance,
            recipe=recipe,
            mapping=np.load(scratch / _D2T_FILE, allow_pickle=False),
            packages=packages,
        )
