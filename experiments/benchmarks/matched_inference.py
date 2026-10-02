# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a small shared checkpoint and run the pinned vLLM comparison locally."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.hero import HeroConfig
from levanter.models.snowball import SnowballConfig
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT
from marin.inference.vllm_server import IsolatedCudaVllm, IsolatedTpuVllm, VllmType
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.benchmarks.matched_comparison import MANIFEST_FILENAME, WORKLOAD_FILENAME, compare_fixture

_WARMUP_BATCHES = 2
_MEASURED_BATCHES = 3
_INITIALIZATION_SEED = 17
_AUXILIARY_SEED = 23
_FIXTURE_MAX_SEQ_LEN = 128
_TINY_KV_CACHE_BYTES = 64 * 1024**2


def export_fixture(root: Path, recipe: str) -> None:
    """Write deterministic BF16 weights, tokenizer, and identical token workload."""
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    dimensions = dict(
        vocab_size=256,
        hidden_dim=256,
        intermediate_dim=256,
        shared_expert_intermediate_dim=256,
        num_experts=16,
        num_experts_per_token=4,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        max_seq_len=_FIXTURE_MAX_SEQ_LEN,
        sliding_window=16,
    )
    config = (
        HeroConfig(**dimensions, local_kv_heads=2, global_kv_heads=1, latent_dim=128)
        if recipe == "hero"
        else SnowballConfig(**dimensions)
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({f"t{i}": i for i in range(config.vocab_size)}, unk_token="t0")),
        unk_token="t0",
    )
    tokenizer.save_pretrained(checkpoint)
    config.to_hf_config(config.vocab_size).to_json_file(checkpoint / "config.json")
    with jax.set_mesh(compact_grug_mesh()):
        model = hax.named_jit(config.build)(
            hax.Axis("vocab", config.vocab_size), key=jax.random.key(_INITIALIZATION_SEED)
        )
        model = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16").cast_to_compute(model)
        state = model.to_state_dict()
        for index, (name, value) in enumerate(state.items()):
            if "sconv" in name or name.endswith(("attn_gate.weight", "router.bias")):
                state[name] = 0.1 * jax.random.normal(
                    jax.random.fold_in(jax.random.key(_AUXILIARY_SEED), index), value.shape, value.dtype
                )
        model = hax.named_jit(lambda m, weights: m.from_state_dict(weights))(model, state)
        # The promoted GrugMoE loader consumes [expert, output, input] banks.
        tensors = {name: np.ascontiguousarray(np.asarray(value)) for name, value in model.to_state_dict().items()}
        weights = checkpoint / "model.safetensors"
        save_file(tensors, str(weights), metadata={"format": "pt"})
        # Verify the exact exported weights through the native checkpoint reader.
        converter = HFCheckpointConverter.from_hf(str(checkpoint))
        loaded = converter.load_pretrained(config.model_type, config=config, dtype=jnp.bfloat16)
        expected = model.to_state_dict()
        for name, value in loaded.to_state_dict().items():
            np.testing.assert_array_equal(np.asarray(value), np.asarray(expected[name]), err_msg=name)
    identity = hashlib.sha256(weights.read_bytes()).hexdigest()
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "checkpoint": identity,
                "model_config": draccus.encode(config),
                "dtype": "bfloat16",
                "synthetic_seed": _INITIALIZATION_SEED,
                "auxiliary_seed": _AUXILIARY_SEED,
                "evidence_kind": "synthetic_checkpoint",
            },
            indent=2,
        )
        + "\n"
    )
    (root / WORKLOAD_FILENAME).write_text(
        json.dumps(
            {
                "prompts": [[1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18]],
                "output_tokens": 16,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"fixture": str(root), "checkpoint_identity": identity}), flush=True)


def run_native(root: Path, hardware_label: str, expert_axis_size: int) -> None:
    """Measure exactly the fixture's checkpoint and token workload in Levanter."""
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    output = root / "native-result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "levanter.main.inference_benchmark",
            "--checkpoint",
            str(root / "checkpoint"),
            "--checkpoint-identity",
            manifest["checkpoint"],
            "--workload",
            str(root / WORKLOAD_FILENAME),
            "--output",
            str(output),
            "--dtype",
            manifest["dtype"],
            "--hardware-label",
            hardware_label,
            "--expert-axis-size",
            str(expert_axis_size),
            "--warmup-batches",
            str(_WARMUP_BATCHES),
            "--measured-batches",
            str(_MEASURED_BATCHES),
        ],
        check=True,
    )
    print(output.read_text(), flush=True)


def run_vllm(
    root: Path,
    hardware_label: str,
    execution_mode: str,
    expert_axis_size: int,
    kv_cache_memory_bytes: int,
    compile_workers: int,
    flashinfer_jit_cache_wheel: str | None,
) -> None:
    """Measure the exported checkpoint using the promoted Marin CUDA fork."""
    if compile_workers < 1:
        raise ValueError("compile_workers must be positive")
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    manifest["hardware_label"] = hardware_label
    provenance = root / "vllm-provenance.json"
    engine_args = root / "vllm-engine.json"
    engine_args.write_text(
        json.dumps(
            {
                "model": str(root / "checkpoint"),
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "data_parallel_size": expert_axis_size,
                "data_parallel_size_local": expert_axis_size,
                "enable_expert_parallel": expert_axis_size > 1,
                "max_model_len": _FIXTURE_MAX_SEQ_LEN,
                "max_num_seqs": 2,
                "max_num_batched_tokens": 256,
                "enable_prefix_caching": False,
                "enforce_eager": execution_mode == "eager",
                "kv_cache_memory_bytes": kv_cache_memory_bytes,
            },
            indent=2,
        )
        + "\n"
    )
    output = root / "vllm-result.json"
    launcher = IsolatedCudaVllm(source=VllmType.MARIN_FORK)
    command = launcher.python_command(_vllm_benchmark_args(root))
    if flashinfer_jit_cache_wheel is not None:
        if "#sha256=" not in flashinfer_jit_cache_wheel:
            raise ValueError("The precompiled FlashInfer cache wheel requires an explicit SHA256 URL fragment")
        command[1:1] = ["--with", f"flashinfer-jit-cache @ {flashinfer_jit_cache_wheel}"]
        manifest["flashinfer_jit_cache_wheel"] = flashinfer_jit_cache_wheel
    repo = Path(__file__).resolve().parents[2]
    paths = [str(repo / "lib/levanter/src"), str(repo / "lib/rigging/src")]
    environment = {**os.environ, **launcher.env()}
    if flashinfer_jit_cache_wheel is not None:
        # Fail on a missing/incompatible precompiled module instead of starting another long build.
        environment["FLASHINFER_DISABLE_JIT"] = "1"
    environment["MAX_JOBS"] = str(compile_workers)
    environment["FLASHINFER_NVCC_THREADS"] = "1"
    manifest["compiler_parallelism"] = {"ninja_workers": compile_workers, "nvcc_threads": 1}
    environment["PYTHONPATH"] = os.pathsep.join([*paths, environment.get("PYTHONPATH", "")])
    # Isolate Triton from cache overrides inherited from the parent JAX process.
    with tempfile.TemporaryDirectory(prefix="matched-vllm-triton-") as cache:
        manifest["triton_cache"] = {
            "policy": "fresh_per_invocation",
            "inherited_directory": environment.get("TRITON_CACHE_DIR"),
            "effective_directory": cache,
        }
        environment["TRITON_CACHE_DIR"] = cache
        provenance.write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run(command, env=environment, check=True)
    print(output.read_text(), flush=True)


def _vllm_benchmark_args(root: Path) -> tuple[str, ...]:
    return (
        "-m",
        "levanter.main.vllm_inference_benchmark",
        "--engine-args",
        str(root / "vllm-engine.json"),
        "--provenance",
        str(root / "vllm-provenance.json"),
        "--workload",
        str(root / WORKLOAD_FILENAME),
        "--output",
        str(root / "vllm-result.json"),
        "--warmup-batches",
        str(_WARMUP_BATCHES),
        "--measured-batches",
        str(_MEASURED_BATCHES),
    )


def run_tpu_vllm(root: Path, hardware_label: str, data_parallel_size: int, kv_cache_memory_bytes: int) -> None:
    """Run the pinned Torchax Snowball path with single-process SPMD data parallelism."""
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    if manifest["model_config"].get("sconv", False):
        raise ValueError("Pinned TPU runtime has no Torchax grug_moe_short_conv implementation; Hero is unsupported")
    launcher = IsolatedTpuVllm(VLLM_FORK_REQUIREMENT, TPU_INFERENCE_FORK_REQUIREMENT)
    repo = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        **launcher.env(),
        "MODEL_IMPL_TYPE": "vllm",
        "TPU_MULTIPROCESS_DP": "0",
        "NEW_MODEL_DESIGN": "1",
    }
    paths = [str(repo / f"lib/{name}/src") for name in ("levanter", "rigging", "finestore")]
    environment["PYTHONPATH"] = os.pathsep.join([*paths, environment.get("PYTHONPATH", "")])
    dependencies = [
        str(repo / "lib/haliax"),
        str(repo / "lib/finestore"),
        "jax==0.11.0",
        "jaxlib==0.11.0",
        "libtpu==0.0.44",
    ]
    command = launcher.python_command(_vllm_benchmark_args(root))
    for dependency in dependencies:
        command[1:1] = ["--with", dependency]
    # Discover in a separate process that exits before the serving worker acquires libtpu.
    discovery = launcher.python_command(
        (
            "-c",
            "import json,jax; print(json.dumps(["
            "{'id': d.id, 'kind': d.device_kind, 'platform': d.platform} for d in jax.devices()]))",
        )
    )
    for dependency in dependencies:
        discovery[1:1] = ["--with", dependency]
    devices = json.loads(subprocess.check_output(discovery, env=environment, text=True))
    if len(devices) != data_parallel_size or any(device["platform"] != "tpu" for device in devices):
        raise ValueError(f"TPU data parallel size must match all visible TPU devices: {devices}")
    manifest.update(
        {
            "hardware_label": hardware_label,
            "tpu_device_discovery": devices,
            "tpu_runtime": {
                "vllm_requirement": launcher.vllm_ref,
                "tpu_inference_requirement": launcher.tpu_inference_ref,
                "dependencies": dependencies,
                "model_impl_type": environment["MODEL_IMPL_TYPE"],
                "multiprocess_dp": environment["TPU_MULTIPROCESS_DP"],
                "new_model_design": environment["NEW_MODEL_DESIGN"],
            },
        }
    )
    (root / "vllm-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    engine_args = {
        "model": str(root / "checkpoint"),
        "dtype": manifest["dtype"],
        "tensor_parallel_size": 1,
        "data_parallel_size": data_parallel_size,
        "data_parallel_size_local": data_parallel_size,
        "enable_expert_parallel": False,
        "max_model_len": _FIXTURE_MAX_SEQ_LEN,
        "max_num_seqs": max(2, data_parallel_size),
        "max_num_batched_tokens": 256,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
    }
    (root / "vllm-engine.json").write_text(json.dumps(engine_args, indent=2) + "\n")
    subprocess.run(command, env=environment, check=True)
    print((root / "vllm-result.json").read_text(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--recipe", choices=["snowball", "hero"], required=True)
    export.add_argument("--output", type=Path, required=True)
    for backend in ("native", "vllm"):
        run = commands.add_parser(backend)
        run.add_argument("--fixture", type=Path, required=True)
        run.add_argument("--hardware-label", required=True)
        run.add_argument("--expert-axis-size", type=int, default=1)
        if backend == "vllm":
            run.add_argument("--execution-mode", choices=["eager", "compiled"], default="eager")
            run.add_argument("--kv-cache-memory-bytes", type=int, default=_TINY_KV_CACHE_BYTES)
            run.add_argument("--compile-workers", type=int, default=2)
            run.add_argument(
                "--flashinfer-jit-cache-wheel",
                help="Exact compatible cache wheel URL with SHA256; requires precompiled modules and disables JIT",
            )
    tpu = commands.add_parser("vllm-tpu")
    tpu.add_argument("--fixture", type=Path, required=True)
    tpu.add_argument("--hardware-label", required=True)
    tpu.add_argument("--data-parallel-size", type=int, required=True)
    tpu.add_argument("--kv-cache-memory-bytes", type=int, default=_TINY_KV_CACHE_BYTES)
    compare = commands.add_parser("compare")
    compare.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_fixture(args.output.resolve(), args.recipe)
    elif args.command == "compare":
        compare_fixture(args.fixture.resolve())
    elif args.command == "native":
        run_native(args.fixture.resolve(), args.hardware_label, args.expert_axis_size)
    elif args.command == "vllm-tpu":
        run_tpu_vllm(args.fixture.resolve(), args.hardware_label, args.data_parallel_size, args.kv_cache_memory_bytes)
    else:
        run_vllm(
            args.fixture.resolve(),
            args.hardware_label,
            args.execution_mode,
            args.expert_axis_size,
            args.kv_cache_memory_bytes,
            args.compile_workers,
            args.flashinfer_jit_cache_wheel,
        )


if __name__ == "__main__":
    main()
