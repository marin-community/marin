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
from marin.inference.vllm_server import IsolatedCudaVllm, VllmType
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.benchmarks.matched_comparison import compare_fixture

_INITIALIZATION_SEED = 17
_AUXILIARY_SEED = 23
_MANIFEST_FILENAME = "manifest.json"
_WORKLOAD_FILENAME = "workload.json"
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
        max_seq_len=128,
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
    (root / _MANIFEST_FILENAME).write_text(
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
    (root / _WORKLOAD_FILENAME).write_text(
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
    manifest = json.loads((root / _MANIFEST_FILENAME).read_text())
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
            str(root / _WORKLOAD_FILENAME),
            "--output",
            str(output),
            "--dtype",
            manifest["dtype"],
            "--hardware-label",
            hardware_label,
            "--expert-axis-size",
            str(expert_axis_size),
            "--warmup-batches",
            "2",
            "--measured-batches",
            "3",
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
) -> None:
    """Measure the exported checkpoint using the promoted Marin CUDA fork."""
    if compile_workers < 1:
        raise ValueError("compile_workers must be positive")
    manifest = json.loads((root / _MANIFEST_FILENAME).read_text())
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
                "max_model_len": 128,
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
    args = (
        "-m",
        "levanter.main.vllm_inference_benchmark",
        "--engine-args",
        str(engine_args),
        "--provenance",
        str(provenance),
        "--workload",
        str(root / _WORKLOAD_FILENAME),
        "--output",
        str(output),
        "--warmup-batches",
        "2",
        "--measured-batches",
        "3",
    )
    repo = Path(__file__).resolve().parents[2]
    paths = [str(repo / "lib/levanter/src"), str(repo / "lib/rigging/src")]
    environment = {**os.environ, **launcher.env()}
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
        subprocess.run(launcher.python_command(args), env=environment, check=True)
    print(output.read_text(), flush=True)


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
    compare = commands.add_parser("compare")
    compare.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_fixture(args.output.resolve(), args.recipe)
    elif args.command == "compare":
        compare_fixture(args.fixture.resolve())
    elif args.command == "native":
        run_native(args.fixture.resolve(), args.hardware_label, args.expert_axis_size)
    else:
        run_vllm(
            args.fixture.resolve(),
            args.hardware_label,
            args.execution_mode,
            args.expert_axis_size,
            args.kv_cache_memory_bytes,
            args.compile_workers,
        )


if __name__ == "__main__":
    main()
