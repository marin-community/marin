# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite tiny-model lifecycle qualification; this does not launch hardware jobs.

The default run requires one host with eight H100s. It compares the prior six
cleanup operations with explicit pinned-JAX backend release. This qualifies
process teardown and checkpoint continuation, not full Snowball67B acceptance.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import haliax as hax
import jax
import jaxlib
import numpy as np
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from marin.rl.levanter_worker_protocol import LearnerProcess
from marin.rl.levanter_worker_runtime import NativePolicyRuntime, _payload_fingerprint
from marin.rl.levanter_worker_runtime import main as worker_main
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

_MODULE = "experiments.post_training.inventory.learner_lifecycle_qualification"


class BaselineRuntime(NativePolicyRuntime):
    """Keep the six cleanup operations used by source8dde, without backend release."""

    def _release_backend_clients(self) -> None:
        pass


def _initializer(path: Path) -> None:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 64)}}, unk_token="<pad>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(path)
    config = SnowballConfig(
        vocab_size=64,
        hidden_dim=128,
        intermediate_dim=256,
        shared_expert_intermediate_dim=128,
        num_layers=2,
        num_experts=32,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=2,
        head_dim=64,
        max_seq_len=8,
        sliding_window=8,
        attention_implementation="reference",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 64), config, key=jax.random.key(31))
        config.hf_checkpoint_converter().replaced(tokenizer=tokenizer).save_pretrained(model, str(path))


def _configuration(initializer: Path, output: Path, platform: str, devices: int) -> dict:
    return {
        "runtime": {
            "allocator": "vmm",
            "preallocate": False,
            "device_memory_fraction": 0.9,
            "numpy_madvise_hugepage": False,
            "validate_checkpoint_roundtrip": False,
            "mesh_axes": {"replica_dcn": 1, "replica": 1, "data": 1, "expert": devices, "context": 1, "model": 1},
            "checkpoint_dir": str(output),
            "projection_block_size": 16,
            "projection_logits_dtype": "bfloat16",
            "attention_implementation": "gpu_fa4_cute" if platform == "gpu" else "reference",
            "moe_implementation": "ring_gather_combine",
            "router_dot_fp32": True,
            "moe_capacity_factor": 8.0,
            "fp32_activations": False,
            "head_gate_fp32": True,
            "xsa_fp32": True,
        },
        "distributed": {
            "num_processes": 1,
            "process_id": 0,
            "local_device_count": devices,
            "coordinator_address": None,
        },
        "initialization_timeout_seconds": 1800,
        "run_id": "native-lifecycle-qualification",
        "seed": 41,
        "bf16": True,
        "num_training_steps": 3,
        "mini_batch_size": devices * 2,
        "micro_batch_size_per_device": 1,
        "eps_clip_low": 0.2,
        "eps_clip_high": 0.2,
        "require_accelerator": platform == "gpu",
        "model": {"path": str(initializer), "revision": None},
        "optimizer": {
            "max_grad_norm": 0.5,
            "lr": 0.001,
            "adam_betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
    }


def _environment(platform: str, python_executable: str) -> dict[str, str]:
    environment = dict(os.environ)
    # Match the production worker's isolation without importing Torch or Ray.
    for name in (
        "CUDA_HOME",
        "NVRTC_HOME",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "JAX_PLATFORM_NAME",
        "JAX_CUDA_VISIBLE_DEVICES",
        "JAX_ROCM_VISIBLE_DEVICES",
        "JAX_LOCAL_DEVICE_IDS",
    ):
        environment.pop(name, None)
    environment["PATH"] = f"{Path(python_executable).parent}{os.pathsep}{os.defpath}"
    environment.pop("XLA_CLIENT_MEM_FRACTION", None)
    environment.update(
        JAX_PLATFORMS="cuda,cpu" if platform == "gpu" else "cpu",
        XLA_PYTHON_CLIENT_ALLOCATOR="vmm",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.9",
        TF_FORCE_UNIFIED_MEMORY="false",
        NUMPY_MADVISE_HUGEPAGE="0",
    )
    return environment


def _frozen_batch(devices: int) -> dict:
    batch = devices * 2
    return {
        "arrays": {
            "tokens": np.tile(np.arange(2, 10, dtype=np.int32), (batch, 1)),
            "attention_mask": np.ones((batch, 8), dtype=np.bool_),
            "position_ids": np.tile(np.arange(8, dtype=np.int32), (batch, 1)),
            "response_mask": np.ones((batch, 2), dtype=np.bool_),
            "advantages": np.tile(np.array([0.5, -0.25], dtype=np.float32), (batch, 1)),
            "policy_weights": np.full((batch, 2), 1 / (batch * 2), dtype=np.float32),
            "kl_weights": np.zeros((batch, 2), dtype=np.float32),
            "reference_logprobs": None,
        },
        "response_length": 2,
        "global_step": 1,
    }


def _start(python: str, environment: dict, configuration: dict, baseline: bool) -> tuple[LearnerProcess, dict]:
    child = LearnerProcess(
        python,
        environment=environment,
        timeout_seconds=1800,
        runtime_module=_MODULE if baseline else "marin.rl.levanter_worker_runtime",
    )
    try:
        identity = child.request("initialize", configuration)
        numpy_policy = identity["numpy_allocation_policy"]
        if (
            identity["jax_version"] != "0.11.0"
            or identity["jaxlib_version"] != "0.11.0"
            or numpy_policy["numpy_version"] != "2.3.5"
            or numpy_policy["observed_madvise_hugepage"]
        ):
            raise RuntimeError("Native child package versions or NumPy allocation policy differ")
        devices = configuration["distributed"]["local_device_count"]
        expected_backend = "gpu" if configuration["require_accelerator"] else "cpu"
        if (
            identity["local_devices"] != devices
            or identity["global_devices"] != devices
            or identity["process_count"] != 1
            or identity["process_id"] != 0
            or identity["mesh"] != configuration["runtime"]["mesh_axes"]
            or identity["backend"] != expected_backend
            or not identity["device_kinds"]
        ):
            raise RuntimeError("Native child backend/device/process/mesh identity differs")
        return child, identity
    except BaseException:
        child.close()
        raise


def _exit(child: LearnerProcess) -> dict:
    pid = child.process.pid
    start = time.monotonic()
    child.request("shutdown", {})
    return {"pid": pid, "exit_code": child.process.returncode, "request_and_exit_seconds": time.monotonic() - start}


def _persist_arm(directory: Path, row: dict, stage: str) -> None:
    row["stage"] = stage
    (directory.parent / f"lifecycle-{directory.name}.json").write_text(json.dumps(row, indent=2) + "\n")


def _arm(python: str, environment: dict, config: dict, directory: Path, baseline: bool) -> dict:
    directory.mkdir(parents=True)
    child = None
    exits = []
    row: dict = {"status": "running", "exits": exits}
    _persist_arm(directory, row, "initialize")
    try:
        child, identity = _start(python, environment, config, baseline)
        row["initial_runtime"] = identity
        _persist_arm(directory, row, "score_frozen_batch")
        if config["require_accelerator"] and any("H100" not in kind for kind in identity["device_kinds"]):
            raise ValueError("Lifecycle qualification requires actual H100 devices")
        payload = _frozen_batch(config["distributed"]["local_device_count"])
        payload["arrays"]["old_logprobs"] = child.request("forward", payload)["output"]
        row["frozen_batch_sha256"] = _payload_fingerprint(payload)
        _persist_arm(directory, row, "first_update")
        update1 = child.request("train", payload)
        if update1["frozen_batch_sha256"] != row["frozen_batch_sha256"]:
            raise RuntimeError("Native update changed the supplied frozen policy batch")
        row["updates"] = [update1]
        _persist_arm(directory, row, "save_checkpoint")
        checkpoint = {"ckpt_dir": str(directory / "step1"), "include_fingerprints": True}
        saved = child.request("save_checkpoint", checkpoint)
        row["checkpoint"] = saved
        _persist_arm(directory, row, "first_natural_exit")
        exits.append(_exit(child))
        child.close()
        _persist_arm(directory, row, "fresh_initialize")
        child, restored_identity = _start(python, environment, config, baseline)
        row["restored_runtime"] = restored_identity
        if child.process.pid == exits[0]["pid"]:
            raise RuntimeError("Checkpoint restart reused the old PID")
        _persist_arm(directory, row, "restore_checkpoint")
        if child.request("load_checkpoint", checkpoint) != saved:
            raise RuntimeError("Model/Adam/RNG/step/averaging checkpoint restoration differs")
        row["restore_exact"] = True
        _persist_arm(directory, row, "continued_update")
        payload["global_step"] = 2
        continued = child.request("train", payload)
        if continued["frozen_batch_sha256"] != update1["frozen_batch_sha256"]:
            raise RuntimeError("Checkpoint continuation changed the frozen policy batch")
        continued_state = child.request("state_fingerprints", {})
        continued_scores = child.request("forward", payload)["output"]
        row["continued_state"] = continued_state
        row["updates"].append(continued)
        _persist_arm(directory, row, "restored_natural_exit")
        exits.append(_exit(child))
        child.close()
        _persist_arm(directory, row, "uninterrupted_control")
        child, _ = _start(python, environment, config, baseline)
        payload["global_step"] = 1
        if child.request("train", payload) != update1:
            raise RuntimeError("Fresh control first update differs")
        payload["global_step"] = 2
        if child.request("train", payload) != continued:
            raise RuntimeError("Continued update differs from uninterrupted control")
        if child.request("state_fingerprints", {}) != continued_state:
            raise RuntimeError("Continued full state differs from uninterrupted control")
        np.testing.assert_array_equal(child.request("forward", payload)["output"], continued_scores)
        row["continuation_exact"] = True
        _persist_arm(directory, row, "control_natural_exit")
        exits.append(_exit(child))
        row["status"] = "passed"
        _persist_arm(directory, row, "complete")
        return row
    except Exception as error:
        row.update(status="failed", error_type=type(error).__name__, error=str(error)[:4000])
        if child is not None:
            row["failed_child"] = {"pid": child.process.pid, "exit_code": child.process.poll()}
        _persist_arm(directory, row, row["stage"])
        return row
    finally:
        if child is not None:
            child.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("IRIS_OUTPUT_DIR", ".")))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu", help="CPU is local validation only")
    parser.add_argument("--local-devices", type=int, default=8)
    args = parser.parse_args()
    if args.platform == "gpu" and args.local_devices != 8:
        parser.error("GPU qualification requires exactly eight local H100 devices")
    if jax.__version__ != "0.11.0" or jaxlib.__version__ != "0.11.0" or np.__version__ != "2.3.5":
        raise RuntimeError("Lifecycle qualification requires JAX/jaxlib0.11.0 and NumPy2.3.5")
    jax.config.update("jax_platforms", "cpu")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initializer = output / "tiny-initializer"
    _initializer(initializer)
    gc.collect()
    config = _configuration(initializer, output, args.platform, args.local_devices)
    receipt = {
        "scope": "tiny native process lifecycle only; not full67B acceptance",
        "platform": args.platform,
        "expected_local_devices": args.local_devices,
        "baseline": "source8dde cleanup-equivalent; whole-file identity differs",
        "post_ack_exit_grace_seconds": 5,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "arms": {},
    }
    for name, baseline in (("baseline", True), ("extension", False)):
        row = _arm(
            args.python_executable, _environment(args.platform, args.python_executable), config, output / name, baseline
        )
        receipt["arms"][name] = row
        (output / f"lifecycle-{name}.json").write_text(json.dumps(row, indent=2) + "\n")
        (output / "lifecycle-qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"arm": name, "status": row["status"]}), flush=True)
    baseline_hash = receipt["arms"]["baseline"].get("frozen_batch_sha256")
    extension_hash = receipt["arms"]["extension"].get("frozen_batch_sha256")
    comparison_available = baseline_hash is not None and extension_hash is not None
    frozen_equal = baseline_hash == extension_hash if comparison_available else None
    candidate_qualified = receipt["arms"]["extension"]["status"] == "passed"
    receipt["candidate_qualified"] = candidate_qualified
    receipt["baseline_comparison_available"] = comparison_available
    receipt["frozen_batch_equal_across_arms"] = frozen_equal
    receipt["comparison_status"] = "equal" if frozen_equal else "different" if comparison_available else "unavailable"
    receipt["status"] = "failed"
    if candidate_qualified and frozen_equal is not False:
        receipt["status"] = "passed" if comparison_available else "candidate_only"
    (output / "lifecycle-qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not candidate_qualified or frozen_equal is False:
        raise RuntimeError("Supported-extension lifecycle candidate failed; inspect lifecycle-qualification.json")
    print(json.dumps({"receipt": str(output / "lifecycle-qualification.json"), "scope": receipt["scope"]}))


if __name__ == "__main__":
    if "--response-fd" in sys.argv:
        worker_main(runtime_factory=BaselineRuntime)
    else:
        main()
