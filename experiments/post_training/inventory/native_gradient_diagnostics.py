# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite reserved-device gradient diagnostics; never launches jobs.

Old tiny-model, direct FA4 VJP, and aligned D128/GQA4 controls remain distinct.
Reference attention is an explicitly labeled diagnostic oracle, never a training
fallback. Raw-gradient graphs are diagnostics; aligned lifecycle uses the normal
learner update/checkpoint path and the unchanged five-second exit requirement.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
from levanter.cutlass_kernel_cache import gpu_compute_capability
from levanter.grug.attention import AttentionMask, attention, reference_attention
from levanter.models.snowball import SnowballConfig
from levanter.trainer_state import trainables_only
from levanter.utils.tree_utils import inference_mode
from marin.rl.levanter_worker_runtime import (
    NativePolicyRuntime,
    _allocator_profile,
    _numpy_allocation_policy,
    _payload_fingerprint,
    _validate_allocator,
)

from experiments.post_training.inventory.actual_attention_probe import (
    capture_backward_residuals,
    capture_raw_case,
    replay_backward_residuals,
    replay_captures,
)
from experiments.post_training.inventory.learner_lifecycle_qualification import (
    PairedOracleRuntime,
    _arm,
    _configuration,
    _environment,
    _frozen_batch,
    _initializer,
    paired_resume_oracle,
    worker_main,
)

_MODULE = "experiments.post_training.inventory.native_gradient_diagnostics"
_INITIALIZER_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
_FA4_CASES = (
    ("diagnostic_s8_d64_mha", 8, 2, 2, 64, "segmented_fallback_64x64"),
    ("s64_d64_mha_control", 64, 2, 2, 64, "segmented_fallback_64x64"),
    ("s128_d128_gqa4_production_heads", 128, 20, 5, 128, "native_sm90_segmented"),
)
# These are the existing FA4 BF16 GPU test thresholds, not a new acceptance tolerance.
ATOL = RTOL = 7e-2


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def initializer_identity(path: Path) -> dict:
    """Hash the five HF files using a canonical filename-to-content digest map."""
    files = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in _INITIALIZER_FILES}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    files = (
        "experiments/post_training/inventory/native_gradient_diagnostics.py",
        "experiments/post_training/inventory/actual_attention_probe.py",
        "experiments/post_training/inventory/learner_lifecycle_qualification.py",
        "lib/marin/src/marin/rl/levanter_worker_runtime.py",
        "lib/marin/src/marin/rl/levanter_worker_protocol.py",
        "lib/levanter/src/levanter/trainer.py",
        "lib/levanter/src/levanter/policy_model.py",
        "lib/levanter/src/levanter/models/snowball.py",
        "lib/levanter/src/levanter/grug/attention/_fa4_cute.py",
        "lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py",
        "lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py",
        "lib/levanter/src/levanter/grug/attention/_fa4_cute_config.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def _array_stats(array) -> dict:
    raw = np.asarray(array)
    value = raw.astype(np.float32)
    finite = np.isfinite(value)
    return {
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "elements": int(raw.size),
        "finite": int(np.count_nonzero(finite)),
        "nan": int(np.count_nonzero(np.isnan(value))),
        "inf": int(np.count_nonzero(np.isinf(value))),
        "finite_abs_max": float(np.max(np.abs(value[finite]))) if np.any(finite) else None,
        "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
    }


def _scalar(value) -> dict:
    value = float(value)
    return {
        "finite": bool(np.isfinite(value)),
        "value": value if np.isfinite(value) else None,
        "nonfinite_kind": None if np.isfinite(value) else "nan" if np.isnan(value) else "inf",
    }


def _device_identity(platform: str) -> dict:
    devices = jax.local_devices()
    expected = "gpu" if platform == "gpu" else "cpu"
    if jax.default_backend() != expected or jax.device_count() != 8 or len(devices) != 8 or jax.process_count() != 1:
        raise RuntimeError("Gradient suite requires one process and exactly eight local reserved devices")
    if platform == "gpu" and any("H100" not in d.device_kind for d in devices):
        raise RuntimeError("Gradient suite requires actual H100 devices")
    if jax.__version__ != "0.11.0" or jaxlib.__version__ != "0.11.0" or np.__version__ != "2.3.5":
        raise RuntimeError("Gradient suite package pins differ from the reviewed native environment")
    return {
        "backend": expected,
        "process_count": jax.process_count(),
        "local_devices": len(devices),
        "global_devices": jax.device_count(),
        "device_kinds": sorted({d.device_kind for d in devices}),
        "packages": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    }


def _raw_case(initializer: Path, output: Path, platform: str, sequence_length: int) -> dict:
    config = _configuration(initializer, output.parent, platform, 8)
    runtime = NativePolicyRuntime()
    result = {
        "status": "running",
        "stage": "initialize",
        "scope": "actual Trainer raw pre-clip diagnostic, not normal update acceptance",
        "sequence_length": sequence_length,
        "initializer": initializer_identity(initializer),
        "source_hashes": _source_hashes(),
    }
    _write(output, result)
    try:
        result["runtime"] = runtime.initialize(config)
        result["devices"] = _device_identity(platform)
        payload = _frozen_batch(8, sequence_length)
        payload["arrays"]["old_logprobs"] = runtime.forward(payload)["output"]
        result["frozen_batch_sha256"] = _payload_fingerprint(payload)
        result["stage"] = "raw_gradients"
        result["geometry"] = {
            "global_batch": 16,
            "execution_microbatch": 8,
            "temporal_microbatches": 2,
            "mesh": result["runtime"]["mesh"],
            "attention_implementation": config["runtime"]["attention_implementation"],
            "moe_implementation": config["runtime"]["moe_implementation"],
            "model_config": dataclasses.asdict(runtime.state.model.config),
        }
        _write(output, result)
        plan = runtime._prepare_projection(payload, "train")
        batch = runtime._batch(payload)
        trainer = runtime.trainer

        def raw_gradients(state, batch):
            key, _ = jax.random.split(state.training_key)
            model = inference_mode(state.model, False)
            loss, grads, _ = trainer._compute_gradients_microbatched(
                trainer.loss_fn, model, batch, key=key, projection_plan=plan
            )
            return loss, grads, optax.global_norm(trainables_only(grads, state.is_trainable))

        compute = hax.named_jit(
            raw_gradients,
            axis_resources=trainer.parameter_axis_mapping,
            out_axis_resources=trainer.parameter_axis_mapping,
        )
        loss, grads, norm = compute(runtime.state, batch)
        jax.block_until_ready((loss, grads, norm))
        rows = []
        for path, value in jax.tree_util.tree_flatten_with_path(grads)[0]:
            if eqx.is_array(value):
                rows.append({"path": jax.tree_util.keystr(path), **_array_stats(value)})
        result.update(
            stage="complete",
            loss=_scalar(loss),
            trainable_global_norm=_scalar(norm),
            gradient_arrays=rows,
            raw_gradient_nonfinite=sum(row["nan"] + row["inf"] for row in rows),
        )
        result["status"] = (
            "finite"
            if result["loss"]["finite"]
            and result["trainable_global_norm"]["finite"]
            and result["raw_gradient_nonfinite"] == 0
            else "nonfinite"
        )
        _write(output, result)
        return result
    except Exception as error:
        result.update(
            status="error",
            error_type=type(error).__name__,
            error=str(error)[:4000],
            traceback=traceback.format_exc()[-8000:],
        )
        _write(output, result)
        raise
    finally:
        runtime.close()


def _aligned_config() -> SnowballConfig:
    return SnowballConfig(
        vocab_size=64,
        hidden_dim=512,
        intermediate_dim=256,
        shared_expert_intermediate_dim=128,
        num_layers=2,
        num_experts=32,
        num_experts_per_token=2,
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        max_seq_len=128,
        sliding_window=128,
        attention_implementation="reference",
    )


def _run_child(args, case: str, initializer: Path, fa4_case: str | None = None) -> dict:
    name = case if fa4_case is None else f"{case}-{fa4_case}"
    output = args.output_dir / (name + ".json")
    command = [
        args.python_executable,
        "-m",
        _MODULE,
        "--case",
        case,
        "--output-dir",
        str(args.output_dir),
        "--python-executable",
        args.python_executable,
        "--platform",
        args.platform,
        "--old-initializer",
        str(args.old_initializer),
        "--old-initializer-sha256",
        args.old_initializer_sha256,
        "--case-initializer",
        str(initializer),
    ]
    if fa4_case is not None:
        command.extend(["--fa4-case", fa4_case])
    if args.capture_replay_states:
        command.append("--capture-replay-states")
    if args.paired_resume_oracle:
        command.append("--paired-resume-oracle")
    with (args.output_dir / (name + ".log")).open("wb") as log:
        try:
            process = subprocess.run(
                command,
                env=_environment(args.platform, args.python_executable),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = None
    if output.exists():
        result = json.loads(output.read_text())
    else:
        result = {"status": "error", "error_type": "missing_case_receipt"}
    result["diagnostic_child_exit_code"] = code
    return result


def _suite(args) -> dict:
    jax.config.update("jax_platforms", "cpu")
    old_identity = initializer_identity(args.old_initializer)
    if old_identity["sha256"] != args.old_initializer_sha256:
        raise ValueError("Archived old initializer differs from the frozen five-file input identity")
    aligned = args.output_dir / "aligned-initializer"
    _initializer(aligned, _aligned_config())
    result = {
        "status": "running",
        "scope": "tiny gradient/kernel diagnostics plus separately qualified aligned lifecycle; not full67B acceptance",
        "platform": args.platform,
        "old_initializer": old_identity,
        "cases": {},
        "aligned_lifecycle_qualified": False,
        "capture_replay_states": args.capture_replay_states,
        "replay_capture_scope": (
            "Added serialization/fingerprint synchronization can perturb GPU timing/layout; "
            "exact acceptance remains unchanged, and a captured pass does not explain the uncaptured mismatch."
            if args.capture_replay_states
            else None
        ),
    }
    result["source_hashes"] = _source_hashes()
    target = args.output_dir / "gradient-suite.json"
    for case, initializer in (
        ("observed-raw" if args.capture_replay_states else "old-raw", args.old_initializer),
        ("fa4-direct", args.old_initializer),
        ("aligned-raw", aligned),
    ):
        if case == "fa4-direct" and args.capture_replay_states:
            result["cases"][case] = {"status": "not_requested", "scope": "Prior direct cases already completed"}
        elif case == "fa4-direct" and args.platform == "cpu":
            result["cases"][case] = {"status": "not_run_cpu", "scope": "No CPU replacement for FA4/CuTe"}
        elif case == "fa4-direct":
            cases = [_run_child(args, case, initializer, definition[0]) for definition in _FA4_CASES]
            result["cases"][case] = {
                "status": "passed" if all(row["status"] == "passed" for row in cases) else "failed",
                "isolated_case_processes": cases,
                "diagnostic_child_exit_code": (
                    0 if all(row["diagnostic_child_exit_code"] == 0 for row in cases) else None
                ),
                "execution_errors_present": any(
                    row.get("execution_errors_present", False) or row["status"] == "error" for row in cases
                ),
            }
            _write(args.output_dir / "fa4-direct.json", result["cases"][case])
        else:
            result["cases"][case] = _run_child(args, case, initializer)
        _write(target, result)
        print(json.dumps({"case": case, "status": result["cases"][case]["status"]}), flush=True)
        if case == "observed-raw" and (args.output_dir / "captures" / "captures.json").exists():
            result["cases"]["actual-replay"] = _run_child(args, "actual-replay", args.old_initializer)
            _write(target, result)
            print(
                json.dumps({"case": "actual-replay", "status": result["cases"]["actual-replay"]["status"]}), flush=True
            )
        if case == "aligned-raw" and args.capture_replay_states:
            result["cases"]["aligned-actual-replay"] = _run_child(args, "aligned-actual-replay", aligned)
            _write(target, result)
            print(
                json.dumps(
                    {"case": "aligned-actual-replay", "status": result["cases"]["aligned-actual-replay"]["status"]}
                ),
                flush=True,
            )
    if not args.raw_only and result["cases"]["aligned-raw"]["status"] == "finite":
        config = _configuration(aligned, args.output_dir, args.platform, 8)
        arms = {}
        requested_arms = (
            (("extension", False),) if args.capture_replay_states else (("baseline", True), ("extension", False))
        )
        for name, baseline in requested_arms:
            arms[name] = _arm(
                args.python_executable,
                _environment(args.platform, args.python_executable),
                config,
                args.output_dir / "aligned-lifecycle" / name,
                baseline,
                sequence_length=128,
                capture_replay_states=args.capture_replay_states,
            )
            result["aligned_lifecycle"] = arms
            _write(target, result)
        hashes = [arm.get("frozen_batch_sha256") for arm in arms.values()]
        result["aligned_lifecycle_qualified"] = (
            all(arm["status"] == "passed" for arm in arms.values()) and hashes[0] is not None and len(set(hashes)) == 1
        )
    result["all_requested_cases_completed"] = all(
        row["status"] != "error"
        and not row.get("execution_errors_present", False)
        and row.get("diagnostic_child_exit_code", 0) == 0
        for row in result["cases"].values()
    )
    result["numerical_findings_present"] = any(
        row["status"] in ("nonfinite", "failed") for row in result["cases"].values()
    )
    result["status"] = "completed" if result["all_requested_cases_completed"] else "completed_with_errors"
    result["aligned_lifecycle_skipped"] = args.raw_only or result["cases"]["aligned-raw"]["status"] != "finite"
    result["aligned_lifecycle_skip_reason"] = (
        "explicit_raw_only"
        if args.raw_only
        else "aligned_raw_not_finite" if result["cases"]["aligned-raw"]["status"] != "finite" else None
    )
    _write(target, result)
    return result


def _paired_suite(args) -> dict:
    """Keep same-A1 qualification separate from the existing stale-score control."""
    jax.config.update("jax_platforms", "cpu")
    old_identity = initializer_identity(args.old_initializer)
    if old_identity["sha256"] != args.old_initializer_sha256:
        raise ValueError("Archived initializer identity differs")
    aligned = args.output_dir / "aligned-initializer"
    _initializer(aligned, _aligned_config())
    result = {
        "status": "running",
        "scope": "Instrumented tiny same-A1 active-gradient resume, not full67B qualification",
        "platform": args.platform,
        "source_hashes": _source_hashes(),
        "old_initializer": old_identity,
        "paired_resume_qualified": False,
        "cases": {},
        "prior_fresh_init_stale_score_diagnostic": (
            "Unchanged separate --capture-replay-states mode; prior failure retained"
        ),
    }
    target = args.output_dir / "paired-probe.json"
    for case in ("backward-observed-raw", "backward-residual-replay"):
        result["cases"][case] = _run_child(args, case, args.old_initializer)
        _write(target, result)
        print(json.dumps({"case": case, "status": result["cases"][case]["status"]}), flush=True)
    config = _configuration(aligned, args.output_dir, args.platform, 8)
    try:
        result["paired_oracle"] = paired_resume_oracle(
            args.python_executable,
            _environment(args.platform, args.python_executable),
            config,
            args.output_dir / "paired-oracle",
            {"source_hashes": result["source_hashes"], "initializer": initializer_identity(aligned)},
        )
    except Exception as error:
        saved = args.output_dir / "paired-oracle" / "paired-resume.json"
        partial = json.loads(saved.read_text()) if saved.exists() else {"exits": []}
        partial.update(
            status="error",
            paired_resume_qualified=False,
            comparison_error={"type": type(error).__name__, "message": str(error)[:4000]},
        )
        result["paired_oracle"] = partial
        _write(saved, partial)
    result["paired_resume_qualified"] = result["paired_oracle"]["paired_resume_qualified"]
    paired = result["paired_oracle"]
    result["paired_execution_complete"] = (
        not any(key in paired for key in ("A_error", "B_error", "comparison_error"))
        and all(key in paired for key in ("A2", "B2", "checks", "state_deviations"))
        and len(paired["exits"]) == 2
        and all(x["exit_code"] == 0 and "error" not in x and "error_type" not in x for x in paired["exits"])
    )
    result["all_requested_cases_completed"] = all(
        row.get("diagnostic_child_exit_code") == 0
        and row["status"] != "error"
        and not row.get("execution_errors_present", False)
        for row in result["cases"].values()
    )
    result["all_requested_cases_completed"] &= result["paired_execution_complete"]
    result["status"] = "completed" if result["all_requested_cases_completed"] else "completed_with_errors"
    result["numerical_findings_present"] = any(
        row["status"] in ("nonfinite", "failed") for row in result["cases"].values()
    ) or (result["paired_execution_complete"] and not result["paired_resume_qualified"])
    _write(target, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("IRIS_OUTPUT_DIR", ".")))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--old-initializer", type=Path, required=True)
    parser.add_argument("--old-initializer-sha256", required=True, help="Canonical five-file directory digest")
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--case",
        choices=(
            "suite",
            "old-raw",
            "aligned-raw",
            "fa4-direct",
            "observed-raw",
            "actual-replay",
            "aligned-actual-replay",
            "backward-observed-raw",
            "backward-residual-replay",
        ),
        default="suite",
    )
    parser.add_argument("--case-initializer", type=Path)
    parser.add_argument("--fa4-case", choices=[definition[0] for definition in _FA4_CASES])
    parser.add_argument(
        "--capture-replay-states",
        action="store_true",
        help="Extension-only aligned replay with four checkpoint trees and exact per-leaf diagnostics",
    )
    parser.add_argument(
        "--raw-only", action="store_true", help="Explicit local diagnostic-only run, no lifecycle acceptance"
    )
    parser.add_argument(
        "--paired-resume-oracle", action="store_true", help="Separate same-A1 active-gradient resume gate"
    )
    args = parser.parse_args()
    if args.paired_resume_oracle and args.capture_replay_states:
        parser.error("Paired oracle and historical fresh-init replay are separate modes")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.case == "suite":
        if args.paired_resume_oracle:
            result = _paired_suite(args)
            print(
                json.dumps(
                    {
                        "receipt": str(args.output_dir / "paired-probe.json"),
                        "status": result["status"],
                        "paired_resume_qualified": result["paired_resume_qualified"],
                    }
                )
            )
            if not result["all_requested_cases_completed"]:
                raise SystemExit(1)
            return
        result = _suite(args)
        print(
            json.dumps(
                {
                    "receipt": str(args.output_dir / "gradient-suite.json"),
                    "status": result["status"],
                    "aligned_lifecycle_qualified": result["aligned_lifecycle_qualified"],
                }
            )
        )
        if not result["all_requested_cases_completed"]:
            raise SystemExit(1)
        return
    numpy_policy = _numpy_allocation_policy(False)
    _validate_allocator({"runtime": {"allocator": "vmm", "preallocate": False, "device_memory_fraction": 0.9}})
    identity = _device_identity(args.platform)
    if args.case == "backward-observed-raw":
        result = capture_backward_residuals(args.case_initializer, args.output_dir, args.platform)
        _write(args.output_dir / "backward-observed-raw.json", result)
    elif args.case == "backward-residual-replay":
        result = replay_backward_residuals(
            args.output_dir / "backward-captures",
            args.output_dir / "backward-residual-replay.json",
            implementation="gpu_fa4_cute" if args.platform == "gpu" else "reference",
        )
        result.update(
            source_hashes=_source_hashes(),
            native_device_identity=identity,
            allocation_policy=_allocator_profile(),
            numpy_allocation_policy=numpy_policy,
        )
        _write(args.output_dir / "backward-residual-replay.json", result)
    elif args.case in ("actual-replay", "aligned-actual-replay"):
        source = args.output_dir / "aligned-observed" if args.case == "aligned-actual-replay" else args.output_dir
        target = args.output_dir / (args.case + ".json")
        result = replay_captures(
            source / "captures",
            target,
            implementation="gpu_fa4_cute" if args.platform == "gpu" else "reference",
        )
        result.update(
            source_hashes=_source_hashes(),
            native_device_identity=identity,
            allocation_policy=_allocator_profile(),
            numpy_allocation_policy=numpy_policy,
        )
        _write(target, result)
    elif args.case == "observed-raw":
        if args.case_initializer is None:
            parser.error("Observed raw diagnostic requires --case-initializer")
        capture_raw_case(args.case_initializer, args.output_dir, args.platform)
    elif args.case == "aligned-raw" and args.capture_replay_states:
        if args.case_initializer is None:
            parser.error("Observed aligned raw diagnostic requires --case-initializer")
        result = capture_raw_case(args.case_initializer, args.output_dir / "aligned-observed", args.platform)
        _write(args.output_dir / "aligned-raw.json", result)
    elif args.case == "fa4-direct":
        name = "fa4-direct" if args.fa4_case is None else f"fa4-direct-{args.fa4_case}"
        target = args.output_dir / (name + ".json")
        result = run_fa4_probe(target, args.fa4_case)
        result.update(
            source_hashes=_source_hashes(),
            native_device_identity=identity,
            allocation_policy=_allocator_profile(),
            numpy_allocation_policy=numpy_policy,
        )
        _write(target, result)
    else:
        if args.case_initializer is None:
            parser.error("Raw diagnostic child requires --case-initializer")
        _raw_case(
            args.case_initializer,
            args.output_dir / (args.case + ".json"),
            args.platform,
            8 if args.case == "old-raw" else 128,
        )


def run_fa4_probe(output_path: str | Path, only_case: str | None = None) -> dict:
    """Persist bounded per-case statistics; return failed status on any mismatch.

    Uses the actual public FA4 selector and a GPU-resident FP32-logits reference
    solely as the numerical oracle. No production implementation is substituted.
    Expected routing is inferred from pinned source, not a runtime route receipt.
    """

    output_path = Path(output_path)
    result = {
        "scope": "direct isolated attention VJP numerical diagnostic only",
        "status": "running",
        "cases": [],
        "source_hashes": _source_hashes(),
        "allocation_policy": _allocator_profile(),
        "numpy_allocation_policy": _numpy_allocation_policy(False),
        "tolerance": {
            "atol": ATOL,
            "rtol": RTOL,
            "source": "lib/levanter/tests/grug/test_fa4_cute_attention.py:492-525",
        },
        "oracle": "GPU reference_attention(logits_dtype=float32); no serving/training fallback",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "nvidia-cutlass-dsl", "flash-attn-4")
        },
        "devices": [{"id": d.id, "kind": d.device_kind, "platform": d.platform} for d in jax.local_devices()],
    }

    def persist():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def stats(array):
        raw = np.asarray(array)
        value = raw.astype(np.float32)
        finite = np.isfinite(value)
        return {
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "elements": int(value.size),
            "finite": int(finite.sum()),
            "nan": int(np.isnan(value).sum()),
            "inf": int(np.isinf(value).sum()),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }

    def comparison(actual, expected):
        a, b = np.asarray(actual).astype(np.float32), np.asarray(expected).astype(np.float32)
        finite = np.isfinite(a) & np.isfinite(b)
        delta = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        return {
            "actual": stats(actual),
            "reference": stats(expected),
            "max_abs_error_finite_entries": float(delta.max()) if delta.size else None,
            "rms_error_finite_entries": float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
            "exact_equal": bool(np.array_equal(a, b)),
            "finite_both": bool(finite.all()),
            "within_existing_tolerance": bool(np.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=False)),
        }

    persist()
    if jax.default_backend() != "gpu" or gpu_compute_capability() != 90:
        raise RuntimeError("This proposal requires the actual H100/SM90 native GPU backend")
    # No ambient global mesh: isolate kernel behavior from EP/shard_map/Trainer.
    device = jax.local_devices()[0]
    with jax.default_device(device):
        for name, seq, hq, hkv, dim, expected_route in _FA4_CASES:
            if only_case is not None and name != only_case:
                continue
            row = {
                "name": name,
                "status": "running",
                "stage": "inputs",
                "q_shape": [1, seq, hq, dim],
                "kv_shape": [1, seq, hkv, dim],
                "expected_source_route": expected_route,
                "seed": 4,
                "mask": "all-valid causal, sliding_window=sequence_length",
            }
            result["cases"].append(row)
            persist()
            try:
                keys = jax.random.split(jax.random.PRNGKey(4), 4)
                q = jax.random.normal(keys[0], (1, seq, hq, dim), dtype=jnp.bfloat16)
                k = jax.random.normal(keys[1], (1, seq, hkv, dim), dtype=jnp.bfloat16)
                v = jax.random.normal(keys[2], (1, seq, hkv, dim), dtype=jnp.bfloat16)
                cotangent = jax.random.normal(keys[3], q.shape, dtype=jnp.bfloat16)
                row["inputs"] = {
                    label: stats(value)
                    for label, value in zip(("q", "k", "v", "cotangent"), (q, k, v, cotangent), strict=True)
                }
                assert all(x["finite"] == x["elements"] for x in row["inputs"].values())
                mask = AttentionMask.causal(sliding_window=seq)

                def actual(q, k, v, mask=mask):
                    return attention(q, k, v, mask, implementation="gpu_fa4_cute")

                def oracle(q, k, v, mask=mask):
                    return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

                def output_vjp(function, q, k, v, tangent):
                    output, pullback = jax.vjp(function, q, k, v)
                    return output, pullback(tangent)

                row["stage"] = "actual_vjp"
                persist()
                got = jax.jit(lambda q, k, v, c: output_vjp(actual, q, k, v, c))(q, k, v, cotangent)
                jax.block_until_ready(got)
                row["actual_finiteness"] = {
                    label: stats(value)
                    for label, value in zip(("output", "dq", "dk", "dv"), (got[0], *got[1]), strict=True)
                }
                row["stage"] = "reference_vjp"
                persist()
                want = jax.jit(lambda q, k, v, c: output_vjp(oracle, q, k, v, c))(q, k, v, cotangent)
                jax.block_until_ready(want)
                row["comparisons"] = {
                    label: comparison(a, b)
                    for label, a, b in zip(
                        ("output", "dq", "dk", "dv"), (got[0], *got[1]), (want[0], *want[1]), strict=True
                    )
                }
                row["status"] = (
                    "passed"
                    if all(x["finite_both"] and x["within_existing_tolerance"] for x in row["comparisons"].values())
                    else "failed"
                )
                row["stage"] = "complete"
            except Exception as exc:
                row.update(
                    status="failed",
                    error_type=type(exc).__name__,
                    error=str(exc)[:4000],
                    traceback=traceback.format_exc()[-8000:],
                )
            persist()
    result["status"] = "passed" if all(x["status"] == "passed" for x in result["cases"]) else "failed"
    result["execution_errors_present"] = any("error_type" in row for row in result["cases"])
    persist()
    return result


if __name__ == "__main__":
    if "--response-fd" in sys.argv:
        worker_main(runtime_factory=PairedOracleRuntime)
    else:
        main()
