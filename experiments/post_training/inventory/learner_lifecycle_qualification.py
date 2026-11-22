# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finite tiny-model lifecycle qualification; this does not launch hardware jobs.

The default run requires one host with eight H100s. It compares the prior six
cleanup operations with explicit pinned-JAX backend release. This qualifies
process teardown and checkpoint continuation, not full Snowball67B acceptance.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax
import tensorstore as ts
from levanter.callbacks import JitCallback
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.tensorstore_serialization import _create_ocdbt_spec
from levanter.tracker import NoopTracker, current_tracker
from levanter.trainer_state import trainables_only
from marin.rl.levanter_worker_protocol import LearnerProcess
from marin.rl.levanter_worker_runtime import NativePolicyRuntime, _payload_fingerprint
from marin.rl.levanter_worker_runtime import main as worker_main
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

_MODULE = "experiments.post_training.inventory.learner_lifecycle_qualification"
_ORACLE_WORKER = "experiments.post_training.inventory.native_gradient_diagnostics"


def _json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def full_batch_identity(payload: dict) -> dict:
    """Bind every array, schema and explicit absent field, including step metadata."""
    if set(payload) != {"arrays", "response_length", "global_step"}:
        raise ValueError("Paired batch identity requires the complete closed payload schema")
    arrays = {}
    for name, value in sorted(payload["arrays"].items()):
        if value is None:
            arrays[name] = None
        else:
            array = np.asarray(value)
            arrays[name] = {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    schema = {"arrays": arrays, "response_length": payload["response_length"], "global_step": payload["global_step"]}
    return {"sha256": _json_digest(schema), "schema": schema}


class GradientEvidence(JitCallback):
    """Summarize the gradients actually consumed by the instrumented Trainer step."""

    def __init__(self, state):
        self.paths = [
            jax.tree_util.keystr(p)
            for p, v in jax.tree_util.tree_flatten_with_path(trainables_only(state.model, state.is_trainable))[0]
            if eqx.is_array(v)
        ]
        self.latest = None
        self.calls = 0

    def inside_step(self, state, inside_info):
        grads = trainables_only(inside_info.grads, state.is_trainable)
        values = [v for v in jax.tree.leaves(grads) if eqx.is_array(v)]
        return {
            "norm": optax.global_norm(grads),
            "rows": jnp.stack(
                [
                    jnp.stack(
                        (
                            jnp.count_nonzero(~jnp.isfinite(v)),
                            jnp.count_nonzero(v),
                            jnp.max(jnp.where(jnp.isfinite(v), jnp.abs(v), 0)),
                        )
                    )
                    for v in values
                ]
            ),
        }

    def on_step(self, step_info, cb_info):
        info = jax.device_get(cb_info)
        norm = float(info["norm"])
        rows = [
            {"path": p, "nonfinite": int(v[0]), "nonzero": int(v[1]), "finite_abs_max": float(v[2])}
            for p, v in zip(self.paths, info["rows"], strict=True)
        ]
        self.latest = {
            "optimizer_step": int(step_info.next_step),
            "global_norm": norm if np.isfinite(norm) else None,
            "norm_finite": bool(np.isfinite(norm)),
            "nonfinite": sum(r["nonfinite"] for r in rows),
            "nonzero": sum(r["nonzero"] for r in rows),
            "leaves": rows,
        }
        self.calls += 1


class LossEvidence(NoopTracker):
    def __init__(self):
        self.metrics = {}

    def log(self, metrics, *, step, commit=None):
        self.metrics.update({str(k): float(v) for k, v in jax.device_get(metrics).items() if k.startswith("train/")})


class PairedOracleRuntime(NativePolicyRuntime):
    """Opt-in experiment worker; its hook graph is outside no-hook qualification."""

    def initialize(self, config):
        if not config["bf16"] or config["runtime"]["projection_logits_dtype"] != "bfloat16":
            raise ValueError("Paired oracle requires the reviewed BF16 projection profile")
        result = super().initialize(config)
        self.gradient_evidence = GradientEvidence(self.state)
        self.oracle_context = {}
        self._oracle_train_active = False
        self._oracle_installed_hook = None
        return result

    def _prepare_projection(self, payload, method):
        # The production guard runs honestly before the known diagnostic hook
        # is installed. The hook never evaluates CE or changes its signatures.
        plan = super()._prepare_projection(payload, method)
        if method == "train" and self._oracle_train_active:
            self.trainer.add_hook(self.gradient_evidence, every=2)
            self._oracle_installed_hook = self.trainer.hooks.jit_hooks[-1]
        return plan

    def train(self, payload):
        if self.trainer.hooks.jit_hooks:
            return super().train(payload)
        identity = full_batch_identity(payload)
        expected = self.oracle_context.get("full_batch_sha256")
        if expected is not None and identity["sha256"] != expected:
            raise ValueError("Paired oracle received a different immutable batch")
        before = self.state_fingerprints()
        before_version = self.version
        count = self.gradient_evidence.calls
        self.gradient_evidence.latest = None
        metrics = LossEvidence()
        self._oracle_train_active = True
        try:
            with current_tracker(metrics):
                result = super().train(payload)
        finally:
            self._oracle_train_active = False
            installed = self._oracle_installed_hook
            for index, hook in enumerate(self.trainer.hooks.jit_hooks):
                if hook is installed:
                    self.trainer.hooks.jit_hooks.pop(index)
                    break
            self._oracle_installed_hook = None
        expected_calls = 1 if result["optimizer_step"] % 2 == 0 else 0
        if self.trainer.hooks.jit_hooks:
            raise RuntimeError("Paired oracle did not restore the empty hook list")
        if self.gradient_evidence.calls != count + expected_calls:
            raise RuntimeError("Actual Trainer gradient callback did not fire exactly once")
        if expected_calls and (
            self.gradient_evidence.latest is None
            or self.gradient_evidence.latest["optimizer_step"] != result["optimizer_step"]
        ):
            raise RuntimeError("Gradient callback step differs from actual update")
        weights = np.asarray(payload["arrays"]["policy_weights"])
        active_mass = float(np.sum(weights.astype(np.float64)))
        unit_mass = metrics.metrics.get("train/ppo_ratio_exact_unit_fraction")
        required = {
            "train/ppo_ratio_exact_unit_fraction",
            "train/log_ratio_abs_max",
            "train/ppo_clip_ratio",
            "train/ppo_clip_pressure_low",
            "train/ppo_clip_pressure_high",
        }
        if not required.issubset(metrics.metrics) or not all(np.isfinite(v) for v in metrics.metrics.values()):
            raise RuntimeError("Actual differentiated loss metrics are missing or nonfinite")
        result["oracle_evidence"] = {
            "context": dict(self.oracle_context),
            "full_batch_identity": identity,
            "pre_state": before,
            "pre_policy_version": before_version,
            "gradients": self.gradient_evidence.latest,
            "actual_callback_count": self.gradient_evidence.calls - count,
            "jit_hooks_restored_empty": True,
            "loss_metrics": metrics.metrics,
            "active_policy_weight_mass": active_mass,
            "active_weight_tokens": int(np.count_nonzero(weights > 0)),
            "unit_ratio_weighted_mass": unit_mass,
            "unit_ratio_fraction_of_active_weight": (
                unit_mass / active_mass if unit_mass is not None and active_mass > 0 else None
            ),
            "instrumentation_scope": (
                "Actual consumed-gradient JitCallback graph; outside original no-hook qualification"
            ),
        }
        return result

    def dispatch(self, method, payload):
        if method == "oracle_context":
            self.oracle_context = dict(payload)
            return _json_digest(self.oracle_context)
        return super().dispatch(method, payload)


class BaselineRuntime(NativePolicyRuntime):
    """Keep the six cleanup operations used by source8dde, without backend release."""

    def _release_backend_clients(self) -> None:
        pass


def _initializer(path: Path, config: SnowballConfig | None = None) -> None:
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 64)}}, unk_token="<pad>")
        ),
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(path)
    config = config or SnowballConfig(
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


def _frozen_batch(devices: int, sequence_length: int = 8) -> dict:
    batch = devices * 2
    return {
        "arrays": {
            "tokens": np.tile(np.arange(sequence_length, dtype=np.int32) % 62 + 2, (batch, 1)),
            "attention_mask": np.ones((batch, sequence_length), dtype=np.bool_),
            "position_ids": np.tile(np.arange(sequence_length, dtype=np.int32), (batch, 1)),
            "response_mask": np.ones((batch, 2), dtype=np.bool_),
            "advantages": np.tile(np.array([0.5, -0.25], dtype=np.float32), (batch, 1)),
            "policy_weights": np.full((batch, 2), 1 / (batch * 2), dtype=np.float32),
            "kl_weights": np.zeros((batch, 2), dtype=np.float32),
            "reference_logprobs": None,
        },
        "response_length": 2,
        "global_step": 1,
    }


def _start(
    python: str, environment: dict, configuration: dict, baseline: bool, runtime_module: str | None = None
) -> tuple[LearnerProcess, dict]:
    child = LearnerProcess(
        python,
        environment=environment,
        timeout_seconds=1800,
        runtime_module=runtime_module or (_MODULE if baseline else "marin.rl.levanter_worker_runtime"),
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


def _checkpoint_deviations(left: Path, right: Path) -> dict:
    """Compare persisted states one leaf pair at a time, without an acceptance tolerance."""
    manifest = json.loads((left / "manifest.json").read_text())
    if manifest != json.loads((right / "manifest.json").read_text()):
        raise RuntimeError("Replay checkpoint manifests differ")
    rows = []
    components = {}
    for entry in manifest["arrays"]:
        path = entry["path"]
        a = ts.open(_create_ocdbt_spec(str(left), path), open=True).result().read().result()
        b = ts.open(_create_ocdbt_spec(str(right), path), open=True).result().read().result()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"Replay checkpoint array schema differs: {path}")
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        finite = finite_a & finite_b
        delta = a[finite].astype(np.float64) - b[finite].astype(np.float64)
        square_sum = float(np.sum(delta * delta))
        maximum = float(np.max(np.abs(delta))) if delta.size else None
        unequal = int(np.count_nonzero(a != b))
        row = {
            "path": path,
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "elements": int(a.size),
            "left_nonfinite": int(a.size - np.count_nonzero(finite_a)),
            "right_nonfinite": int(b.size - np.count_nonzero(finite_b)),
            "unequal": unequal,
            "max_abs_finite": maximum,
            "rms_finite": float(np.sqrt(square_sum / delta.size)) if delta.size else None,
        }
        rows.append(row)
        component = components.setdefault(
            path.split("/")[0],
            {
                "arrays": 0,
                "elements": 0,
                "left_nonfinite": 0,
                "right_nonfinite": 0,
                "unequal": 0,
                "finite_pairs": 0,
                "square_sum": 0.0,
                "max_abs_finite": None,
            },
        )
        component["arrays"] += 1
        for name in ("elements", "left_nonfinite", "right_nonfinite", "unequal"):
            component[name] += row[name]
        component["finite_pairs"] += int(delta.size)
        component["square_sum"] += square_sum
        if maximum is not None:
            component["max_abs_finite"] = max(component["max_abs_finite"] or 0.0, maximum)
    for component in components.values():
        squares = component.pop("square_sum")
        count = component["finite_pairs"]
        component["rms_finite"] = float(np.sqrt(squares / count)) if count else None
    return {
        "left": str(left),
        "right": str(right),
        "components": components,
        "arrays": rows,
        "exact_arrays": all(row["unequal"] == 0 for row in rows),
        "tolerance": None,
    }


def _capture_control(child: LearnerProcess, payload: dict, directory: Path, row: dict) -> tuple:
    """Preserve both control steps before any comparison can discard the evidence."""
    row["control_updates"] = []
    row["control_checkpoints"] = []
    for step in (1, 2):
        payload["global_step"] = step
        row["control_updates"].append(child.request("train", payload))
        saved = child.request(
            "save_checkpoint", {"ckpt_dir": str(directory / f"control-step{step}"), "include_fingerprints": True}
        )
        row["control_checkpoints"].append(saved)
        _persist_arm(directory, row, f"control_step{step}_captured")
    scores = child.request("forward", payload)["output"]
    np.save(directory / "control-scores.npy", scores, allow_pickle=False)
    row["control_scores"] = str(directory / "control-scores.npy")
    _persist_arm(directory, row, "control_scores_captured")
    return row["control_updates"], row["control_checkpoints"][-1]["state_fingerprints"], scores


def _arm(
    python: str,
    environment: dict,
    config: dict,
    directory: Path,
    baseline: bool,
    sequence_length: int = 8,
    capture_replay_states: bool = False,
) -> dict:
    directory.mkdir(parents=True)
    child = None
    exits = []
    row: dict = {"status": "running", "exits": exits}
    if capture_replay_states:
        row["replay_capture_scope"] = (
            "Added checkpoint serialization and fingerprint synchronization may perturb GPU execution timing/layout. "
            "The exact gate is unchanged; a captured pass cannot explain the earlier uncaptured mismatch. "
            "Control forward remains after the second train."
        )
    _persist_arm(directory, row, "initialize")
    try:
        child, identity = _start(python, environment, config, baseline)
        row["initial_runtime"] = identity
        _persist_arm(directory, row, "score_frozen_batch")
        if config["require_accelerator"] and any("H100" not in kind for kind in identity["device_kinds"]):
            raise ValueError("Lifecycle qualification requires actual H100 devices")
        payload = _frozen_batch(config["distributed"]["local_device_count"], sequence_length)
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
        if capture_replay_states:
            row["continued_checkpoint"] = child.request(
                "save_checkpoint", {"ckpt_dir": str(directory / "resumed-step2"), "include_fingerprints": True}
            )
            np.save(directory / "continued-scores.npy", continued_scores, allow_pickle=False)
            row["continued_scores"] = str(directory / "continued-scores.npy")
        _persist_arm(directory, row, "restored_natural_exit")
        exits.append(_exit(child))
        child.close()
        _persist_arm(directory, row, "uninterrupted_control")
        child, _ = _start(python, environment, config, baseline)
        if capture_replay_states:
            updates, control_state, control_scores = _capture_control(child, payload, directory, row)
            row["state_comparisons"] = {
                "step1": _checkpoint_deviations(directory / "step1", directory / "control-step1"),
                "step2": _checkpoint_deviations(directory / "resumed-step2", directory / "control-step2"),
                "step1_fingerprints_exact": (
                    saved["state_fingerprints"] == row["control_checkpoints"][0]["state_fingerprints"]
                ),
                "step2_fingerprints_exact": continued_state == control_state,
                "scores_exact": bool(np.array_equal(continued_scores, control_scores)),
            }
            _persist_arm(directory, row, "control_comparison_captured")
            # A failed exact comparison is still a failed qualification; natural
            # process exit is recorded separately before reporting the mismatch.
            exits.append(_exit(child))
            comparisons = row["state_comparisons"]
            if (
                updates != [update1, continued]
                or not comparisons["step1_fingerprints_exact"]
                or not comparisons["step2_fingerprints_exact"]
                or not comparisons["step1"]["exact_arrays"]
                or not comparisons["step2"]["exact_arrays"]
            ):
                raise RuntimeError("Continued update or full state differs from uninterrupted control")
            np.testing.assert_array_equal(control_scores, continued_scores)
            row["continuation_exact"] = True
            row["status"] = "passed"
            _persist_arm(directory, row, "complete")
            return row
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


def paired_resume_oracle(python: str, environment: dict, config: dict, directory: Path, source_identity: dict) -> dict:
    """Compare live/restored step2 from one A1 and one immutable, fresh-score batch."""
    directory.mkdir(parents=True)
    row = {
        "status": "running",
        "paired_resume_qualified": False,
        "exits": [],
        "scope": "Instrumented same-A1 resume gate; separate from fresh-init/stale-score diagnostic",
        "limitations": (
            "Actual-gradient JitCallback changes compiled body; original no-hook profile guard runs before attachment. "
            "Scores, serialization and fingerprints synchronize execution and may perturb GPU timing/layout. "
            "No full67B qualification or explanation of prior failures follows from this instrumented tiny-model gate."
        ),
        "source_identity": source_identity,
        "config_sha256": _json_digest(config),
    }

    def persist(stage):
        row["stage"] = stage
        (directory / "paired-resume.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")

    def context(child, branch, payload, **binding):
        identity = full_batch_identity(payload)
        child.request(
            "oracle_context",
            {
                "branch": branch,
                "full_batch_sha256": identity["sha256"],
                "config_sha256": row["config_sha256"],
                **binding,
            },
        )
        return identity

    def finish(child, branch):
        try:
            row["exits"].append({"branch": branch, **_exit(child)})
        except Exception as error:
            row["exits"].append(
                {
                    "branch": branch,
                    "pid": child.process.pid,
                    "exit_code": child.process.poll(),
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                }
            )
        finally:
            child.close()
            persist(branch + "_exit_attempted")

    def after_step(child, payload, branch):
        before = child.request("state_fingerprints", {})
        scores = child.request("forward", payload)["output"]
        after_score = child.request("state_fingerprints", {})
        np.save(directory / (branch + "-scores.npy"), scores, allow_pickle=False)
        saved = child.request(
            "save_checkpoint",
            {
                "ckpt_dir": str(directory / branch),
                "include_fingerprints": True,
            },
        )
        after_save = child.request("state_fingerprints", {})
        row[branch] = {
            "before_score": before,
            "after_score": after_score,
            "checkpoint": saved,
            "after_save": after_save,
            "score_and_save_pure": before == after_score == saved["state_fingerprints"] == after_save,
            "scores": str(directory / (branch + "-scores.npy")),
        }
        persist(branch + "_captured")
        return scores

    child = None
    batch2 = None
    persist("A_initialize")
    try:
        child, row["A_runtime"] = _start(python, environment, config, False, _ORACLE_WORKER)
        if config["require_accelerator"] and any("H100" not in k for k in row["A_runtime"]["device_kinds"]):
            raise RuntimeError("Paired oracle requires actual H100 devices")
        payload = _frozen_batch(config["distributed"]["local_device_count"], 128)
        payload["arrays"]["old_logprobs"] = child.request("forward", payload)["output"]
        row["batch1_identity"] = context(child, "A1", payload)
        row["A1_update"] = child.request("train", payload)
        a1_before_score = child.request("state_fingerprints", {})
        old_scores = child.request("forward", payload)["output"]
        a1_after_score = child.request("state_fingerprints", {})
        row["A1_checkpoint"] = child.request(
            "save_checkpoint",
            {
                "ckpt_dir": str(directory / "A1"),
                "include_fingerprints": True,
            },
        )
        a1_after_save = child.request("state_fingerprints", {})
        row["A1_purity"] = {
            "before_score": a1_before_score,
            "after_score": a1_after_score,
            "after_save": a1_after_save,
            "score_and_save_pure": (
                a1_before_score == a1_after_score == a1_after_save == row["A1_checkpoint"]["state_fingerprints"]
            ),
        }
        batch2 = {
            "arrays": {k: None if v is None else np.array(v, copy=True) for k, v in payload["arrays"].items()},
            "response_length": payload["response_length"],
            "global_step": 2,
        }
        batch2["arrays"]["old_logprobs"] = np.array(old_scores, copy=True)
        for value in batch2["arrays"].values():
            if value is not None:
                value.setflags(write=False)
        row["batch2_identity"] = full_batch_identity(batch2)
        row["batch2_binding"] = {
            "config_sha256": row["config_sha256"],
            "source_identity": source_identity,
            "A1_checkpoint": row["A1_checkpoint"],
            "full_batch_identity": row["batch2_identity"],
        }
        row["batch2_binding_sha256"] = _json_digest(row["batch2_binding"])
        np.savez(directory / "batch2.npz", **{k: v for k, v in batch2["arrays"].items() if v is not None})
        persist("batch2_frozen_once")
        context(child, "A2", batch2, binding_sha256=row["batch2_binding_sha256"])
        row["A2_update"] = child.request("train", batch2)
        after_step(child, batch2, "A2")
    except Exception as error:
        row["A_error"] = {"type": type(error).__name__, "message": str(error)[:4000]}
        persist("A_failed")
    finally:
        if child is not None:
            finish(child, "A")
    child = None
    if batch2 is not None:
        try:
            persist("B_initialize")
            child, row["B_runtime"] = _start(python, environment, config, False, _ORACLE_WORKER)
            if config["require_accelerator"] and any("H100" not in k for k in row["B_runtime"]["device_kinds"]):
                raise RuntimeError("Paired oracle requires actual H100 devices")
            saved = row["A1_checkpoint"]
            loaded = child.request("load_checkpoint", {"ckpt_dir": saved["checkpoint"], "include_fingerprints": True})
            before = child.request("state_fingerprints", {})
            scores = child.request("forward", batch2)["output"]
            after = child.request("state_fingerprints", {})
            row["B1_loaded"] = loaded
            row["B1_purity"] = {
                "before_score": before,
                "after_score": after,
                "load_exact": loaded == saved and before == saved["state_fingerprints"],
                "score_pure": before == after,
                "scores_exact_frozen_batch2": bool(np.array_equal(scores, batch2["arrays"]["old_logprobs"])),
            }
            np.save(directory / "B1-scores.npy", scores, allow_pickle=False)
            persist("B1_load_and_score_captured")
            context(child, "B2", batch2, binding_sha256=row["batch2_binding_sha256"])
            row["B2_update"] = child.request("train", batch2)
            after_step(child, batch2, "B2")
        except Exception as error:
            row["B_error"] = {"type": type(error).__name__, "message": str(error)[:4000]}
            persist("B_failed")
        finally:
            if child is not None:
                finish(child, "B")
    if "A1_checkpoint" in row:
        row["A1_finiteness"] = _checkpoint_deviations(directory / "A1", directory / "A1")
    if "A2" in row and "B2" in row:
        row["state_deviations"] = _checkpoint_deviations(directory / "A2", directory / "B2")
        update_a, update_b = row["A2_update"], row["B2_update"]
        score_a, score_b = np.load(directory / "A2-scores.npy"), np.load(directory / "B2-scores.npy")
        row["checks"] = {
            "consumed_batch2_exact": all(
                u["oracle_evidence"]["full_batch_identity"] == row["batch2_identity"] for u in [update_a, update_b]
            ),
            "consumed_prestate_exact": all(
                u["oracle_evidence"]["pre_state"] == row["A1_checkpoint"]["state_fingerprints"]
                and u["oracle_evidence"]["pre_policy_version"] == 1
                for u in [update_a, update_b]
            ),
            "consumed_binding_exact": all(
                u["oracle_evidence"]["context"].get("binding_sha256") == row["batch2_binding_sha256"]
                for u in [update_a, update_b]
            ),
            "A1_finite": all(
                c["left_nonfinite"] == c["right_nonfinite"] == 0 for c in row["A1_finiteness"]["components"].values()
            ),
            "post_states_finite": all(
                c["left_nonfinite"] == c["right_nonfinite"] == 0 for c in row["state_deviations"]["components"].values()
            ),
            "A1_pure": row["A1_purity"]["score_and_save_pure"],
            "B1_load_exact": row["B1_purity"]["load_exact"],
            "B1_score_pure": row["B1_purity"]["score_pure"],
            "B1_scores_equal_immutable_batch2": row["B1_purity"]["scores_exact_frozen_batch2"],
            "batch2_unchanged": full_batch_identity(batch2) == row["batch2_identity"],
            "post_steps_pure": row["A2"]["score_and_save_pure"] and row["B2"]["score_and_save_pure"],
            "post_state_exact": (
                row["A2"]["after_save"] == row["B2"]["after_save"] and row["state_deviations"]["exact_arrays"]
            ),
            "post_scores_exact": bool(np.array_equal(score_a, score_b)),
            "scalar_metrics_exact": (
                {k: v for k, v in update_a.items() if k != "oracle_evidence"}
                == {k: v for k, v in update_b.items() if k != "oracle_evidence"}
            ),
            "loss_metrics_exact": (
                update_a["oracle_evidence"]["loss_metrics"] == update_b["oracle_evidence"]["loss_metrics"]
            ),
            "fresh_gradients_finite_nonzero": all(
                u["oracle_evidence"]["actual_callback_count"] == 1
                and u["oracle_evidence"]["jit_hooks_restored_empty"]
                and u["oracle_evidence"]["gradients"]["norm_finite"]
                and u["oracle_evidence"]["gradients"]["global_norm"] > 0
                and u["oracle_evidence"]["gradients"]["nonfinite"] == 0
                and u["oracle_evidence"]["gradients"]["nonzero"] > 0
                for u in [update_a, update_b]
            ),
            "new_pid": row["A_runtime"]["pid"] != row["B_runtime"]["pid"],
            "natural_exits_zero": (
                len(row["exits"]) == 2
                and all(x["exit_code"] == 0 and "error" not in x and "error_type" not in x for x in row["exits"])
            ),
        }
        row["paired_resume_qualified"] = all(row["checks"].values())
    row["status"] = "passed" if row["paired_resume_qualified"] else "failed"
    persist("complete")
    return row


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
