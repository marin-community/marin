# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repeat one public compiled, pure Trainer raw-gradient program at fixed A1.

This module is an isolated diagnostic child, not an optimizer-gradient capture.
It reads the current paired oracle's exact A1/config/batch2. No update, donation,
key advance, alternate kernel, format conversion, or checkpoint rewrite occurs.
"""

import argparse
import dataclasses
import gzip
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import equinox as eqx
import haliax as hax
import jax
import jaxlib
import numpy as np
from haliax.partitioning import infer_resource_partitions
from levanter.utils.tree_utils import inference_mode
from marin.rl.levanter_worker_runtime import NativePolicyRuntime

from experiments.post_training.inventory.learner_lifecycle_qualification import full_batch_identity

_GRADIENT_BUDGET = 256 * 2**20
_REPORT_BUDGET = 16 * 2**20
_FINITE_COMPONENTS = ("model", "opt_state", "training_key", "step", "model_averaging")
_SCOPE = (
    "Separate pure differentiated Trainer program, not actual gradients consumed by the optimizer. "
    "Different repeats establish variability in this program, not a specific FA4/backward cause. "
    "Matching repeats do not clear the actual optimizer-body or paired-resume failure. "
    "Host hashing/serialization and synchronization may perturb timing."
)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_digest(value: Any) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sources() -> dict:
    root = Path(__file__).resolve().parents[3]
    names = (
        "experiments/post_training/inventory/fixed_gradient_repeat_probe.py",
        "experiments/post_training/inventory/learner_lifecycle_qualification.py",
        "lib/marin/src/marin/rl/levanter_worker_runtime.py",
        "lib/levanter/src/levanter/trainer.py",
        "lib/levanter/src/levanter/grad_accum.py",
        "lib/levanter/src/levanter/trainer_state.py",
        "lib/levanter/src/levanter/policy_model.py",
        "lib/levanter/src/levanter/models/snowball.py",
        "lib/levanter/src/levanter/grug/attention/_fa4_cute.py",
        "lib/levanter/src/levanter/grug/attention/_fa4_cute_backend.py",
        "lib/levanter/src/levanter/grug/attention/_fa4_cute_kernels.py",
        "lib/levanter/src/levanter/grug/attention/_fa4_cute_config.py",
        "lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py",
        "lib/haliax/src/haliax/partitioning.py",
    )
    return {name: _file_hash(root / name) for name in names}


class _Reports:
    def __init__(self, directory: Path):
        self.directory = directory

    def write(self, name: str, value: dict) -> None:
        raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        other = sum(
            p.stat().st_size for p in self.directory.iterdir() if p.is_file() and p.name != name and p.suffix != ".npz"
        )
        if other + len(raw) >= _REPORT_BUDGET:
            raise RuntimeError("Fixed-gradient reports exceed the sixteen-MiB bound")
        (self.directory / name).write_bytes(raw)


def _sharding(array: Any) -> dict:
    value = array.sharding
    result = {"type": type(value).__name__, "text": str(value)}
    if hasattr(value, "spec"):
        result["spec"] = str(value.spec)
    if hasattr(value, "mesh"):
        result["mesh_axes"] = list(value.mesh.axis_names)
        result["mesh_shape"] = dict(value.mesh.shape)
    return result


def _index(index: Any) -> list:
    return [{"start": v.start, "stop": v.stop, "step": v.step} if isinstance(v, slice) else int(v) for v in index]


def _format(array: Any) -> dict:
    try:
        value = array.format
        return {"available": True, "text": str(value)}
    except Exception as error:
        return {"available": False, "reason": type(error).__name__}


def _axes(tree: Any) -> list:
    return [
        {"path": jax.tree_util.keystr(path), "axes": [[axis.name, axis.size] for axis in value.axes]}
        for path, value in jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: isinstance(x, hax.NamedArray))[0]
        if isinstance(value, hax.NamedArray)
    ]


def _snapshot(tree: Any, *, formats: bool = False) -> dict:
    leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    rows = []
    for path, value in leaves:
        if not eqx.is_array(value):
            continue
        host = np.asarray(value)
        row = {
            "path": jax.tree_util.keystr(path),
            "shape": list(host.shape),
            "dtype": str(host.dtype),
            "dtype_str": host.dtype.str,
            "bytes": host.nbytes,
            "sha256": _digest(host.tobytes()),
            "sharding": _sharding(value) if isinstance(value, jax.Array) else {"type": "host"},
        }
        if formats and isinstance(value, jax.Array):
            row["public_format"] = _format(value)
            row["addressable_shards"] = [
                {
                    "device_id": shard.device.id,
                    "process_index": shard.device.process_index,
                    "index": _index(shard.index),
                    "shape": list(shard.data.shape),
                }
                for shard in value.addressable_shards
            ]
        rows.append(row)
    payload = {"arrays": rows, "logical_axes": _axes(tree), "tree_structure": str(structure)}
    return {"sha256": _json_digest(payload), "array_bytes": sum(row["bytes"] for row in rows), **payload}


def _checkpoint_files(path: Path) -> dict:
    paths = [p for p in sorted(path.rglob("*")) if p.is_file()]
    if not paths or sum(p.stat().st_size for p in paths) > 2**30:
        raise ValueError("Fixed-A1 probe requires a bounded existing tiny checkpoint")
    files = {str(p.relative_to(path)): {"bytes": p.stat().st_size, "sha256": _file_hash(p)} for p in paths}
    return {"sha256": _json_digest(files), "files": files}


def _load_payload(directory: Path, config: dict) -> tuple[dict, dict]:
    row = json.loads((directory / "paired-resume.json").read_text())
    if row["config_sha256"] != _json_digest(config):
        raise ValueError("Probe config differs from the exact paired config")
    binding = row["batch2_binding"]
    if _json_digest(binding) != row["batch2_binding_sha256"] or binding["config_sha256"] != row["config_sha256"]:
        raise ValueError("Paired batch2 binding differs")
    if binding["A1_checkpoint"] != row["A1_checkpoint"] or binding["full_batch_identity"] != row["batch2_identity"]:
        raise ValueError("Batch2 is not bound to the exact saved A1")
    if Path(row["A1_checkpoint"]["checkpoint"]).resolve() != (directory / "A1").resolve():
        raise ValueError("Probe A1 path differs from the exact paired checkpoint binding")
    schema = row["batch2_identity"]["schema"]
    path = directory / "batch2.npz"
    if path.stat().st_size > 16 * 2**20:
        raise ValueError("Probe batch exceeds the tiny diagnostic bound")
    with np.load(path, allow_pickle=False) as archive:
        expected = {key for key, value in schema["arrays"].items() if value is not None}
        if set(archive.files) != expected:
            raise ValueError("Batch2 NPZ keys differ from full frozen schema")
        arrays = {
            key: None if value is None else np.array(archive[key], copy=True) for key, value in schema["arrays"].items()
        }
    for value in arrays.values():
        if value is not None:
            value.setflags(write=False)
    payload = {"arrays": arrays, "response_length": schema["response_length"], "global_step": schema["global_step"]}
    if full_batch_identity(payload) != row["batch2_identity"]:
        raise ValueError("Batch2 arrays/schema/metadata differ from the frozen paired payload")
    return payload, row


def _result_stats(result: Any) -> dict:
    rows = []
    for path, value in jax.tree_util.tree_flatten_with_path(result)[0]:
        if not eqx.is_array(value):
            continue
        host = np.asarray(value)
        numeric = host.astype(np.float64)
        finite = np.isfinite(numeric)
        rows.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(host.shape),
                "dtype": str(host.dtype),
                "sha256": _digest(host.tobytes()),
                "elements": host.size,
                "nonfinite": int(np.count_nonzero(~finite)),
                "finite_min": float(numeric[finite].min()) if finite.any() else None,
                "finite_max": float(numeric[finite].max()) if finite.any() else None,
                "scalar": float(numeric) if numeric.shape == () and bool(finite) else None,
            }
        )
    return {"arrays": rows, "nonfinite": sum(row["nonfinite"] for row in rows)}


def _compare(left: Any, right: Any) -> dict:
    a, adef = jax.tree_util.tree_flatten_with_path(left)
    b, bdef = jax.tree_util.tree_flatten_with_path(right)
    structure_equal = adef == bdef
    rows = []
    if not structure_equal:
        return {"exact": False, "structure_equal": False, "arrays": [], "axes_equal": _axes(left) == _axes(right)}
    for (path, x), (other_path, y) in zip(a, b, strict=True):
        if not eqx.is_array(x):
            continue
        xx, yy = np.asarray(x), np.asarray(y)
        schema = xx.shape == yy.shape and xx.dtype == yy.dtype and path == other_path
        exact = schema and _digest(xx.tobytes()) == _digest(yy.tobytes())
        shardings_equal = (
            isinstance(x, jax.Array) and isinstance(y, jax.Array) and x.sharding.is_equivalent_to(y.sharding, x.ndim)
        )
        delta = None
        if schema:
            xf, yf = xx.astype(np.float64), yy.astype(np.float64)
            finite = np.isfinite(xf) & np.isfinite(yf)
            delta = float(np.abs(xf[finite] - yf[finite]).max(initial=0))
        rows.append(
            {
                "path": jax.tree_util.keystr(path),
                "schema_equal": schema,
                "bytes_exact": exact,
                "shardings_equivalent": shardings_equal,
                "max_abs_finite_difference": delta,
            }
        )
    axes_equal = _axes(left) == _axes(right)
    return {
        "exact": structure_equal and axes_equal and all(row["bytes_exact"] for row in rows),
        "structure_equal": structure_equal,
        "axes_equal": axes_equal,
        "shardings_equivalent": all(row["shardings_equivalent"] for row in rows),
        "arrays": rows,
    }


def _save_gradients(grads: Any, path: Path) -> dict:
    arrays = {
        f"g{index:05d}": np.asarray(value)
        for index, (_, value) in enumerate(jax.tree_util.tree_flatten_with_path(grads)[0])
        if eqx.is_array(value)
    }
    np.savez(path, **arrays)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_hash(path),
        "array_keys": list(arrays),
        "schema": _snapshot(grads),
        "reconstruction": (
            "Full returned raw-gradient arrays in tree-flatten path order; "
            "schema/axes and A1 model template bind static structure."
        ),
    }


def _compile_diagnostics(compiled: Any, output: Path) -> dict:
    result = {}
    try:
        text = compiled.as_text()
        if text is None:
            result["optimized_hlo"] = {"available": False, "reason": "as_text returned None"}
        else:
            raw = text.encode()
            compressed = gzip.compress(raw)
            if len(raw) > 64 * 2**20 or len(compressed) > 4 * 2**20:
                result["optimized_hlo"] = {
                    "available": False,
                    "reason": "bounded report omitted",
                    "source_bytes": len(raw),
                    "sha256": _digest(raw),
                }
            else:
                path = output / "same-compiled-object.optimized-hlo.txt.gz"
                path.write_bytes(compressed)
                result["optimized_hlo"] = {
                    "available": True,
                    "path": str(path),
                    "bytes": len(compressed),
                    "source_bytes": len(raw),
                    "sha256": _digest(compressed),
                }
    except Exception as error:
        result["optimized_hlo"] = {"available": False, "reason": type(error).__name__}
    try:
        stats = compiled.memory_analysis()
        result["memory_analysis"] = {"available": stats is not None}
        if stats is not None:
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
                "host_argument_size_in_bytes",
                "host_output_size_in_bytes",
                "host_temp_size_in_bytes",
            ):
                value = getattr(stats, name, None)
                if value is not None:
                    result["memory_analysis"][name] = int(value)
    except Exception as error:
        result["memory_analysis"] = {"available": False, "reason": type(error).__name__}
    return result


def compile_raw_program(raw: Any, state: Any, batch: Any, mapping: dict, mesh: Any) -> tuple:
    """Public arrays-only JAX AOT with the existing physical input/output resources."""
    dynamic, static = eqx.partition((state, batch), eqx.is_array)

    def arrays_only(arrays):
        raw_state, raw_batch = eqx.combine(arrays, static)
        return raw(raw_state, raw_batch)

    with jax.set_mesh(mesh), hax.axis_mapping(mapping):
        shape = jax.eval_shape(arrays_only, dynamic)
        gradient_bytes = sum(
            int(np.prod(x.shape)) * np.dtype(x.dtype).itemsize
            for x in jax.tree_util.tree_leaves(shape[1])
            if isinstance(x, jax.ShapeDtypeStruct)
        )
        if 2 * gradient_bytes + 2 * 2**20 > _GRADIENT_BUDGET:
            raise ValueError("Two complete gradient outputs exceed the256MiB bound")
        input_shardings = jax.tree_util.tree_map(lambda x: x.sharding, dynamic)
        output_shardings = infer_resource_partitions(shape, mapping, mesh=mesh)
        public_jit = jax.jit(
            arrays_only, in_shardings=(input_shardings,), out_shardings=output_shardings, donate_argnums=()
        )
        compiled = public_jit.lower(dynamic).compile()
    return compiled, dynamic, gradient_bytes


def run_probe(
    config: dict,
    paired_oracle_dir: Path,
    output_dir: Path,
    *,
    platform: str,
    repeats: int = 2,
    validate_reference: bool = False,
) -> dict:
    """Run only in a fresh diagnostic child with its native environment pre-set."""
    output_dir.mkdir(parents=True, exist_ok=False)
    reports = _Reports(output_dir)
    receipt = {
        "status": "running",
        "execution_errors_present": False,
        "numerical_findings_present": False,
        "raw_repeats_exact": False,
        "scope": _SCOPE,
        "repeats": repeats,
        "calls": [],
        "source_hashes": _sources(),
        "platform": platform,
    }
    target = "fixed-gradient-repeat.json"
    runtime = NativePolicyRuntime()
    try:
        if repeats != 2:
            raise ValueError("This bounded probe requires exactly two calls")
        if validate_reference and platform != "cpu":
            raise ValueError("Reference comparison is CPU proof only; GPU probe uses one compiled object")
        if jax.__version__ != "0.11.0" or jaxlib.__version__ != "0.11.0" or np.__version__ != "2.3.5":
            raise ValueError("Probe requires exact JAX/jaxlib0.11.0 and NumPy2.3.5")
        if (
            config["distributed"]["local_device_count"] != 8
            or config["mini_batch_size"] != 16
            or config["runtime"]["mesh_axes"]["expert"] != 8
            or config["runtime"]["mesh_axes"]["data"] != 1
        ):
            raise ValueError("Probe requires the unchanged tiny one-host eight-device EP8 geometry")
        initializer = Path(config["model"]["path"])
        weights = list(initializer.glob("*.safetensors")) if initializer.is_dir() else []
        if not weights or sum(p.stat().st_size for p in weights) > 128 * 2**20:
            raise ValueError("Full models are outside the bounded tiny probe")
        payload, paired = _load_payload(paired_oracle_dir, config)
        checkpoint = paired_oracle_dir / "A1"
        checkpoint_before = _checkpoint_files(checkpoint)
        receipt["binding"] = {
            "config_sha256": _json_digest(config),
            "batch2_identity": full_batch_identity(payload),
            "batch2_binding_sha256": paired["batch2_binding_sha256"],
            "checkpoint_tree_sha256": checkpoint_before["sha256"],
        }
        receipt["runtime"] = runtime.initialize(config)
        expected = "gpu" if platform == "gpu" else "cpu"
        if jax.default_backend() != expected or jax.process_count() != 1 or len(jax.local_devices()) != 8:
            raise ValueError("Probe requires one process and eight actual devices on requested backend")
        if platform == "gpu" and any("H100" not in d.device_kind for d in jax.local_devices()):
            raise ValueError("GPU diagnostic requires actual H100 devices")
        loaded = runtime.checkpoint("load_checkpoint", {"ckpt_dir": str(checkpoint), "include_fingerprints": True})
        if loaded["state_fingerprints"] != paired["A1_checkpoint"]["state_fingerprints"]:
            raise ValueError("Current full A1 fingerprint dictionary differs after exact restore")
        receipt["A1_loaded_fingerprints"] = loaded["state_fingerprints"]
        plan = runtime._prepare_projection(payload, "train")
        trainer = runtime.trainer
        state = runtime.state
        batch = runtime._batch(payload)
        if trainer.hooks.jit_hooks:
            raise ValueError("Pure repeat probe must not install Trainer hooks")
        plan_identity = repr(plan)
        receipt["projection_plan"] = {"representation": plan_identity, "sha256": _digest(plan_identity.encode())}
        mapping = trainer.parameter_axis_mapping
        loss_fn = trainer.loss_fn
        receipt["geometry"] = {
            "mesh": dict(trainer.device_mesh.shape),
            "parameter_mapping": dict(mapping),
            "compute_mapping": dict(trainer.compute_axis_mapping),
            "microbatch_size": trainer.config.microbatch_size,
            "model_config": dataclasses.asdict(state.model.config),
        }

        def raw(raw_state, raw_batch):
            with hax.axis_mapping(mapping):
                key, _ = jax.random.split(raw_state.training_key)
                model = inference_mode(raw_state.model, False)
                return trainer._compute_gradients_microbatched(loss_fn, model, raw_batch, key=key, projection_plan=plan)

        before = _snapshot((state, batch))
        with jax.set_mesh(trainer.device_mesh), hax.axis_mapping(mapping):
            compiled, dynamic, gradient_bytes = compile_raw_program(raw, state, batch, mapping, trainer.device_mesh)
            receipt["compilation"] = {
                "public_api": "jax.jit.lower.compile",
                "lower_count": 1,
                "compile_count": 1,
                "donate_argnums": [],
                "same_compiled_object_id": id(compiled),
                "gradient_bytes_per_call": gradient_bytes,
                **_compile_diagnostics(compiled, output_dir),
            }
            reports.write(target, receipt)
            results = []
            for index in range(repeats):
                if (
                    _snapshot((state, batch)) != before
                    or full_batch_identity(payload) != receipt["binding"]["batch2_identity"]
                ):
                    raise RuntimeError("State/full input changed before pure compiled call")
                formats = _snapshot((state, batch), formats=True)
                reports.write(f"inputs-before-call-{index}.json", formats)
                result = compiled(dynamic)
                jax.block_until_ready(result)
                stats = _result_stats(result)
                receipt["numerical_findings_present"] |= bool(stats["nonfinite"])
                row = {
                    "index": index,
                    "compiled_object_id": id(compiled),
                    "dynamic_input_object_id": id(dynamic),
                    "state_and_full_input_pure": False,
                    "result": stats,
                }
                receipt["calls"].append(row)
                reports.write(target, receipt)
                if _snapshot((state, batch)) != before or repr(plan) != plan_identity:
                    raise RuntimeError("Pure compiled call mutated state/input/plan")
                row["state_and_full_input_pure"] = True
                saved = _save_gradients(result[1], output_dir / f"raw-gradients-{index}.npz")
                row["raw_gradients"] = saved
                if sum(p.stat().st_size for p in output_dir.glob("raw-gradients-*.npz")) > _GRADIENT_BUDGET:
                    raise RuntimeError("Serialized raw gradients exceed the256MiB bound")
                results.append(result)
                reports.write(target, receipt)
            difference = _compare(results[0], results[1])
            receipt["repeat_comparison"] = difference
            receipt["raw_repeats_exact"] = difference["exact"]
            receipt["numerical_findings_present"] |= not difference["exact"]
            if validate_reference:
                reference = hax.named_jit(raw, axis_resources=mapping, out_axis_resources=mapping)(state, batch)
                jax.block_until_ready(reference)
                comparison = _compare(results[0], reference)
                receipt["CPU_reference"] = {
                    "scope": "Actual named_jit Trainer raw path, not optimizer consumed gradients",
                    **comparison,
                }
                receipt["numerical_findings_present"] |= not comparison["exact"]
        if _snapshot((state, batch)) != before or full_batch_identity(payload) != receipt["binding"]["batch2_identity"]:
            raise RuntimeError("State/full input changed across probe/reference comparison")
        if _checkpoint_files(checkpoint) != checkpoint_before:
            raise RuntimeError("Immutable A1 checkpoint files changed")
        if _sources() != receipt["source_hashes"]:
            raise RuntimeError("Working-tree source changed during probe; final-source proof is invalid")
        receipt.update(status="completed", state_and_full_input_pure=True, immutable_A1_files_unchanged=True)
    except Exception as error:
        receipt.update(
            status="error",
            execution_errors_present=True,
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc()[-8000:],
        )
    finally:
        try:
            runtime.close()
        except Exception as error:
            receipt.update(
                status="error",
                execution_errors_present=True,
                cleanup_error_type=type(error).__name__,
                cleanup_error=str(error),
            )
        reports.write(target, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paired-oracle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--repeats", type=int, choices=(2,), default=2)
    parser.add_argument("--validate-reference", action="store_true", help="CPU8 proof only")
    args = parser.parse_args()
    result = run_probe(
        json.loads(args.config.read_text()),
        args.paired_oracle_dir,
        args.output_dir,
        platform=args.platform,
        repeats=args.repeats,
        validate_reference=args.validate_reference,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "execution_errors_present": result["execution_errors_present"],
                "numerical_findings_present": result["numerical_findings_present"],
                "raw_repeats_exact": result["raw_repeats_exact"],
            }
        ),
        flush=True,
    )
    if result["execution_errors_present"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
