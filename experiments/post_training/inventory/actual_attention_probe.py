# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment-only identity taps around the closed S8/S128 Snowball diagnostic cases.

No kernel is replaced: both custom VJPs return their incoming cotangents exactly.
Callbacks can still change compilation/layouts. Interpret replay only if the
observed raw-gradient failure persists. Reference attention is a probe oracle.
"""

import contextlib
import dataclasses
import hashlib
import json
import os
import threading
import traceback
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import levanter.models.snowball as snowball
import numpy as np
from levanter.grug.attention import AttentionMask, attention, fa4_cute_segment_bounds, reference_attention
from levanter.grug.attention import _fa4_cute_backend as fa4_backend
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig, Flash4CuteSm90BackwardConfig

ATOL = RTOL = 7e-2  # Existing tests/grug/test_fa4_cute_attention.py GPU thresholds.


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _stats(array):
    value = np.asarray(array).astype(np.float32)
    finite = np.isfinite(value)
    return {
        "shape": list(value.shape),
        "dtype": str(np.asarray(array).dtype),
        "sha256": hashlib.sha256(np.asarray(array).tobytes()).hexdigest(),
        "elements": int(value.size),
        "finite": int(finite.sum()),
        "nan": int(np.isnan(value).sum()),
        "inf": int(np.isinf(value).sum()),
        "finite_abs_max": float(np.abs(value[finite]).max()) if finite.any() else None,
    }


def _digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode())
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


class AttentionCapture:
    """Closed two-layer probe, bounded at 64 saved callbacks/8MiB host payload.

    The long-mask branch is layer1/NoPE for exactly this model. Callback ordinal
    is not a microbatch or device ID. Partitioned callbacks may observe local or
    logical arrays depending on JAX lowering; actual observed shapes are stored.
    """

    def __init__(
        self, directory: Path, *, geometry=(8, 2, 2, 64), max_per_kind=None, max_callbacks=64, max_bytes=8 << 20
    ):
        self.directory = Path(directory)
        self.geometry = geometry
        self.max_per_kind = max_per_kind
        self.omitted = {}
        self.max_callbacks = max_callbacks
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.records = []
        self.bytes = 0
        self.original = None

    def _capture(self, kind, q, k, v, a, b, c, bounds, valid):
        values = {
            name: np.asarray(value)
            for name, value in zip(
                ("q", "k", "v", "a", "b", "c", "bounds", "valid"), (q, k, v, a, b, c, bounds, valid), strict=True
            )
        }
        # The closed fixture has the same all-valid causal mask in every row.
        # Replicated mask arguments can arrive with a different batch size than
        # partitioned QKV. Never guess a row for a heterogeneous mask.
        for name in ("bounds", "valid"):
            value = values[name]
            if value.shape[0] != values["q"].shape[0]:
                if not np.array_equal(value, np.broadcast_to(value[:1], value.shape)):
                    raise ValueError("Cannot align heterogeneous callback mask to local QKV")
                values[name] = np.broadcast_to(value[:1], (values["q"].shape[0], value.shape[1])).copy()
        size = sum(value.nbytes for value in values.values())
        if size > 1 << 20:
            raise ValueError("Attention callback exceeds 1MiB per-record budget")
        with self.lock:
            if len(self.records) >= self.max_callbacks or self.bytes + size > self.max_bytes:
                raise ValueError("Attention capture exceeds bounded callback budget")
            if self.max_per_kind is not None and sum(r["kind"] == kind for r in self.records) >= self.max_per_kind:
                self.omitted[kind] = self.omitted.get(kind, 0) + 1
                _write(self.directory / "captures.json", self.receipt())
                return
            index = len(self.records)
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"capture-{index:03d}.npz"
            encoded, dtypes = {}, {}
            for name, value in values.items():
                dtypes[name] = str(value.dtype)
                encoded[name] = value.view(np.uint16) if str(value.dtype) == "bfloat16" else value
            np.savez(path, **encoded)
            row = {
                "ordinal": index,
                "kind": kind,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": size,
                "file_bytes": path.stat().st_size,
                "dtypes": dtypes,
                "qkv_sha256": _digest({name: values[name] for name in ("q", "k", "v")}),
                "stats": {name: _stats(value) for name, value in values.items()},
                "layer": 1,
                "branch": "long_NoPE",
                "pid": os.getpid(),
                "callback_thread_id": threading.get_ident(),
                "device_id": None,
                "temporal_microbatch_id": None,
            }
            self.records.append(row)
            self.bytes += size
            _write(self.directory / "captures.json", self.receipt())

    def receipt(self):
        return {
            "scope": "closed two-layer S8/S128 identity-tap observer; not production instrumentation",
            "records": list(self.records),
            "payload_bytes": self.bytes,
            "max_callbacks": self.max_callbacks,
            "max_payload_bytes": self.max_bytes,
            "geometry": list(self.geometry),
            "max_saved_per_kind": self.max_per_kind,
            "omitted_callbacks_by_kind": dict(self.omitted),
            "identity_gradient_contract": "Both taps return the exact incoming cotangent; no replacement VJP",
            "partitioned_callback": True,
            "observer_can_perturb_compilation": True,
            "labels": "ordinal/thread are not device or temporal microbatch IDs",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def _attention(self, q, k, v, mask, **kwargs):
        # Snowball scans layers and traces both cond branches. Only the executed
        # long branch emits backward callbacks in the closed two-layer model.
        target = isinstance(mask, AttentionMask) and mask.sliding_window is None
        if not target:
            return self.original(q, k, v, mask, **kwargs)
        seq, hq, hkv, dim = self.geometry
        if q.shape[1:] != (seq, hq, dim) or k.shape[1:] != (seq, hkv, dim):
            raise ValueError("Actual attention capture differs from the reviewed fixture geometry")
        bounds, valid = mask.fa4_bounds or fa4_cute_segment_bounds(
            mask, batch_size=q.shape[0], seq_len=seq, sliding_window=None
        )
        capture = self._capture

        @jax.custom_vjp
        def inputs(q, k, v, bounds, valid):
            return q, k, v

        def inputs_fwd(q, k, v, bounds, valid):
            return (q, k, v), (q, k, v, bounds, valid)

        def inputs_bwd(residual, grads):
            q, k, v, bounds, valid = residual
            dq, dk, dv = grads
            jax.debug.callback(
                partial(capture, "input_cotangents"), q, k, v, dq, dk, dv, bounds, valid, partitioned=True
            )
            return dq, dk, dv, None, None

        inputs.defvjp(inputs_fwd, inputs_bwd)
        q_tap, k_tap, v_tap = inputs(q, k, v, bounds, valid)
        output = self.original(q_tap, k_tap, v_tap, mask, **kwargs)

        @jax.custom_vjp
        def outputs(output, q, k, v, bounds, valid):
            return output

        def outputs_fwd(output, q, k, v, bounds, valid):
            return output, (output, q, k, v, bounds, valid)

        def outputs_bwd(residual, grad):
            output, q, k, v, bounds, valid = residual
            jax.debug.callback(
                partial(capture, "output_cotangent"),
                q,
                k,
                v,
                grad,
                output,
                jnp.zeros_like(output),
                bounds,
                valid,
                partitioned=True,
            )
            return grad, None, None, None, None, None

        outputs.defvjp(outputs_fwd, outputs_bwd)
        return outputs(output, q, k, v, bounds, valid)

    @contextlib.contextmanager
    def installed(self):
        if self.original is not None:
            raise RuntimeError("Attention observer is already installed")
        self.original = snowball.attention
        snowball.attention = self._attention
        try:
            yield self
        finally:
            snowball.attention = self.original
            self.original = None


def capture_raw_case(initializer: Path, output_dir: Path, platform: str):
    """Run existing actual-Trainer raw graph with identity taps in this process."""
    from experiments.post_training.inventory.native_gradient_diagnostics import _raw_case  # noqa: PLC0415 -- probe cycle

    cfg = json.loads((Path(initializer) / "config.json").read_text())
    # HF names, not native config names. Layer tag depends on this closed model.
    geometry = (cfg["max_position_embeddings"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"])
    if cfg["num_hidden_layers"] != 2 or geometry not in ((8, 2, 2, 64), (128, 4, 1, 128)):
        raise ValueError("Observer requires one of the two reviewed two-layer fixtures")
    output_dir = Path(output_dir)
    capture = AttentionCapture(
        output_dir / "captures", geometry=geometry, max_per_kind=2 if geometry[0] == 128 else None
    )
    with capture.installed():
        result = _raw_case(Path(initializer), output_dir / "observed-raw.json", platform, geometry[0])
    _write(output_dir / "observer.json", capture.receipt())
    if not capture.records:
        raise RuntimeError("Attention observer emitted no backward captures")
    return result


def _load_record(directory, row):
    path = directory / row["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Captured tensor archive hash differs")
    with np.load(path, allow_pickle=False) as archive:
        result = {}
        for name, dtype in row["dtypes"].items():
            value = jnp.asarray(archive[name])
            result[name] = jax.lax.bitcast_convert_type(value, jnp.bfloat16) if dtype == "bfloat16" else value
        return result


@partial(jax.jit, static_argnames=("implementation",))
def _replay(q, k, v, grad, bounds, valid, *, implementation):
    mask = AttentionMask.causal().with_fa4_bounds(bounds, valid)
    if implementation == "reference":

        def function(q, k, v):
            return reference_attention(q, k, v, mask, logits_dtype=jnp.float32)

    else:

        def function(q, k, v):
            return attention(q, k, v, mask, implementation=implementation)

    output, pullback = jax.vjp(function, q, k, v)
    return output, pullback(grad)


def replay_captures(directory: Path, output: Path, *, implementation="gpu_fa4_cute"):
    """Replay actual local QKV/dO outside any global mesh; reference is an oracle."""
    directory = Path(directory)
    receipt = json.loads((directory / "captures.json").read_text())
    result = {
        "scope": "actual captured attention input/cotangent replay only",
        "cases": [],
        "implementation": implementation,
        "atol": ATOL,
        "rtol": RTOL,
        "matching_recorded_input_cotangents": "Not inferred from callback arrival order",
        "observer_can_perturb_compilation": True,
    }
    for row in receipt["records"]:
        if row["kind"] != "output_cotangent":
            continue
        case = {"capture_ordinal": row["ordinal"], "qkv_sha256": row["qkv_sha256"], "status": "running"}
        result["cases"].append(case)
        _write(Path(output), result)
        try:
            values = _load_record(directory, row)
            args = tuple(values[name] for name in ("q", "k", "v", "a", "bounds", "valid"))
            got = _replay(*args, implementation=implementation)
            jax.block_until_ready(got)
            repeat = _replay(*args, implementation=implementation)
            jax.block_until_ready(repeat)
            want = _replay(*args, implementation="reference")
            jax.block_until_ready(want)
            repeats = {}
            for name, first, second in zip(
                ("output", "dq", "dk", "dv"), (got[0], *got[1]), (repeat[0], *repeat[1]), strict=True
            ):
                a, b = np.asarray(first).astype(np.float64), np.asarray(second).astype(np.float64)
                finite = np.isfinite(a) & np.isfinite(b)
                delta = np.abs(a[finite] - b[finite])
                repeats[name] = {
                    "first": _stats(first),
                    "second": _stats(second),
                    "exact_equal": bool(np.array_equal(a, b)),
                    "finite_both": bool(finite.all()),
                    "max_abs_finite": float(delta.max()) if delta.size else None,
                    "rms_finite": float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
                }
            comparisons = {}
            for name, actual, expected in zip(
                ("output", "dq", "dk", "dv"), (got[0], *got[1]), (want[0], *want[1]), strict=True
            ):
                a, b = np.asarray(actual).astype(np.float64), np.asarray(expected).astype(np.float64)
                finite = np.isfinite(a) & np.isfinite(b)
                delta = np.abs(a[finite] - b[finite])
                comparisons[name] = {
                    "actual": _stats(actual),
                    "reference": _stats(expected),
                    "finite_both": bool(finite.all()),
                    "max_abs_finite": float(delta.max()) if delta.size else None,
                    "rms_finite": float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
                    "within_existing_tolerance": bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)),
                }
            case.update(
                status=(
                    "passed"
                    if all(c["finite_both"] and c["within_existing_tolerance"] for c in comparisons.values())
                    else "failed"
                ),
                input_stats={name: row["stats"][name] for name in ("q", "k", "v", "a")},
                comparisons=comparisons,
                same_inputs_repeat=repeats,
                repeat_exactness_informational=True,
            )
        except Exception as error:
            case.update(
                status="error",
                error_type=type(error).__name__,
                error=str(error)[:4000],
                traceback=traceback.format_exc()[-8000:],
            )
        _write(Path(output), result)
    result["status"] = (
        "passed" if result["cases"] and all(case["status"] == "passed" for case in result["cases"]) else "failed"
    )
    result["execution_errors_present"] = any(case["status"] == "error" for case in result["cases"])
    _write(Path(output), result)
    return result


class BackwardResidualCapture:
    """Same-call backward residuals and gradients; closed S8/D64/MHA only.

    The real custom-VJP backward looks up fa4_backend's global function. This
    wrapper calls that original once, records both sides together, and returns
    its gradients unchanged. The callback can still perturb buffer lifetimes.
    """

    names = ("q", "k", "v", "out", "dout", "lse", "bounds", "valid", "q_offset", "dq", "dk", "dv")

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.lock = threading.Lock()
        self.records = []
        self.bytes = 0
        self.original = None

    def receipt(self):
        return {
            "scope": "original FA4 backward same-call residual/gradient observer; no recomputed forward",
            "records": list(self.records),
            "payload_bytes": self.bytes,
            "max_payload_bytes": 8 << 20,
            "max_records": 64,
            "hook": "levanter.grug.attention._fa4_cute_backend.segmented_flash_attention_backward",
            "original_calls_per_wrapped_invocation": 1,
            "gradient_return_contract": "Original dq/dk/dv returned unchanged",
            "observer_can_perturb_compilation_layout_and_timing": True,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def _capture(self, metadata, *arrays):
        values = {name: np.asarray(value) for name, value in zip(self.names, arrays, strict=True)}
        size = sum(value.nbytes for value in values.values())
        if size > 1 << 20:
            raise ValueError("Backward residual record exceeds 1MiB")
        with self.lock:
            if len(self.records) >= 64 or self.bytes + size > 8 << 20:
                raise ValueError("Backward residual capture exceeds 64 records/8MiB")
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"backward-{len(self.records):03d}.npz"
            encoded, dtypes = {}, {}
            for name, value in values.items():
                dtypes[name] = str(value.dtype)
                encoded[name] = value.view(np.uint16) if str(value.dtype) == "bfloat16" else value
            np.savez(path, **encoded)
            self.records.append(
                {
                    "ordinal": len(self.records),
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": size,
                    "file_bytes": path.stat().st_size,
                    "dtypes": dtypes,
                    "qkv_sha256": _digest({name: values[name] for name in ("q", "k", "v")}),
                    "all_backward_inputs_sha256": _digest({name: values[name] for name in self.names[:9]}),
                    "stats": {name: _stats(value) for name, value in values.items()},
                    "metadata": metadata,
                    "pid": os.getpid(),
                    "callback_thread_id": threading.get_ident(),
                    "device_id": None,
                    "layer_id": None,
                    "temporal_microbatch_id": None,
                    "pairing": "All residuals and gradients in this record come from ONE original backward invocation",
                }
            )
            self.bytes += size
            _write(self.directory / "captures.json", self.receipt())

    def wrap(self, original):
        """Common wrapper seam; CPU tests use an explicitly labeled reference stand-in."""

        def wrapped(q, k, v, out, dout, lse, lower_bounds, valid, *, softmax_scale, kernel_config, q_offset):
            if q.shape[1:] != (8, 2, 64) or k.shape[1:] != (8, 2, 64):
                raise ValueError("Backward residual observer only supports the reviewed S8/D64/MHA fixture")
            grads = original(
                q,
                k,
                v,
                out,
                dout,
                lse,
                lower_bounds,
                valid,
                softmax_scale=softmax_scale,
                kernel_config=kernel_config,
                q_offset=q_offset,
            )
            metadata = {
                "softmax_scale": float(softmax_scale),
                "kernel_config": dataclasses.asdict(kernel_config),
                "dispatch_branch_from_original_conditions": "segmented_fallback",
                "dispatch_evidence": "D64/ratio1 cannot enter native SM90 or SM100 branches",
                "kernel_trace_instrumentation": False,
            }
            jax.debug.callback(
                partial(self._capture, metadata),
                q,
                k,
                v,
                out,
                dout,
                lse,
                lower_bounds,
                valid,
                q_offset,
                *grads,
                partitioned=True,
            )
            return grads

        return wrapped

    @contextlib.contextmanager
    def installed(self):
        if self.original is not None:
            raise RuntimeError("Backward observer already installed")
        self.original = fa4_backend.segmented_flash_attention_backward
        fa4_backend.segmented_flash_attention_backward = self.wrap(self.original)
        try:
            yield self
        finally:
            fa4_backend.segmented_flash_attention_backward = self.original
            self.original = None


def capture_backward_residuals(initializer: Path, output_dir: Path, platform: str):
    """Run the old actual-Trainer graph with only the original-backward hook.

    CPU Trainer uses reference attention and never executes this GPU-only hook.
    That is reported explicitly; common-wrapper CPU tests are separate evidence.
    """
    from experiments.post_training.inventory.native_gradient_diagnostics import _raw_case  # noqa: PLC0415 -- probe cycle

    cfg = json.loads((Path(initializer) / "config.json").read_text())
    geometry = (
        cfg["num_hidden_layers"],
        cfg["hidden_size"],
        cfg["max_position_embeddings"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    if geometry != (2, 128, 8, 2, 2, 64):
        raise ValueError("Original-backward capture requires the exact reviewed old fixture")
    output_dir = Path(output_dir)
    capture = BackwardResidualCapture(output_dir / "backward-captures")
    with capture.installed():
        result = _raw_case(Path(initializer), output_dir / "observed-raw.json", platform, 8)
    receipt = capture.receipt()
    receipt["status"] = "captured" if capture.records else "not_run_cpu_original_gpu_backward"
    _write(output_dir / "backward-observer.json", receipt)
    _write(capture.directory / "captures.json", receipt)
    if platform == "gpu" and not capture.records:
        raise RuntimeError("Original FA4 backward hook emitted no records")
    return result


def _decode_kernel_config(data):
    data = dict(data)
    if data["sm100_forward"] is not None or data["sm100_backward"] is not None:
        raise ValueError("Closed backward replay does not accept SM100 config")
    data["forward_tile"] = tuple(data["forward_tile"])
    data["backward_tile"] = tuple(data["backward_tile"])
    if data["sm90_backward"] is not None:
        sm90 = dict(data["sm90_backward"])
        sm90["tile"] = tuple(sm90["tile"])
        data["sm90_backward"] = Flash4CuteSm90BackwardConfig(**sm90)
    return Flash4CuteKernelConfig(**data)


@partial(jax.jit, static_argnames=("softmax_scale", "kernel_config"))
def _replay_original_backward(q, k, v, out, dout, lse, bounds, valid, q_offset, *, softmax_scale, kernel_config):
    # There is intentionally NO forward call here. OUT/LSE are saved originals.
    return fa4_backend.segmented_flash_attention_backward(
        q,
        k,
        v,
        out,
        dout,
        lse,
        bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        q_offset=q_offset,
    )


def _compare_arrays(actual, expected):
    a, b = np.asarray(actual).astype(np.float64), np.asarray(expected).astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a[finite] - b[finite])
    return {
        "actual": _stats(actual),
        "reference": _stats(expected),
        "finite_both": bool(finite.all()),
        "exact_equal": bool(np.array_equal(a, b)),
        "max_abs_finite": float(delta.max()) if delta.size else None,
        "rms_finite": float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
        "within_existing_tolerance": bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)),
    }


def replay_backward_residuals(directory: Path, output: Path, *, implementation="gpu_fa4_cute"):
    """Direct backward on the SAME saved residuals; reference remains a separate oracle."""
    directory, output = Path(directory), Path(output)
    receipt = json.loads((directory / "captures.json").read_text())
    result = {
        "scope": "exact ORIGINAL backward residual replay, not regenerated OUT/LSE",
        "cases": [],
        "atol": ATOL,
        "rtol": RTOL,
        "implementation": implementation,
        "observer_can_perturb_compilation_layout_and_timing": True,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if implementation == "reference":
        result.update(
            status="not_run_cpu",
            execution_errors_present=False,
            reason="Original FA4 backward cannot execute on CPU; no reference replacement of this test",
        )
        _write(output, result)
        return result
    if implementation != "gpu_fa4_cute" or jax.default_backend() != "gpu":
        raise RuntimeError("Original backward residual replay requires the native GPU backend")
    for row in receipt["records"]:
        case = {
            "capture_ordinal": row["ordinal"],
            "qkv_sha256": row["qkv_sha256"],
            "all_backward_inputs_sha256": row["all_backward_inputs_sha256"],
            "status": "running",
            "metadata": row["metadata"],
            "original_gradient_stats": {name: row["stats"][name] for name in ("dq", "dk", "dv")},
            "input_stats": {name: row["stats"][name] for name in BackwardResidualCapture.names[:9]},
        }
        result["cases"].append(case)
        _write(output, result)
        try:
            values = _load_record(directory, row)
            metadata = row["metadata"]
            config = _decode_kernel_config(metadata["kernel_config"])
            scale = metadata["softmax_scale"]
            args = tuple(values[name] for name in BackwardResidualCapture.names[:9])
            first = _replay_original_backward(*args, softmax_scale=scale, kernel_config=config)
            jax.block_until_ready(first)
            second = _replay_original_backward(*args, softmax_scale=scale, kernel_config=config)
            jax.block_until_ready(second)
            # The reference is an independent oracle. It is NOT substituted for
            # the saved original OUT/LSE in either direct backward call above.
            if scale != 1 / np.sqrt(64) or np.any(np.asarray(values["q_offset"]) != 0):
                raise ValueError("Reference oracle only supports reviewed default scale and zero query offset")
            oracle = _replay(
                values["q"],
                values["k"],
                values["v"],
                values["dout"],
                values["bounds"],
                values["valid"],
                implementation="reference",
            )
            jax.block_until_ready(oracle)
            comparisons = {
                name: _compare_arrays(a, b) for name, a, b in zip(("dq", "dk", "dv"), first, oracle[1], strict=True)
            }
            case.update(
                comparisons=comparisons,
                same_saved_residuals_repeat={
                    name: _compare_arrays(a, b) for name, a, b in zip(("dq", "dk", "dv"), first, second, strict=True)
                },
                original_recorded_vs_direct_replay={
                    name: _compare_arrays(values[name], a) for name, a in zip(("dq", "dk", "dv"), first, strict=True)
                },
                original_saved_output_vs_reference=_compare_arrays(values["out"], oracle[0]),
                input_stats={name: row["stats"][name] for name in BackwardResidualCapture.names[:9]},
                original_gradient_stats={name: row["stats"][name] for name in ("dq", "dk", "dv")},
                repeat_and_original_exactness_informational=True,
                status=(
                    "passed"
                    if all(c["finite_both"] and c["within_existing_tolerance"] for c in comparisons.values())
                    else "failed"
                ),
            )
        except Exception as error:
            case.update(
                status="error",
                error_type=type(error).__name__,
                error=str(error)[:4000],
                traceback=traceback.format_exc()[-8000:],
            )
        _write(output, result)
    result["status"] = (
        "passed" if result["cases"] and all(c["status"] == "passed" for c in result["cases"]) else "failed"
    )
    result["execution_errors_present"] = any(c["status"] == "error" for c in result["cases"])
    result["original_nonfinite_events_present"] = any(
        any(q["nan"] or q["inf"] for q in c.get("original_gradient_stats", {}).values()) for c in result["cases"]
    )
    _write(output, result)
    return result
