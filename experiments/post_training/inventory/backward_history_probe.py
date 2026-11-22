# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded F/F/NaN-dO/F/F replay of one pinned finite FA4 residual packet.

One public compiled object, fixed finite argument objects, dynamic dO, unchanged
backward/kernel settings. Reference mode is only a CPU API-contract stand-in.
This experiment never launches jobs and never establishes a race mechanism.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from levanter.grug.attention import AttentionMask, reference_attention

from experiments.post_training.inventory.actual_attention_probe import (
    ATOL,
    RTOL,
    BackwardResidualCapture,
    _compare_arrays,
    _decode_kernel_config,
    _digest,
    _stats,
    _write,
    fa4_backend,
)

_NAMES = BackwardResidualCapture.names[:9]
_SOURCE_HEAD = "d3fecc6bf743e25ae26f42bac8df8b5602d3242b"
_ORIGIN_ARCHIVE_SHA = "1b9da4e63b963e80fbf55f8d919382d0c219d771792858dacf952c0f5cb31720"
_INPUT_DIGEST = "adef6a04db05cb0df58d270ce8bd6359ba575e8b913de04d9e4c414041fa3dc7"
_PACKET_SHA = "bbf1153225253d46ec84ff3d722c9eb259878ec338794b2ff3573ff0f4d3de56"
_OUTPUT_LIMIT = 16 << 20
_FLASH_SOURCE_SHA = {
    "flash_attn/cute/flash_bwd_preprocess.py": "b43ec797cb57938ee887a1576eb05aff0c1ead686e073ed4f426b47dc7a7c830",
    "flash_attn/cute/flash_bwd_postprocess.py": "df88b299cb938197abc145c3a85262515d1d810633229d0f929239e8291e8ce8",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_packet(packet, metadata, packet_sha256, metadata_sha256):
    if Path(packet).stat().st_size != 19374 or Path(metadata).stat().st_size > 64 << 10:
        raise ValueError("Pinned packet or metadata exceeds its bounded input contract")
    if _sha(packet) != packet_sha256 or _sha(metadata) != metadata_sha256:
        raise ValueError("Pinned packet or metadata SHA differs")
    data = json.loads(Path(metadata).read_text())
    row = data["record"]
    if (
        data["source_head"] != _SOURCE_HEAD
        or data["archive_sha256"] != _ORIGIN_ARCHIVE_SHA
        or row["all_backward_inputs_sha256"] != _INPUT_DIGEST
        or packet_sha256 != _PACKET_SHA
        or row["ordinal"] != 27
        or row["sha256"] != packet_sha256
    ):
        raise ValueError("Packet differs from the explicit original finite record27 contract")
    result = {}
    with np.load(packet, allow_pickle=False) as archive:
        if set(archive.files) != set(BackwardResidualCapture.names):
            raise ValueError("Unexpected residual packet fields")
        for name, dtype in row["dtypes"].items():
            raw = archive[name]
            if dtype == "bfloat16":
                if raw.dtype != np.uint16:
                    raise ValueError("BF16 packet must contain raw uint16 bits")
                # Host-only exact bit decoding; do not initialize a backend yet.
                value = (raw.astype(np.uint32) << 16).view(np.float32)
            else:
                if str(raw.dtype) != dtype:
                    raise ValueError("Residual dtype differs from metadata")
                value = raw
            if (
                list(raw.shape) != row["stats"][name]["shape"]
                or hashlib.sha256(raw.tobytes()).hexdigest() != row["stats"][name]["sha256"]
                or not np.isfinite(value).all()
            ):
                raise ValueError("Packet schema/hash/finiteness differs")
            result[name] = np.array(raw, copy=True)
    if row["metadata"]["softmax_scale"] != 0.125 or np.any(result["q_offset"] != 0):
        raise ValueError("Probe requires reviewed default scale and zero query offset")
    return data, result


def _array_formats(array):
    form = array.format
    layout = form.layout
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "format": str(form),
        "sharding": str(form.sharding),
        "layout": (
            None
            if layout is None
            else {
                "major_to_minor": list(layout.major_to_minor),
                "tiling": layout.tiling,
                "sub_byte_element_size_in_bits": layout.sub_byte_element_size_in_bits,
            }
        ),
    }


def _identity(mode):
    devices = jax.local_devices()
    expected = "gpu" if mode == "gpu" else "cpu"
    if jax.default_backend() != expected or jax.process_count() != 1 or not devices:
        raise RuntimeError("History probe requires the requested single-process backend")
    if mode == "gpu" and "H100" not in devices[0].device_kind:
        raise RuntimeError("History probe requires an actual H100")
    if (jax.__version__, jaxlib.__version__, np.__version__) != ("0.11.0", "0.11.0", "2.3.5"):
        raise RuntimeError("Native JAX/NumPy package identity differs")
    package_sources = {}
    for package, files in {
        "flash-attn-4": ("flash_attn/cute/flash_bwd_preprocess.py", "flash_attn/cute/flash_bwd_postprocess.py"),
        "nvidia-cutlass-dsl": (),
    }.items():
        try:
            distribution = importlib.metadata.distribution(package)
            package_sources[package] = {
                "version": distribution.version,
                "files": {name: _sha(distribution.locate_file(name)) for name in files},
            }
        except importlib.metadata.PackageNotFoundError:
            package_sources[package] = {"status": "not_installed_cpu"}
    if mode == "gpu" and any(
        package_sources[p].get("version") != v
        for p, v in (("flash-attn-4", "4.0.0b28"), ("nvidia-cutlass-dsl", "4.6.2"))
    ):
        raise RuntimeError("Original GPU dependency versions differ")
    if mode == "gpu" and package_sources["flash-attn-4"]["files"] != _FLASH_SOURCE_SHA:
        raise RuntimeError("Installed original preprocess/postprocess source differs from the pinned wheel")
    backend_root = Path(fa4_backend.__file__).parent
    backend_sources = {
        name: _sha(backend_root / name)
        for name in ("_fa4_cute_backend.py", "_fa4_cute_kernels.py", "_fa4_cute_segmented_bwd.py", "_fa4_cute_config.py")
    }
    return {
        "mode": mode,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "local_device_count": len(devices),
        "process_count": jax.process_count(),
        "selected_device": {"id": devices[0].id, "kind": devices[0].device_kind, "platform": devices[0].platform},
        "dependency_sources": package_sources,
        "backend_sources": backend_sources,
    }


def _save_text(root, name, text):
    encoded = text.encode()
    info = {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    if len(encoded) <= 2 << 20:
        (root / name).write_bytes(encoded)
        info["file"] = name
    else:
        info["file_omitted_due_to_2MiB_budget"] = True
    return info


def _host_values(values):
    return {name: np.asarray(array) for name, array in zip(_NAMES, values, strict=True)}


def run_history_probe(
    packet: Path, metadata: Path, output_dir: Path, *, packet_sha256: str, metadata_sha256: str, mode="gpu"
):
    """Compile once, then keep all outputs alive through the fixed five calls."""
    if mode not in ("gpu", "cpu_reference"):
        raise ValueError("Unsupported history probe mode")
    data, host = _read_packet(packet, metadata, packet_sha256, metadata_sha256)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "history-probe.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite a prior history probe")
    result = {
        "status": "running",
        "execution_errors_present": False,
        "numerical_findings_present": False,
        "scope": "fixed saved-residual invocation history only; not learner qualification",
        "mode": mode,
        "packet_sha256": packet_sha256,
        "metadata_sha256": metadata_sha256,
        "origin": {
            "source_head": data["source_head"],
            "archive_sha256": data["archive_sha256"],
            "record_ordinal": data["record"]["ordinal"],
            "input_digest": _INPUT_DIGEST,
        },
        "sequence": [],
        "atol": ATOL,
        "rtol": RTOL,
        "source_sha256": _sha(__file__),
        "helper_source_sha256": _sha(Path(__file__).with_name("actual_attention_probe.py")),
        "static_backward_config": data["record"]["metadata"],
        "donate_argnums": [],
        "output_limit_bytes": _OUTPUT_LIMIT,
        "kept_all_outputs_alive_until_sequence_complete": True,
        "scientific_limit": "Any reproduced history/placement dependence does not identify a race/allocator mechanism",
    }
    _write(target, result)
    try:
        result["runtime"] = _identity(mode)
        config = _decode_kernel_config(data["record"]["metadata"]["kernel_config"])
        scale = data["record"]["metadata"]["softmax_scale"]
        with jax.default_device(jax.local_devices()[0]):
            values = {
                name: (
                    jax.lax.bitcast_convert_type(jnp.asarray(host[name]), jnp.bfloat16)
                    if data["record"]["dtypes"][name] == "bfloat16"
                    else jnp.asarray(host[name])
                )
                for name in _NAMES
            }
            finite_args = tuple(values[name] for name in _NAMES)
            poison = jnp.full_like(values["dout"], jnp.nan)
            poison_args = (*finite_args[:4], poison, *finite_args[5:])
            jax.block_until_ready((finite_args, poison))
            result["finite_input_before"] = {
                name: _stats(array) for name, array in zip(_NAMES, finite_args, strict=True)
            }
            result["input_formats"] = {
                name: _array_formats(array) for name, array in zip(_NAMES, finite_args, strict=True)
            }
            if _digest(_host_values(finite_args)) != _INPUT_DIGEST:
                raise RuntimeError("Device finite residual values differ from pinned packet")
            trace_calls = []

            def backward(q, k, v, out, dout, lse, bounds, valid, q_offset):
                trace_calls.append("one_public_lower_trace")
                if mode == "cpu_reference":
                    mask = AttentionMask.causal().with_fa4_bounds(bounds, valid)
                    _, pullback = jax.vjp(
                        lambda q, k, v: reference_attention(q, k, v, mask, logits_dtype=jnp.float32), q, k, v
                    )
                    return pullback(dout)
                return fa4_backend.segmented_flash_attention_backward(
                    q, k, v, out, dout, lse, bounds, valid, softmax_scale=scale, kernel_config=config, q_offset=q_offset
                )

            lowered = jax.jit(backward).lower(*finite_args)
            compiled = lowered.compile()
            if len(trace_calls) != 1:
                raise RuntimeError("Probe did not trace the backward exactly once")
            result["compilation"] = {
                "public_api": "jax.jit(backward).lower(...).compile()",
                "compile_count": 1,
                "python_trace_count": len(trace_calls),
                "compiled_object_id_process_local": id(compiled),
                "input_formats": str(compiled.input_formats),
                "output_formats": str(compiled.output_formats),
                "lowered": _save_text(root, "lowered.mlir", lowered.as_text()),
            }
            compiled_text = compiled.as_text()
            result["compilation"]["compiled"] = (
                _save_text(root, "compiled.hlo", compiled_text) if compiled_text else None
            )
            result["compilation"]["input_avals"] = str(compiled.in_avals)
            _write(target, result)
            outputs = []
            # dO is an ordinary dynamic argument. All other argument OBJECTS and
            # the compiled callable are reused. No donation or recompilation.
            for label, args in (
                ("F1", finite_args),
                ("F2", finite_args),
                ("N", poison_args),
                ("F3", finite_args),
                ("F4", finite_args),
            ):
                grads = compiled(*args)
                jax.block_until_ready(grads)
                outputs.append(grads)
                arrays = {name: np.asarray(array) for name, array in zip(("dq", "dk", "dv"), grads, strict=True)}
                path = root / (label + "-gradients.npz")
                np.savez(
                    path,
                    **{
                        name: value.view(np.uint16) if str(value.dtype) == "bfloat16" else value
                        for name, value in arrays.items()
                    },
                )
                row = {
                    "label": label,
                    "compiled_object_id_process_local": id(compiled),
                    "diagnostic_NaN_dO": label == "N",
                    "dO": _stats(args[4]),
                    "gradient_stats": {name: _stats(value) for name, value in arrays.items()},
                    "gradient_file": path.name,
                    "gradient_file_sha256": _sha(path),
                }
                if label != "N" and (
                    any(v["nan"] or v["inf"] for v in row["gradient_stats"].values())
                    or (
                        result["sequence"]
                        and any(
                            row["gradient_stats"][name]["sha256"]
                            != result["sequence"][0]["gradient_stats"][name]["sha256"]
                            for name in ("dq", "dk", "dv")
                        )
                    )
                ):
                    result["numerical_findings_present"] = True
                result["sequence"].append(row)
                _write(target, result)
            result["finite_input_after"] = {name: _stats(array) for name, array in zip(_NAMES, finite_args, strict=True)}
            result["finite_inputs_pure"] = result["finite_input_before"] == result["finite_input_after"]
            result["finite_input_after_digest"] = _digest(_host_values(finite_args))
            if not result["finite_inputs_pure"] or result["finite_input_after_digest"] != _INPUT_DIGEST:
                result["numerical_findings_present"] = True
                raise RuntimeError("Backward calls mutated fixed finite residual inputs")
            mask = AttentionMask.causal().with_fa4_bounds(values["bounds"], values["valid"])
            _, reference_pullback = jax.vjp(
                lambda q, k, v: reference_attention(q, k, v, mask, logits_dtype=jnp.float32),
                values["q"],
                values["k"],
                values["v"],
            )
            reference_grads = reference_pullback(values["dout"])
            jax.block_until_ready(reference_grads)
            result["reference_gradient_stats"] = {
                name: _stats(g) for name, g in zip(("dq", "dk", "dv"), reference_grads, strict=True)
            }
            for i, row in enumerate(result["sequence"]):
                row["versus_F1"] = {
                    name: _compare_arrays(g, base)
                    for name, g, base in zip(("dq", "dk", "dv"), outputs[i], outputs[0], strict=True)
                }
                if row["label"] != "N":
                    row["versus_reference"] = {
                        name: _compare_arrays(g, ref)
                        for name, g, ref in zip(("dq", "dk", "dv"), outputs[i], reference_grads, strict=True)
                    }
            result["finite_calls_all_finite_within_reference_tolerance"] = all(
                c["finite_both"] and c["within_existing_tolerance"]
                for row in result["sequence"]
                if row["label"] != "N"
                for c in row["versus_reference"].values()
            )
            result["finite_before_vs_after_bitwise_equal"] = all(
                row["gradient_stats"][name]["sha256"] == result["sequence"][0]["gradient_stats"][name]["sha256"]
                for row in result["sequence"]
                if row["label"] != "N"
                for name in ("dq", "dk", "dv")
            )
        result["python_trace_count_after_sequence"] = len(trace_calls)
        result["pre_N_F_calls_healthy"] = all(
            c["finite_both"] and c["within_existing_tolerance"]
            for row in result["sequence"][:2]
            for c in row["versus_reference"].values()
        )
        result["post_N_F_differs_bitwise_from_F1"] = any(
            row["gradient_stats"][name]["sha256"] != result["sequence"][0]["gradient_stats"][name]["sha256"]
            for row in result["sequence"][3:]
            for name in ("dq", "dk", "dv")
        )
        result["numerical_findings_present"] = result["numerical_findings_present"] or not (
            result["finite_calls_all_finite_within_reference_tolerance"]
            and result["finite_before_vs_after_bitwise_equal"]
        )
        result["status"] = "completed"
        result["actual_GPU_backward_executed"] = mode == "gpu"
        result["CPU_scope"] = (
            "reference API/compile/serialization stand-in only; GPU backward NOT_RUN_CPU" if mode != "gpu" else None
        )
    except Exception as error:
        result.update(
            status="error", execution_errors_present=True, error_type=type(error).__name__, error=str(error)[:4000]
        )
        _write(target, result)
        raise
    _write(target, result)
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if total >= _OUTPUT_LIMIT:
        result.update(
            status="error", execution_errors_present=True, error_type="OutputBudgetExceeded", output_bytes=total
        )
        _write(target, result)
        raise RuntimeError("History probe output exceeds 16MiB")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("gpu", "cpu_reference"), default="gpu")
    args = parser.parse_args()
    run_history_probe(
        args.packet,
        args.metadata,
        args.output_dir,
        packet_sha256=args.packet_sha256,
        metadata_sha256=args.metadata_sha256,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
