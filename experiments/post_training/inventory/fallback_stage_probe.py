# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Locate D64/MHA fallback history corruption without changing production code.

Run control, stages and prefill in separate fresh processes. The control delegates
to the unchanged history probe. Stage copies intentionally perturb the launch
graph and can suppress an ordering problem; a healthy observer is not a remedy.
CPU mode exercises serialization/compiled-call contracts with a reference stand-in,
not the GPU stages or their clearing/order contract.
"""

import argparse
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from levanter.cutlass_kernel_cache import cutlass_call
from levanter.grug.attention import AttentionMask, reference_attention
from levanter.grug.attention import _fa4_cute_kernels as kernels

from experiments.post_training.inventory.actual_attention_probe import (
    ATOL,
    RTOL,
    _compare_arrays,
    _decode_kernel_config,
    _digest,
    _stats,
    _write,
    fa4_backend,
)
from experiments.post_training.inventory.backward_history_probe import (
    _NAMES,
    _host_values,
    _identity,
    _read_packet,
    _save_text,
    _sha,
    run_history_probe,
)

_SNAPSHOTS = (
    "dq_after_preprocess",
    "dq_after_main",
    "dpsum_after_preprocess",
    "lse_after_preprocess",
)
_OUTPUTS = ("dq", "dk", "dv", *_SNAPSHOTS, "addresses")
_LIMIT = 4 << 20

_DSL_SOURCES = {
    "base_dsl/dsl.py": "e3b98e040fc44f6067ed396107b5881fa8cb5b9eb10b0dfe0205e075f7b9c1d5",
    "cute/typing.py": "e08e02c14ea72e718e905fc8f4d97b02a1040bb7914d1e9c4d82b410f8e65c5d",
    "cutlass_dsl/cutlass.py": "60a81d4516ad0b90c9fb29c3a0b02edc8cbc162c62677d4a15223bebfbb0535c",
}


def _installed_dsl_contract():
    distribution = importlib.metadata.distribution("nvidia-cutlass-dsl-libs-core")
    actual = {
        name: _sha(distribution.locate_file("nvidia_cutlass_dsl/dsl_packages/cutlass/" + name)) for name in _DSL_SOURCES
    }
    if distribution.version != "4.6.2" or actual != _DSL_SOURCES:
        raise RuntimeError("Installed CuTe launch/pointer API differs from reviewed 4.6.2 source")
    return {"version": distribution.version, "sources": actual}


def _launcher(modules: Any, *, prefill: bool):
    """Original D64/ratio1 constructors and arguments, with explicit observer copies."""
    deps = kernels._import_cute_dependencies(modules)
    cutlass, cute, cuda = deps.cutlass, deps.cute, deps.cuda
    pre_module = importlib.import_module("flash_attn.cute.flash_bwd_preprocess")
    post_module = importlib.import_module("flash_attn.cute.flash_bwd_postprocess")
    main_module = importlib.import_module("levanter.grug.attention._fa4_cute_segmented_bwd")
    # Closed geometry from the original pinned packet. These match the production
    # fallback factory; no generalized attention launcher is being introduced.
    preprocess = pre_module.FlashAttentionBackwardPreprocess(
        cutlass.BFloat16, 64, 64, 64, num_threads=128, use_padded_offsets=True
    )
    backward = main_module.SegmentedFlashAttentionBackwardSm120(
        cutlass.BFloat16,
        64,
        64,
        1,
        64,
        64,
        2,
        2,
        128,
        False,
        True,
        False,
        False,
        False,
        4,
        4,
        4,
        V_in_regs=False,
        score_mod=None,
        score_mod_bwd=None,
    )
    postprocess = post_module.FlashAttentionBackwardPostprocess(
        cutlass.BFloat16, 64, 120, 64, num_threads=128, AtomLayoutMdQ=4
    )
    as_gmem = kernels._native_gmem_tensor_view(modules)

    class Snapshot:
        @cute.jit
        def copy(self, source: cute.Tensor, destination: cute.Tensor, stream: cuda.CUstream):
            self.copy_kernel(source, destination).launch(
                grid=[cute.ceil_div(cute.size(source), 128), 1, 1],
                block=[128, 1, 1],
                stream=stream,
                use_pdl=False,
            )

        @cute.kernel
        def copy_kernel(self, source: cute.Tensor, destination: cute.Tensor):
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            src = cute.make_tensor(source.iterator, cute.make_layout(cute.size(source)))
            dst = cute.make_tensor(destination.iterator, cute.make_layout(cute.size(destination)))
            index = cutlass.Int64(block) * 128 + cutlass.Int64(thread)
            if index < cute.size(src):
                dst[index] = src[index]

        @cute.jit
        def poison(self, destination: cute.Tensor, stream: cuda.CUstream):
            self.poison_kernel(destination).launch(
                grid=[cute.ceil_div(cute.size(destination), 128), 1, 1],
                block=[128, 1, 1],
                stream=stream,
                use_pdl=False,
            )

        @cute.kernel
        def poison_kernel(self, destination: cute.Tensor):
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            dst = cute.make_tensor(destination.iterator, cute.make_layout(cute.size(destination)))
            index = cutlass.Int64(block) * 128 + cutlass.Int64(thread)
            if index < cute.size(dst):
                dst[index] = cutlass.Float32(float("nan"))

        @cute.jit
        def addresses(
            self,
            scratch: cute.Tensor,
            snapshot: cute.Tensor,
            result: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.address_kernel(scratch, snapshot, result).launch(
                grid=[1, 1, 1], block=[1, 1, 1], stream=stream, use_pdl=False
            )

        @cute.kernel
        def address_kernel(self, scratch: cute.Tensor, snapshot: cute.Tensor, result: cute.Tensor):
            a = cutlass.Uint64(scratch.iterator.toint())
            b = cutlass.Uint64(snapshot.iterator.toint())
            result[0] = cutlass.Uint32(a & 0xFFFFFFFF)
            result[1] = cutlass.Uint32(a >> 32)
            result[2] = cutlass.Uint32(b & 0xFFFFFFFF)
            result[3] = cutlass.Uint32(b >> 32)

    snapshot = Snapshot()

    @cute.jit
    def launch(
        stream: cuda.CUstream,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        dout: cute.Tensor,
        lse: cute.Tensor,
        bounds: cute.Tensor,
        valid: cute.Tensor,
        q_offset: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dpsum: cute.Tensor,
        lse_log2: cute.Tensor,
        dq_accum: cute.Tensor,
        dk_accum: cute.Tensor,
        dv_accum: cute.Tensor,
        after_pre: cute.Tensor,
        after_main: cute.Tensor,
        dpsum_pre: cute.Tensor,
        lse_pre: cute.Tensor,
        addresses: cute.Tensor,
        *,
        softmax_scale: cutlass.Float32,
    ):
        if cutlass.const_expr(prefill):
            snapshot.poison(dq_accum, stream)
        preprocess(
            out,
            dout,
            dpsum,
            lse,
            lse_log2,
            dq_accum,
            None,
            None,
            None,
            None,
            None,
            softmax_scale,
            None,
            stream,
        )
        snapshot.copy(dq_accum, after_pre, stream)
        snapshot.copy(dpsum, dpsum_pre, stream)
        snapshot.copy(lse_log2, lse_pre, stream)
        snapshot.addresses(dq_accum, after_pre, addresses, stream)
        backward(
            q,
            k,
            v,
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk,
            dv,
            bounds,
            valid,
            q_offset,
            softmax_scale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            stream,
        )
        snapshot.copy(dq_accum, after_main, stream)
        postprocess(as_gmem(dq_accum), dq, softmax_scale, None, None, None, None, stream)

    contract = {
        "preprocess_class": type(preprocess).__name__,
        "preprocess_use_pdl_at_construction": bool(preprocess.use_pdl),
        "main_class": type(backward).__name__,
        "main_use_pdl": False,
        "main_launch_attribute_basis": "Pinned adapted launch omits use_pdl; pinned DSL default is False.",
        "postprocess_class": type(postprocess).__name__,
        "postprocess_use_pdl": False,
        "postprocess_launch_attribute_basis": "Pinned postprocess launch omits use_pdl; pinned DSL default is False.",
        "observer_copy_use_pdl": False,
        "stream": "Every stage receives the same incoming CUDA stream argument; numeric handle not recorded.",
        "pointer_record": "dq_accum and independent after-preprocess snapshot addresses, low/high uint32 words",
        "prefill_before_original_preprocess": prefill,
    }
    return launch, contract


def _gpu_backward(prefill: bool):
    modules = fa4_backend._import_cutlass_cute()
    launcher, contract = _launcher(modules, prefill=prefill)
    scratch = jax.ShapeDtypeStruct((1, 2, 4096), jnp.float32)
    sums = jax.ShapeDtypeStruct((1, 2, 64), jnp.float32)
    ptrs = jax.ShapeDtypeStruct((4,), jnp.uint32)
    spec = modules.cjax.TensorSpec
    extra_specs = (spec(mode=(0, 1, 2), static=True),) * 4 + (spec(mode=(0,), static=True),)
    input_specs, output_specs = fa4_backend._cutlass_attention_backward_specs(
        modules, vector_elems=8, qhead_per_kvhead=1
    )

    def backward(q, k, v, out, dout, lse, bounds, valid, q_offset):
        shapes = fa4_backend._cutlass_attention_backward_output_shapes(q, k, v, (64, 64))
        call = cutlass_call(
            launcher,
            output_shape_dtype=(*shapes, scratch, scratch, sums, sums, ptrs),
            input_spec=input_specs,
            output_spec=(*output_specs, *extra_specs),
            use_static_tensors=True,
            softmax_scale=0.125,
        )
        values = call(q, k, v, out, dout, lse, bounds, valid.astype(jnp.int32), q_offset)
        return (*values[:3], *values[8:])

    return backward, contract


def _cpu_backward(q, k, v, out, dout, lse, bounds, valid, q_offset):
    del out, lse, q_offset
    mask = AttentionMask.causal().with_fa4_bounds(bounds, valid)
    _, pullback = jax.vjp(
        lambda q, k, v: reference_attention(q, k, v, mask, logits_dtype=jnp.float32),
        q,
        k,
        v,
    )
    gradients = pullback(dout)
    # Deliberate synthetic snapshots: CPU cannot exercise the CUDA scratch ABI.
    scratch = jnp.zeros((1, 2, 4096), dtype=jnp.float32)
    sums = jnp.zeros((1, 2, 64), dtype=jnp.float32)
    return (*gradients, scratch, scratch, sums, sums, jnp.zeros((4,), dtype=jnp.uint32))


def _regions(array, *, feature_size: int):
    host = np.asarray(array).reshape(1, 2, 64, feature_size)
    names = (
        ("active_query_rows", "padded_query_rows")
        if feature_size == 1
        else ("physical_prefix_1024_elements", "physical_remaining_7168_elements")
    )
    return {name: _stats(values) for name, values in zip(names, (host[:, :, :8], host[:, :, 8:]), strict=True)}


def run_stage_probe(
    packet,
    metadata,
    output_dir,
    *,
    packet_sha256,
    metadata_sha256,
    mode="gpu",
    case="stages",
):
    if case == "control":
        return run_history_probe(
            packet,
            metadata,
            output_dir,
            packet_sha256=packet_sha256,
            metadata_sha256=metadata_sha256,
            mode=mode,
        )
    if case not in ("stages", "prefill") or mode not in ("gpu", "cpu_reference"):
        raise ValueError("Unsupported stage probe case or mode")
    data, host = _read_packet(packet, metadata, packet_sha256, metadata_sha256)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "stage-probe.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing stage probe")
    receipt = {
        "status": "running",
        "execution_errors_present": False,
        "numerical_findings_present": False,
        "mode": mode,
        "case": case,
        "actual_GPU_stages_executed": False,
        "source_sha256": _sha(Path(__file__)),
        "packet_sha256": packet_sha256,
        "metadata_sha256": metadata_sha256,
        "output_limit_bytes": _LIMIT,
        "origin": {k: data[k] for k in ("source_head", "archive_sha256")},
        "scope": "D64/ratio1 fallback only; no native D128/GQA claim or learner acceptance",
        "perturbation": "Extra copies and buffers may suppress ordering/history corruption.",
        "CPU_scope": "CPU reference: SYNTHETIC snapshots; GPU scratch clearing NOT_RUN_CPU.",
        "sequence": [],
    }
    _write(target, receipt)
    try:
        receipt["runtime"] = _identity(mode)
        config = _decode_kernel_config(data["record"]["metadata"]["kernel_config"])
        if config.backward_tile != (64, 64) or config.num_threads != 128:
            raise ValueError("Packet fallback geometry changed")
        with jax.default_device(jax.local_devices()[0]):
            values = {
                name: (
                    jax.lax.bitcast_convert_type(jnp.asarray(host[name]), jnp.bfloat16)
                    if data["record"]["dtypes"][name] == "bfloat16"
                    else jnp.asarray(host[name])
                )
                for name in _NAMES
            }
            finite = tuple(values[name] for name in _NAMES)
            poison = (*finite[:4], jnp.full_like(values["dout"], jnp.nan), *finite[5:])
            jax.block_until_ready((finite, poison))
            receipt["input_before"] = {name: _stats(a) for name, a in zip(_NAMES, finite, strict=True)}
            if _digest(_host_values(finite)) != data["record"]["all_backward_inputs_sha256"]:
                raise RuntimeError("Device values differ from the pinned input packet")
            if mode == "gpu":
                receipt["runtime"]["installed_DSL_contract"] = _installed_dsl_contract()
                backward, receipt["launch_contract"] = _gpu_backward(case == "prefill")
            else:
                backward = _cpu_backward
                receipt["launch_contract"] = {
                    "status": "NOT_RUN_CPU",
                    "snapshots": "synthetic",
                    "prefill": "NOT_RUN_CPU",
                }
            traces = []

            def call(*args):
                traces.append(1)
                return backward(*args)

            lowered = jax.jit(call).lower(*finite)
            compiled = lowered.compile()
            receipt["compilation"] = {
                "compile_count": 1,
                "python_trace_count": len(traces),
                "id_process_local": id(compiled),
                "lowered": _save_text(root, "lowered.mlir", lowered.as_text()),
                "compiled": _save_text(root, "compiled.hlo", compiled.as_text() or ""),
                "input_formats": str(compiled.input_formats),
                "output_formats": str(compiled.output_formats),
                "donation": [],
            }
            retained = []
            sequence = (
                ("F1", finite),
                ("F2", finite),
                ("N", poison),
                ("F3", finite),
                ("F4", finite),
            )
            if case == "prefill":
                sequence = (("finite_after_whole_scratch_NaN_prefill", finite),)
            for label, args in sequence:
                output = compiled(*args)
                jax.block_until_ready(output)
                receipt["actual_GPU_stages_executed"] = mode == "gpu"
                retained.append(output)
                arrays = {name: np.asarray(v) for name, v in zip(_OUTPUTS, output, strict=True)}
                path = root / f"{label}-stages.npz"
                row = {
                    "label": label,
                    "compiled_id_process_local": id(compiled),
                    "file": path.name,
                    "file_status": "pending",
                    "stats": {name: _stats(a) for name, a in arrays.items()},
                    "dq_after_preprocess_regions": _regions(arrays["dq_after_preprocess"], feature_size=64),
                    "dq_after_main_regions": _regions(arrays["dq_after_main"], feature_size=64),
                    "dpsum_preprocess_regions": _regions(arrays["dpsum_after_preprocess"], feature_size=1),
                    "lse_preprocess_regions": _regions(arrays["lse_after_preprocess"], feature_size=1),
                    "dq_scratch_regions_basis": "Physical ranges only; not MMA logical query rows.",
                    "whole_dq_scratch_zero_after_preprocess": bool(np.all(arrays["dq_after_preprocess"] == 0)),
                    "addresses": [int(x) for x in arrays["addresses"]],
                    "addresses_meaning": "low/high32 scratch, low/high32 independent pre-snapshot; allzero CPU stand-in",
                }
                if label != "N":
                    first_stats = receipt["sequence"][0]["stats"] if receipt["sequence"] else row["stats"]
                    row["observed_numerical_findings"] = {
                        "nonfinite_gradients": any(
                            row["stats"][name]["nan"] or row["stats"][name]["inf"] for name in ("dq", "dk", "dv")
                        ),
                        "nonzero_scratch_after_preprocess": not row["whole_dq_scratch_zero_after_preprocess"],
                        "nonfinite_active_preprocess_outputs": any(
                            row[key]["active_query_rows"][kind]
                            for key in (
                                "dpsum_preprocess_regions",
                                "lse_preprocess_regions",
                            )
                            for kind in ("nan", "inf")
                        ),
                        "gradient_bits_differ_from_F1": any(
                            row["stats"][name]["sha256"] != first_stats[name]["sha256"] for name in ("dq", "dk", "dv")
                        ),
                    }
                    receipt["numerical_findings_present"] |= any(row["observed_numerical_findings"].values())
                else:
                    row["intentional_NaN_dO_excluded_from_numerical_findings"] = True
                receipt["sequence"].append(row)
                # Findings survive a later artifact, invocation or oracle failure.
                _write(target, receipt)
                np.savez(
                    path,
                    **{name: a.view(np.uint16) if str(a.dtype) == "bfloat16" else a for name, a in arrays.items()},
                )
                row.update(file_status="written", file_sha256=_sha(path))
                _write(target, receipt)
            # Keep the five-call history free of interleaved reference GPU work.
            _, pullback = jax.vjp(
                lambda q, k, v: reference_attention(
                    q,
                    k,
                    v,
                    AttentionMask.causal().with_fa4_bounds(values["bounds"], values["valid"]),
                    logits_dtype=jnp.float32,
                ),
                *finite[:3],
            )
            reference = tuple(np.asarray(v) for v in pullback(finite[4]))
            baseline = receipt["sequence"][0]["stats"]
            for row, output in zip(receipt["sequence"], retained, strict=True):
                if row["label"] == "N":
                    continue
                row["versus_reference"] = {
                    n: _compare_arrays(np.asarray(v), ref)
                    for n, v, ref in zip(("dq", "dk", "dv"), output[:3], reference, strict=True)
                }
                row["gradients_bitwise_equal_F1"] = all(
                    row["stats"][n]["sha256"] == baseline[n]["sha256"] for n in ("dq", "dk", "dv")
                )
                receipt["numerical_findings_present"] |= not (
                    row["whole_dq_scratch_zero_after_preprocess"]
                    and row["gradients_bitwise_equal_F1"]
                    and all(c["within_existing_tolerance"] for c in row["versus_reference"].values())
                )
            receipt["input_after"] = {name: _stats(a) for name, a in zip(_NAMES, finite, strict=True)}
            receipt["finite_inputs_pure"] = receipt["input_before"] == receipt["input_after"]
            receipt["all_outputs_retained_until_sequence_end"] = len(retained) == len(sequence)
            receipt["python_trace_count_after_sequence"] = len(traces)
            receipt["actual_GPU_stages_executed"] = mode == "gpu"
            receipt["existing_tolerances"] = {"atol": ATOL, "rtol": RTOL}
            if not receipt["finite_inputs_pure"] or len(traces) != 1:
                raise RuntimeError("Input purity or single-compiled-call contract failed")
            receipt["status"] = "completed"
    except Exception as error:
        receipt.update(
            status="error",
            execution_errors_present=True,
            error_type=type(error).__name__,
            error=str(error)[:4000],
        )
        _write(target, receipt)
        raise
    _write(target, receipt)
    total = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    if total >= _LIMIT:
        receipt.update(
            status="error",
            execution_errors_present=True,
            error_type="OutputBudgetExceeded",
            output_bytes=total,
        )
        _write(target, receipt)
        raise RuntimeError("Stage probe exceeded its bounded output budget")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("gpu", "cpu_reference"), default="gpu")
    parser.add_argument("--case", choices=("control", "stages", "prefill"), required=True)
    args = parser.parse_args()
    run_stage_probe(
        args.packet,
        args.metadata,
        args.output_dir,
        packet_sha256=args.packet_sha256,
        metadata_sha256=args.metadata_sha256,
        mode=args.mode,
        case=args.case,
    )


if __name__ == "__main__":
    main()
