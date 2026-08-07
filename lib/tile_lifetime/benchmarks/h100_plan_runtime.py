# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile StableHLO and execute its selected dense RegionPlan on H100."""

import argparse
import math
import statistics
from collections.abc import Mapping

import torch
from flash_attn_interface import flash_attn_3_gpu
from quack.epilogue.library import rms_partial_epi, rstd_swiglu_epi, swiglu_mod
from quack.epilogue.rotary import make_interleaved_inv_freq, rope_posfreq_epi, rstd_rope_posfreq_epi
from quack.operand_transform import a_transform, transform_a_operand
from quack.rms_final_reduce import _rms_final_reduce_out

from tile_lifetime import (
    DType,
    GemmSkeleton,
    NumericalPolicy,
    ReductionSkeleton,
    RMSScalePlacement,
    RuntimeBufferSpec,
    StreamingAttentionSkeleton,
    TensorBinding,
    compile_stablehlo_dense_transformer_region,
    execute_region_plan,
    required_input_specs,
)
from tile_lifetime.reference import DENSE_REGION_INPUT_NAMES, DenseDebugConfig, export_debug_dense_region

TILE_M = 128
TILE_N = 256
TILE_K = 64


@a_transform(vec_size=8, args={"inverse_rms": "colvec_ktile_fp32"})
def _scale_a_by_fp32_inverse_rms(activation, inverse_rms):
    return activation * inverse_rms


def _tensor(bindings: Mapping[str, TensorBinding], name: str) -> torch.Tensor:
    handle = bindings[name].handle
    assert isinstance(handle, torch.Tensor)
    return handle


class H100DenseBackend:
    """QuACK/CODA plus official-FA3 implementation of the bounded plan contract."""

    def __init__(self, config: DenseDebugConfig) -> None:
        self.config = config
        self.buffers: dict[str, torch.Tensor] = {}
        self.rope_validated = False
        positions = torch.arange(config.sequence, dtype=torch.float32, device="cuda").repeat(config.batch)
        inverse_frequency = 10_000.0 ** (
            -torch.arange(config.head_dimension // 2, dtype=torch.float64, device="cuda") / (config.head_dimension // 2)
        )
        query_key_width = (config.query_heads + config.key_value_heads) * config.head_dimension
        key_value_width = config.key_value_heads * config.head_dimension
        self.rope_args = {
            "pos": positions,
            "freq": make_interleaved_inv_freq(inverse_frequency, query_key_width, key_value_width),
        }
        table_frequency = 10_000.0 ** (
            -torch.arange(config.head_dimension // 2, dtype=torch.float32, device="cuda") / (config.head_dimension // 2)
        )
        table_angle = torch.arange(config.sequence, dtype=torch.float32, device="cuda")[:, None] * table_frequency
        self.canonical_rope_sine = torch.sin(table_angle).bfloat16()
        self.canonical_rope_cosine = torch.cos(table_angle).bfloat16()

    def allocate(self, spec: RuntimeBufferSpec) -> TensorBinding:
        tensor = self.buffers.get(spec.name)
        if tensor is None:
            dtype = torch.bfloat16 if spec.dtype is DType.BF16 else torch.float32
            tensor = torch.empty(spec.shape, dtype=dtype, device="cuda")
            self.buffers[spec.name] = tensor
        return TensorBinding(handle=tensor, shape=spec.shape, dtype=spec.dtype)

    def alias(self, spec: RuntimeBufferSpec, source: TensorBinding) -> TensorBinding:
        source_tensor = source.handle
        assert isinstance(source_tensor, torch.Tensor)
        if source_tensor.numel() == math.prod(spec.shape):
            view = source_tensor.view(spec.shape)
        else:
            view = self._packed_qkv_view(spec, source_tensor)
        return TensorBinding(handle=view, shape=spec.shape, dtype=spec.dtype)

    def _packed_qkv_view(self, spec: RuntimeBufferSpec, packed: torch.Tensor) -> torch.Tensor:
        width = math.prod(spec.shape[-2:])
        query_width = self.config.query_heads * self.config.head_dimension
        key_value_width = self.config.key_value_heads * self.config.head_dimension
        if "query" in spec.name:
            offset = 0
        elif "key" in spec.name:
            offset = query_width
        else:
            offset = query_width + key_value_width
        return packed[..., offset : offset + width].view(spec.shape)

    def run_gemm(self, skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        if skeleton.backend == "quack_sm90_rope_posfreq":
            self._run_qkv(skeleton, bindings)
            return
        if skeleton.backend == "coda_cute_h100":
            self._run_residual_rms(skeleton, bindings)
            return
        if "swiglu" in (skeleton.backend or ""):
            self._run_gate_up(skeleton, bindings)
            return
        if "rope_posfreq" in (skeleton.backend or ""):
            self._run_qkv(skeleton, bindings)
            return
        raise ValueError(f"unsupported GEMM backend {skeleton.backend!r}")

    def _run_residual_rms(self, skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        residual_attachment, gamma_attachment, partial_attachment = skeleton.epilogue
        rms_partial_epi.gemm(
            _tensor(bindings, skeleton.input).view(skeleton.shape[0], skeleton.shape[2]),
            _tensor(bindings, skeleton.weight),
            _tensor(bindings, skeleton.output),
            _tensor(bindings, residual_attachment.inputs[1]),
            epi_args={
                "weight": _tensor(bindings, gamma_attachment.inputs[1]),
                "resid_out": _tensor(bindings, residual_attachment.outputs[0]),
                "sqsum": _tensor(bindings, partial_attachment.outputs[0]),
            },
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=skeleton.cluster_shape[0],
            cluster_N=skeleton.cluster_shape[1],
            pingpong=skeleton.pingpong,
        )

    def _run_gate_up(self, skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        activated = _tensor(bindings, skeleton.output)
        if skeleton.backend == "quack_sm90_fp32_a_transform_swiglu_dead_preact":
            inverse_rms = _tensor(bindings, skeleton.prologue[0].inputs[1])
            strip = self._scale_strip(skeleton.name, skeleton.shape[2], skeleton.shape[0])
            strip.copy_(inverse_rms[None, :])
            bundle = transform_a_operand(
                _scale_a_by_fp32_inverse_rms,
                _tensor(bindings, skeleton.input),
                {"inverse_rms": strip},
                TILE_M,
                TILE_K,
            )
            swiglu_mod.gemm(
                bundle,
                _tensor(bindings, skeleton.weight),
                None,
                epi_args={"postact": activated},
                transform_a=_scale_a_by_fp32_inverse_rms,
                tile_M=TILE_M,
                tile_N=TILE_N,
                cluster_M=skeleton.cluster_shape[0],
                cluster_N=skeleton.cluster_shape[1],
                pingpong=skeleton.pingpong,
            )
            return
        rstd_swiglu_epi.gemm(
            _tensor(bindings, skeleton.input),
            _tensor(bindings, skeleton.weight),
            None,
            epi_args={"rstd": _tensor(bindings, skeleton.epilogue[0].inputs[1]), "postact": activated},
            tile_M=TILE_M,
            tile_N=TILE_N,
            cluster_M=skeleton.cluster_shape[0],
            cluster_N=skeleton.cluster_shape[1],
            pingpong=skeleton.pingpong,
        )

    def _run_qkv(self, skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        self._validate_canonical_rope(skeleton, bindings)
        output = _tensor(bindings, skeleton.output).view(skeleton.shape[0], skeleton.shape[1])
        common = {
            "tile_M": TILE_M,
            "tile_N": TILE_N,
            "cluster_M": skeleton.cluster_shape[0],
            "cluster_N": skeleton.cluster_shape[1],
            "pingpong": skeleton.pingpong,
        }
        if skeleton.backend == "quack_sm90_fp32_a_transform_rope_posfreq":
            inverse_rms = _tensor(bindings, skeleton.prologue[0].inputs[1])
            strip = self._scale_strip(skeleton.name, skeleton.shape[2], skeleton.shape[0])
            strip.copy_(inverse_rms[None, :])
            bundle = transform_a_operand(
                _scale_a_by_fp32_inverse_rms,
                _tensor(bindings, skeleton.input),
                {"inverse_rms": strip},
                TILE_M,
                TILE_K,
            )
            rope_posfreq_epi.gemm(
                bundle,
                _tensor(bindings, skeleton.weight),
                output,
                epi_args=self.rope_args,
                transform_a=_scale_a_by_fp32_inverse_rms,
                **common,
            )
            return
        if skeleton.backend == "quack_sm90_rstd_rope_posfreq":
            rstd_rope_posfreq_epi.gemm(
                _tensor(bindings, skeleton.input),
                _tensor(bindings, skeleton.weight),
                output,
                epi_args={**self.rope_args, "rstd": _tensor(bindings, skeleton.epilogue[0].inputs[1])},
                **common,
            )
            return
        rope_posfreq_epi.gemm(
            _tensor(bindings, skeleton.input).view(skeleton.shape[0], skeleton.shape[2]),
            _tensor(bindings, skeleton.weight),
            output,
            epi_args=self.rope_args,
            **common,
        )

    def _validate_canonical_rope(
        self,
        skeleton: GemmSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> None:
        if self.rope_validated:
            return
        rope_attachment = next(
            attachment for attachment in skeleton.epilogue if attachment.operation == "pairwise_rope_q"
        )
        sine = _tensor(bindings, rope_attachment.inputs[1])
        cosine = _tensor(bindings, rope_attachment.inputs[2])
        if not torch.equal(sine, self.canonical_rope_sine) or not torch.equal(cosine, self.canonical_rope_cosine):
            raise ValueError("the measured pos/frequency backend requires canonical Llama base-10000 RoPE tables")
        self.rope_validated = True

    def _scale_strip(self, name: str, reduction_width: int, rows: int) -> torch.Tensor:
        key = f"{name}.fp32_scale_strip"
        strip = self.buffers.get(key)
        if strip is None:
            strip = torch.empty(reduction_width // TILE_K, rows, dtype=torch.float32, device="cuda")
            self.buffers[key] = strip
        return strip

    def run_attention(
        self,
        skeleton: StreamingAttentionSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> None:
        output = _tensor(bindings, skeleton.output)
        returned, *_ = flash_attn_3_gpu.fwd(
            _tensor(bindings, skeleton.query),
            _tensor(bindings, skeleton.key),
            _tensor(bindings, skeleton.value),
            out=output,
            softmax_scale=skeleton.scale,
            is_causal=skeleton.causal,
            is_rotary_interleaved=True,
            num_splits=1,
            pack_gqa=skeleton.pack_gqa,
        )
        if returned.data_ptr() != output.data_ptr():
            raise RuntimeError("official FA3 did not honor the plan's preallocated output")

    def run_reduction(self, skeleton: ReductionSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        body = skeleton.operator.removeprefix("rsqrt(sum / ").removesuffix(")")
        hidden_text, epsilon_text = body.split(" + ")
        _rms_final_reduce_out(
            _tensor(bindings, skeleton.input),
            _tensor(bindings, skeleton.output),
            1.0 / int(hidden_text),
            float(epsilon_text),
        )


def _input_bindings(plan, config: DenseDebugConfig) -> dict[str, TensorBinding]:
    specifications = required_input_specs(plan)
    torch.manual_seed(0)
    query_width = config.query_heads * config.head_dimension
    key_value_width = config.key_value_heads * config.head_dimension
    qkv_width = query_width + 2 * key_value_width
    tensors = {
        "x": torch.randn(config.tokens, config.hidden, dtype=torch.bfloat16, device="cuda"),
        "qkv_weight": (
            torch.randn(qkv_width, config.hidden, dtype=torch.bfloat16, device="cuda") / math.sqrt(config.hidden)
        ),
        "output_weight": (
            torch.randn(config.hidden, config.hidden, dtype=torch.bfloat16, device="cuda") / math.sqrt(config.hidden)
        ),
        "mlp_gamma": torch.randn(config.hidden, dtype=torch.bfloat16, device="cuda"),
        "gate_up_weight": (
            torch.randn(2 * config.intermediate, config.hidden, dtype=torch.bfloat16, device="cuda")
            / math.sqrt(config.hidden)
        ),
        "down_weight": (
            torch.randn(config.hidden, config.intermediate, dtype=torch.bfloat16, device="cuda")
            / math.sqrt(config.intermediate)
        ),
        "next_gamma": torch.randn(config.hidden, dtype=torch.bfloat16, device="cuda"),
        "next_qkv_weight": (
            torch.randn(qkv_width, config.hidden, dtype=torch.bfloat16, device="cuda") / math.sqrt(config.hidden)
        ),
    }
    positions = torch.arange(config.sequence, dtype=torch.float32, device="cuda")[:, None]
    inverse_frequency = 10_000.0 ** (
        -torch.arange(config.head_dimension // 2, dtype=torch.float32, device="cuda") / (config.head_dimension // 2)
    )
    angle = positions * inverse_frequency[None, :]
    tensors["rope_sine"] = torch.sin(angle).bfloat16()
    tensors["rope_cosine"] = torch.cos(angle).bfloat16()
    return {
        name: TensorBinding(handle=tensors[name], shape=spec.shape, dtype=spec.dtype)
        for name, spec in specifications.items()
    }


def _benchmark_variants(
    plans,
    inputs,
    backends,
    *,
    warmups: int,
    repeats: int,
    iterations: int,
) -> dict[str, tuple[float, float]]:
    variants = tuple(plans.items())
    for _ in range(warmups):
        for name, plan in variants:
            execute_region_plan(plan, inputs, backends[name])
    torch.cuda.synchronize()
    samples = {name: [] for name, _ in variants}
    for repeat in range(repeats):
        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
        for name, plan in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                execute_region_plan(plan, inputs, backends[name])
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)
    return {name: (statistics.median(values), min(values)) for name, values in samples.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    config = DenseDebugConfig(
        sequence=args.sequence,
        hidden=4096,
        intermediate=14336,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
    )
    artifact = export_debug_dense_region(config)
    plans = {
        placement.value: compile_stablehlo_dense_transformer_region(
            artifact,
            input_names=DENSE_REGION_INPUT_NAMES,
            gemm_accumulation_dtype=DType.FP32,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
            rms_scale_placement=placement,
        )
        for placement in (RMSScalePlacement.CONSUMER_PROLOGUE, RMSScalePlacement.CONSUMER_EPILOGUE)
    }
    inputs = _input_bindings(plans[RMSScalePlacement.CONSUMER_PROLOGUE.value], config)
    backends = {name: H100DenseBackend(config) for name in plans}
    outputs = {}
    for name, plan in plans.items():
        result = execute_region_plan(plan, inputs, backends[name])
        outputs[name] = (
            _tensor(result.bindings, "x2").clone(),
            _tensor(result.bindings, "next_qkv").clone(),
        )
    prologue_x2, prologue_qkv = outputs[RMSScalePlacement.CONSUMER_PROLOGUE.value]
    delayed_x2, delayed_qkv = outputs[RMSScalePlacement.CONSUMER_EPILOGUE.value]
    x2_difference = (prologue_x2.float() - delayed_x2.float()).abs()
    qkv_difference = (prologue_qkv.float() - delayed_qkv.float()).abs()
    print(f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    print(f"stablehlo_bytes={len(artifact)} skeletons={len(plans[RMSScalePlacement.CONSUMER_PROLOGUE.value].skeletons)}")
    print(
        f"prologue_vs_delayed x2_max_abs={x2_difference.max().item():.6f} "
        f"x2_mean_abs={x2_difference.mean().item():.6f} "
        f"next_qkv_max_abs={qkv_difference.max().item():.6f} "
        f"next_qkv_mean_abs={qkv_difference.mean().item():.6f}"
    )
    measurements = _benchmark_variants(
        plans,
        inputs,
        backends,
        warmups=args.warmups,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    for name in plans:
        median_ms, minimum_ms = measurements[name]
        print(f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f}")


if __name__ == "__main__":
    main()
