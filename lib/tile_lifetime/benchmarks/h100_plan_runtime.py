# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile StableHLO and execute its selected dense RegionPlan on H100."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from types import ModuleType

import torch
from quack.operand_transform import transform_a_operand
from quack.rms_final_reduce import _rms_final_reduce_out

try:
    from backends.h100.cute_streaming_emitter import compile_h100_streaming_program
except ModuleNotFoundError:
    compile_h100_streaming_program = None

try:
    from flash_attn_interface import flash_attn_3_gpu
except ModuleNotFoundError:
    flash_attn_3_gpu = None

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
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND, compile_gemm_program
from tile_lifetime.quack_gemm_codegen import (
    GeneratedQuackGemm,
    QuackOperand,
    QuackOperandKind,
    generate_quack_gemm,
    safe_module_name,
)
from tile_lifetime.reference import DENSE_REGION_INPUT_NAMES, DenseDebugConfig, export_debug_dense_region
from tile_lifetime.streaming_attention import (
    StreamingTileSchedule,
    apply_causal_score_mask,
    build_attention_tensor_program,
    derive_streaming_attention,
    scaled_score_map,
)
from tile_lifetime.tile_program import TileProgramStage


class AttentionBackend(StrEnum):
    """Explicit physical attention choice for the dense replay."""

    GENERATED_SM90 = "generated_sm90"
    OFFICIAL_FA3_ORACLE = "official_fa3_oracle"


def _tensor(bindings: Mapping[str, TensorBinding], name: str) -> torch.Tensor:
    handle = bindings[name].handle
    assert isinstance(handle, torch.Tensor)
    return handle


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gemm_schedule(
    skeleton: GemmSkeleton,
) -> tuple[tuple[int, int, int], tuple[int, int, int], bool]:
    tile_shape = skeleton.physical_tile_shape
    cluster_shape = skeleton.cluster_shape
    pingpong = skeleton.pingpong
    if tile_shape is None or cluster_shape is None or pingpong is None:
        raise ValueError(f"generated GEMM {skeleton.name!r} is missing an explicit physical schedule")
    return tile_shape, cluster_shape, pingpong


class H100DenseBackend:
    """Generated QuACK/CuTe GEMMs plus the bounded attention implementation."""

    def __init__(
        self,
        config: DenseDebugConfig,
        *,
        attention_backend: AttentionBackend = AttentionBackend.GENERATED_SM90,
        generated_source_directory: Path | None = None,
    ) -> None:
        self.config = config
        self.attention_backend = attention_backend
        self.buffers: dict[str, torch.Tensor] = {}
        self.generated_source_directory = generated_source_directory or Path("/tmp/shuttle-quack-generated")
        self.generated_source_directory.mkdir(parents=True, exist_ok=True)
        self.generated_modules: dict[str, ModuleType] = {}
        self.generated_sources: dict[str, str] = {}
        self.gemm_launches: dict[GemmSkeleton, Callable[[], None]] = {}
        self.compiled_attention: dict[
            str,
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
        ] = {}

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
        if skeleton.backend != GENERIC_H100_GEMM_BACKEND:
            self._run_named_oracle(skeleton, bindings)
            return
        launch = self.gemm_launches.get(skeleton)
        if launch is None:
            launch = self._build_gemm_launch(skeleton, bindings)
            self.gemm_launches[skeleton] = launch
        launch()

    def _build_gemm_launch(
        self,
        skeleton: GemmSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> Callable[[], None]:
        tile_shape, cluster_shape, pingpong = _gemm_schedule(skeleton)
        program = compile_gemm_program(skeleton)
        generated = generate_quack_gemm(program)
        module = self._generated_module(generated)
        epilogue_arguments = self._epilogue_arguments(generated, skeleton, bindings)
        activation_tensor = _tensor(bindings, skeleton.input).view(skeleton.shape[0], skeleton.shape[2])
        activation: object = activation_tensor
        transform = None
        preparation_operands = tuple(
            operand for operand in generated.operands if operand.stage is TileProgramStage.PREPARATION
        )
        scale_copy: Callable[[], None] | None = None
        if generated.has_transform:
            if len(preparation_operands) != 1 or preparation_operands[0].kind is not QuackOperandKind.COLUMN:
                raise ValueError("measured H100 A-transform path requires exactly one FP32 row-scale operand")
            transform = module.generated_transform
            scale = _tensor(bindings, preparation_operands[0].sources[0])
            strip = self._scale_strip(generated.digest, skeleton.shape[2], skeleton.shape[0], tile_shape[2])

            def copy_scale() -> None:
                strip.copy_(scale[None, :])

            scale_copy = copy_scale
            scale_copy()
            # This constructs QuACK's (A, auxiliary-view) operand bundle. The
            # generated transform itself is applied exactly once in the GEMM
            # mainloop when ``transform_a`` is traced.
            activation = transform_a_operand(
                transform,
                activation_tensor,
                {preparation_operands[0].parameter: strip},
                tile_shape[0],
                tile_shape[2],
            )
        output = (
            _tensor(bindings, skeleton.output).view(skeleton.shape[0], skeleton.shape[1])
            if generated.writes_main_output
            else None
        )
        residual_c = (
            _tensor(bindings, generated.c_source).view(skeleton.shape[0], skeleton.shape[1])
            if generated.c_source is not None
            else None
        )
        weight = _tensor(bindings, skeleton.weight)

        def launch() -> None:
            if scale_copy is not None:
                scale_copy()
            module.generated_epilogue.gemm(
                activation,
                weight,
                output,
                residual_c,
                epi_args=epilogue_arguments,
                transform_a=transform,
                tile_M=tile_shape[0],
                tile_N=tile_shape[1],
                cluster_M=cluster_shape[0],
                cluster_N=cluster_shape[1],
                pingpong=pingpong,
            )

        return launch

    def _run_named_oracle(self, skeleton: GemmSkeleton, bindings: Mapping[str, TensorBinding]) -> None:
        """Keep historical named kernels available only as explicit comparison oracles."""
        raise ValueError(f"named GEMM oracle {skeleton.backend!r} is not enabled by the generated-plan runtime")

    def _scale_strip(self, name: str, reduction_width: int, rows: int, tile_k: int) -> torch.Tensor:
        key = f"{name}.fp32_scale_strip"
        strip = self.buffers.get(key)
        if strip is None:
            strip = torch.empty(reduction_width // tile_k, rows, dtype=torch.float32, device="cuda")
            self.buffers[key] = strip
        return strip

    def _generated_module(self, generated: GeneratedQuackGemm) -> ModuleType:
        cached = self.generated_modules.get(generated.digest)
        if cached is not None:
            return cached
        module_name = safe_module_name(generated.digest)
        source_path = self.generated_source_directory / f"{module_name}.py"
        source_path.write_text(generated.source)
        specification = importlib.util.spec_from_file_location(module_name, source_path)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"could not import generated QuACK module {source_path}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        self.generated_modules[generated.digest] = module
        self.generated_sources[generated.digest] = generated.source
        return module

    def _epilogue_arguments(
        self,
        generated: GeneratedQuackGemm,
        skeleton: GemmSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> dict[str, torch.Tensor]:
        arguments: dict[str, torch.Tensor] = {}
        for operand in generated.operands:
            if operand.stage is TileProgramStage.PREPARATION:
                continue
            arguments[operand.parameter] = self._physical_operand(operand, skeleton, bindings)
        for output in generated.outputs:
            tensor = _tensor(bindings, output.destination)
            arguments[output.parameter] = tensor.view(skeleton.shape[0], -1)
        return arguments

    def _physical_operand(
        self,
        operand: QuackOperand,
        skeleton: GemmSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> torch.Tensor:
        if operand.kind is not QuackOperandKind.PAIR_COEFFICIENT_TILE:
            if len(operand.sources) != 1:
                raise ValueError(f"operand {operand.parameter} requires one logical source")
            tensor = _tensor(bindings, operand.sources[0])
            if operand.kind is QuackOperandKind.TILE:
                return tensor.view(skeleton.shape[0], skeleton.shape[1])
            return tensor
        sine = _tensor(bindings, operand.sources[0])
        cosine = _tensor(bindings, operand.sources[1])
        return self._pair_coefficient_tile(sine, cosine, skeleton)

    def _pair_coefficient_tile(
        self,
        coefficient_1: torch.Tensor,
        coefficient_0: torch.Tensor,
        skeleton: GemmSkeleton,
    ) -> torch.Tensor:
        partition = next(attachment for attachment in skeleton.epilogue if attachment.operation == "partition")
        segment_extents = ast.literal_eval(dict(partition.attributes)["segment_extents"])
        if not isinstance(segment_extents, tuple) or not all(isinstance(extent, int) for extent in segment_extents):
            raise ValueError("partition segment extents must be a tuple of integers")
        mapped_values = {
            attachment.inputs[0] for attachment in skeleton.epilogue if attachment.operation == "pairwise_linear_map"
        }
        base = torch.stack((coefficient_0, coefficient_1), dim=-1).flatten(-2)
        segments: list[torch.Tensor] = []
        for name, extent in zip(partition.outputs, segment_extents, strict=True):
            if name in mapped_values:
                if extent % base.shape[-1] != 0:
                    raise ValueError("pairwise-map segment width is not divisible by its coefficient-tile width")
                segments.append(base.repeat(1, extent // base.shape[-1]))
                continue
            identity = torch.empty((base.shape[0], extent), dtype=base.dtype, device=base.device)
            identity[..., 0::2] = 1
            identity[..., 1::2] = 0
            segments.append(identity)
        one_batch = torch.cat(segments, dim=-1).contiguous()
        if skeleton.shape[0] % one_batch.shape[0] != 0:
            raise ValueError("coefficient tile rows do not divide the Contract output rows")
        key = f"pair_coefficients.{coefficient_0.data_ptr()}.{coefficient_1.data_ptr()}.{segment_extents}"
        table = self.buffers.get(key)
        if table is None:
            table = one_batch.repeat(skeleton.shape[0] // one_batch.shape[0], 1)
            self.buffers[key] = table
        return table

    def _rotary_table(self, sine: torch.Tensor, cosine: torch.Tensor) -> torch.Tensor:
        """Construct the legacy named-oracle table outside the generated path."""
        key = f"rotary_table.{sine.data_ptr()}.{cosine.data_ptr()}"
        table = self.buffers.get(key)
        if table is None:
            one_head = torch.stack((cosine, sine), dim=-1).flatten(-2)
            rotated_heads = self.config.query_heads + self.config.key_value_heads
            rotated = one_head.repeat(1, rotated_heads)
            value_width = self.config.key_value_heads * self.config.head_dimension
            identity = torch.empty((self.config.sequence, value_width), dtype=sine.dtype, device=sine.device)
            identity[..., 0::2] = 1
            identity[..., 1::2] = 0
            table = torch.cat((rotated, identity), dim=-1).contiguous()
            self.buffers[key] = table
        return table

    def write_generated_manifest(self, path: Path) -> None:
        """Persist exact generated sources used by this backend instance."""
        path.write_text(
            json.dumps(
                {
                    digest: {
                        "source_file": f"{safe_module_name(digest)}.py",
                        "source_sha256": digest,
                    }
                    for digest in sorted(self.generated_sources)
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def run_attention(
        self,
        skeleton: StreamingAttentionSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> None:
        if self.attention_backend is AttentionBackend.GENERATED_SM90:
            self._run_generated_attention(skeleton, bindings)
            return
        self._run_official_fa3_oracle(skeleton, bindings)

    def _run_generated_attention(
        self,
        skeleton: StreamingAttentionSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> None:
        query = _tensor(bindings, skeleton.query)
        key = _tensor(bindings, skeleton.key)
        value = _tensor(bindings, skeleton.value)
        output = _tensor(bindings, skeleton.output)
        lse_name = f"{skeleton.name}.generated_lse"
        log_sum_exp = self.buffers.get(lse_name)
        if log_sum_exp is None:
            log_sum_exp = torch.empty(
                query.shape[0], query.shape[2], query.shape[1], dtype=torch.float32, device=query.device
            )
            self.buffers[lse_name] = log_sum_exp
        compiled = self.compiled_attention.get(skeleton.name)
        if compiled is None:
            if compile_h100_streaming_program is None:
                raise RuntimeError("generated SM90 attention dependencies are unavailable")
            score_map = scaled_score_map(skeleton.scale)
            if skeleton.causal:
                score_map = apply_causal_score_mask(score_map)
            source = build_attention_tensor_program(
                batch_size=query.shape[0],
                query_length=query.shape[1],
                key_length=key.shape[1],
                query_heads=query.shape[2],
                key_value_heads=key.shape[2],
                key_dimension=query.shape[3],
                value_dimension=value.shape[3],
                score_map=score_map,
                input_dtype=DType.BF16,
            )
            program = derive_streaming_attention(
                source,
                schedule=StreamingTileSchedule(
                    skeleton.query_block_size,
                    skeleton.key_value_block_size,
                    skeleton.pipeline_stages,
                ),
            )
            compiled = compile_h100_streaming_program(
                program,
                query=query,
                key=key,
                value=value,
                output=output,
                log_sum_exp=log_sum_exp,
            )
            self.compiled_attention[skeleton.name] = compiled
        compiled(query, key, value, output, log_sum_exp)

    def _run_official_fa3_oracle(
        self,
        skeleton: StreamingAttentionSkeleton,
        bindings: Mapping[str, TensorBinding],
    ) -> None:
        if flash_attn_3_gpu is None:
            raise RuntimeError("the explicitly requested official FA3 oracle is unavailable")
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
) -> dict[str, dict[str, float | list[float]]]:
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
    return {
        name: {
            "samples_ms": values,
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
        }
        for name, values in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--shuttle-revision", required=False)
    parser.add_argument("--quack-revision", required=False)
    parser.add_argument(
        "--attention-backend",
        type=AttentionBackend,
        choices=tuple(AttentionBackend),
        default=AttentionBackend.GENERATED_SM90,
    )
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
    backends = {name: H100DenseBackend(config, attention_backend=args.attention_backend) for name in plans}
    outputs = {}
    deterministic_hashes: dict[str, dict[str, str]] = {}
    for name, plan in plans.items():
        result = execute_region_plan(plan, inputs, backends[name])
        outputs[name] = (
            _tensor(result.bindings, "x2").clone(),
            _tensor(result.bindings, "next_qkv").clone(),
        )
        first_hashes = {
            "x2": _tensor_sha256(_tensor(result.bindings, "x2")),
            "next_qkv": _tensor_sha256(_tensor(result.bindings, "next_qkv")),
        }
        repeated = execute_region_plan(plan, inputs, backends[name])
        second_hashes = {
            "x2": _tensor_sha256(_tensor(repeated.bindings, "x2")),
            "next_qkv": _tensor_sha256(_tensor(repeated.bindings, "next_qkv")),
        }
        if first_hashes != second_hashes:
            raise AssertionError(f"generated dense plan {name!r} is not bitwise deterministic")
        deterministic_hashes[name] = first_hashes
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
        median_ms = float(measurements[name]["median_ms"])
        minimum_ms = float(measurements[name]["minimum_ms"])
        print(f"  {name}: median_ms={median_ms:.4f} min_ms={minimum_ms:.4f}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        generated_root = args.json_output.parent / f"{args.json_output.stem}-generated-source"
        generated_root.mkdir(parents=True, exist_ok=True)
        generated_sources: dict[str, str] = {}
        for backend in backends.values():
            generated_sources.update(backend.generated_sources)
        for digest, source in generated_sources.items():
            (generated_root / f"{safe_module_name(digest)}.py").write_text(source)
        gemm_backends = {
            skeleton.backend
            for plan in plans.values()
            for skeleton in plan.skeletons
            if isinstance(skeleton, GemmSkeleton)
        }
        generated_only = gemm_backends == {GENERIC_H100_GEMM_BACKEND}
        attention_manifests = {
            placement: {
                name: {
                    "program_sha256": hashlib.sha256(repr(getattr(compiled, "program", None)).encode()).hexdigest(),
                    "schedule": repr(getattr(compiled, "schedule", None)),
                }
                for name, compiled in backend.compiled_attention.items()
            }
            for placement, backend in backends.items()
        }
        manifest = {
            "generated_gemm_source_sha256": sorted(generated_sources),
            "gemm_backends": sorted(str(backend) for backend in gemm_backends),
            "named_gemm_callback_selected": not generated_only,
            "attention_backend": args.attention_backend.value,
            "official_fa3_oracle_selected": args.attention_backend is AttentionBackend.OFFICIAL_FA3_ORACLE,
            "attention_programs": attention_manifests,
        }
        (generated_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        telemetry = subprocess.check_output(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.current.memory",
                "--format=csv,noheader,nounits",
                "--id=0",
            ),
            text=True,
        ).strip()
        historical_oracle_ms = {2048: 1.4561, 4096: 3.0080}.get(args.sequence)
        payload = {
            "schema_version": 1,
            "workload": {
                "sequence": args.sequence,
                "hidden": config.hidden,
                "intermediate": config.intermediate,
                "query_heads": config.query_heads,
                "key_value_heads": config.key_value_heads,
                "head_dimension": config.head_dimension,
                "dtype": "bfloat16",
                "accumulation_dtype": "float32",
            },
            "benchmark": {
                "attention_backend": args.attention_backend.value,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "iterations_per_sample": args.iterations,
                "interleaved_order": True,
                "measurements": measurements,
            },
            "correctness": {
                "prologue_vs_delayed_x2_max_abs": x2_difference.max().item(),
                "prologue_vs_delayed_x2_mean_abs": x2_difference.mean().item(),
                "prologue_vs_delayed_next_qkv_max_abs": qkv_difference.max().item(),
                "prologue_vs_delayed_next_qkv_mean_abs": qkv_difference.mean().item(),
                "bitwise_deterministic_output_sha256": deterministic_hashes,
            },
            "comparison": {
                "historical_manual_oracle_ms": historical_oracle_ms,
                "generated_delayed_ratio": (
                    float(measurements[RMSScalePlacement.CONSUMER_EPILOGUE.value]["median_ms"]) / historical_oracle_ms
                    if historical_oracle_ms is not None
                    else None
                ),
            },
            "sources": {
                "shuttle_revision": args.shuttle_revision,
                "quack_revision": args.quack_revision,
                "generated_source_sha256": sorted(generated_sources),
                "stablehlo_sha256": hashlib.sha256(artifact).hexdigest(),
                "quack_patch_sha256": "40318b9b390e111c38f4838a50cf8913695c9f94142122b374bf09c220cfd9a1",
                "implementation_file_sha256": {
                    str(path.relative_to(Path(__file__).parents[1])): _file_sha256(path)
                    for path in (
                        Path(__file__),
                        Path(__file__).parents[1] / "src/tile_lifetime/quack_gemm_codegen.py",
                        Path(__file__).parents[1] / "src/tile_lifetime/h100_streaming_lowering.py",
                        Path(__file__).parents[1] / "backends/h100/cute_streaming_emitter.py",
                        Path(__file__).parents[1] / "backends/h100/cute_streaming_sm90.py",
                    )
                },
            },
            "environment": {
                "gpu_telemetry": telemetry,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
                "flash_attn_4": importlib.metadata.version("flash-attn-4"),
            },
            "generated_path_manifest": manifest,
        }
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
