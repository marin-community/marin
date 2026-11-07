"""Quantize received expert shards with vLLM's online FP8 layer processor."""

from collections.abc import Iterator
from copy import copy
from enum import StrEnum

import torch
import torch.distributed as dist
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    LAYERWISE_INFO,
    _copy_and_restore_kernel_tensors,
    _place_kernel_tensors,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.model_executor.model_loader.reload.meta import materialize_layer

from skyrl_train.weight_sync.expert_block.schedule import BF16, DenseBroadcast, ExpertBroadcast, Schedule
from skyrl_train.weight_sync.expert_block.source_views import ROUTED_EXPERTS, dtype_name, is_widened_router
from skyrl_train.weight_sync.expert_block.stream import InstallReport, Landing, Stream
from skyrl_train.weight_sync.expert_block.verify_weights import (
    REPLAY_COMPARE_CHUNK_BYTES,
    ReplayReport,
    mismatched_bytes,
)


class UpdateState(StrEnum):
    READY = "ready"
    INSTALL = "install"
    VERIFY = "verify"
    FAILED = "failed"


def online_fp8_modules(model) -> Iterator[tuple[str, torch.nn.Module]]:
    """Yield online tensorwise FP8 layers whose checkpoint tensors can be staged."""
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, (Fp8PerTensorOnlineLinearMethod, Fp8PerTensorOnlineMoEMethod)):
            continue
        if module not in LAYERWISE_INFO or not LAYERWISE_INFO[module].restore_metadata[0]:
            raise ValueError(f"Online FP8 layer {name} has no checkpoint metadata for reloading")
        yield name, module


def checkpoint_parameters(model) -> dict[str, torch.Tensor]:
    """Report BF16 checkpoint shapes, excluding derived FP8 scales and transposes."""
    parameters = dict(model.named_parameters())
    for name, module in online_fp8_modules(model):
        for parameter_name, _ in module.named_parameters(recurse=False):
            del parameters[f"{name}.{parameter_name}"]
        parameters.update(
            (f"{name}.{parameter_name}", parameter)
            for parameter_name, parameter in LAYERWISE_INFO[module].restore_metadata[0].items()
        )
    return parameters


class OnlineFp8Stream(Stream):
    """Keep expert-block routing while staging and quantizing one leaf layer at a time.

    Only quantized layers enter vLLM's reload lifecycle. Unquantized weights keep
    the direct install path, including the FP32 router. Replay runs the same
    quantizer into temporary tensors and compares weights AND scales, leaving
    serving buffers and quant-method bindings untouched.
    """

    def __init__(
        self,
        participant: int,
        schedule: Schedule,
        groups: dict[str, dist.ProcessGroup],
        *,
        parameters: dict[str, torch.Tensor],
        expert_maps: dict[str, tuple[int, ...]],
        device: torch.device,
        model: torch.nn.Module,
        vllm_config: VllmConfig,
    ):
        super().__init__(participant, schedule, groups, parameters=parameters, expert_maps=expert_maps, device=device)
        self.vllm_config = vllm_config
        self.modules = dict(online_fp8_modules(model))
        self.state = UpdateState.READY
        self.started: set[str] = set()
        self.completed: set[str] = set()
        self.loaded: dict[str, int] = {}
        self.kernel_bytes = 0
        self.mismatches = torch.zeros((), dtype=torch.int64, device=device)
        self.checkpoint = dict(self.parameters)
        self.expected = {
            name: sum(parameter.numel() for parameter in LAYERWISE_INFO[module].restore_metadata[0].values())
            for name, module in self.modules.items()
        }
        scheduled = dict.fromkeys(self.modules, 0)
        for item, landing in self.transfers():
            name = self.module_name(item)
            if landing is not None and name in scheduled:
                scheduled[name] += landing.installed.numel()
        if scheduled != self.expected:
            raise ValueError(
                f"Expert-block schedule does not cover FP8 checkpoint layers: {scheduled} != {self.expected}"
            )

    @staticmethod
    def module_name(item: ExpertBroadcast | DenseBroadcast) -> str:
        if isinstance(item, ExpertBroadcast):
            return f"model.layers.{item.entry.layer}.{ROUTED_EXPERTS}"
        return item.source.hf_name.rpartition(".")[0]

    def _prepare_layer(self, name: str) -> None:
        if name not in self.modules or name in self.started:
            return
        module = self.modules[name]
        initialize_layerwise_reload(module)
        info = LAYERWISE_INFO[module]
        materialize_layer(module, info)
        for parameter_name in info.restore_metadata[0]:
            self.parameters[f"{name}.{parameter_name}"] = getattr(module, parameter_name)
        self.started.add(name)

    def prepare(self, item: ExpertBroadcast | DenseBroadcast, landing: Landing) -> Landing:
        name = self.module_name(item)
        if name not in self.modules:
            return landing
        self._prepare_layer(name)
        return self.expert_landing(item) if isinstance(item, ExpertBroadcast) else self.dense_landing(item)

    def wire_tensor(self, landing: Landing) -> torch.Tensor:
        if self.state == UpdateState.VERIFY:
            return self.scratch.narrow(0, 0, landing.nbytes).view(landing.wire_dtype).view(landing.installed.shape)
        return super().wire_tensor(landing)

    def install(self, item, landing: Landing, tensor: torch.Tensor) -> None:
        name = self.module_name(item)
        if self.state == UpdateState.VERIFY and name not in self.modules:
            self.mismatches.add_(
                mismatched_bytes(tensor, landing.installed.to(landing.wire_dtype), REPLAY_COMPARE_CHUNK_BYTES)
            )
            return
        if self.state == UpdateState.VERIFY:
            landing.installed.copy_(tensor)
        else:
            super().install(item, landing, tensor)
        if name not in self.modules:
            return
        self.loaded[name] = self.loaded.get(name, 0) + landing.installed.numel()
        if self.loaded[name] > self.expected[name]:
            raise RuntimeError(f"Expert-block FP8 layer {name} received duplicate checkpoint elements")
        if self.loaded[name] == self.expected[name]:
            self.finish_layer(name)

    def finish_layer(self, name: str) -> None:
        module = self.modules[name]
        info = LAYERWISE_INFO[module]
        original_method = module.quant_method
        # Post-load processing rebuilds MoE kernels and can retag linear kernels.
        # A verification must not replace the serving method's cached bindings.
        if self.state == UpdateState.VERIFY:
            module.quant_method = copy(original_method)
            if isinstance(original_method, Fp8PerTensorOnlineLinearMethod):
                module.quant_method.fp8_linear = copy(original_method.fp8_linear)
        try:
            module.__dict__.pop("_already_called_process_weights_after_loading", None)
            module.quant_method.process_weights_after_loading(module)
            if self.state == UpdateState.VERIFY:
                parameters, buffers = info.kernel_tensors
                for parameter_name, actual in parameters.items():
                    expected = getattr(module, parameter_name)
                    self.mismatches.add_(mismatched_bytes(actual, expected, REPLAY_COMPARE_CHUNK_BYTES))
                    self.kernel_bytes += actual.numel() * actual.element_size()
                for buffer_name, actual in buffers.items():
                    if buffer_name in info.kernel_non_persistent_buffers:
                        continue
                    expected = getattr(module, buffer_name)
                    self.mismatches.add_(mismatched_bytes(actual, expected, REPLAY_COMPARE_CHUNK_BYTES))
                    self.kernel_bytes += actual.numel() * actual.element_size()
            else:
                _copy_and_restore_kernel_tensors(module, info)
        finally:
            if info.kernel_tensors is not None:
                _place_kernel_tensors(module, info)
            module.quant_method = original_method
            info.reset()
            finalize_layerwise_reload(module, self.vllm_config.model_config)
            for parameter_name in info.restore_metadata[0]:
                key = f"{name}.{parameter_name}"
                self.parameters[key] = self.checkpoint[key]
        self.completed.add(name)

    def _update(self, version: int, state: UpdateState) -> InstallReport:
        if self.state != UpdateState.READY:
            raise RuntimeError(f"Cannot start expert-block FP8 update while receiver is {self.state}")
        self.state = state
        self.started.clear()
        self.completed.clear()
        self.loaded.clear()
        self.kernel_bytes = 0
        self.mismatches.zero_()
        try:
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                report = super().run(version)
            if self.completed != self.modules.keys():
                raise RuntimeError("Expert-block FP8 update ended before all quantized layers were installed")
        except BaseException:
            self.state = UpdateState.FAILED
            raise
        self.state = UpdateState.READY
        return report

    def run(self, version: int) -> InstallReport:
        return self._update(version, UpdateState.INSTALL)

    def verify(self, version: int) -> ReplayReport:
        scratch = self.scratch
        self.scratch = torch.empty(
            max(landing.nbytes for _, landing in self.transfers()), dtype=torch.uint8, device=self.device
        )
        try:
            report = self._update(version, UpdateState.VERIFY)
        finally:
            self.scratch = scratch
        parameter_bytes = sum(
            value.numel() * (2 if is_widened_router(name, BF16, dtype_name(value.dtype)) else value.element_size())
            for name, value in self.checkpoint.items()
        )
        return ReplayReport(
            self.participant,
            version,
            report.wire_bytes,
            parameter_bytes,
            int(self.mismatches.item()),
            self.kernel_bytes,
        )
