# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl diagnostic plugin: the dtypes and memory an FSDP2 run actually uses (pod-side).

Logs, from every rank's first step: the FSDP2 mixed-precision policy (param, reduce,
output dtypes) of the root and of one wrapped block; the dtypes of the first MoE experts
module's parameters and inputs at call time; the bytes saved for backward by dtype during
the first forward; and per-GPU memory after wrap, after the first forward, after the first
backward and after the first optimizer step. Printed with a ``[precision-probe]`` prefix.
"""

import collections

import torch
from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

PREFIX = "[precision-probe]"


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _say(message: str) -> None:
    print(f"{PREFIX} rank={_rank()} {message}", flush=True)


def _memory(phase: str) -> None:
    _say(
        f"memory {phase}: allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
        f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
    )


def _mp_policy(module) -> str:
    state = getattr(module, "_get_fsdp_state", None)
    if state is None:
        return "not an FSDP2 module"
    policy = state()._mp_policy
    return f"param={policy.param_dtype} reduce={policy.reduce_dtype} output={policy.output_dtype}"


class _Probe(TrainerCallback):
    def __init__(self):
        self._saved: collections.Counter = collections.Counter()
        self._hooks = None
        self._done = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        _memory("after wrap")
        _say(f"root mixed precision: {_mp_policy(model)}")
        blocks = [module for module in model.modules() if type(module).__name__.endswith("DecoderLayer")]
        if blocks:
            _say(f"block mixed precision: {_mp_policy(blocks[0])}")
        experts = next((m for m in model.modules() if type(m).__name__.endswith("Experts")), None)
        if experts is not None:
            experts.register_forward_pre_hook(self._log_experts, with_kwargs=True)
        model.register_forward_pre_hook(self._start_forward)
        model.register_forward_hook(self._end_forward)

    def _log_experts(self, module, args, kwargs):
        if self._done:
            return
        inputs = {type(a).__name__ if not isinstance(a, torch.Tensor) else str(a.dtype) for a in args}
        params = {name: str(p.dtype) for name, p in module.named_parameters(recurse=False)}
        _say(
            f"experts {type(module).__name__}: params {params}, inputs {inputs}, autocast {torch.is_autocast_enabled()}"
        )

    def _pack(self, tensor):
        self._saved[str(tensor.dtype)] += tensor.numel() * tensor.element_size()
        return tensor

    def _start_forward(self, module, args):
        if self._done or self._hooks is not None:
            return
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda tensor: tensor)
        self._hooks.__enter__()

    def _end_forward(self, module, args, output):
        if self._done or self._hooks is None:
            return
        self._hooks.__exit__(None, None, None)
        self._hooks = None
        _say(f"saved for backward by dtype (GiB): { {k: round(v / 2**30, 2) for k, v in self._saved.items()} }")
        _memory("after first forward")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self._done:
            _memory("after first backward")

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self._done:
            _memory("after first optimizer step")
            self._done = True


class PrecisionProbePlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        return [_Probe()]
