# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Axolotl plugin: NemotronH load-time checks and Mamba path (pod-side, image environment).

The hub ``mamba-ssm`` kernel's fused training path (``mamba_split_conv1d_scan_combined``)
calls the ``causal_conv1d_cuda`` extension of the pip ``causal-conv1d`` package, which the
task image lacks and which has no prebuilt wheel for its torch. The plugin switches every
Mamba-2 mixer to the unfused training path: the hub ``causal_conv1d_fn`` kernel, then the
Triton ``mamba_chunk_scan_combined`` scan. Both compute the same function; the unfused path
keeps the conv output for backward, which FSDP's per-block activation checkpointing frees.

Three NemotronH failure modes are silent in transformers and would only show up as a slow
or wrong run:

- The hub repo ships its own remote modeling code; Stage 0 must run transformers' class, and
  the export drops the base config's ``auto_map`` so that serving loads it too.
- Without the hub ``mamba-ssm`` and ``causal-conv1d`` kernels, the Mamba-2 mixers fall back
  to a pure-torch chunked scan, which at 32k tokens per sequence is far slower and holds
  far more activation memory.
- The router's ``e_score_correction_bias`` is a buffer, not a parameter. Stage 0 freezes it
  at its pretrained value (the recorded deviation), so after FSDP's RAM-efficient load
  (rank 0 reads the checkpoint and broadcasts it) every rank must hold the same nonzero
  biases. Before the broadcast only rank 0 holds real values.
"""

import logging

import torch
import torch.distributed as dist
from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback
from transformers.models.nemotron_h import modeling_nemotron_h

# Under the "axolotl" logger, which Axolotl configures at INFO; a bare module logger is dropped.
LOG = logging.getLogger(f"axolotl.{__name__}")


def _correction_biases(model) -> list[torch.Tensor]:
    return [module.e_score_correction_bias for module in model.modules() if hasattr(module, "e_score_correction_bias")]


def _zeroed(biases: list[torch.Tensor]) -> int:
    return sum(int(not bias.any()) for bias in biases)


class _RouterBiasCheck(TrainerCallback):
    """At the first step, every rank's router biases are nonzero and identical."""

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step != 0:
            return
        biases = _correction_biases(model)
        zeroed = _zeroed(biases)
        checksum = torch.stack([bias.double().sum() for bias in biases]).sum().reshape(1)
        if dist.is_initialized():
            gathered = [torch.zeros_like(checksum) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, checksum)
            checksums = {value.item() for value in gathered}
        else:
            checksums = {checksum.item()}
        LOG.info(
            "NemotronH router biases at step 0: %d buffers, %d all-zero, checksums %s", len(biases), zeroed, checksums
        )
        if zeroed or len(checksums) != 1:
            raise RuntimeError(f"router biases differ across ranks or are zeroed: {zeroed} zero, checksums {checksums}")


class NemotronHChecksPlugin(BasePlugin):
    """Fail the run on remote NemotronH code, missing Mamba kernels, or lost router biases."""

    def post_model_load(self, cfg, model):
        if type(model).__module__ != modeling_nemotron_h.__name__:
            raise RuntimeError(f"expected transformers' NemotronH, got {type(model).__module__}")
        # The base config's auto_map names the hub repo's remote code, which the export does
        # not ship; vLLM serves with trust_remote_code and would look for it.
        if getattr(model.config, "auto_map", None):
            del model.config.auto_map
        if not modeling_nemotron_h.is_fast_path_available:
            raise RuntimeError(
                "NemotronH Mamba kernels (hub mamba-ssm / causal-conv1d) did not load; "
                "the naive scan is not usable at Stage 0 sequence lengths"
            )
        mixers = [module for module in model.modules() if isinstance(module, modeling_nemotron_h.NemotronHMamba2Mixer)]
        for mixer in mixers:
            mixer.use_mem_eff_path = False
        biases = _correction_biases(model)
        if not biases:
            raise RuntimeError("no e_score_correction_bias buffers found on a NemotronH model")
        if not dist.is_initialized() or dist.get_rank() == 0:
            zeroed = _zeroed(biases)
            LOG.info(
                "NemotronH checks: Mamba fast path available, %d mixers on the unfused path; "
                "%d router biases, %d all-zero",
                len(mixers),
                len(biases),
                zeroed,
            )
            if zeroed:
                raise RuntimeError(f"{zeroed} of {len(biases)} loaded e_score_correction_bias buffers are all zero")

    def add_callbacks_post_trainer(self, cfg, trainer):
        return [_RouterBiasCheck()]
