# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint policies for native JAX layer scans."""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

import equinox as eqx
import jax


@dataclasses.dataclass(frozen=True)
class ScanCheckpointPolicy:
    """Choose which layer inputs and intermediates survive the forward pass.

    Carries and extra inputs receive separate checkpoint names. Named block
    internals can be saved or offloaded independently; ``nested`` groups layers
    into a second rematerialized scan to reduce saved carry memory.
    """

    save_carries: bool | Literal["offload"] = True
    save_inputs: bool | Literal["offload"] = False
    save_block_internals: bool | list[str] = False
    offload_block_internals: list[str] = dataclasses.field(default_factory=list)
    prevent_cse: bool = False
    disable: bool = False
    simple: bool = False
    nested: bool | int = False

    @staticmethod
    def from_bool_or_str(remat_policy: bool | str):
        """Resolve a shorthand into a scan rematerialization policy.

        ``True`` checkpoints each block; ``False`` disables checkpointing.
        ``offload`` moves saved carries/inputs to pinned host memory, ``full``
        (or ``recompute``) saves no intermediates, ``save_all`` saves everything,
        and ``nested`` additionally checkpoints groups of layers.
        """
        if remat_policy == "offload":
            return ScanCheckpointPolicy(save_carries="offload", save_inputs="offload", save_block_internals=False)
        elif remat_policy == "recompute" or remat_policy == "full":
            return ScanCheckpointPolicy(save_carries=False, save_inputs=False, save_block_internals=False)
        elif remat_policy == "save_all":
            return ScanCheckpointPolicy(save_carries=True, save_inputs=True, save_block_internals=True)
        elif remat_policy == "nested":
            return ScanCheckpointPolicy(nested=True)
        elif remat_policy is True:
            return ScanCheckpointPolicy(simple=True)
        elif remat_policy is False:
            return ScanCheckpointPolicy(save_carries=True, save_inputs=True, save_block_internals=True, disable=True)
        else:
            raise ValueError(f"Invalid checkpoint policy {remat_policy}")

    @staticmethod
    def _mk(remat_policy: bool | str | "ScanCheckpointPolicy") -> "ScanCheckpointPolicy":
        if isinstance(remat_policy, ScanCheckpointPolicy):
            return remat_policy
        else:
            return ScanCheckpointPolicy.from_bool_or_str(remat_policy)

    def checkpoint(self, carry_name: str, input_name: str, callable):
        if self.disable:
            return callable
        elif self.simple:
            return eqx.filter_checkpoint(callable, prevent_cse=self.prevent_cse)
        else:
            policy = self._to_jax_policy(carry_name, input_name)
            return eqx.filter_checkpoint(callable, policy=policy, prevent_cse=self.prevent_cse)

    def _to_jax_policy(self, carry_name: str, input_name: str):
        names_to_save = []
        names_to_offload = []
        names_to_recompute = []
        for name, setting in ((carry_name, self.save_carries), (input_name, self.save_inputs)):
            if setting is True:
                names_to_save.append(name)
            elif setting == "offload":
                names_to_offload.append(name)
            else:
                assert setting is False, f"Invalid checkpoint setting {setting}"
                names_to_recompute.append(name)

        if isinstance(self.save_block_internals, Sequence):
            names_to_save.extend(self.save_block_internals)
        names_to_offload.extend(self.offload_block_internals)
        if names_to_offload:
            if self.save_block_internals is True:
                raise ValueError("Can't save all block internals and offload too. Use a list of names instead.")
            return jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=names_to_save,
                names_which_can_be_offloaded=names_to_offload,
                offload_src="device",
                offload_dst="pinned_host",
            )
        if self.save_block_internals is True:
            return jax.checkpoint_policies.save_anything_except_these_names(*names_to_recompute)
        if names_to_save:
            return jax.checkpoint_policies.save_only_these_names(*names_to_save)
        return None


ScanCheckpointSpec = ScanCheckpointPolicy | bool | Literal["offload", "recompute", "full", "save_all", "nested"]
