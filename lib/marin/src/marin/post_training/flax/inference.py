# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from:
# https://github.com/Sea-Snell/llama3_train/blob/fixed_fast_inference/llama3_even_faster_inference_clean_script.py
import copy
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator
from functools import partial
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax import struct
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as PS
from jaxtyping import PyTree
from scalax.sharding import MeshShardingHelper, TreePathShardingRule
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .llama3 import (
    LLAMA_STANDARD_CONFIGS,
    FlaxLLaMAForCausalLM,
    LLaMAConfig,
)
from .utils import (
    get_float_dtype_by_name,
    jax_distributed_barrier,
    jax_distributed_initalize,
    load_checkpoint,
    open_with_bucket,
    prepare_prompt_tokens,
    profile_fn,
    select_tokens,
)


@struct.dataclass
class GenerationConfig:
    temperature: float = 1.0
    max_output_length: int = 1024
    stop_tokens: tuple[tuple[int, ...], ...] | None = None
    n_generations: int = 1

    def to_dict(self):
        return flax.serialization.to_state_dict(self)


@struct.dataclass
class GenerationState:
    kv_cache: PyTree
    tokens: jnp.ndarray  # [bsize, max_output_length]
    logprobs: jnp.ndarray  # [bsize, max_output_length]
    dones: jnp.ndarray  # [bsize]
    next_tokens: jnp.ndarray  # [bsize]
    prefix_lengths: jnp.ndarray  # [bsize]
    generation_lengths: jnp.ndarray  # [bsize]
    temperature: jnp.ndarray  # scalar
    max_output_length: jnp.ndarray  # scalar
    # [n_stop_seqs, max_stop_seq_len] -- left padded with pad_token_id
    stop_tokens: jnp.ndarray | None
    # [bsize, max_trail_seq_len] -- left padded with pad_token_id
    trail_tokens: jnp.ndarray | None
    pad_token_id: jnp.ndarray  # scalar
    frozen_items: jnp.ndarray  # [bsize]


@partial(struct.dataclass, frozen=False)
class SamplingState:
    """Manages the state of the sampling loop."""

    is_done: bool
    to_prefill: list[int]
    unique_prompts_state: dict[str, dict[str, Any]]
    idx_to_item_id: dict[int, str]
    batch_iterator: Iterator[tuple[str, str]]
    n_generations: int

    @classmethod
    def init(
        cls,
        batch_iterable: Iterable[tuple[str, str]],
        bsize: int,
        n_generations: int,
    ):
        return cls(
            is_done=False,
            to_prefill=list(range(bsize)),
            unique_prompts_state=dict(),
            idx_to_item_id=dict(),
            batch_iterator=iter(batch_iterable),
            n_generations=n_generations,
        )

    def add_unique_prompt(
        self,
        item_id: str,
        last_prefill_token: int,
        source_idx: int,
    ):
        """Add a new unique prompt to track for multiple generations."""
        if self.n_generations > 1:
            self.unique_prompts_state[item_id] = {
                "generation_counts": 1,
                "last_prefill_token": last_prefill_token,
                "source_idx": source_idx,
            }
        self.idx_to_item_id[source_idx] = item_id

    def increment_generation_count(self, item_id: str) -> bool:
        """Increment generation count for an item and return True if max generations reached."""
        state = self.unique_prompts_state[item_id]
        state["generation_counts"] += 1
        if state["generation_counts"] == self.n_generations:
            self.unique_prompts_state.pop(item_id)
            return True
        return False


class GenerationManager:
    """Manages generation state operations."""

    def __init__(self, mesh: MeshShardingHelper, replica_axis_name: str | tuple[str, ...], per_replica_bsize: int):
        self.mesh = mesh
        self.replica_axis_name = replica_axis_name
        self.per_replica_bsize = per_replica_bsize

    def get_insert_kv_indices_fn(self, kv_cache_sharding_rules):
        @partial(
            shard_map,
            mesh=self.mesh.get_global_mesh(),
            in_specs=(
                kv_cache_sharding_rules,
                kv_cache_sharding_rules,
                PS(self.replica_axis_name),
                PS(self.replica_axis_name),
            ),
            out_specs=kv_cache_sharding_rules,
        )
        def insert_kv_indices(
            source_kv_cache,
            target_kv_cache,
            indices,
            mask,
        ):
            def insert_source_into_target(source, target):
                return target.at[indices, : source.shape[1]].set(
                    jnp.where(mask[..., None, None, None], source, target[indices, : source.shape[1]])
                )

            local_scattered = jax.tree.map(
                lambda x, y: insert_source_into_target(x, y) if len(x.shape) == 4 else y,
                source_kv_cache,
                target_kv_cache,
            )
            return local_scattered

        return insert_kv_indices

    def get_swap_kv_indices_fn(self, kv_cache_sharding_rules):
        @partial(
            shard_map,
            mesh=self.mesh.get_global_mesh(),
            in_specs=(kv_cache_sharding_rules, PS(self.replica_axis_name), PS(self.replica_axis_name)),
            out_specs=kv_cache_sharding_rules,
        )
        def swap_kv_indices(
            kv_cache,
            from_index,
            to_index,
        ):
            def swap(x):
                return x.at[to_index].set(x[from_index])

            local_selected = jax.tree.map(lambda x: swap(x) if len(x.shape) == 4 else x, kv_cache)
            return local_selected

        return swap_kv_indices

    def clear_indices(
        self,
        generation_state: GenerationState,
        indices: jnp.ndarray,  # [bsize]
        freeze: jnp.ndarray,  # [bsize]
        next_tokens: jnp.ndarray | None = None,  # [bsize]
        masked_items: jnp.ndarray | None = None,  # [bsize]
    ):
        if masked_items is None:
            masked_items = jnp.full((indices.shape[0],), True, dtype=jnp.bool_)
        new_tokens = generation_state.tokens.at[indices].set(
            jnp.where(masked_items[..., None], generation_state.pad_token_id, generation_state.tokens[indices])
        )
        new_logprobs = generation_state.logprobs.at[indices, :].set(
            jnp.where(masked_items[..., None], 0.0, generation_state.logprobs[indices])
        )
        new_dones = generation_state.dones.at[indices].set(
            jnp.where(masked_items, False, generation_state.dones[indices])
        )
        new_next_tokens = generation_state.next_tokens.at[indices].set(
            jnp.where(
                masked_items,
                generation_state.pad_token_id if next_tokens is None else next_tokens,
                generation_state.next_tokens[indices],
            )
        )
        new_generation_lengths = generation_state.generation_lengths.at[indices].set(
            jnp.where(masked_items, 0, generation_state.generation_lengths[indices])
        )
        new_trail_tokens = generation_state.trail_tokens
        if generation_state.stop_tokens is not None:
            assert new_trail_tokens is not None
            new_trail_tokens = new_trail_tokens.at[indices].set(
                jnp.where(masked_items[..., None], generation_state.pad_token_id, new_trail_tokens[indices])
            )
        new_frozen_items = generation_state.frozen_items.at[indices].set(
            jnp.where(masked_items, freeze, generation_state.frozen_items[indices])
        )
        return generation_state.replace(
            tokens=MeshShardingHelper.with_sharding_constraint(new_tokens, PS()),
            logprobs=MeshShardingHelper.with_sharding_constraint(new_logprobs, PS()),
            dones=MeshShardingHelper.with_sharding_constraint(new_dones, PS()),
            next_tokens=MeshShardingHelper.with_sharding_constraint(new_next_tokens, PS()),
            generation_lengths=MeshShardingHelper.with_sharding_constraint(new_generation_lengths, PS()),
            trail_tokens=MeshShardingHelper.with_sharding_constraint(new_trail_tokens, PS()),
            frozen_items=MeshShardingHelper.with_sharding_constraint(new_frozen_items, PS()),
        )

    def insert_tokens(
        self,
        generation_state: GenerationState,
        kv_cache: PyTree,
        tokens: jnp.ndarray,  # [bsize]
        logprobs: jnp.ndarray,  # [bsize]
    ):
        all_frozen = jnp.logical_or(generation_state.frozen_items, generation_state.dones)
        tokens = jnp.where(all_frozen, generation_state.pad_token_id, tokens)
        new_tokens = generation_state.tokens.at[jnp.arange(tokens.shape[0]), generation_state.generation_lengths].set(
            tokens
        )
        new_logprobs = generation_state.logprobs.at[
            jnp.arange(tokens.shape[0]), generation_state.generation_lengths
        ].set(
            jnp.where(all_frozen, 0.0, logprobs),
        )
        new_generation_lengths = generation_state.generation_lengths + (1 - all_frozen)
        new_dones = jnp.logical_or(
            generation_state.dones,
            new_generation_lengths >= generation_state.max_output_length,
        )
        new_trail_tokens = generation_state.trail_tokens
        if generation_state.stop_tokens is not None:
            assert new_trail_tokens is not None
            new_trail_tokens = jnp.concatenate((new_trail_tokens[:, 1:], tokens[:, None]), axis=1)
            stop_tokens_mask = generation_state.stop_tokens != generation_state.pad_token_id
            new_dones = jnp.logical_or(
                new_dones,
                jnp.any(
                    ((new_trail_tokens[:, None, :] == generation_state.stop_tokens[None, :, :]) * stop_tokens_mask).sum(
                        axis=-1
                    )
                    == stop_tokens_mask.sum(axis=-1),
                    axis=-1,
                ),
            )
        return generation_state.replace(
            kv_cache=kv_cache,
            tokens=MeshShardingHelper.with_sharding_constraint(new_tokens, PS()),
            logprobs=MeshShardingHelper.with_sharding_constraint(new_logprobs, PS()),
            dones=MeshShardingHelper.with_sharding_constraint(new_dones, PS()),
            next_tokens=MeshShardingHelper.with_sharding_constraint(tokens, PS()),
            generation_lengths=MeshShardingHelper.with_sharding_constraint(new_generation_lengths, PS()),
            trail_tokens=MeshShardingHelper.with_sharding_constraint(new_trail_tokens, PS()),
        )


class FlaxSampler:
    """High-level sampling interface that manages model generation."""

    def __init__(
        self,
        prefill_model: FlaxLLaMAForCausalLM,
        generate_model: FlaxLLaMAForCausalLM,
        tokenizer: AutoTokenizer,
        generation_config: GenerationConfig,
        bsize: int,
        prefill_bsize: int,
        max_input_length: int,
        params_sharding_rules: PyTree,
        intermediate_sharding_rules: PyTree,
        replica_axis_name: str | None | tuple[str, ...],
        tp_axis_name: str | None | tuple[str, ...],
        mesh: MeshShardingHelper,
        pad_token_id: int,
    ):
        self.prefill_model = prefill_model
        self.generate_model = generate_model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.bsize = bsize
        self.prefill_bsize = prefill_bsize
        self.max_input_length = max_input_length
        self.pad_token_id = pad_token_id
        self.mesh = mesh

        if isinstance(replica_axis_name, tuple):
            n_replicas = 1
            for axis_name in replica_axis_name:
                n_replicas *= mesh.get_global_mesh().shape[axis_name]
        else:
            n_replicas = mesh.get_global_mesh().shape[replica_axis_name]

        assert bsize % n_replicas == 0, f"bsize must be divisible by n_replicas, got {bsize} and {n_replicas}"
        assert (
            prefill_bsize % n_replicas == 0
        ), f"prefill_bsize must be divisible by n_replicas, got {prefill_bsize} and {n_replicas}"

        self.n_replicas = n_replicas
        self.per_replica_bsize = bsize // n_replicas
        self.per_replica_prefill_bsize = prefill_bsize // n_replicas
        self.replica_axis_name = replica_axis_name

        # Initialize generation manager
        self.generation_manager = GenerationManager(mesh, replica_axis_name, self.per_replica_bsize)

        # Setup sharding
        self._setup_sharding(params_sharding_rules, intermediate_sharding_rules, tp_axis_name)

        # Compile JAX functions
        self._compile_functions()

    def _setup_sharding(self, params_sharding_rules, intermediate_sharding_rules, tp_axis_name):
        """Setup sharding configurations."""
        self.kv_cache_shape = jax.eval_shape(
            lambda: self.prefill_model.init_cache(
                self.bsize, self.max_input_length + self.generation_config.max_output_length - 1
            )
        )
        self.prefill_kv_cache_shape = jax.eval_shape(
            lambda: self.prefill_model.init_cache(self.prefill_bsize, self.max_input_length)
        )

        self.kv_cache_sharding_rules = jax.tree.map(
            lambda x: PS(self.replica_axis_name, None, tp_axis_name, None) if len(x.shape) == 4 else PS(),
            self.kv_cache_shape,
        )

        self.generation_state_sharding_rules = GenerationState(
            self.kv_cache_sharding_rules,
            PS(),
            PS(),
            PS(),
            PS(),
            PS(),
            PS(),
            PS(),
            PS(),
            PS(),
            PS(),
            PS(),
            PS(),
        )

        self.params_sharding_rules = params_sharding_rules
        self.intermediate_sharding_rules = intermediate_sharding_rules

    def _compile_functions(self):
        """Compile JAX functions for sampling."""
        do_sample = self.generation_config.temperature > 0.0
        select_tokens_fn = partial(select_tokens, do_sample=do_sample)

        @partial(
            self.mesh.sjit,
            out_shardings=self.generation_state_sharding_rules,
        )
        def init_generation_state():
            kv_cache = jax.tree.map(
                lambda x, y: MeshShardingHelper.with_sharding_constraint(jnp.zeros(x.shape, dtype=x.dtype), y),
                self.kv_cache_shape,
                self.kv_cache_sharding_rules,
            )
            stop_tokens = self.generation_config.stop_tokens
            trail_tokens = None
            if stop_tokens is not None:
                max_stop_seq_len = max(len(stop_seq) for stop_seq in stop_tokens)
                stop_tokens = [
                    [self.pad_token_id] * (max_stop_seq_len - len(stop_seq)) + list(stop_seq) for stop_seq in stop_tokens
                ]
                stop_tokens = jnp.asarray(stop_tokens, dtype=jnp.int32)
                trail_tokens = jnp.full((self.bsize, max_stop_seq_len), self.pad_token_id, dtype=jnp.int32)
            return GenerationState(
                kv_cache=kv_cache,
                tokens=jnp.full(
                    (self.bsize, self.generation_config.max_output_length), self.pad_token_id, dtype=jnp.int32
                ),
                logprobs=jnp.zeros((self.bsize, self.generation_config.max_output_length), dtype=jnp.float32),
                dones=jnp.full((self.bsize,), False, dtype=jnp.bool_),
                next_tokens=jnp.full((self.bsize,), self.pad_token_id, dtype=jnp.int32),
                prefix_lengths=jnp.zeros((self.bsize,), dtype=jnp.int32),
                generation_lengths=jnp.zeros((self.bsize,), dtype=jnp.int32),
                temperature=jnp.asarray(self.generation_config.temperature, dtype=jnp.float32),
                max_output_length=jnp.asarray(self.generation_config.max_output_length, dtype=jnp.int32),
                stop_tokens=stop_tokens,
                trail_tokens=trail_tokens,
                pad_token_id=jnp.asarray(self.pad_token_id, dtype=jnp.int32),
                frozen_items=jnp.full((self.bsize,), False, dtype=jnp.bool_),
            )

        @partial(profile_fn, name="prefill")
        @partial(
            self.mesh.sjit,
            in_shardings=(
                self.params_sharding_rules,
                PS(),
                PS(),
                PS(),
                PS(),
                PS(),
                self.generation_state_sharding_rules,
            ),
            out_shardings=self.generation_state_sharding_rules,
            args_sharding_constraint=(
                self.params_sharding_rules,
                PS(self.replica_axis_name),
                PS(self.replica_axis_name),
                PS(self.replica_axis_name),
                PS(),
                PS(),
                self.generation_state_sharding_rules,
            ),
            donate_argnums=(6,),
            annotation_shardings=self.intermediate_sharding_rules,
        )
        def prefill(
            params: PyTree,
            input_ids: jnp.ndarray,
            attention_mask: jnp.ndarray,
            position_ids: jnp.ndarray,
            map_indices: jnp.ndarray,
            freeze: jnp.ndarray,
            generation_state: GenerationState,
        ) -> GenerationState:
            prefill_kv_cache = jax.tree.map(
                lambda x, y: MeshShardingHelper.with_sharding_constraint(jnp.zeros(x.shape, dtype=x.dtype), y),
                self.prefill_kv_cache_shape,
                self.kv_cache_sharding_rules,
            )
            model_output = self.prefill_model(
                input_ids,
                attention_mask,
                position_ids,
                params=params,
                train=False,
                past_key_values=prefill_kv_cache,
            )
            prefill_kv_cache = model_output.past_key_values
            prefix_lengths = attention_mask.sum(axis=-1)
            masked_items = prefix_lengths >= 1
            kv_cache = self.generation_manager.get_insert_kv_indices_fn(self.kv_cache_sharding_rules)(
                prefill_kv_cache,
                generation_state.kv_cache,
                map_indices % self.per_replica_bsize,
                masked_items,
            )
            return self.generation_manager.clear_indices(
                generation_state.replace(
                    kv_cache=MeshShardingHelper.with_sharding_constraint(kv_cache, self.kv_cache_sharding_rules),
                    prefix_lengths=MeshShardingHelper.with_sharding_constraint(
                        generation_state.prefix_lengths.at[map_indices].set(
                            jnp.where(masked_items, prefix_lengths, generation_state.prefix_lengths[map_indices])
                        ),
                        PS(),
                    ),
                ),
                MeshShardingHelper.with_sharding_constraint(map_indices, PS()),
                MeshShardingHelper.with_sharding_constraint(
                    jnp.where(masked_items, freeze, generation_state.frozen_items[map_indices]), PS()
                ),
                MeshShardingHelper.with_sharding_constraint(
                    jnp.where(
                        masked_items,
                        input_ids[jnp.arange(input_ids.shape[0]), prefix_lengths - 1],
                        generation_state.next_tokens[map_indices],
                    ),
                    PS(),
                ),
                MeshShardingHelper.with_sharding_constraint(masked_items, PS()),
            )

        @partial(profile_fn, name="generate_step")
        @partial(
            self.mesh.sjit,
            in_shardings=(
                self.params_sharding_rules,
                self.generation_state_sharding_rules,
                PS(),
            ),
            out_shardings=self.generation_state_sharding_rules,
            args_sharding_constraint=(
                self.params_sharding_rules,
                self.generation_state_sharding_rules,
                PS(),
            ),
            donate_argnums=(1,),
            annotation_shardings=self.intermediate_sharding_rules,
        )
        def generate_step(
            params: PyTree,
            generation_state: GenerationState,
            prng_key: jnp.ndarray,
        ) -> GenerationState:
            position_ids = generation_state.prefix_lengths + generation_state.generation_lengths - 1
            model_output = self.generate_model(
                input_ids=generation_state.next_tokens[..., None],
                attention_mask=None,
                position_ids=position_ids[..., None],
                params=params,
                train=False,
                past_key_values=generation_state.kv_cache,
            )
            logits = model_output.logits
            kv_cache = model_output.past_key_values
            selected_tokens, selected_logprobs = select_tokens_fn(
                logits[:, -1, :],
                prng_key,
                generation_state.temperature,
            )
            selected_tokens = MeshShardingHelper.with_sharding_constraint(selected_tokens, PS())
            selected_logprobs = MeshShardingHelper.with_sharding_constraint(selected_logprobs, PS())
            kv_cache = MeshShardingHelper.with_sharding_constraint(kv_cache, self.kv_cache_sharding_rules)
            return self.generation_manager.insert_tokens(
                generation_state,
                kv_cache,
                selected_tokens,
                selected_logprobs,
            )

        # Store compiled functions
        self.init_generation_state = init_generation_state
        self.prefill = prefill
        self.generate_step = generate_step

        # Additional compiled functions for state management
        @partial(profile_fn, name="extract_dones")
        @partial(
            self.mesh.sjit,
            in_shardings=(self.generation_state_sharding_rules,),
            out_shardings=PS(),
            args_sharding_constraint=(self.generation_state_sharding_rules,),
        )
        def extract_dones(
            generation_state: GenerationState,
        ) -> tuple[bool, dict[str, jnp.ndarray]]:
            argmax_idx = jnp.argmax(generation_state.dones)
            return generation_state.dones[argmax_idx], {
                "tokens": generation_state.tokens[argmax_idx],
                "logprobs": generation_state.logprobs[argmax_idx],
                "generation_length": generation_state.generation_lengths[argmax_idx],
                "idx": argmax_idx,
            }

        @partial(profile_fn, name="all_finished")
        @partial(
            self.mesh.sjit,
            in_shardings=(self.generation_state_sharding_rules,),
            out_shardings=PS(),
            args_sharding_constraint=(self.generation_state_sharding_rules,),
        )
        def all_finished(
            generation_state: GenerationState,
        ) -> bool:
            return jnp.all(generation_state.frozen_items)

        @partial(profile_fn, name="clear_indices_compiled")
        @partial(
            self.mesh.sjit,
            in_shardings=(self.generation_state_sharding_rules, PS(), PS(), PS()),
            out_shardings=self.generation_state_sharding_rules,
            args_sharding_constraint=(self.generation_state_sharding_rules, PS(), PS(), PS()),
            donate_argnums=(0,),
        )
        def clear_indices_compiled(
            generation_state: GenerationState,
            indices: jnp.ndarray,
            freeze: jnp.ndarray,
            next_tokens: jnp.ndarray | None = None,
        ) -> GenerationState:
            return self.generation_manager.clear_indices(
                generation_state,
                indices,
                freeze,
                next_tokens,
            )

        self.extract_dones = extract_dones
        self.all_finished = all_finished
        self.clear_indices_compiled = clear_indices_compiled

        # Additional helper function for copying cache indices
        @partial(profile_fn, name="_copy_index_across_cache")
        @partial(
            self.mesh.sjit,
            in_shardings=(self.generation_state_sharding_rules, PS(), PS(), PS(), PS(), PS()),
            out_shardings=self.generation_state_sharding_rules,
            args_sharding_constraint=(
                self.generation_state_sharding_rules,
                PS(self.replica_axis_name),
                PS(self.replica_axis_name),
                PS(),
                PS(),
                PS(),
            ),
            donate_argnums=(0,),
        )
        def _copy_index_across_cache(
            generation_state: GenerationState,
            from_idx: jnp.ndarray,
            to_idx: jnp.ndarray,
            initial_token: jnp.ndarray,
            freeze: jnp.ndarray,
            masked_items: jnp.ndarray | None = None,
        ) -> GenerationState:
            if masked_items is None:
                masked_items = jnp.full((from_idx.shape[0],), True, dtype=jnp.bool_)
            kv_cache = self.generation_manager.get_swap_kv_indices_fn(self.kv_cache_sharding_rules)(
                generation_state.kv_cache,
                MeshShardingHelper.with_sharding_constraint(
                    from_idx % self.per_replica_bsize, PS(self.replica_axis_name)
                ),
                MeshShardingHelper.with_sharding_constraint(to_idx % self.per_replica_bsize, PS(self.replica_axis_name)),
            )
            return self.generation_manager.clear_indices(
                generation_state.replace(
                    kv_cache=MeshShardingHelper.with_sharding_constraint(kv_cache, self.kv_cache_sharding_rules),
                    prefix_lengths=MeshShardingHelper.with_sharding_constraint(
                        generation_state.prefix_lengths.at[to_idx].set(
                            jnp.where(
                                masked_items,
                                generation_state.prefix_lengths[from_idx],
                                generation_state.prefix_lengths[to_idx],
                            )
                        ),
                        PS(),
                    ),
                ),
                MeshShardingHelper.with_sharding_constraint(to_idx, PS()),
                MeshShardingHelper.with_sharding_constraint(freeze, PS()),
                MeshShardingHelper.with_sharding_constraint(initial_token, PS()),
                MeshShardingHelper.with_sharding_constraint(masked_items, PS()),
            )

        self._copy_index_across_cache = _copy_index_across_cache

    def copy_single_index_across_cache(
        self,
        generation_state: GenerationState,
        from_idx: int,
        to_idx: int,
        initial_token: int,
        freeze: bool,
    ) -> GenerationState:
        """Copy cache from one index to another."""
        assert to_idx // self.per_replica_bsize == from_idx // self.per_replica_bsize
        to_idx_list = [replica_idx * self.per_replica_bsize for replica_idx in range(self.n_replicas)]
        to_idx_list[to_idx // self.per_replica_bsize] = to_idx
        from_idx_list = [replica_idx * self.per_replica_bsize for replica_idx in range(self.n_replicas)]
        from_idx_list[from_idx // self.per_replica_bsize] = from_idx
        mask_list = [False] * self.n_replicas
        mask_list[to_idx // self.per_replica_bsize] = True
        return self._copy_index_across_cache(
            generation_state,
            jnp.asarray(from_idx_list, dtype=jnp.int32),
            jnp.asarray(to_idx_list, dtype=jnp.int32),
            jnp.asarray([initial_token] * self.n_replicas, dtype=jnp.int32),
            jnp.asarray([freeze] * self.n_replicas, dtype=jnp.bool_),
            jnp.asarray(mask_list, dtype=jnp.bool_),
        )

    def __call__(
        self,
        params: PyTree,
        batch_iterable: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
        prng_key: jnp.ndarray,
    ) -> Generator[dict[str, jnp.ndarray], None, None]:
        """Run the sampling loop."""
        generation_state = self.init_generation_state()
        sampling_loop_state = SamplingState.init(
            batch_iterable=batch_iterable,
            bsize=self.bsize,
            n_generations=self.generation_config.n_generations,
        )

        # Implementation of the full sampling loop would go here
        # This is a simplified version - the full implementation would include
        # all the prefilling, generation, and state management logic
        yield from self._sampling_loop_impl(params, generation_state, sampling_loop_state, prng_key)

    def _sampling_loop_impl(self, params, generation_state, sampling_loop_state, prng_key):
        """Implementation of the sampling loop."""

        def build_prefill_batch(sampling_loop_state: SamplingState):
            batch_input_ids, batch_attention_mask, batch_position_ids = [], [], []
            idxs, freeze = [], []
            sorted_to_prefill = sorted(sampling_loop_state.to_prefill, reverse=True)
            unused_idxs = np.arange(self.bsize).reshape(self.n_replicas, self.per_replica_bsize).tolist()

            def add_pad_item(prefill_idx: int):
                batch_input_ids.append(np.full((1, self.max_input_length), self.pad_token_id, dtype=jnp.int32))
                batch_attention_mask.append(np.zeros((1, self.max_input_length), dtype=jnp.int32))
                batch_position_ids.append(np.zeros((1, self.max_input_length), dtype=jnp.int32))
                idxs.append(unused_idxs[prefill_idx // self.per_replica_prefill_bsize].pop())
                freeze.append(True)

            for prefill_idx in range(self.prefill_bsize):
                if len(sorted_to_prefill) == 0:
                    break

                new_idx = sorted_to_prefill.pop()
                while new_idx // self.per_replica_bsize < prefill_idx // self.per_replica_prefill_bsize:
                    if len(sorted_to_prefill) == 0:
                        new_idx = None
                        break
                    new_idx = sorted_to_prefill.pop()

                if new_idx is None:
                    break

                if new_idx // self.per_replica_bsize > prefill_idx // self.per_replica_prefill_bsize:
                    add_pad_item(prefill_idx)
                    sorted_to_prefill.append(new_idx)
                    continue

                assert new_idx // self.per_replica_bsize == prefill_idx // self.per_replica_prefill_bsize
                try:
                    prompt_str, item_id = next(sampling_loop_state.batch_iterator)
                    curr_input_ids, curr_attention_mask, curr_position_ids = prepare_prompt_tokens(
                        self.tokenizer,
                        prompt_str,
                        self.max_input_length,
                    )
                except StopIteration:
                    sampling_loop_state.is_done = True
                    break

                batch_input_ids.append(curr_input_ids)
                batch_attention_mask.append(curr_attention_mask)
                batch_position_ids.append(curr_position_ids)
                idxs.append(new_idx)
                freeze.append(False)
                sampling_loop_state.add_unique_prompt(
                    item_id=item_id,
                    last_prefill_token=curr_input_ids[0][curr_attention_mask[0].sum() - 1],
                    source_idx=new_idx,
                )
                unused_idxs[prefill_idx // self.per_replica_prefill_bsize].remove(new_idx)
                sampling_loop_state.to_prefill.remove(new_idx)

            # pad remaining slots if needed
            if len(batch_input_ids) < self.prefill_bsize:
                for prefill_idx in range(len(batch_input_ids), self.prefill_bsize):
                    add_pad_item(prefill_idx)

            # concatenate and validate batch
            batch_input_ids = np.concatenate(batch_input_ids, axis=0)
            batch_attention_mask = np.concatenate(batch_attention_mask, axis=0)
            batch_position_ids = np.concatenate(batch_position_ids, axis=0)
            idxs = np.asarray(idxs, dtype=jnp.int32)
            freeze = np.asarray(freeze, dtype=jnp.bool_)

            assert all(
                len(x) == self.prefill_bsize
                for x in [batch_input_ids, batch_attention_mask, batch_position_ids, idxs, freeze]
            )

            return batch_input_ids, batch_attention_mask, batch_position_ids, idxs, freeze

        def make_copies_from_prefill(
            sampling_loop_state: SamplingState,
            generation_state: GenerationState,
        ) -> GenerationState:
            sorted_unique_prompts_state = sorted(
                sampling_loop_state.unique_prompts_state.items(), key=lambda x: x[1]["generation_counts"]
            )
            for item_id, prompt_state in sorted_unique_prompts_state:
                to_prefill_on_device = [
                    idx
                    for idx in sampling_loop_state.to_prefill
                    if idx // self.per_replica_bsize == prompt_state["source_idx"] // self.per_replica_bsize
                ]
                while (
                    len(to_prefill_on_device) > 0
                    and prompt_state["generation_counts"] < sampling_loop_state.n_generations
                ):
                    to_idx = to_prefill_on_device.pop()
                    generation_state = self.copy_single_index_across_cache(
                        generation_state,
                        prompt_state["source_idx"],
                        to_idx,
                        prompt_state["last_prefill_token"],
                        False,
                    )
                    prompt_state["generation_counts"] += 1
                    sampling_loop_state.idx_to_item_id[to_idx] = item_id
                    sampling_loop_state.to_prefill.remove(to_idx)
                if prompt_state["generation_counts"] == sampling_loop_state.n_generations:
                    sampling_loop_state.unique_prompts_state.pop(item_id)
                if len(sampling_loop_state.to_prefill) == 0:
                    break
            return generation_state

        def can_prefill(sampling_loop_state: SamplingState):
            replica_counts = [0 for _ in range(self.n_replicas)]
            for idx in sampling_loop_state.to_prefill:
                replica_counts[idx // self.per_replica_bsize] += 1
            return any([count >= self.per_replica_prefill_bsize for count in replica_counts])

        while not sampling_loop_state.is_done:
            # prefill until we can't anymore
            while (not sampling_loop_state.is_done) and can_prefill(sampling_loop_state):
                batch_input_ids, batch_attention_mask, batch_position_ids, idxs, freeze = build_prefill_batch(
                    sampling_loop_state
                )
                generation_state = self.prefill(
                    params,
                    batch_input_ids,
                    batch_attention_mask,
                    batch_position_ids,
                    idxs,
                    freeze,
                    generation_state,
                )
                generation_state = make_copies_from_prefill(sampling_loop_state, generation_state)

            # if done prefilling, clear any remaining indices
            while sampling_loop_state.is_done and len(sampling_loop_state.to_prefill) > 0:
                generation_state = self.clear_indices_compiled(
                    generation_state,
                    jnp.asarray([sampling_loop_state.to_prefill.pop()], dtype=jnp.int32),
                    jnp.asarray([True], dtype=jnp.bool_),
                    None,
                )

            # generation steps
            while True:
                # first check if any indices are done generating
                has_done, done_info = self.extract_dones(generation_state)
                while has_done:
                    idx = int(done_info.pop("idx"))
                    item_id = sampling_loop_state.idx_to_item_id[idx]
                    done_info["item_id"] = item_id
                    yield done_info
                    if item_id not in sampling_loop_state.unique_prompts_state:
                        # we have completed generations for this item
                        on_device_unique_prompts_state = list(
                            filter(
                                lambda x: x[1]["source_idx"] // self.per_replica_bsize == idx // self.per_replica_bsize,
                                sampling_loop_state.unique_prompts_state.items(),
                            )
                        )
                        if len(on_device_unique_prompts_state) == 0:
                            # clear the index and add to prefill queue
                            sampling_loop_state.to_prefill.append(idx)
                            generation_state = self.clear_indices_compiled(
                                generation_state,
                                jnp.asarray([idx], dtype=jnp.int32),
                                jnp.asarray([True], dtype=jnp.bool_),
                                None,
                            )
                        else:
                            # copy kv from another unique prompt on device
                            min_id, min_prompt_state = min(
                                on_device_unique_prompts_state, key=lambda x: x[1]["generation_counts"]
                            )
                            generation_state = self.copy_single_index_across_cache(
                                generation_state,
                                min_prompt_state["source_idx"],
                                idx,
                                min_prompt_state["last_prefill_token"],
                                False,
                            )
                            sampling_loop_state.idx_to_item_id[idx] = min_id
                            sampling_loop_state.increment_generation_count(min_id)
                    else:
                        # reset and re-generate at this index
                        generation_state = self.clear_indices_compiled(
                            generation_state,
                            jnp.asarray([idx], dtype=jnp.int32),
                            jnp.asarray([False], dtype=jnp.bool_),
                            jnp.asarray(
                                [sampling_loop_state.unique_prompts_state[item_id]["last_prefill_token"]],
                                dtype=jnp.int32,
                            ),
                        )
                        sampling_loop_state.increment_generation_count(item_id)

                    has_done, done_info = self.extract_dones(generation_state)

                if (can_prefill(sampling_loop_state) and (not sampling_loop_state.is_done)) or self.all_finished(
                    generation_state
                ):
                    break

                # step generation
                prng_key, new_key = jax.random.split(prng_key)
                generation_state = self.generate_step(
                    params,
                    generation_state,
                    new_key,
                )


def build_sampler(**kwargs) -> FlaxSampler:
    """Factory function to build a sampler."""
    return FlaxSampler(**kwargs)


def batch_inference(
    sampler: FlaxSampler,
    params: PyTree,
    prompts: list[str],
    prng_key: jnp.ndarray,
    n_generations: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run batch inference on a list of prompts."""

    def create_batch_iterable(prompts: list[str]) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        for i, prompt in enumerate(prompts):
            yield prompt, f"{i}"

    all_results = []
    sampling_iter = sampler(
        params,
        create_batch_iterable(prompts),
        prng_key,
    )

    for _, result in tqdm(
        enumerate(sampling_iter),
        total=len(prompts) * n_generations if n_generations is not None else None,
        disable=not verbose,
    ):
        all_results.append(result)

    all_results_by_id = defaultdict(list)
    for result in tqdm(all_results, total=len(all_results), disable=not verbose):
        generation_length = int(result["generation_length"])
        all_results_by_id[int(result["item_id"])].append(
            {
                "tokens": result["tokens"][:generation_length].tolist(),
                "logprobs": result["logprobs"][:generation_length].tolist(),
            }
        )

    all_results_ordered = [all_results_by_id[int(item_id)] for item_id in sorted(map(int, all_results_by_id.keys()))]
    return all_results_ordered


class FlaxInferenceContext:
    """Wraps Flax inference components and hides model parameters."""

    def __init__(
        self,
        params: PyTree,
        sampler: FlaxSampler,
        prng_key: jnp.ndarray,
        tokenizer,
        get_logprobs_fn,
        reference_logprobs_bsize: int = 32,
    ):
        """Initialize with all inference components.

        Args:
            params: Model parameters (hidden from external use)
            sampler: Configured FlaxSampler (already has max_input_length, bsize, etc.)
            prng_key: JAX PRNG key for sampling
            tokenizer: Tokenizer
            get_logprobs_fn: Function to compute logprobs with params
            reference_logprobs_bsize: Batch size for logprobs computation (internal)
        """
        self._params = params
        self._sampler = sampler
        self._prng_key = prng_key
        self._tokenizer = tokenizer
        self._get_logprobs_fn = get_logprobs_fn
        self._reference_logprobs_bsize = reference_logprobs_bsize

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(
        self,
        prompts: list[str],
        temperature: float = 1.0,
        n_generations: int = 1,
    ) -> list[list[dict]]:
        """Generate using hidden params and sampler."""

        # Split key for this generation
        self._prng_key, subkey = jax.random.split(self._prng_key)

        # The sampler already knows its max_input_length, bsize, etc.
        return batch_inference(
            self._sampler,
            self._params,
            prompts,
            subkey,
            n_generations=n_generations,
            verbose=False,
        )

    def compute_logprobs(
        self,
        input_tokens: np.ndarray,
        input_attention_mask: np.ndarray,
        target_tokens: np.ndarray,
        target_attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute log probabilities."""

        # Handle batching internally if needed
        batch_size = input_tokens.shape[0]
        if batch_size > self._reference_logprobs_bsize:
            # Process in chunks internally
            all_logprobs = []
            for i in range(0, batch_size, self._reference_logprobs_bsize):
                end = min(i + self._reference_logprobs_bsize, batch_size)
                batch_input_tokens = input_tokens[i:end]
                batch_input_attention_mask = input_attention_mask[i:end]
                batch_target_tokens = target_tokens[i:end]
                batch_target_attention_mask = target_attention_mask[i:end]

                batch_logprobs = self._get_logprobs_fn(
                    self._params,
                    batch_input_tokens,
                    batch_input_attention_mask,
                    batch_target_tokens,
                    batch_target_attention_mask,
                )
                all_logprobs.append(batch_logprobs)
            return np.concatenate(all_logprobs, axis=0)
        else:
            return self._get_logprobs_fn(
                self._params,
                input_tokens,
                input_attention_mask,
                target_tokens,
                target_attention_mask,
            )

    def update_params(self, new_params: PyTree):
        """Update the hidden model parameters (used by training loop)."""
        self._params = new_params


def main(
    load_model: str,
    data_path: str,
    bsize: int,
    prefill_bsize: int,
    max_input_length: int,
    save_path: str,
    sharding: str,
    param_dtype: str = "bf16",
    activation_dtype: str = "fp32",
    generation_config: str = "{}",
    model_config_override: str = "{}",
    tokenizer_override: str = "{}",
    prefill_attention_kernel_config: str = "splash:{}",
    generate_attention_kernel_config: str = "paged:{}",
    jax_distributed_initalize_config: str = "{}",
    physical_axis_splitting: bool = False,
    pad_token_id: int = 128002,
):
    """Main inference script."""
    assert bsize % prefill_bsize == 0
    args_dict = dict(locals())
    print(args_dict)
    sharding: list[int] = list(map(lambda x: int(x.strip()), sharding.split(",")))

    param_dtype = get_float_dtype_by_name(param_dtype)
    activation_dtype = get_float_dtype_by_name(activation_dtype)

    generation_config: dict[str, Any] = json.loads(generation_config)
    model_config_override: dict[str, Any] = json.loads(model_config_override)
    tokenizer_override: dict[str, Any] = json.loads(tokenizer_override)
    jax_distributed_initalize_config: dict[str, Any] = json.loads(jax_distributed_initalize_config)

    jax_distributed_initalize(**jax_distributed_initalize_config)
    jax_distributed_barrier()

    # Load attention kernel configurations

    if prefill_attention_kernel_config.startswith("splash:"):
        prefill_attention_kernel_config = json.loads(prefill_attention_kernel_config[len("splash:") :])
        prefill_attention_kernel = "splash"
    elif prefill_attention_kernel_config.startswith("default:"):
        prefill_attention_kernel_config = json.loads(prefill_attention_kernel_config[len("default:") :])
        prefill_attention_kernel = "default"
    else:
        raise ValueError(f"Unknown prefill attention kernel config: {prefill_attention_kernel_config}")

    if generate_attention_kernel_config.startswith("paged:"):
        generate_attention_kernel_config = json.loads(generate_attention_kernel_config[len("paged:") :])
        generate_attention_kernel = "paged"
    elif generate_attention_kernel_config.startswith("default:"):
        generate_attention_kernel_config = json.loads(generate_attention_kernel_config[len("default:") :])
        generate_attention_kernel = "default"
    else:
        raise ValueError(f"Unknown generate attention kernel config: {generate_attention_kernel_config}")

    mesh = MeshShardingHelper(
        sharding, ["replica", "fsdp", "sequence", "tensor"], mesh_axis_splitting=physical_axis_splitting
    )

    with mesh.get_context():
        print("loading model ...")

        if load_model.startswith("paths:"):
            model_paths = json.loads(load_model[len("paths:") :])
            if "remove_dict_prefix" not in model_paths:
                model_paths["remove_dict_prefix"] = None
        else:
            raise ValueError(f"Unknown model info type: {load_model}")

        config_is_temp = False
        if "config" in model_paths and model_paths["config"].startswith("gs://"):
            temp_file = tempfile.NamedTemporaryFile("wb", delete=False)
            with open_with_bucket(model_paths["config"], "rb") as f:
                temp_file.write(f.read())
            temp_file.close()
            model_paths["config"] = temp_file.name
            config_is_temp = True

        if "config" in model_paths:
            config = LLaMAConfig.from_pretrained(model_paths["config"], **model_config_override)
        elif "default_config_name" in model_paths:
            config = LLaMAConfig(**LLAMA_STANDARD_CONFIGS[model_paths["default_config_name"]], **model_config_override)
        else:
            config = LLaMAConfig(**model_config_override)

        prefill_config = copy.deepcopy(config)
        prefill_config.attention_kernel = prefill_attention_kernel
        prefill_config.attention_kernel_settings = prefill_attention_kernel_config

        generate_config = copy.deepcopy(config)

        if config_is_temp:
            os.remove(model_paths["config"])

        prefill_model = FlaxLLaMAForCausalLM(
            prefill_config,
            dtype=activation_dtype,
            _do_init=False,
            param_dtype=param_dtype,
            input_shape=(prefill_bsize, max_input_length),
        )
        generate_model = FlaxLLaMAForCausalLM(
            generate_config,
            dtype=activation_dtype,
            _do_init=False,
            param_dtype=param_dtype,
            input_shape=(prefill_bsize, max_input_length),
        )
        generate_model.config.attention_kernel = generate_attention_kernel
        generate_model.config.attention_kernel_settings = generate_attention_kernel_config

        tokenizer_is_temp = False
        if model_paths["tokenizer"].startswith("gs://"):
            temp_file = tempfile.NamedTemporaryFile("wb", delete=False)
            with open_with_bucket(model_paths["tokenizer"], "rb") as f:
                temp_file.write(f.read())
            temp_file.close()
            model_paths["tokenizer"] = temp_file.name
            tokenizer_is_temp = True

        tokenizer_kwargs = dict(
            truncation_side="right",
            padding_side="right",
            pad_token="<|reserved_special_token_0|>",
        )
        tokenizer_kwargs.update(tokenizer_override)
        tokenizer = AutoTokenizer.from_pretrained(model_paths["tokenizer"], **tokenizer_kwargs)

        if tokenizer_is_temp:
            os.remove(model_paths["tokenizer"])

        # load params sharded
        params_sharding_rules = TreePathShardingRule(
            *prefill_config.get_partition_rules(
                model_all_gather_axis=None,
            )
        )
        params_shape = jax.eval_shape(
            lambda: prefill_model.init_weights(jax.random.PRNGKey(0), (prefill_bsize, max_input_length))
        )
        shard_param_fns, gather_param_fns = mesh.make_shard_and_gather_fns(params_shape, params_sharding_rules)
        intermediate_sharding_rules = config.get_intermediate_sharding_rules(
            data_axis="replica",
            sequence_axis=None,
        )

        @partial(
            mesh.sjit,
            in_shardings=(PS(),),
            out_shardings=params_sharding_rules,
            annotation_shardings=intermediate_sharding_rules,
        )
        def init_fn(rng):
            return prefill_model.init_weights(rng, (prefill_bsize, max_input_length))

        if "params" in model_paths:
            params = load_checkpoint(
                model_paths["params"],
                shard_fns=shard_param_fns,
                remove_dict_prefix=model_paths["remove_dict_prefix"],
            )
        else:
            print("WARNING: no params path provided, initializing with random weights...")
            params = init_fn(jax.random.PRNGKey(0))

        print("model loaded.")

        generation_config = GenerationConfig(**generation_config)
        sampler = build_sampler(
            prefill_model=prefill_model,
            generate_model=generate_model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            bsize=bsize,
            prefill_bsize=prefill_bsize,
            max_input_length=max_input_length,
            params_sharding_rules=params_sharding_rules,
            intermediate_sharding_rules=intermediate_sharding_rules,
            replica_axis_name="replica",
            tp_axis_name="tensor",
            mesh=mesh,
            pad_token_id=pad_token_id,
        )

        print("loading data ...")
        if data_path.endswith(".jsonl"):
            examples = []
            with open_with_bucket(data_path, "r") as f:
                for line in f:
                    examples.append(json.loads(line.strip()))
        else:
            with open_with_bucket(data_path, "r") as f:
                examples = json.load(f)
        print("data loaded.")

        print("running generation ...")
        all_results = batch_inference(
            sampler,
            params,
            [example["prompt"] for example in examples],
            jax.random.PRNGKey(0),
            n_generations=generation_config.n_generations,
            verbose=True,
        )
        print("finished generation.")

        jax_distributed_barrier()

        print(f"saving results to {save_path} ...")
        with open_with_bucket(save_path, "w") as f:
            json.dump(
                {
                    "results": all_results,
                    "model_config": config.to_dict(),
                    "generation_config": generation_config.to_dict(),
                    "model_paths": model_paths,
                    "n_examples": len(examples),
                    "raw_args": args_dict,
                },
                f,
            )
        print("saved.")

        jax_distributed_barrier()


if __name__ == "__main__":
    tyro.cli(main)
