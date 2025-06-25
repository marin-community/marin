from typing import Optional, List, Dict, Any, Union, NamedTuple, Tuple, Iterator, Iterable, Generator, Callable
import json
from functools import partial
import os
import tempfile
import copy
from collections import defaultdict, deque
import time
import re
import collections
import itertools
from collections import Counter
import random

import tyro
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from scalax.sharding import MeshShardingHelper, TreePathShardingRule
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from jax.sharding import PartitionSpec as PS
import numpy as np
from flax import struct
import flax.serialization
from jax.experimental.shard_map import shard_map
from optax import softmax_cross_entropy_with_integer_labels
from flax.training.train_state import TrainState
import optax
import pickle as pkl

from utils import (
    load_checkpoint, open_with_bucket,
    get_float_dtype_by_name, jax_distributed_initalize,
    jax_distributed_barrier, float_to_dtype,
    get_weight_decay_mask, global_norm, WandbLogger,
    delete_with_bucket, save_checkpoint, load_attention_kernel_config,
    validate_format, checkpointer,
)
from llama3 import (
    LLaMAConfig, FlaxLLaMAForCausalLM,
    LLAMA_STANDARD_CONFIGS,
)
from inference import build_sampler, GenerationConfig, batch_inference
from optimizer import load_adamw_optimizer
from environments.math_env import MathEnv
from environments.marin_env import MarinEnv


class Trainer:
    """RL trainer"""

    def __init__(
        self,
        config: Dict[str, Any],
        mesh: MeshShardingHelper,
        models: Dict[str, FlaxLLaMAForCausalLM],
        tokenizer: AutoTokenizer,
        environment: MarinEnv,
        logger: WandbLogger,
    ):
        self.config = config
        self.mesh = mesh
        self.models = models
        self.tokenizer = tokenizer
        self.environment = environment
        self.logger = logger

        # Extract frequently used config values
        self.max_input_length = config['max_input_length']
        self.max_output_length = config['max_output_length']
        self.train_bsize = config['train_bsize']
        self.reference_logprobs_bsize = config['reference_logprobs_bsize']
        self.pad_token_id = config['pad_token_id']
        self.kl_coef = config['kl_coef']

        # Setup training components
        self._setup_optimizer()
        self._setup_models()
        self._setup_samplers()
        self._compile_functions()

    def _setup_optimizer(self):
        """Setup optimizer configuration."""
        optim_config = self.config['optim_config']
        if optim_config.startswith('adamw:'):
            optim_config = json.loads(optim_config[len('adamw:'):])
            optim_config['weight_decay_mask'] = get_weight_decay_mask(optim_config.pop('weight_decay_exclusions', tuple()))
            self.grad_accum_steps = optim_config.pop('grad_accum_steps', 1)
            optimizer, self.optimizer_info = load_adamw_optimizer(**optim_config)
        else:
            raise ValueError(f'Unknown optimizer config: {optim_config}')

        if self.grad_accum_steps > 1:
            optimizer = optax.MultiSteps(optimizer, self.grad_accum_steps)

        self.optimizer = optimizer

    def _setup_models(self):
        """Setup model configurations and sharding."""
        self.train_model = self.models['train']
        self.prefill_model = self.models['prefill']
        self.generate_model = self.models['generate']

        # Setup sharding rules
        config = self.train_model.config
        self.train_params_sharding_rules = TreePathShardingRule(*config.get_partition_rules(
            model_all_gather_axis=('fsdp', 'sequence'),
        ))
        self.inference_params_sharding_rules = TreePathShardingRule(*config.get_partition_rules(
            model_all_gather_axis=None,
        ))
        self.train_intermediate_sharding_rules = config.get_intermediate_sharding_rules(
            data_axis=('replica', 'fsdp'),
            sequence_axis='sequence',
        )
        self.inference_intermediate_sharding_rules = config.get_intermediate_sharding_rules(
            data_axis=('replica', 'fsdp'),
            sequence_axis=None,
        )

    def _setup_samplers(self):
        """Setup sampling configurations."""
        generation_config = GenerationConfig(**self.config['generation_config'])
        test_generation_config = GenerationConfig(**self.config['test_generation_config'])

        sampler_kwargs = {
            'prefill_model': self.prefill_model,
            'generate_model': self.generate_model,
            'tokenizer': self.tokenizer,
            'bsize': self.config['decode_bsize'],
            'prefill_bsize': self.config['prefill_bsize'],
            'max_input_length': self.max_input_length,
            'params_sharding_rules': self.inference_params_sharding_rules,
            'intermediate_sharding_rules': self.inference_intermediate_sharding_rules,
            'replica_axis_name': ('replica', 'fsdp'),
            'tp_axis_name': 'tensor',
            'mesh': self.mesh,
            'pad_token_id': self.pad_token_id,
        }

        self.sampler = build_sampler(generation_config=generation_config, **sampler_kwargs)
        self.test_sampler = build_sampler(generation_config=test_generation_config, **sampler_kwargs)

    def _compile_functions(self):
        """Compile JAX functions for training."""
        @partial(
            self.mesh.sjit,
            in_shardings=(self.train_params_sharding_rules,),
            out_shardings=self.train_params_sharding_rules,
            annotation_shardings=self.train_intermediate_sharding_rules,
        )
        def create_train_state_from_params(params):
            return TrainState.create(params=params, tx=self.optimizer, apply_fn=None)

        @partial(
            self.mesh.sjit,
            in_shardings=(PS(),),
            out_shardings=self.train_params_sharding_rules,
            annotation_shardings=self.train_intermediate_sharding_rules,
        )
        def init_fn(rng):
            params = self.train_model.init_weights(rng, (self.train_bsize, self.max_input_length+self.max_output_length-1))
            return create_train_state_from_params(params)

        @partial(
            self.mesh.sjit,
            in_shardings=(self.train_params_sharding_rules,),
            out_shardings=self.inference_params_sharding_rules,
            args_sharding_constraint=(self.train_params_sharding_rules,),
        )
        def reshard_params(params):
            params = float_to_dtype(params, self.config['inference_param_dtype'])
            return params

        @partial(
            self.mesh.sjit,
            in_shardings=(
                self.train_params_sharding_rules,
                PS(),
                PS(),
                PS(),
                PS(),
            ),
            out_shardings=PS(),
            args_sharding_constraint=(
                self.train_params_sharding_rules,
                PS(('replica', 'fsdp')),
                PS(('replica', 'fsdp')),
                PS(('replica', 'fsdp')),
                PS(('replica', 'fsdp')),
            ),
        )
        def get_logprobs(
            params,
            input_tokens,
            input_attention_mask,
            target_tokens,
            target_attention_mask,
        ):
            full_tokens = jnp.concatenate([input_tokens, target_tokens], axis=1)
            full_attention_mask = jnp.concatenate([input_attention_mask, target_attention_mask], axis=1)
            full_position_ids = jnp.maximum(jnp.cumsum(full_attention_mask, axis=1) - 1, 0)
            logits = self.train_model(
                full_tokens[:, :-1],
                full_attention_mask[:, :-1],
                full_position_ids[:, :-1],
                params=params,
                train=False,
            ).logits
            logits = logits[:, input_tokens.shape[1]-1:]
            logprobs = -softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), target_tokens.astype(jnp.int32))
            logprobs = MeshShardingHelper.with_sharding_constraint(logprobs, PS(('replica', 'fsdp'), None))
            return logprobs

        @partial(
            self.mesh.sjit,
            in_shardings=(self.train_params_sharding_rules, PS(), PS()),
            out_shardings=(self.train_params_sharding_rules, PS()),
            args_sharding_constraint=(self.train_params_sharding_rules, None, PS(('replica', 'fsdp'))),
            donate_argnums=(0,),
            annotation_shardings=self.train_intermediate_sharding_rules,
        )
        def train_step(train_state, rng, batch):
            def loss(params):
                logits = self.train_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    position_ids=batch['position_ids'],
                    params=params,
                    dropout_rng=rng,
                    train=True,
                ).logits
                logits = logits.astype(jnp.float32)
                token_loss = softmax_cross_entropy_with_integer_labels(logits, batch['target_ids'])
                log_ratio = jnp.exp((-token_loss) - jax.lax.stop_gradient(-token_loss))
                weighted_log_ratio = log_ratio * batch['loss_weights']
                reinforce_loss = jnp.mean(-weighted_log_ratio, where=batch['loss_masks'] > 0.0)
                ref_log_ratio = batch['reference_logprobs'] + token_loss
                kl_loss = jnp.exp(ref_log_ratio) - 1 - ref_log_ratio
                kl_loss = jnp.mean(kl_loss, where=batch['loss_masks'] > 0.0)
                loss = reinforce_loss + self.kl_coef * kl_loss
                return loss, {
                    'reinforce_loss': reinforce_loss,
                    'kl_loss': kl_loss,
                }

            grad_fn = jax.value_and_grad(loss, has_aux=True)
            (l, aux), grads = grad_fn(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            metrics = dict(
                loss=l,
                reinforce_loss=aux['reinforce_loss'],
                kl_loss=aux['kl_loss'],
                learning_rate=self.optimizer_info['learning_rate_schedule'](train_state.step),
                gradient_norm=global_norm(grads),
                param_norm=global_norm(train_state.params),
            )
            return train_state, metrics

        # Store compiled functions
        self.create_train_state_from_params = create_train_state_from_params
        self.init_fn = init_fn
        self.reshard_params = reshard_params
        self.get_logprobs = get_logprobs
        self.train_step = train_step

    def compute_rloo_advantages_for_group(self, rewards: np.ndarray) -> np.ndarray:
        """Compute RLOO advantages for a group of rewards."""
        advantages = (rewards - rewards.mean()) / np.clip(rewards.std(), 1e-8, None)
        return advantages

    def prepare_data_from_env_step(self, env_step) -> Dict[str, np.ndarray]:
        """Prepare training data from environment step."""
        examples = env_step.examples
        samples = env_step.samples
        rewards = env_step.rewards

        # Prepare data to compute reference logprobs
        batch_items = []
        for i, example in enumerate(examples):
            # Use pre-tokenized data instead of tokenizing again
            prompt_tokens = example['prompt_tokens']
            prompt_attention_mask = example['prompt_attention_mask']
            
            for sample in samples[i]:
                answer_tokens = sample['tokens'][:self.max_output_length]
                answer_attention_mask = [1] * len(answer_tokens) + [0] * (self.max_output_length - len(answer_tokens))
                answer_tokens = answer_tokens + [self.pad_token_id] * (self.max_output_length - len(answer_tokens))
                answer_logprobs = sample['logprobs'][:self.max_output_length]
                answer_logprobs = answer_logprobs + [0] * (self.max_output_length - len(answer_logprobs))
                batch_items.append({
                    'prompt_tokens': prompt_tokens[None],
                    'prompt_attention_mask': prompt_attention_mask[None],
                    'answer_tokens': np.asarray(answer_tokens)[None],
                    'answer_attention_mask': np.asarray(answer_attention_mask)[None],
                    'answer_logprobs': np.asarray(answer_logprobs)[None],
                })

        true_batch_items_len = len(batch_items)
        if true_batch_items_len % self.reference_logprobs_bsize != 0:
            for _ in range(self.reference_logprobs_bsize - (true_batch_items_len % self.reference_logprobs_bsize)):
                batch_items.append({
                    'prompt_tokens': np.full((1, self.max_input_length), self.pad_token_id, dtype=np.int32),
                    'prompt_attention_mask': np.zeros((1, self.max_input_length), dtype=np.int32),
                    'answer_tokens': np.full((1, self.max_output_length), self.pad_token_id, dtype=np.int32),
                    'answer_attention_mask': np.zeros((1, self.max_output_length), dtype=np.int32),
                    'answer_logprobs': np.zeros((1, self.max_output_length), dtype=np.float32),
                })

        # Compute reference logprobs
        all_reference_logprobs, all_logprobs = [], []
        prompt_tokens, prompt_masks = [], []
        output_tokens, output_masks = [], []
        for i in tqdm(range(0, len(batch_items), self.reference_logprobs_bsize)):
            curr_batch = batch_items[i:(i+self.reference_logprobs_bsize)]
            curr_batch = {
                k: np.concatenate([item[k] for item in curr_batch], axis=0)
                for k in curr_batch[0].keys()
            }
            reference_logprobs = np.asarray(self.get_logprobs(
                self.reference_params,
                curr_batch['prompt_tokens'],
                curr_batch['prompt_attention_mask'],
                curr_batch['answer_tokens'],
                curr_batch['answer_attention_mask'],
            ))
            if (i // self.reference_logprobs_bsize) == (len(batch_items) // self.reference_logprobs_bsize) - 1:
                true_batch_size = true_batch_items_len % self.reference_logprobs_bsize
                if true_batch_size == 0:
                    true_batch_size = reference_logprobs.shape[0]
            else:
                true_batch_size = reference_logprobs.shape[0]
            for x in range(true_batch_size):
                all_reference_logprobs.append(reference_logprobs[x])
                all_logprobs.append(curr_batch['answer_logprobs'][x])
                output_masks.append(curr_batch['answer_attention_mask'][x])
                output_tokens.append(curr_batch['answer_tokens'][x])
                prompt_tokens.append(curr_batch['prompt_tokens'][x])
                prompt_masks.append(curr_batch['prompt_attention_mask'][x])

        all_reference_logprobs = np.stack(all_reference_logprobs, axis=0)
        all_logprobs = np.stack(all_logprobs, axis=0)
        output_masks = np.stack(output_masks, axis=0)
        output_tokens = np.stack(output_tokens, axis=0)
        prompt_tokens = np.stack(prompt_tokens, axis=0)
        prompt_masks = np.stack(prompt_masks, axis=0)

        # Compute RLOO advantages
        all_rloo_advantages = []
        for rewards_group in rewards:
            all_rloo_advantages.append(self.compute_rloo_advantages_for_group(rewards_group))
        all_rloo_advantages = np.concatenate(all_rloo_advantages, axis=0)

        # Compute returns
        all_returns = jnp.repeat(all_rloo_advantages[..., None], output_masks.shape[1], axis=1)

        return {
            'returns': all_returns,
            'policy_logprobs': all_logprobs,
            'reference_logprobs': all_reference_logprobs,
            'prompt_tokens': prompt_tokens,
            'prompt_masks': prompt_masks,
            'output_tokens': output_tokens,
            'output_masks': output_masks,
        }

    def prepare_data_from_environment(self, params, prng_key):
        """Prepare training data using environment."""
        inference_params = self.reshard_params(params)

        # Get environment step
        env_step = self.environment.step(
            sampler=self.sampler,
            params=inference_params,
            n_examples=self.config['n_prompts_per_step'],
            prng_key=prng_key,
            mode="train",
            n_generations=self.config['generation_config']['n_generations'],
        )

        del inference_params

        # Prepare training data from environment step
        dataset = self.prepare_data_from_env_step(env_step)
        return dataset, env_step.metrics

    def evaluate_data_from_environment(self, params, prng_key):
        """Evaluate model using environment."""
        inference_params = self.reshard_params(params)

        # Get evaluation examples from environment
        eval_examples = self.environment.get_eval_examples(self.config['num_eval_examples'])

        # Generate samples for evaluation
        samples = batch_inference(
            self.test_sampler,
            inference_params,
            [example['prompt'] for example in eval_examples],
            prng_key,
            self.config['test_generation_config']['n_generations'],
            verbose=True,
        )
        del inference_params

        # Compute rewards using environment's reward computation
        _, metrics = self.environment._compute_rewards(eval_examples, samples)

        # Rename metrics for evaluation
        eval_metrics = {}
        for k, v in metrics.items():
            eval_metrics[k.replace('train_', 'test_')] = v
        return eval_metrics

    def prepare_training_data_iterable(
        self,
        data_items: Dict[str, np.ndarray],
        bsize: int,
        shuffle: bool = True,
        loop: bool = True
    ) -> Iterator[Dict[str, np.ndarray]]:
        """Create an iterable over processed training data - moved from Dataset class."""
        N = data_items['returns'].shape[0]
        rng = jax.random.PRNGKey(0)
        
        while True:
            with jax.default_device(jax.devices('cpu')[0]):
                idxs = []
                for _ in range((bsize + (N - 1)) // N):
                    if shuffle:
                        rng, subrng = jax.random.split(rng)
                        curr_idxs = jax.random.permutation(subrng, np.arange(N))
                        idxs.extend(curr_idxs.tolist())
                    else:
                        curr_idxs = np.arange(N)
                        idxs.extend(curr_idxs.tolist())
                idxs = np.asarray(idxs)

                for batch_idx in range(len(idxs) // bsize):
                    batch_idxs = idxs[batch_idx*bsize:(batch_idx+1)*bsize]
                    batch_examples = {
                        k: np.asarray([data_items[k][idx] for idx in batch_idxs])
                        for k in data_items.keys()
                    }
                    batch = self._prepare_rloo_examples(batch_examples)
                    yield batch

                if not loop:
                    break
    
    def _prepare_rloo_examples(self, examples: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Prepare examples for RLOO training - moved from Dataset class."""
        full_tokens = np.concatenate((
            examples['prompt_tokens'],
            examples['output_tokens'],
        ), axis=1)
        full_attention_mask = np.concatenate((
            examples['prompt_masks'],
            examples['output_masks'],
        ), axis=1)
        full_position_ids = np.maximum(
            np.cumsum(full_attention_mask, axis=1) - 1,
            0,
        )
        input_tokens = full_tokens[:, :-1]
        input_attention_mask = full_attention_mask[:, :-1]
        target_tokens = full_tokens[:, 1:]
        position_ids = full_position_ids[:, :-1]
        loss_masks = np.concatenate([
            np.zeros((
                examples['prompt_masks'].shape[0],
                examples['prompt_masks'].shape[1]-1,
            ), dtype=np.float32),
            examples['output_masks'].astype(np.float32),
        ], axis=1)
        loss_weights = np.concatenate([
            np.zeros((
                examples['prompt_masks'].shape[0],
                examples['prompt_masks'].shape[1]-1,
            ), dtype=np.float32),
            examples['returns'].astype(np.float32),
        ], axis=1)
        reference_logprobs = np.concatenate([
            np.zeros((
                examples['prompt_masks'].shape[0],
                examples['prompt_masks'].shape[1]-1,
            ), dtype=np.float32),
            examples['reference_logprobs'].astype(np.float32),
        ], axis=1)
        return {
            'input_ids': input_tokens,
            'attention_mask': input_attention_mask,
            'position_ids': position_ids,
            'target_ids': target_tokens,
            'loss_masks': loss_masks,
            'loss_weights': loss_weights,
            'reference_logprobs': reference_logprobs,
        }

    def save_checkpoint(self, train_state, step):
        """Save model checkpoint."""
        if (self.config['max_checkpoints'] is not None) and (len(self.checkpoint_queue) >= self.config['max_checkpoints']):
            old_step = self.checkpoint_queue.popleft()
            if self.logger.can_save():
                old_path = os.path.join(self.logger.output_dir, 'checkpoints', f'step_{old_step}')
                delete_with_bucket(old_path, recursive=True)

        if self.logger.can_save():
            print(f'saving checkpoint at step {step} ...')

            metadata = dict(
                step=step,
                args_dict=self.config,
            )

            checkpointer(
                path=os.path.join(self.logger.output_dir, 'checkpoints', f'step_{step}'),
                train_state=train_state,
                config=self.train_model.config.to_dict(),
                gather_fns=self.train_state_gather_fns,
                metadata=metadata,
                active=self.logger.can_save(),
                **self.config.get('checkpointer_config', {}),
            )

            self.checkpoint_queue.append(step)
            print('saved.')

    def train(self):
        """Main training loop."""
        # Initialize training state shapes and sharding functions
        train_state_shape = jax.eval_shape(lambda: self.init_fn(jax.random.PRNGKey(0)))
        inference_params_shape = jax.eval_shape(lambda: self.prefill_model.init_weights(jax.random.PRNGKey(0), (self.config['decode_bsize'], self.max_input_length+self.max_output_length-1)))
        self.train_state_shard_fns, self.train_state_gather_fns = self.mesh.make_shard_and_gather_fns(train_state_shape, self.train_params_sharding_rules)
        inference_param_shard_fns, inference_param_gather_fns = self.mesh.make_shard_and_gather_fns(inference_params_shape, self.inference_params_sharding_rules)

        # Initialize training state
        if 'params' in self.config['model_paths']:
            train_state = self.create_train_state_from_params(load_checkpoint(
                self.config['model_paths']['params'],
                shard_fns=self.train_state_shard_fns.params,
                remove_dict_prefix=self.config['model_paths']['remove_dict_prefix'],
                convert_to_dtypes=jax.tree_util.tree_map(lambda x: self.config['training_param_dtype'], train_state_shape.params),
            ))
        elif 'train_state' in self.config['model_paths']:
            train_state = load_checkpoint(
                self.config['model_paths']['train_state'],
                shard_fns=self.train_state_shard_fns,
                remove_dict_prefix=self.config['model_paths']['remove_dict_prefix'],
                convert_to_dtypes=jax.tree_util.tree_map(lambda x: self.config['training_param_dtype'], train_state_shape),
            )
        else:
            print('WARNING: no params path provided, initializing with random weights...')
            train_state = self.init_fn(jax.random.PRNGKey(0))

        self.reference_params = float_to_dtype(train_state.params, self.config['inference_param_dtype'])
        self.checkpoint_queue = deque()

        if self.config.get('save_initial_checkpoint', False):
            self.save_checkpoint(train_state, 0)

        # Training loop
        rng = jax.random.PRNGKey(0)

        for step in tqdm(range(self.config['num_train_steps']), total=self.config['num_train_steps']):
            rng, subrng = jax.random.split(rng)

            # Get training data from environment
            generated_data, dataset_metrics = self.prepare_data_from_environment(
                train_state.params,
                subrng,
            )

            # Use the training data directly without creating Dataset object
            generated_data_iterable = self.prepare_training_data_iterable(
                generated_data, 
                self.train_bsize, 
                shuffle=True, 
                loop=False
            )

            for generated_train_sub_batch in tqdm(generated_data_iterable):
                train_state, metrics = self.train_step(train_state, subrng, generated_train_sub_batch)

            if self.config['log_freq'] > 0 and ((step+1) % self.config['log_freq'] == 0 or (self.config.get('log_initial_step', True) and step == 0)):
                if self.config['num_eval_examples'] > 0:
                    rng, subrng = jax.random.split(rng)
                    metrics.update(self.evaluate_data_from_environment(train_state.params, subrng))

                log_metrics = {"step": step+1}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                self.logger.log(log_metrics)
                print(log_metrics)

            if self.config['save_model_freq'] > 0 and (step+1) % self.config['save_model_freq'] == 0:
                self.save_checkpoint(train_state, step+1)

        if self.config['save_model_freq'] > 0 and (self.config['num_train_steps'] not in self.checkpoint_queue):
            self.save_checkpoint(train_state, self.config['num_train_steps'])


def main(
    load_model: str,
    output_dir: Optional[str],
    sharding: str,
    num_train_steps: int,
    max_input_length: int,
    max_output_length: int,
    train_bsize: int,
    decode_bsize: int,
    prefill_bsize: int,
    reference_logprobs_bsize: int,
    n_prompts_per_step: int,
    log_freq: int,
    num_eval_examples: int,
    save_model_freq: int,
    wandb_project: str,
    inference_param_dtype: str='bf16',
    inference_activation_dtype: str='bf16',
    training_param_dtype: str='fp32',
    training_activation_dtype: str='fp32',
    optim_config: str='adamw:{}',
    logger_config: str='{}',
    checkpointer_config: str='{}',
    generation_config: str='{}',
    test_generation_config: str='{}',
    model_config_override: str='{}',
    tokenizer_override: str='{}',
    train_attention_kernel_config: str='splash:{}',
    prefill_attention_kernel_config: str='splash:{}',
    generate_attention_kernel_config: str='paged:{}',
    jax_distributed_initalize_config:str='{}',
    save_initial_checkpoint: bool=False,
    log_initial_step: bool=True,
    max_checkpoints: Optional[int]=None,
    physical_axis_splitting: bool=False,
    pad_token_id: int=128002,
    kl_coef: float=0.0,
):
    """Main training script with environment."""
    # Parse configurations
    args_dict = dict(locals())
    print(args_dict)
    sharding: List[int] = list(map(lambda x: int(x.strip()), sharding.split(',')))

    # Parse dtype configurations
    inference_param_dtype = get_float_dtype_by_name(inference_param_dtype)
    inference_activation_dtype = get_float_dtype_by_name(inference_activation_dtype)
    training_param_dtype = get_float_dtype_by_name(training_param_dtype)
    training_activation_dtype = get_float_dtype_by_name(training_activation_dtype)

    # Parse JSON configurations
    logger_config: Dict[str, Any] = json.loads(logger_config)
    checkpointer_config: Dict[str, Any] = json.loads(checkpointer_config)
    generation_config: Dict[str, Any] = json.loads(generation_config)
    test_generation_config: Dict[str, Any] = json.loads(test_generation_config)
    model_config_override: Dict[str, Any] = json.loads(model_config_override)
    tokenizer_override: Dict[str, Any] = json.loads(tokenizer_override)
    jax_distributed_initalize_config: Dict[str, Any] = json.loads(jax_distributed_initalize_config)

    # Initialize JAX distributed
    jax_distributed_initalize(**jax_distributed_initalize_config)
    jax_distributed_barrier()

    # Load attention kernel configurations
    prefill_attention_kernel, prefill_attention_kernel_config = load_attention_kernel_config(prefill_attention_kernel_config, ['splash', 'default'])
    generate_attention_kernel, generate_attention_kernel_config = load_attention_kernel_config(generate_attention_kernel_config, ['paged', 'default'])
    train_attention_kernel, train_attention_kernel_config = load_attention_kernel_config(train_attention_kernel_config, ['splash', 'default', 'ring', 'ring_jax'])

    # Setup mesh
    mesh = MeshShardingHelper(sharding, ['replica', 'fsdp', 'sequence', 'tensor'], mesh_axis_splitting=physical_axis_splitting)

    with mesh.get_context():
        # Load model configuration and paths
        if load_model.startswith('paths:'):
            model_paths = json.loads(load_model[len('paths:'):])
            if 'remove_dict_prefix' not in model_paths:
                model_paths['remove_dict_prefix'] = None
        else:
            raise ValueError(f'Unknown model info type: {load_model}')

        # Load model config
        config_is_temp = False
        if 'config' in model_paths and model_paths['config'].startswith('gs://'):
            temp_file = tempfile.NamedTemporaryFile('wb', delete=False)
            with open_with_bucket(model_paths['config'], 'rb') as f:
                temp_file.write(f.read())
            temp_file.close()
            model_paths['config'] = temp_file.name
            config_is_temp = True

        if 'config' in model_paths:
            config = LLaMAConfig.from_pretrained(model_paths['config'], **model_config_override)
        elif 'default_config_name' in model_paths:
            config = LLaMAConfig(**LLAMA_STANDARD_CONFIGS[model_paths['default_config_name']], **model_config_override)
        else:
            config = LLaMAConfig(**model_config_override)

        # Create model configurations
        prefill_config = copy.deepcopy(config)
        prefill_config.attention_kernel = prefill_attention_kernel
        prefill_config.attention_kernel_settings = prefill_attention_kernel_config

        generate_config = copy.deepcopy(config)
        train_config = copy.deepcopy(config)

        if config_is_temp:
            os.remove(model_paths['config'])

        # Initialize models
        prefill_model = FlaxLLaMAForCausalLM(prefill_config, dtype=inference_activation_dtype, _do_init=False, param_dtype=inference_param_dtype, input_shape=(prefill_bsize, max_input_length))
        generate_model = FlaxLLaMAForCausalLM(generate_config, dtype=inference_activation_dtype, _do_init=False, param_dtype=inference_param_dtype, input_shape=(decode_bsize, max_input_length+max_output_length-1))
        train_model = FlaxLLaMAForCausalLM(train_config, dtype=training_activation_dtype, _do_init=False, param_dtype=training_param_dtype, input_shape=(train_bsize, max_input_length+max_output_length-1))
        generate_model.config.attention_kernel = generate_attention_kernel
        generate_model.config.attention_kernel_settings = generate_attention_kernel_config
        train_model.config.attention_kernel = train_attention_kernel
        train_model.config.attention_kernel_settings = train_attention_kernel_config

        models = {
            'prefill': prefill_model,
            'generate': generate_model,
            'train': train_model,
        }

        # Load tokenizer
        tokenizer_is_temp = False
        if model_paths['tokenizer'].startswith('gs://'):
            temp_file = tempfile.NamedTemporaryFile('wb', delete=False)
            with open_with_bucket(model_paths['tokenizer'], 'rb') as f:
                temp_file.write(f.read())
            temp_file.close()
            model_paths['tokenizer'] = temp_file.name
            tokenizer_is_temp = True

        tokenizer_kwargs = dict(
            truncation_side='right',
            padding_side='right',
            pad_token="<|reserved_special_token_0|>",
        )
        tokenizer_kwargs.update(tokenizer_override)
        tokenizer = AutoTokenizer.from_pretrained(model_paths['tokenizer'], **tokenizer_kwargs)

        if tokenizer_is_temp:
            os.remove(model_paths['tokenizer'])

        # Initialize environment with tokenization parameters
        environment = MathEnv(
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            pad_token_id=pad_token_id
        )

        # Initialize logger
        if 'enable' not in logger_config:
            logger_config['enable'] = (jax.process_index() == 0)
        if 'config_to_log' in logger_config:
            logger_config['config_to_log'].update(args_dict)
        else:
            logger_config['config_to_log'] = args_dict
        logger = WandbLogger(wandb_project, output_dir=output_dir, **logger_config)

        # Create trainer configuration
        trainer_config = {
            **args_dict,
            'model_paths': model_paths,
            'optim_config': optim_config,
            'generation_config': generation_config,
            'test_generation_config': test_generation_config,
            'checkpointer_config': checkpointer_config,
            'inference_param_dtype': inference_param_dtype,
            'inference_activation_dtype': inference_activation_dtype,
            'training_param_dtype': training_param_dtype,
            'training_activation_dtype': training_activation_dtype,
        }

        # Initialize and run trainer
        trainer = Trainer(trainer_config, mesh, models, tokenizer, environment, logger)
        trainer.train()

        jax_distributed_barrier()
        logger.finish()
        jax_distributed_barrier()


if __name__ == '__main__':
    tyro.cli(main)
