"""
RL Trainer class. Supports PPO-style training with RLOO advantages and KL regularization.
"""

import copy
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from flax.training.train_state import TrainState
from optax import softmax_cross_entropy_with_integer_labels
import optax

from dataset import DataProto
from utils import global_norm, float_to_dtype
from scalax.sharding import MeshShardingHelper


class Trainer:
    """
    Modular trainer for RLHF with RLOO advantages and KL regularization.
    """
    
    def __init__(self, 
                 model,
                 train_state: TrainState,
                 reference_params: PyTree,
                 tokenizer,
                 config: Dict[str, Any],
                 mesh: MeshShardingHelper,
                 max_input_length: int,
                 max_output_length: int,
                 train_bsize: int,
                 pad_token_id: int = 128002,
                 kl_coef: float = 0.0,
                 **kwargs):
        """
        Initialize the trainer.
        
        Args:
            model: The Flax model for training
            train_state: TrainState containing params, optimizer, etc.
            reference_params: Reference model parameters for KL computation
            tokenizer: Tokenizer for encoding/decoding
            config: Model configuration
            mesh: JAX mesh for sharding
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            train_bsize: Training batch size
            pad_token_id: Padding token ID
            kl_coef: KL regularization coefficient
        """
        self.model = model
        self.train_state = train_state
        self.reference_params = reference_params
        self.tokenizer = tokenizer
        self.config = config
        self.mesh = mesh
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.train_bsize = train_bsize
        self.pad_token_id = pad_token_id
        self.kl_coef = kl_coef
        
        # Set up sharded computation functions
        self._setup_sharded_functions()
    
    def _setup_sharded_functions(self):
        """Setup sharded JAX functions for training"""
        
        # Get sharding rules from config
        train_intermediate_sharding_rules = self.config.get_intermediate_sharding_rules(
            data_axis=('replica', 'fsdp'),
            sequence_axis='sequence',
        )
        
        from jax.sharding import PartitionSpec as PS
        from scalax.sharding import TreePathShardingRule
        
        train_params_sharding_rules = TreePathShardingRule(*self.config.get_partition_rules(
            model_all_gather_axis=('fsdp', 'sequence'),
        ))
        
        @partial(
            self.mesh.sjit,
            in_shardings=(
                train_params_sharding_rules,
                PS(),
                PS(),
                PS(),
                PS(),
            ),
            out_shardings=PS(),
            args_sharding_constraint=(
                train_params_sharding_rules,
                PS(('replica', 'fsdp')),
                PS(('replica', 'fsdp')),
                PS(('replica', 'fsdp')),
                PS(('replica', 'fsdp')),
            ),
        )
        def _get_logprobs(params, input_tokens, input_attention_mask, target_tokens, target_attention_mask):
            return self._compute_logprobs_impl(
                params, input_tokens, input_attention_mask, target_tokens, target_attention_mask
            )
        
        @partial(
            self.mesh.sjit,
            in_shardings=(train_params_sharding_rules, PS(), PS()),
            out_shardings=(train_params_sharding_rules, PS()),
            args_sharding_constraint=(train_params_sharding_rules, None, PS(('replica', 'fsdp'))),
            donate_argnums=(0,),
            annotation_shardings=train_intermediate_sharding_rules,
        )
        def _train_step(train_state, rng, batch):
            return self._train_step_impl(train_state, rng, batch)
        
        self._get_logprobs_fn = _get_logprobs
        self._train_step_fn = _train_step
    
    def _compute_logprobs_impl(self, params, input_tokens, input_attention_mask, target_tokens, target_attention_mask):
        """Compute log probabilities for target tokens given input context"""
        full_tokens = jnp.concatenate([input_tokens, target_tokens], axis=1)
        full_attention_mask = jnp.concatenate([input_attention_mask, target_attention_mask], axis=1)
        full_position_ids = jnp.maximum(jnp.cumsum(full_attention_mask, axis=1) - 1, 0)
        
        logits = self.model(
            full_tokens[:, :-1],
            full_attention_mask[:, :-1],
            full_position_ids[:, :-1],
            params=params,
            train=False,
        ).logits
        
        logits = logits[:, input_tokens.shape[1]-1:]
        logprobs = -softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), 
            target_tokens.astype(jnp.int32)
        )
        logprobs = MeshShardingHelper.with_sharding_constraint(
            logprobs, 
            jax.sharding.PartitionSpec(('replica', 'fsdp'), None)
        )
        return logprobs
    
    def compute_logits(self, batch: DataProto) -> jnp.ndarray:
        """
        Compute logits for a batch of data.
        
        Args:
            batch: DataProto containing input_ids, attention_mask, position_ids
            
        Returns:
            Logits array of shape (batch_size, seq_len, vocab_size)
        """
        logits = self.model(
            input_ids=batch.batch['input_ids'],
            attention_mask=batch.batch['attention_mask'],
            position_ids=batch.batch['position_ids'],
            params=self.train_state.params,
            train=True,
        ).logits
        return logits
    
    def loss(self, batch: DataProto, rng: jax.Array) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Compute RLOO loss with KL regularization.
        
        Args:
            batch: DataProto containing training data
            rng: Random key for dropout
            
        Returns:
            Tuple of (loss, auxiliary_info)
        """
        def loss_fn(params):
            logits = self.model(
                input_ids=batch.batch['input_ids'],
                attention_mask=batch.batch['attention_mask'],
                position_ids=batch.batch['position_ids'],
                params=params,
                dropout_rng=rng,
                train=True,
            ).logits
            
            logits = logits.astype(jnp.float32)
            token_loss = softmax_cross_entropy_with_integer_labels(logits, batch.batch['target_ids'])
            
            # Compute importance weights for RLOO
            log_ratio = jnp.exp((-token_loss) - jax.lax.stop_gradient(-token_loss))
            weighted_log_ratio = log_ratio * batch.batch['loss_weights']
            
            # REINFORCE loss
            reinforce_loss = jnp.mean(-weighted_log_ratio, where=batch.batch['loss_masks'] > 0.0)
            
            # KL divergence regularization
            ref_log_ratio = batch.batch['reference_logprobs'] + token_loss
            kl_loss = jnp.exp(ref_log_ratio) - 1 - ref_log_ratio
            kl_loss = jnp.mean(kl_loss, where=batch.batch['loss_masks'] > 0.0)
            
            total_loss = reinforce_loss + self.kl_coef * kl_loss
            
            return total_loss, {
                'reinforce_loss': reinforce_loss,
                'kl_loss': kl_loss,
            }
        
        return loss_fn(self.train_state.params)
    
    def _train_step_impl(self, train_state: TrainState, rng: jax.Array, batch_dict: Dict[str, jnp.ndarray]) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Implementation of a single training step"""
        # Convert dict to DataProto for consistency
        batch = DataProto(batch=batch_dict)
        
        def loss_fn(params):
            logits = self.model(
                input_ids=batch.batch['input_ids'],
                attention_mask=batch.batch['attention_mask'],
                position_ids=batch.batch['position_ids'],
                params=params,
                dropout_rng=rng,
                train=True,
            ).logits
            
            logits = logits.astype(jnp.float32)
            token_loss = softmax_cross_entropy_with_integer_labels(logits, batch.batch['target_ids'])
            
            log_ratio = jnp.exp((-token_loss) - jax.lax.stop_gradient(-token_loss))
            weighted_log_ratio = log_ratio * batch.batch['loss_weights']
            
            reinforce_loss = jnp.mean(-weighted_log_ratio, where=batch.batch['loss_masks'] > 0.0)
            
            ref_log_ratio = batch.batch['reference_logprobs'] + token_loss
            kl_loss = jnp.exp(ref_log_ratio) - 1 - ref_log_ratio
            kl_loss = jnp.mean(kl_loss, where=batch.batch['loss_masks'] > 0.0)
            
            total_loss = reinforce_loss + self.kl_coef * kl_loss
            
            return total_loss, {
                'reinforce_loss': reinforce_loss,
                'kl_loss': kl_loss,
            }
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_val, aux), grads = grad_fn(train_state.params)
        
        train_state = train_state.apply_gradients(grads=grads)
        
        metrics = {
            'loss': loss_val,
            'reinforce_loss': aux['reinforce_loss'],
            'kl_loss': aux['kl_loss'],
            'gradient_norm': global_norm(grads),
            'param_norm': global_norm(train_state.params),
        }
        
        return train_state, metrics
    
    def train_step(self, batch: DataProto, rng: jax.Array) -> Dict[str, jnp.ndarray]:
        """
        Perform a single training step.
        
        Args:
            batch: DataProto containing training data
            rng: Random key for dropout
            
        Returns:
            Dictionary of training metrics
        """
        self.train_state, metrics = self._train_step_fn(self.train_state, rng, batch.batch)
        return metrics
    
    def compute_reference_logprobs(self, 
                                   input_tokens: jnp.ndarray,
                                   input_attention_mask: jnp.ndarray,
                                   target_tokens: jnp.ndarray,
                                   target_attention_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Compute reference model log probabilities for KL regularization.
        
        Args:
            input_tokens: Input token IDs
            input_attention_mask: Input attention mask
            target_tokens: Target token IDs  
            target_attention_mask: Target attention mask
            
        Returns:
            Reference log probabilities
        """
        return self._get_logprobs_fn(
            self.reference_params,
            input_tokens,
            input_attention_mask,
            target_tokens,
            target_attention_mask,
        )
    
    def compute_rloo_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute RLOO (Reinforcement Learning with Leave-One-Out) advantages.
        
        Args:
            rewards: Array of rewards for each sample in the group
            
        Returns:
            RLOO advantages (standardized rewards)
        """
        advantages = (rewards - rewards.mean()) / np.clip(rewards.std(), 1e-8, None)
        return advantages
    
    def prepare_training_batch(self, 
                               examples: List[Dict[str, Any]],
                               samples: List[List[Dict[str, np.ndarray]]],
                               rewards: np.ndarray,
                               reference_logprobs_bsize: int) -> DataProto:
        """
        Prepare training batch from samples and rewards.
        
        Args:
            examples: List of example prompts and answers
            samples: Generated samples for each example
            rewards: Rewards for each sample
            reference_logprobs_bsize: Batch size for computing reference logprobs
            
        Returns:
            DataProto ready for training
        """
        # Prepare data for reference logprob computation
        batch_items = []
        for i, example in enumerate(examples):
            prompt_tokens = self.tokenizer.encode(example['prompt'], add_special_tokens=True)[-self.max_input_length:]
            prompt_attention_mask = [0]*(self.max_input_length - len(prompt_tokens)) + [1]*len(prompt_tokens)
            prompt_tokens = [self.pad_token_id] * (self.max_input_length - len(prompt_tokens)) + prompt_tokens
            
            for sample in samples[i]:
                answer_tokens = sample['tokens'][:self.max_output_length]
                answer_attention_mask = [1] * len(answer_tokens) + [0] * (self.max_output_length - len(answer_tokens))
                answer_tokens = answer_tokens + [self.pad_token_id] * (self.max_output_length - len(answer_tokens))
                answer_logprobs = sample['logprobs'][:self.max_output_length]
                answer_logprobs = answer_logprobs + [0] * (self.max_output_length - len(answer_logprobs))
                
                batch_items.append({
                    'prompt_tokens': np.asarray(prompt_tokens)[None],
                    'prompt_attention_mask': np.asarray(prompt_attention_mask)[None],
                    'answer_tokens': np.asarray(answer_tokens)[None],
                    'answer_attention_mask': np.asarray(answer_attention_mask)[None],
                    'answer_logprobs': np.asarray(answer_logprobs)[None],
                })
        
        # Pad if necessary
        true_batch_items_len = len(batch_items)
        if true_batch_items_len % reference_logprobs_bsize != 0:
            pad_size = reference_logprobs_bsize - (true_batch_items_len % reference_logprobs_bsize)
            for _ in range(pad_size):
                batch_items.append({
                    'prompt_tokens': np.full((1, self.max_input_length), self.pad_token_id, dtype=np.int32),
                    'prompt_attention_mask': np.zeros((1, self.max_input_length), dtype=np.int32),
                    'answer_tokens': np.full((1, self.max_output_length), self.pad_token_id, dtype=np.int32),
                    'answer_attention_mask': np.zeros((1, self.max_output_length), dtype=np.int32),
                    'answer_logprobs': np.zeros((1, self.max_output_length), dtype=np.float32),
                })
        
        # Compute reference logprobs in batches
        all_reference_logprobs = []
        all_logprobs = []
        output_masks = []
        output_tokens = []
        prompt_tokens = []
        prompt_masks = []
        
        for i in range(0, len(batch_items), reference_logprobs_bsize):
            curr_batch = batch_items[i:(i+reference_logprobs_bsize)]
            curr_batch = {
                k: np.concatenate([item[k] for item in curr_batch], axis=0)
                for k in curr_batch[0].keys()
            }
            
            reference_logprobs = np.asarray(self.compute_reference_logprobs(
                curr_batch['prompt_tokens'],
                curr_batch['prompt_attention_mask'],
                curr_batch['answer_tokens'],
                curr_batch['answer_attention_mask'],
            ))
            
            # Handle the last batch which might be padded
            if (i // reference_logprobs_bsize) == (len(batch_items) // reference_logprobs_bsize) - 1:
                true_batch_size = true_batch_items_len % reference_logprobs_bsize
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
            all_rloo_advantages.append(self.compute_rloo_advantages(rewards_group))
        all_rloo_advantages = np.concatenate(all_rloo_advantages, axis=0)
        
        # Compute returns (advantages repeated for each token)
        all_returns = np.repeat(all_rloo_advantages[..., None], output_masks.shape[1], axis=1)
        
        # Prepare final training data
        return self._prepare_rloo_examples({
            'returns': all_returns,
            'policy_logprobs': all_logprobs,
            'reference_logprobs': all_reference_logprobs,
            'prompt_tokens': prompt_tokens,
            'prompt_masks': prompt_masks,
            'output_tokens': output_tokens,
            'output_masks': output_masks,
        })
    
    def _prepare_rloo_examples(self, examples: Dict[str, np.ndarray]) -> DataProto:
        """Convert processed examples into training format"""
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
        
        batch_dict = {
            'input_ids': jnp.array(input_tokens),
            'attention_mask': jnp.array(input_attention_mask),
            'position_ids': jnp.array(position_ids),
            'target_ids': jnp.array(target_tokens),
            'loss_masks': jnp.array(loss_masks),
            'loss_weights': jnp.array(loss_weights),
            'reference_logprobs': jnp.array(reference_logprobs),
        }
        
        return DataProto(batch=batch_dict)
    
    def get_params(self) -> PyTree:
        """Get current model parameters"""
        return self.train_state.params
    
    def get_step(self) -> int:
        """Get current training step"""
        return int(self.train_state.step)
    
    def update_reference_params(self, new_reference_params: PyTree):
        """Update reference model parameters"""
        self.reference_params = new_reference_params
