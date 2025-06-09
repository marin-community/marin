"""
PPO Trainer implementation for LLM token-level training.
"""

from typing import Dict, Tuple, Optional, Any, Callable
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
import optax
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from functools import partial

from marin.post_training.trainer.ppo.core_algos import (
    PPOParams,
    PPOTrainerConfig,
    compute_gae_advantage_return,
    compute_ppo_loss,
    AdvantageEstimator
)

@struct.dataclass
class PPOTrainerConfig:
    """Configuration for PPO trainer."""
    learning_rate: float = 3e-4
    num_epochs: int = 10
    batch_size: int = 64
    num_minibatches: int = 4
    ppo_params: PPOParams = PPOParams()
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    update_epochs: int = 4
    advantage_estimator: str = "gae"
    loss_agg_mode: str = "token-mean"

class PPOTrainer:
    """PPO Trainer class that handles token-level training."""
    
    def __init__(
        self,
        config: PPOTrainerConfig,
        policy_fn: Callable,
        value_fn: Callable,
        optimizer: Optional[optax.GradientTransformation] = None,
    ):
        """Initialize PPO trainer.
        
        Args:
            config: PPO trainer configuration
            policy_fn: Function that computes policy logits
            value_fn: Function that computes value estimates
            optimizer: Optional optimizer. If None, uses Adam with config.learning_rate
        """
        self.config = config
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        
        if optimizer is None:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=config.learning_rate)
            )
        else:
            self.optimizer = optimizer
            
        self.train_state = None
        
    def init_train_state(self, rng: jnp.ndarray, dummy_input: Any) -> None:
        """Initialize the training state.
        
        Args:
            rng: Random number generator key
            dummy_input: Dummy input to initialize the model
        """
        policy_params = self.policy_fn.init(rng, dummy_input)
        value_params = self.value_fn.init(rng, dummy_input)
        
        params = {
            'policy': policy_params,
            'value': value_params
        }
        
        self.train_state = TrainState.create(
            apply_fn=None,  # Not used
            params=params,
            tx=self.optimizer
        )
        
    @partial(jit, static_argnums=(0,))
    def _update_epoch(
        self,
        train_state: TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Update the model for one epoch.
        
        Args:
            train_state: Current training state
            batch: Dictionary containing batch data
            rng: Random number generator key
            
        Returns:
            train_state: Updated training state
            metrics: Dictionary containing metrics
        """
        def loss_fn(params):
            # Compute policy logits and value estimates
            policy_logits = self.policy_fn.apply(params['policy'], batch['input_ids'])
            values = self.value_fn.apply(params['value'], batch['input_ids'])
            
            # Compute action log probabilities
            log_probs = jax.nn.log_softmax(policy_logits)
            action_log_probs = jnp.take_along_axis(
                log_probs,
                batch['response_ids'][..., None],
                axis=-1
            ).squeeze(-1)
            
            # Compute PPO loss
            loss, metrics = compute_ppo_loss(
                self.config.ppo_params,
                action_log_probs,
                batch['old_log_probs'],
                values,
                batch['returns'],
                batch['advantages'],
                batch['response_mask'],
                batch.get('ref_log_probs'),
                self.config.loss_agg_mode
            )
            
            return loss, metrics
            
        grad_fn = grad(loss_fn, has_aux=True)
        grads, metrics = grad_fn(train_state.params)
        
        train_state = train_state.apply_gradients(grads=grads)
        
        return train_state, metrics
    
    def update(
        self,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Update the model for multiple epochs.
        
        Args:
            batch: Dictionary containing batch data
            rng: Random number generator key
            
        Returns:
            train_state: Updated training state
            metrics: Dictionary containing metrics
        """
        train_state = self.train_state
        metrics = {}
        
        for _ in range(self.config.update_epochs):
            # Shuffle the batch
            rng, shuffle_rng = jax.random.split(rng)
            batch_size = batch['input_ids'].shape[0]
            indices = jax.random.permutation(shuffle_rng, batch_size)
            shuffled_batch = jax.tree_map(
                lambda x: x[indices],
                batch
            )
            
            # Split into minibatches
            minibatch_size = batch_size // self.config.num_minibatches
            for i in range(self.config.num_minibatches):
                start_idx = i * minibatch_size
                end_idx = start_idx + minibatch_size
                minibatch = jax.tree_map(
                    lambda x: x[start_idx:end_idx],
                    shuffled_batch
                )
                
                # Update on minibatch
                train_state, minibatch_metrics = self._update_epoch(
                    train_state,
                    minibatch,
                    rng
                )
                
                # Aggregate metrics
                metrics = jax.tree_map(
                    lambda x, y: x + y / self.config.num_minibatches,
                    metrics,
                    minibatch_metrics
                )
        
        self.train_state = train_state
        return train_state, metrics
    
    def compute_advantages(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        response_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute advantages and returns.
        
        Args:
            rewards: Array of shape (batch_size, response_length)
            values: Array of shape (batch_size, response_length)
            response_mask: Array of shape (batch_size, response_length)
            
        Returns:
            advantages: Array of shape (batch_size, response_length)
            returns: Array of shape (batch_size, response_length)
        """
        if self.config.advantage_estimator == AdvantageEstimator.GAE:
            advantages, returns = compute_gae_advantage_return(
                rewards,
                values,
                response_mask,
                self.config.ppo_params.gamma,
                self.config.ppo_params.gae_lambda
            )
        else:
            raise NotImplementedError(
                f"Advantage estimator {self.config.advantage_estimator} not implemented"
            )
            
        return advantages, returns
    
    def prepare_batch(
        self,
        input_ids: jnp.ndarray,
        response_ids: jnp.ndarray,
        rewards: jnp.ndarray,
        response_mask: jnp.ndarray,
        old_log_probs: jnp.ndarray,
        ref_log_probs: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Prepare a batch for training.
        
        Args:
            input_ids: Array of shape (batch_size, input_length)
            response_ids: Array of shape (batch_size, response_length)
            rewards: Array of shape (batch_size, response_length)
            response_mask: Array of shape (batch_size, response_length)
            old_log_probs: Array of shape (batch_size, response_length)
            ref_log_probs: Optional array of shape (batch_size, response_length)
            
        Returns:
            batch: Dictionary containing prepared batch data
        """
        # Compute value estimates
        values = self.value_fn.apply(self.train_state.params['value'], input_ids)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(
            rewards, values, response_mask
        )
        
        batch = {
            'input_ids': input_ids,
            'response_ids': response_ids,
            'rewards': rewards,
            'response_mask': response_mask,
            'old_log_probs': old_log_probs,
            'values': values,
            'returns': returns,
            'advantages': advantages
        }
        
        if ref_log_probs is not None:
            batch['ref_log_probs'] = ref_log_probs
            
        return batch 