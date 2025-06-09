"""
Example script demonstrating how to use the PPO implementation with LLM tokens.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from flax import linen as nn

from marin.post_training.trainer.ppo.trainer import PPOTrainer, PPOTrainerConfig
from marin.post_training.trainer.ppo.core_algos import PPOParams

class TransformerPolicy(nn.Module):
    """Transformer-based policy network."""
    
    vocab_size: int
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    @nn.compact
    def __call__(self, x):
        # Add positional embeddings
        x = nn.Embed(self.vocab_size, self.hidden_size)(x)
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Self-attention
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads
            )(x, x)
            x = x + attn_output
            
            # Feed-forward
            x = x + nn.Dense(self.hidden_size * 4)(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_size)(x)
        
        # Output projection
        logits = nn.Dense(self.vocab_size)(x)
        return logits

class TransformerValue(nn.Module):
    """Transformer-based value network."""
    
    vocab_size: int
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    
    @nn.compact
    def __call__(self, x):
        # Add positional embeddings
        x = nn.Embed(self.vocab_size, self.hidden_size)(x)
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Self-attention
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads
            )(x, x)
            x = x + attn_output
            
            # Feed-forward
            x = x + nn.Dense(self.hidden_size * 4)(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_size)(x)
        
        # Output projection
        value = nn.Dense(1)(x)
        return value.squeeze(-1)

def create_dummy_batch(
    batch_size: int,
    input_length: int,
    response_length: int,
    vocab_size: int,
) -> Dict[str, jnp.ndarray]:
    """Create a dummy batch for testing.
    
    Args:
        batch_size: Batch size
        input_length: Input sequence length
        response_length: Response sequence length
        vocab_size: Vocabulary size
        
    Returns:
        batch: Dictionary containing dummy batch data
    """
    input_ids = jnp.zeros((batch_size, input_length), dtype=jnp.int32)
    response_ids = jnp.zeros((batch_size, response_length), dtype=jnp.int32)
    rewards = jnp.zeros((batch_size, response_length))
    response_mask = jnp.ones((batch_size, response_length))
    old_log_probs = jnp.zeros((batch_size, response_length))
    
    return {
        'input_ids': input_ids,
        'response_ids': response_ids,
        'rewards': rewards,
        'response_mask': response_mask,
        'old_log_probs': old_log_probs
    }

def main():
    # Set random seed
    rng = jax.random.PRNGKey(0)
    
    # Create dummy environment parameters
    batch_size = 32
    input_length = 512
    response_length = 128
    vocab_size = 50257  # GPT-2 vocabulary size
    
    # Create networks
    policy_network = TransformerPolicy(
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12
    )
    
    value_network = TransformerValue(
        vocab_size=vocab_size,
        hidden_size=768,
        num_layers=12,
        num_heads=12
    )
    
    # Create PPO trainer
    config = PPOTrainerConfig(
        learning_rate=3e-4,
        num_epochs=10,
        batch_size=batch_size,
        num_minibatches=4,
        ppo_params=PPOParams(),
        max_grad_norm=0.5,
        update_epochs=4,
        advantage_estimator="gae",
        loss_agg_mode="token-mean"
    )
    
    trainer = PPOTrainer(
        config=config,
        policy_fn=policy_network,
        value_fn=value_network
    )
    
    # Initialize training state
    dummy_input = jnp.zeros((1, input_length), dtype=jnp.int32)
    trainer.init_train_state(rng, dummy_input)
    
    # Create dummy batch
    batch = create_dummy_batch(
        batch_size=batch_size,
        input_length=input_length,
        response_length=response_length,
        vocab_size=vocab_size
    )
    
    # Prepare batch for training
    prepared_batch = trainer.prepare_batch(
        input_ids=batch['input_ids'],
        response_ids=batch['response_ids'],
        rewards=batch['rewards'],
        response_mask=batch['response_mask'],
        old_log_probs=batch['old_log_probs']
    )
    
    # Update model
    train_state, metrics = trainer.update(prepared_batch, rng)
    
    # Print metrics
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 