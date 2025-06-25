"""
Example usage of the modular Trainer and DataProto classes.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any

from trainer import Trainer
from dataset import DataProto


def example_usage():
    """Example of how to use the Trainer and DataProto classes"""
    
    # 1. Create sample training data using DataProto
    batch_size = 4
    seq_len = 128
    vocab_size = 32000
    
    # Sample JAX arrays for training
    input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, vocab_size)
    attention_mask = jnp.ones((batch_size, seq_len))
    position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    target_ids = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), 0, vocab_size)
    loss_masks = jnp.ones((batch_size, seq_len))
    loss_weights = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len))
    reference_logprobs = jax.random.normal(jax.random.PRNGKey(3), (batch_size, seq_len))
    
    # Create DataProto from tensors
    training_data = DataProto.from_dict({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'target_ids': target_ids,
        'loss_masks': loss_masks,
        'loss_weights': loss_weights,
        'reference_logprobs': reference_logprobs,
    })
    
    print(f"Training data batch size: {len(training_data)}")
    print(f"Training data keys: {list(training_data.batch.keys())}")
    
    # 2. Example of chunking data
    chunks = training_data.chunk(2)
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk 0 size: {len(chunks[0])}")
    print(f"Chunk 1 size: {len(chunks[1])}")
    
    # 3. Example of concatenating data
    concatenated = DataProto.concat(chunks)
    print(f"Concatenated size: {len(concatenated)}")
    
    # 4. Example of creating mini-batch iterator
    rng_key = jax.random.PRNGKey(42)
    mini_batch_size = 2
    epochs = 1
    
    iterator = training_data.make_iterator(
        mini_batch_size=mini_batch_size,
        epochs=epochs,
        shuffle=True,
        rng_key=rng_key
    )
    
    print("\nIterating through mini-batches:")
    for i, batch in enumerate(iterator):
        print(f"Mini-batch {i}: size={len(batch)}")
        if i >= 2:  # Just show first few
            break
    
    # 5. Example of data selection and manipulation
    selected_data = training_data.select(
        batch_keys=['input_ids', 'attention_mask'],
        deepcopy=True
    )
    print(f"\nSelected data keys: {list(selected_data.batch.keys())}")
    
    # 6. Example with non-tensor data
    non_tensor_data = {
        'prompts': np.array(['Hello world', 'How are you', 'What is AI', 'Nice day'], dtype=object),
        'metadata': np.array([{'id': i} for i in range(batch_size)], dtype=object)
    }
    
    data_with_non_tensors = DataProto.from_dict(
        tensors={'input_ids': input_ids},
        non_tensors=non_tensor_data,
        meta_info={'dataset': 'example', 'version': '1.0'}
    )
    
    print(f"\nData with non-tensors:")
    print(f"Tensor keys: {list(data_with_non_tensors.batch.keys())}")
    print(f"Non-tensor keys: {list(data_with_non_tensors.non_tensor_batch.keys())}")
    print(f"Meta info: {data_with_non_tensors.meta_info}")
    
    # 7. Example of accessing individual items
    item = data_with_non_tensors[0]
    print(f"\nFirst item:")
    print(f"Input IDs shape: {item.batch['input_ids'].shape}")
    print(f"Prompt: {item.non_tensor_batch['prompts']}")
    print(f"Metadata: {item.non_tensor_batch['metadata']}")


def example_trainer_setup():
    """Example of how to set up and use the Trainer class"""
    
    # Note: This is a conceptual example. In practice, you would:
    # 1. Load your actual model, tokenizer, and configuration
    # 2. Set up proper sharding and mesh
    # 3. Initialize the train state with real optimizer
    
    print("\nTrainer setup example (conceptual):")
    
    # Mock objects for demonstration
    class MockModel:
        def __call__(self, input_ids, attention_mask, position_ids, params, train=False, dropout_rng=None):
            # Mock logits output
            batch_size, seq_len = input_ids.shape
            vocab_size = 32000
            
            class MockOutput:
                def __init__(self):
                    self.logits = jax.random.normal(
                        jax.random.PRNGKey(0), 
                        (batch_size, seq_len, vocab_size)
                    )
            
            return MockOutput()
    
    class MockConfig:
        def get_partition_rules(self, model_all_gather_axis):
            return []
        
        def get_intermediate_sharding_rules(self, data_axis, sequence_axis):
            return {}
    
    class MockMesh:
        def sjit(self, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    
    class MockTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3, 4, 5]  # Mock token IDs
        
        def decode(self, tokens, skip_special_tokens=True):
            return "mock decoded text"
    
    # Mock train state
    from flax.training.train_state import TrainState
    import optax
    
    # Create mock parameters
    mock_params = {'layer1': jnp.ones((10, 10))}
    
    train_state = TrainState.create(
        apply_fn=lambda x: x,
        params=mock_params,
        tx=optax.adam(1e-4)
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=MockModel(),
        train_state=train_state,
        reference_params=mock_params,  # Same as initial params for demo
        tokenizer=MockTokenizer(),
        config=MockConfig(),
        mesh=MockMesh(),
        max_input_length=512,
        max_output_length=128,
        train_bsize=4,
        pad_token_id=0,
        kl_coef=0.1
    )
    
    print(f"Trainer initialized with step: {trainer.get_step()}")
    
    # Example of computing RLOO advantages
    rewards = np.array([0.8, 0.6, 0.9, 0.7])
    advantages = trainer.compute_rloo_advantages(rewards)
    print(f"Rewards: {rewards}")
    print(f"RLOO advantages: {advantages}")
    
    print("\nTrainer setup complete!")


if __name__ == "__main__":
    print("=== DataProto Usage Example ===")
    example_usage()
    
    print("\n=== Trainer Setup Example ===")
    example_trainer_setup()
