# RL Dataset System

This module implements a new design for RL training data management that separates concerns between experience accumulation and batch creation.

## Architecture Overview

The system consists of three main components:

1. **ReplayBuffer** - Accumulates experiences, writes them to disk, and handles all I/O operations
2. **BatchMaker** - Abstract interface for creating batches of RL examples (purely in-memory)
3. **RLExample** - Data structure for individual training examples

## Components

### ReplayBuffer

The `ReplayBuffer` class accumulates experiences from environment steps, writes them to disk, and handles all I/O operations including batch storage. Each buffer handles one environment type and can optionally forward rollouts to a `BatchMaker`.

```python
from marin.rl.replay_buffer import ReplayBuffer
from marin.rl.batch_maker import GrpoBatchMaker

# Create a batch maker (no I/O responsibilities)
batch_maker = GrpoBatchMaker(rng_seed=42)  # Optional seed for reproducible sampling

# Create a replay buffer with batch maker
replay_buffer = ReplayBuffer.remote(
    root_path="/path/to/storage",
    compression="zstd",
    batch_maker=batch_maker
)

# Add rollouts (they'll be automatically forwarded to the batch maker)
replay_buffer.add_rollout.remote(rollout_record)

# Create and store batches
batch_id = replay_buffer.create_and_store_batch.remote(batch_size=32)
```

**Features:**
- Automatically flushes to disk when groups are sealed
- Creates timestamped parquet files for each sealed group
- Organizes storage by environment type
- Faithfully preserves all inbound data
- Can forward rollouts to a BatchMaker for processing
- Handles all batch I/O operations (creation and storage)

### BatchMaker

The `BatchMaker` is an abstract base class that defines the interface for creating batches of RL examples. This class is purely in-memory and does not handle I/O operations.

#### GrpoBatchMaker

The `GrpoBatchMaker` implements a GRPO-style batching strategy:

```python
from marin.rl.batch_maker import GrpoBatchMaker

batch_maker = GrpoBatchMaker(
    environment_name="math_problems"
)

# Add rollouts to the batch maker
batch_maker.add_rollout(rollout_record)

# Create a batch of examples (in-memory only)
batch = batch_maker.create_batch(batch_size=32)

# Get batch metadata for storage purposes
metadata = batch_maker.get_batch_metadata(batch)
```

**Features:**
- Groups rollouts by `(environment, example_id, policy_version)`
- Applies retention policy (keeps only latest policy version per problem)
- Computes advantages using RLOO method
- Maintains reservoir of rollouts with non-zero advantage (includes both positive and negative examples)
- Randomly shuffles and samples from reservoir for diverse training batches
- Optional RNG seed for reproducible sampling
- Purely in-memory - no I/O operations
- Provides metadata for batch storage

### RLExample

Each `RLExample` contains:

- `tokens`: `np.ndarray` of token IDs (i32["pos"])
- `loss_mask`: `np.ndarray` of boolean values indicating which positions to compute loss on (bool["pos"])
- `advantage`: `np.ndarray` of advantage values for each position (float["pos"])
- `generator_log_probs`: `np.ndarray` of log probabilities from the generator model (float["pos"])

## Data Flow

1. **Experience Collection**: Rollouts are added to `ReplayBuffer` instances
2. **Experience Storage**: Experiences are automatically written to disk as parquet files when groups are sealed
3. **Rollout Processing**: Rollouts are automatically forwarded to `BatchMaker` instances
4. **Batch Creation**: `BatchMaker` creates batches of `RLExample` objects (in-memory)
5. **Batch Storage**: `ReplayBuffer` handles writing completed batches to disk as parquet files

## File Organization

The system creates the following directory structure:

```
root_path/
├── part-uuid.parquet          # Sealed rollout groups (existing functionality)
└── batches/                   # Training batches (new functionality)
    └── batch_uuid_timestamp.parquet
```

## Usage Example

See `example_usage.py` for a complete demonstration of the system.

## Integration with Existing System

The new system integrates seamlessly with existing `marin.rl` infrastructure:

- Uses existing `RolloutRecord`, `Turn`, and other datatypes
- Works with existing Ray-based `ReplayBuffer` actor
- Maintains backward compatibility
- No modifications to `post_training` module
- Clean separation: BatchMaker handles logic, ReplayBuffer handles I/O

## Configuration

Key configuration options:

- **Buffer Size**: Number of rollouts before sealing groups (default: 8)
- **Compression**: Parquet compression format (default: "zstd")
- **Retention Policy**: How to handle rollouts from different policy versions
- **Advantage Computation**: Method for computing advantages (currently RLOO)

## Extending the System

To create new batching strategies:

1. Inherit from `BatchMaker`
2. Implement `create_batch()` and `get_batch_metadata()` methods
3. Add any custom logic for grouping, filtering, or processing rollouts
4. The ReplayBuffer will handle all I/O operations

Example:

```python
class CustomBatchMaker(BatchMaker):
    def create_batch(self, batch_size: int) -> Optional[List[RLExample]]:
        # Custom batching logic
        pass

    def get_batch_metadata(self, batch: List[RLExample]) -> Dict[str, Any]:
        # Custom metadata logic
        pass
```

## Running the Example

To run the example script:

```bash
cd src/marin/rl
python -m example_usage
```

This will demonstrate the complete workflow of adding rollouts, creating batches, and storing them to disk using the ReplayBuffer for all I/O operations.
