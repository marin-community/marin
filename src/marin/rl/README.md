# Marin RL Overview


## Architecture Overview


```
              +-------------------+
              |  Weight           |
     +------> |  Broadcaster      | --------------------------+
     |        +-------------------+                           |
     |                        |                               |
     |  New Weights           |  Metrics                      |
     |  + Generation          |                               |
     |                        v                               v
+-----------+          +-------------------+         +------------------------+
| Learner   |----+---> |    Overseer       | <------ | OAI Inference Server(s)|
+-----------+ Metrics  +-------------------+ Metrics +------------------------+ 
     ^                    ^         ^                          ^
     |        Metrics     |         |        Metrics           |
     |    +---------------+         +---------------------+    | Turns
     |    |                                               |    |
     |    |                                               V    v
+----------+                                             +------------+
|  Replay  |            <---------------                 |   Env(s)   |
|  Buffer  |                 Rollouts                    |            |
+----------+                                             +------------+ 
```

## Components

Probably this diagram is upside down relative to how one should think about it.

### Overseer

At the center is the Overseer. It is responsible for:

- Spinning up the other components, including the number of replicas for each env.
- Coordinating the training process and rollout generation
- Receiving and logging metrics from the other components

### Envs (Environments)

Envs are responsible for:

- Generating rollouts and sending them to the ReplayBuffer

Envs operate asynchronously and continously, similar to how Magistral works.
Envs can range from the fairly simple (e.g. multiple choice problems) to the complex (e.g. multiturn coding problems with tool use).

Envs typically will have:

- a set of prompts/problems (often with a solution)
- a verifier (e.g. a reward function to compare against a solution)

### ReplayBuffer

The ReplayBuffer is responsible for:

- Receiving rollouts from the Envs
- Writing rollouts to disk
- Creating and saving batches of RlExamples and giving them to the Learner

### Learner

The Learner receives batches of RlExamples and updates the weights of the model. It then sends the new weights to the Weight Broadcaster along with a new policy version.

### Weight Broadcaster

The Weight Broadcaster is responsible for broadcasting the new weights to the OAI Inference Server(s).

### OAI Inference Server(s)

The OAI Inference Server(s) are responsible for:

- Receiving new weights from the Weight Broadcaster
- Generating responses


## Data Structures


### RlExample

An RlExample is a single example of a rollout. It contains:

- `tokens`: `np.ndarray` of token IDs (i32["pos"])
- `loss_mask`: `np.ndarray` of boolean values indicating which positions to compute loss on (bool["pos"])
- `segment_ids`: `np.ndarray` of integer values indicating which segment each position belongs to (i32["pos"]). This is for sequence packing primarily.
- `advantage`: `np.ndarray` of advantage values for each position (float["pos"])
- `generator_log_probs`: `np.ndarray` of log probabilities from the generator model (float["pos"])
- `reference_log_probs`: `np.ndarray` of log probabilities from the reference model (float["pos"]) (TODO: should we do this on the fly or precompute?)




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
