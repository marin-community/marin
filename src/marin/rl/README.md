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
| Learner   |--------> |    Orchestrator   | <------ | OAI Inference Server(s)|
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

### Orchestrator

At the center is the Orchestrator. It is responsible for:

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



### Rollout

A Rollout is a sequence of Turns.

# TODO: add multiple reward functions to Turn and then a linearized reward function

```python
@dataclass(slots=True, frozen=True)
class Turn:
    """A single message-level interaction within a rollout."""

    message: str
    role: str
    tokens: list[str] | None = None
    logprobs: np.ndarray | None = None
    reward: float | None = None
    inference_metadata: InferenceMetadata | None = None
    timestamp: int | None = None


@dataclass(slots=True, frozen=True)
class Rollout:
    """A sequence of :class:`Turn` objects plus auxiliary metadata."""

    environment: str
    problem_id: str
    rollout_uid: str

    turns: list[Turn]
    metadata: dict[str, Any]

    def __iter__(self):
        return iter(self.turns)
```

## Creating `RlExample`s


An RlExample is the packed representation of one or more rollouts. It is what the learner needs.
Rollouts need to be flattened and packed into a set of RlExamples. 


```python
class RlExample(eqx.Module):
    input_ids: ht.i32[NamedArray, "batch position"]  # type: ignore
    loss_mask: ht.bool_[NamedArray, "batch position"]  # type: ignore
    """indicates prompt vs not prompt"""
    segment_ids: ht.i32[NamedArray, "batch position"]  # type: ignore
    """mostly 1/0 for padding"""
    advantages: ht.f32[NamedArray, "batch position"]  # type: ignore
    """RLOO advantages or similar"""
    policy_logprobs: ht.Float[NamedArray, "batch position"]
    reference_logprobs: ht.Float[NamedArray, "batch position"] # TODO: should we do this on the fly or precompute?
```


So how do we create `RlExample`s? We have a number of environments producing rollouts, at potentially very different rates.
Also, rollouts may vary quite a lot in length (across env types and over the course of training with thinking).
So we need to think about how to allocate our tokens to the rollouts from each env.

We set a *budget* for each env type (we may let this vary over time). This is the fraction of the total budget that we want to allocate to that env type.
The budget need only sum to less than 1. For any remaining budget, we'll follow some TBD strategy to add more tokens.

### ReplayBuffer

The goal of `ReplayBuffer` is to hold for a particular env type and be able to select batches of rollouts to suit some budget.
*ReplayBuffer is also responsible for computing advantages/baselines for each rollout*, though this is delegated to an *AdvantageFunction*.
(This only works for policy-based RL of course.)

We let each env type have its own ReplayBuffer (though we could do something more sophisticated and group some env types together).
ReplayBuffers maintain a pool of rollouts that could be used for training.

ReplayBuffers have:

- a "max age" for rollouts. Rollouts older than this (in steps) are discarded. For now, we'll set it to 1.
- min group size: the minimum number of rollouts for a group to be included in a batch.
- an advantage estimation function. For now, RLOO

#### Metrics

ReplayBuffer publish the following metrics:

- replays/${env_type}/rollouts_in_buffer
- replays/${env_type}/tokens_in_buffer
- replays/${env_type}/frac_on_policy

- replays/${env_type}/new_rollouts
- replays/${env_type}/new_tokens # includes prompt tokens and generated tokens
- replays/${env_type}/new_generated_tokens # generated tokens only
- replays/${env_type}/reward/mean
- replays/${env_type}/reward/std
- replays/${env_type}/frac_truncated

- replays/${env_type}/rewards/${reward_name}/mean
- replays/${env_type}/rewards/${reward_name}/std
- replays/${env_type}/frac_used_in_batch

It reports them when a batch is created.(?)

### Batcher

#### v0: simple round-robin

- Inputs: total_token_budget, env_fractions, max_seq_len 
- Algorithm: simple round-robin over envs by token deficit; newest groups first; no fancy priority.
- Spillover: if an env can’t meet its share, block

We pack rollouts into batches

#### Metrics

- batches/${env_type}/rollouts_used
- batches/${env_type}/tokens_used
- batches/${env_type}/frac_used


### Learner

Learner has a model and whatever opt state etc. It takes batches of RlExamples and updates the model.
Using a standard PPO-ish loss function.


### Weight Broadcaster
