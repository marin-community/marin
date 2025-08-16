## Inspiration

- https://github.com/willccbb/verifiers
- https://github.com/google/tunix/blob/main/tunix/rl/grpo/grpo_learner.py

## Meta

- [x] understand charlie's code
- [ ] do we want to do disaggegrated inference?
- [ ] unify naming: prefer `datatypes.py` and remove/alias `types.py`
- [ ] fix minor typing nits (e.g., `Turn.reward: float | None | None` -> `float | None`)
- [ ] README in `marin.rl/` describing the new async push-based design and how it maps to `post_training`

## Transfer
- [x] Implement weight transfer coordinator
- [x] make benchmark_weight_transfer to use new coordinator
- [ ] Multiple servers (one per training node) p3
- [ ] Trainer publishes weights every N steps using `process_weight_transfers(...)` p1
- [ ] Inference server receiver pulls weights via `receive_weight_transfers(...)` and hot-swaps sampler params p1
- [ ] Add retry/backoff + metrics for transfer failures/timeouts p3

## Coordination
- [ ] parameterize and initialize environments (via `AbstractEnvConfig`) p1
- [ ] central runner that: 
  - [ ] starts Ray, weight transfer server, and a named `WeightTransferCoordinator` p1
  - [x] instantiates env actors per `MarinRlConfig.envs` with varying seeds p1
  - [x] wires a `RolloutSink` to Parquet (see Parquet section) and/or ReplayBuffer p1
  - [ ] exposes health/metrics endpoint and graceful shutdown
- [ ] gather and log rollouts (counts, eps, per-env throughput)
- [ ] allow multiple coordinators (one per training node) and envs selecting nearest coordinator



## ReplayBuffer

- [x] Make ReplayBuffer skeleton p0
  - [ ] Ray actor: concurrent append; sampling by group for GRPO; capacity/eviction p1
  - [ ] Support persistence: optional Parquet append on a background thread p1
  - [ ] API: `add(groups: list[RolloutGroup])`, `sample(num_groups: int, strategy="by_group")` p1
  - [ ] Metrics: occupancy, enqueue/dequeue rate, drop count



## Training
- [x] skeleton loop
- [x] initialize model
- [x] objective function
- [ ] verify objective function works
- [x] optimizer
- [x] aux losses (kl) and logging
- [ ] integrate real data path (replace dummy dataset)
  - Option A (fastest): use `legacy_adapter.NewStyleEnvWrapper` + `post_training.rl_dataset.create_dataset_from_environment`
  - Option B (clean): port `post_training.rl_dataset` into `marin.rl` and consume `RolloutGroup`
- [ ] implement sampler (prefill/generate) or reuse `post_training.inference.build_sampler`
- [ ] wire `reference_logprobs` computation path (current `get_logprobs`)
- [ ] periodic evaluation on held-out prompts (see Evaluation)
- [ ] scheduler hooks to `process_weight_transfers(...)`


## Metrics
- cf https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo
- [ ] **eps**: Tracks the number of episodes per second.
- [x] **objective/kl**: The mean Kullback-Leibler (KL) divergence between the current policy and reference policy.
- [ ] **objective/entropy**: The mean entropy of the policy, indicating the randomness of the actions chosen by the policy.
- [ ] **objective/non_score_reward**: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where beta is the KL penalty coefficient and kl is the per-token KL divergence.
- [ ] **objective/rlhf_reward**: The mean RLHF reward, which is `score - non_score_reward`.
- [ ] **objective/scores**: The mean scores returned by the reward model / environment.
- [ ] **policy/approxkl_avg**: The average approximate KL divergence between consecutive PPO policies. Note that this is not the same as `objective/kl`.
- [ ] **policy/clipfrac_avg**: The average fraction of policy updates that are clipped, indicating how often the policy updates are constrained to prevent large changes.
- [ ] **loss/policy_avg**: The average policy loss, indicating how well the policy is performing.
- [ ] **val/clipfrac_avg**: The average fraction of value function updates that are clipped, similar to `policy/clipfrac_avg` but for the value function.
- [ ] **policy/entropy_avg**: The average entropy of the policy during training, indicating how diverse the policy’s actions are.
- [ ] **val/ratio**: The mean ratio of the current policy probability to the old policy probability, providing a measure of how much the policy has changed.
- [ ] **val/ratio_var**: The variance of the `val/ratio`, indicating the variability in policy changes.
- [ ] **val/num_eos_tokens**: The number of end-of-sequence (EOS) tokens generated, which can indicate the number of complete responses.
- [ ] **lr**: The current learning rate used by the optimizer.
- [ ] **episode**: The current global step or episode count in the training process.
- [ ] **train/rewards**
- [ ] **train/format_rewards**
- [ ] **train/correct_rewards**
- [ ] **train/output_length**
- [ ] weight transfer metrics: bytes/transfer, time_elapsed, transfers/step
- [ ] env metrics: per-env eps, errors, formatting rate



## Environments
- [x] hello world
- [x] chat echo (OpenAI-compatible)
- [x] math env (via OAI, correctness by `post_training` utils)
- [ ] port additional `post_training.environments` as needed (swe-bench, olympiad, etc.) to async API
- [ ] add option to emit to Parquet sink directly
- [ ] handle environments where groups come from multiple replicas
  - [ ] eval-style envs that only run once per replica (e.g., SWE-bench, olympiad)
  - [ ] coordinate group assembly across replicas before calling `rollout_sink`
  - [ ] ensure `RolloutGroup` metadata includes replica info for proper batching
  - [ ] consider adding a "group coordinator" actor that collects from multiple env replicas


## Sketches
- RLExample:

```python
import haliax.haxtyping as ht
import jaxtyping as jt
from marin.rl.datatypes import Rollout

@dataclass(frozen=True)
class ProcessedRollout:
    source: str
    id: str
    input_ids: jt.i32[np.ndarray, "batch position"]
    loss_mask: jt.bool_[np.ndarray, "batch position"]
    segment_ids: jt.i32[np.ndarray, "batch position"]
    returns: jt.f32[np.ndarray, "batch"]
    reference_logprobs: jt.f32[np.ndarray, "batch position"]
    policy_logprobs: jt.f32[np.ndarray, "batch position"]


def process_rollout_group(
    tokenizer: "HfTokenizer",  # type: ignore
    rollout: RolloutGroup,  # type: ignore
    max_length: int,
    apply_chat_template: bool = True,
) -> "ProcessedRollout":  # type: ignore
    """Process a rollout into a format suitable for training."""
    rollout_items = []
    for rollout in rollout.groups:
        if apply_chat_template:
            messages = [{"role": turn.role, "content": turn.message} for turn in rollout.turns]
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_attenteon_mask=True,
                return_assistant_tokens_mask=True,
                max_length=max_length,
                truncation=True,
            )
            input_ids = np.array(tokenized["input_ids"], dtype=np.int32)
            loss_mask = np.array(tokenized["assistant_masks"], dtype=np.bool_)
            segment_ids = np.ones_like(input_ids, dtype=np.int32)

        else:
            # Simple concatenation of turns without a chat template.
            # This is a simplified approach. A more robust implementation would handle special tokens and roles more carefully.
            all_input_ids = []
            all_loss_mask = []
            all_segment_ids = []

            for turn in rollout.turns:
                turn_ids = tokenizer(turn.message, add_special_tokens=False)["input_ids"]
                all_input_ids.extend(turn_ids)
                if turn.role == "assistant":
                    all_loss_mask.extend([True] * len(turn_ids))
                else:
                    all_loss_mask.extend([False] * len(turn_ids))
                all_segment_ids.extend([1] * len(turn_ids))

            # Pad or truncate
            input_ids = np.array(all_input_ids[:max_length], dtype=np.int32)
            loss_mask = np.array(all_loss_mask[:max_length], dtype=np.bool_)
            segment_ids = np.array(all_segment_ids[:max_length], dtype=np.int32)

        if len(input_ids) < max_length:
            padding_len = max_length - len(input_ids)
            input_ids = np.pad(input_ids, (0, padding_len), constant_values=tokenizer.pad_token_id)
            loss_mask = np.pad(loss_mask, (0, padding_len), constant_values=False)
            segment_ids = np.pad(segment_ids, (0, padding_len), constant_values=0)

    total_reward = sum(turn.reward for turn in rollout.turns if turn.reward is not None)
    returns = np.array([total_reward], dtype=np.float32)

    policy_logprobs = np.zeros_like(input_ids, dtype=np.float32)
    reference_logprobs = np.zeros_like(input_ids, dtype=np.float32)

    # The ProcessedRollout is defined to not be batched, so we add a batch dimension.
    return ProcessedRollout(
        input_ids=input_ids[None, :],
        loss_mask=loss_mask[None, :],
        segment_ids=segment_ids[None, :],
        returns=returns,
        reference_logprobs=reference_logprobs[None, :],
        policy_logprobs=policy_logprobs[None, :],
    )

def compute_rloo_advantages_for_group(rewards: np.ndarray) -> np.ndarray:
    """Compute RLOO advantages for a group of rewards.

    Args:
        rewards: Array of rewards for a group

    Returns:
        Normalized advantages
    """
    advantages = (rewards - rewards.mean()) / np.clip(rewards.std(), 1e-8, None)
    return advantages



```

## Data processing & storage
- [x] Parquet IO for `RolloutGroup`
- [x] add `RolloutSink` that writes groups via `parquet_store.write_rollout_groups`
- [ ] implement `process_rollout_group(...)` to produce training batches (port from `post_training.rl_dataset`)

## Evaluation
- [ ] hook for periodic eval using env-provided prompts or fixed eval sets
- [ ] mirror `post_training.Trainer.evaluate_data_from_environment`
- [ ] log eval metrics under `eval/*`

## Orchestration & CLI
- [ ] CLI: `marin-rl-train` main that loads `tiny_grpo.yaml`-style config, launches runner, starts training
- [ ] YAML schema: extend `tiny_grpo.yaml` to include `envs`, `inference`, `learner_resources`, `parquet_path`, `replay_buffer`
- [ ] integrate Ray resource configs (`RayResources`) and `ResourceConfig` for learner actor

## Parity with `post_training`
Map of what to copy/reuse vs. re-implement:
- **Sampler & inference**: reuse `post_training.inference.build_sampler`
- **Dataset shaping**: reuse/port `post_training.rl_dataset.RLDataset` and `create_dataset_from_environment`
- **Optimizer & schedules**: reuse `post_training.optimizer.load_adamw_optimizer`
- **Utils (checkpointing, dtype, logging)**: reuse `post_training.utils` (logger, save/load)
- **Environment wrappers**: keep `legacy_adapter.NewStyleEnvWrapper` for compatibility

## Prioritized next steps (execute in order)
1) Wire training to real data quickly: use `NewStyleEnvWrapper(MathEnvConfig)` + `post_training.rl_dataset.create_dataset_from_environment` in `simple_train.py` (replace dummy dataset).
2) Add a minimal runner that: starts Ray, weight transfer server + coordinator, and launches a few `MathEnv` actors writing to Parquet and/or ReplayBuffer.
3) Add a `RolloutSink` in runner that calls `parquet_store.write_rollout_groups` and increments eps metrics.
4) Enable weight publishing from trainer every K steps using `process_weight_transfers(...)` and have envs pull via `receive_weight_transfers(...)` to hot-swap sampler params.
5) Implement metrics logging: eps, reinforce_loss, kl_loss, transfer stats; log to WandB via Levanter tracker or `post_training.utils.WandbLogger`.

## Inference (OpenAI-compatible layer)
- Goal: Provide an OAI-compatible server that `openai.Client` can talk to via `base_url`, supporting `/v1/chat/completions` for our envs. Prefer Ray Serve for scalability/co-location per the design doc.

- MVP endpoints
  - [ ] `GET /healthz` and `GET /readyz`
  - [ ] `GET /v1/models`: return configured model IDs; the active weight version must be returned as a model ID
  - [ ] `POST /v1/chat/completions`: subset of fields: `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `n`, `stream`, `logprobs`
  - [ ] (Optional) `POST /v1/completions` compatibility shim (maps to chat)

- Server implementation
  - [ ] New module `marin.inference.oai_server`
    - Option A: Ray Serve deployment(s) for `chat.completions` with autoscaling
    - Option B (dev): FastAPI + Uvicorn single-process runner
  - [ ] Pydantic request/response schemas (minimal OpenAI v1)
  - [ ] Auth: Bearer token; 401 on missing/invalid
  - [ ] CORS, request size limits, timeouts
  - [ ] Streaming: SSE/chunking with `data: {"choices":[{"delta":{"content":"..."}}]}` and terminating `[DONE]`
  - [ ] Non-streaming: assemble and return one-shot response
  - [ ] Response metadata: deterministic `id`, `created`, `model` set to current weight version

- Model, tokenizer, sampler
  - [ ] Load tokenizer; initialize sampler via `post_training.inference.build_sampler`
  - [ ] Chat templating: map `messages` → prompt string; start with a simple role template
  - [ ] Params loading: checkpoint paths (or random tiny for dev); support attention kernel configs like post-training

- Logprobs support
  - [ ] If `logprobs` requested, compute token logprobs for returned tokens using a `get_logprobs` function
  - [ ] Return per-token `logprobs` (OpenAI-compatible) or omit initially and return aggregate in `usage`

- Weight broadcaster integration (hot-swap)
  - [ ] Start local JAX `TransferServer`
  - [ ] Background task periodically `receive_weight_transfers(...)` from a named `WeightTransferCoordinator`
  - [ ] Maintain `current_params` and `current_weight_version`; update atomically behind an RW lock
  - [ ] Expose `weight_version` as the OpenAI `model` field in responses
  - [ ] Metrics: bytes/transfer, transfer latency, time since last update

- Configuration & CLI
  - [ ] Pydantic settings: model/tokenizer paths, sharding, dtypes, coordinator name/address, auth token, host/port
  - [ ] CLI `marin-oai-serve` to launch server with config/env vars
  - [ ] Optional Ray Serve deployment script to scale replicas, set resource requirements

- Observability & reliability
  - [ ] Structured logs: request latency, tokens, stream vs batch, error codes
  - [ ] Optional Prometheus metrics endpoint
  - [ ] Basic rate limiting (token bucket)
  - [ ] Graceful shutdown and in-flight request draining

- Testing
  - [ ] Unit tests: chat templating; auth; model id = weight version
  - [ ] E2E: start server; call via `openai.Client(base_url=...)` (reuse env tests shape)
  - [ ] Streaming test: multiple chunks + final `[DONE]`
  - [ ] Logprobs test: sanity check with tiny model

- Stretch
  - [ ] Tools/function-calling no-op support (accept/ignore)
  - [ ] JSON mode `response_format`
  - [ ] Batch endpoint to amortize overhead under high QPS

## Overseer
- [ ] Implement an `Overseer` Ray actor that:
  - [ ] Boots the learner and a Ray Serve (or FastAPI) inference deployment with proper resources
  - [ ] Launches env actors based on `MarinRlConfig.envs`, wiring sinks to Parquet/ReplayBuffer
  - [ ] Ties the learner’s weight publisher to the coordinator; registers inference servers
  - [ ] Aggregates metrics from all components and logs to WandB
  - [ ] Exposes health, topology, and simple control operations (pause/resume/scale)

## Spec alignment with design doc
- [ ] `Turn.inference_metadata` should be `dict[str, Any]` and include at least `model_version` (aka weight version), temperature/top_p, etc.
- [ ] Response adapters from OAI → `RolloutGroup` should populate `turn.logprobs` when available and always include `model_version`
- [ ] Ensure envs are push-based and can be wrapped from pull-based via `legacy_adapter.NewStyleEnvWrapper`
