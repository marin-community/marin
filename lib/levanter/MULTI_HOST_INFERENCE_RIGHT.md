# Multi-Host Inference (Right Way): Plan

This document is a plan plus a critique of the current code paths. The goal is to make multi-host inference boring:
every host runs the same program with the same shapes, so XLA collectives line up.

## Goal

Run inference **across all hosts together** using the **globally sharded model**, with deterministic inputs and shapes
so every host compiles and executes the same program. This avoids the failure modes seen in `MULTI_HOST.md` and
`codex_MULTI_HOST.md`.

## Background: How Inference Works Today (Single Host)

The canonical single-host entrypoint is `src/levanter/main/sample_lm.py`.

High level:
1) Establish a device mesh and axis mapping:
   - `with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):`
2) Load a model (checkpoint or HF).
3) Tokenize prompts on the host.
4) Build `Request(prompt_tokens, SeqDecodingParams, n_generations)` objects.
5) Create an `InferenceEngine`:
   - `InferenceEngine.from_model_with_config(model, tokenizer, config.engine)`
6) Call `engine.generate(requests)` to run prefill + decode.

Important implementation notes:
- Engine construction allocates the KV cache via `hax.named_jit(model.initial_cache)(...)`.
- Decode runs via `hax.named_jit(_run_generation_loop)` which calls `model.decode(...)` and performs sampling.
- Host extraction is done by `jax.device_get(_DecodeOutputs)` and appending tokens into Python lists.

Multi-host sensitivity comes from anything that causes hosts to compile different programs:
- different shapes (KV cache sizes, prefill buffer sizes, stop token buffer sizes, etc.)
- different host code paths that trigger compilation/lowering on only some hosts
- different data-dependent compilation behavior (e.g., dynamic sizes inferred from local HBM)

## Constraints: What Multi-Host Inference Requires

- **Global sharding means global participation.** If the model parameters are sharded across hosts, then any jit that
  touches them (including `model.initial_cache` and `model.decode`) requires all hosts to participate. A leader-only
  forward pass will hang.
- **Deterministic shapes, everywhere.** Any shape inference that depends on per-host state (e.g., free HBM) can cause
  different compiled programs across hosts, leading to launch-group mismatch errors (e.g., “unexpected peer /
  different launch id”).
- **All hosts must take the same compilation-triggering path.** It is fine if only host 0 prints/logs; it is not fine
  if only host 0 performs JAX lowering/compilation steps that involve collectives.
- **Host sync must be deterministic.** Prefer `barrier_sync_with_tag(tag)` over counter-based barriers to avoid
  per-host drift. (See `src/levanter/utils/jax_utils.py`.)

## Critique of the Previous Plan (What Was Missing / Risky)

The original plan direction is correct (global inference across all hosts), but it missed several codebase realities:

1) **`InferenceEngine.generate()` admits prompts only once.**
   - `_prefill_prompts()` can `break` when it hits capacity (`max_seqs_in_prefill` or `max_prefill_size`), which can
     silently skip requests. The engine then runs decode and may "finish" with partial or empty outputs.
   - `generate()` currently does *not* loop and re-run prefill for remaining requests.
   - Consequence: “it ran without hanging” does not imply correctness; you must validate that all prompts were
     admitted.

2) **`engine.max_pages` being explicit is necessary, but not sufficient.**
   - Several other config fields drive static buffer shapes and therefore compilation determinism:
     `max_seqs`, `max_seqs_in_prefill`, `max_prefill_size`, `max_rounds`, `page_size`, `max_stop_seqs`,
     `max_stop_tokens`, and (if set) `max_tokens_per_round`.
   - The plan only enforced two of these.

3) **Broadcasting tokenized prompts is good, but has a deadlock failure mode.**
   - `multihost_broadcast_sync()` is a barrier rendezvous. If host 0 throws before calling it, other hosts will hang.
   - If you broadcast, you need an error-broadcast pattern or a guarantee that leader never fails before broadcast.

4) **“Only host 0 writes profiling/jaxprs” may be unsafe.**
   - `InferenceEngine.write_kernel_jaxprs()` calls `.lower()` which can trigger compilation/lowering paths.
   - If lowering involves collectives or results in different compilation timing, doing this on only one host can
     reintroduce launch-id mismatch risk. You should treat “lower/compile” as something all hosts do or none do.

5) **`InferenceEngineConfig.devices` does not do what it sounds like.**
   - In the current implementation, `devices` is only used for HBM budget inference (`_available_hbm_budget_bytes`),
     not to restrict which devices the engine runs on.
   - It should not be used as a strategy for single-host inference with a globally-sharded model.

6) **`sample_lm.py` has a tokenizer-argument bug that you do not want to copy.**
   - It computes `tok_string` (falling back to `hf_checkpoint.model_name_or_path`) but then calls
     `load_tokenizer(config.tokenizer)` anyway. If `config.tokenizer is None` this is wrong.
   - A new entrypoint should either require `tokenizer` explicitly or call `load_tokenizer(tok_string)`.

## Strategy Choice: Tradeoffs (Pick the Direction First)

There are multiple plausible “multi-host inference” strategies; the right one depends on what you are optimizing for.

### Option A: Global-sharded inference across all hosts (this document's default)

Pros:
- No weight replication; minimal extra memory.
- Exactly matches the training sharding layout (same weights, same mesh).
- No per-host divergence in results if inputs and PRNG are deterministic.

Cons / risks:
- Requires strict determinism in shapes and compilation across hosts.
- If you do this inside a training callback, you can perturb the training execution context and hit launch-group
  errors (see `MULTI_HOST.md` history).
- Operationally: every host must run inference; you cannot “just run it on leader.”

### Option B: Replicate the model per host and run inference on local devices only

Pros:
- Each host can run inference independently; avoids multi-host collectives during inference.
- Often avoids XLA launch-group mismatch when interleaving training and inference (because inference becomes local).

Cons / risks:
- Requires an all-gather (replication) of the model shards; expensive and can be infeasible for large models.
- Increased HBM usage; can OOM when combined with training state.
- Results can diverge across hosts unless you are careful, and you still pay compilation cost per host.

### Option C: Run inference as a separate job (not interleaved with training)

Pros:
- Clean separation of concerns; avoids “training context got corrupted” classes of failure.
- Much easier to reason about determinism and compilation.

Cons:
- Not “live samples during training”; you trade off latency (need checkpoints / eval artifacts).

### Option D: Leader-only inference

Only works if the model is replicated (Option B) or the inference model is built as replicated from the start.
With a globally sharded model, leader-only inference will hang.

## Plan (Updated, Thorough)

### Step 0: Codify “Determinism Contract” for Multi-Host Runs

Before writing code, define what is allowed to differ across hosts:
- Allowed: host-side printing/logging, local filesystem paths (if never used to drive JAX compilation).
- Not allowed: any change that affects JAX tracing, lowering, compilation, or runtime shapes.

Concrete “contract” checks you should enforce:
- All hosts see identical `engine` sizing config (see Step 2).
- All hosts see identical token IDs for prompts and stop sequences (see Step 1).
- All hosts call the same jitted functions the same number of times in the same order.

### Step 1: New Multi-Host Entrypoint (Keep Single-Host Behavior Intact)

Create a new file:

`src/levanter/main/sample_lm_multihost.py`

Key requirements vs `sample_lm.py`:
- Fix the tokenizer source-of-truth issue (do not call `load_tokenizer(None)`).
- Run everything under the global mesh (`trainer.use_device_mesh`) and compute axis mapping.
- Ensure all hosts build identical `Request` objects (same prompt token IDs, same stop IDs, same PRNG keys).
- Avoid compilation-triggering work on only one host (especially `.lower()`, `.as_text()`, `write_kernel_jaxprs()`).

#### Deterministic prompt tokenization: two viable patterns

Pattern 1 (recommended for determinism): **leader tokenizes, broadcasts token IDs**
- Pros: guarantees identical prompt IDs, avoids per-host tokenizer/model/version drift.
- Cons: if leader fails before broadcast, other hosts hang; requires careful error handling.

Pattern 2: **all hosts tokenize locally**
- Pros: no multihost broadcast step (fewer deadlocks).
- Cons: you are assuming tokenization is identical across hosts (same HF files, same versions, same settings).

If you choose Pattern 1, implement a safe broadcast:
- leader computes payload
- leader broadcasts payload
- all hosts validate payload shape/contents
- if leader hits an error before broadcast, it should broadcast an error payload so others can raise instead of hang

#### Suggested entrypoint sketch (focus: determinism + “admission fits” validation)

This is intentionally host-only logic; keep jitted code untouched.

```python
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import levanter
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.inference.engine import InferenceEngine, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.utils.jax_utils import barrier_sync_with_tag, multihost_broadcast_sync


def _require(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError(msg)


def main(config):
    levanter.initialize(config)

    is_multihost = jax.process_count() > 1
    is_leader = jax.process_index() == 0

    if is_multihost:
        barrier_sync_with_tag("sample_lm_multihost_start")

    # Tokenizer: do not copy sample_lm.py's tok_string bug.
    tok_string = config.tokenizer
    if tok_string is None and config.hf_checkpoint is not None:
        tok_string = config.hf_checkpoint.model_name_or_path
    _require(tok_string is not None, "Must specify tokenizer or hf_checkpoint with tokenizer")
    tokenizer = load_tokenizer(tok_string)

    # Global sharding: run under the global mesh + compute axis mapping.
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        model = ...  # load the sharded model here

        prompts = config.prompts
        if isinstance(prompts, str):
            prompts = [prompts]

        # Shape contract: max_pages must be explicit (avoid per-host HBM-based inference).
        # For the other knobs, defaults are deterministic but it is safer to set/validate them intentionally.
        _require(config.engine.max_pages is not None, "engine.max_pages must be explicit for multi-host inference")

        if is_multihost:
            if is_leader:
                try:
                    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
                    payload = {"ok": True, "prompt_ids": prompt_ids, "base_seed": int(config.engine.seed)}
                except Exception as e:
                    payload = {"ok": False, "error": repr(e)}
            else:
                payload = None
            payload = multihost_broadcast_sync(payload, is_source=is_leader)
            if not payload["ok"]:
                raise RuntimeError(f"Leader tokenization failed: {payload['error']}")
            prompt_ids = payload["prompt_ids"]
            base_seed = payload["base_seed"]
        else:
            prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            base_seed = int(config.engine.seed)

        # Admission-fit validation (critical with today's engine behavior).
        total_prompt_tokens = sum(len(t) for t in prompt_ids)
        _require(
            config.engine.max_seqs_in_prefill >= len(prompt_ids),
            "engine.max_seqs_in_prefill must be >= number of prompts (engine admits only one prefill batch)",
        )
        _require(
            config.engine.max_seqs >= len(prompt_ids) * int(config.n_generations),
            "engine.max_seqs must be >= total sequences (including n_generations clones)",
        )
        if config.engine.max_prefill_size is not None:
            _require(
                config.engine.max_prefill_size >= total_prompt_tokens,
                "engine.max_prefill_size must be >= sum(prompt lengths) (packed queue capacity)",
            )
        _require(
            config.engine.max_rounds <= config.engine.max_seq_len,
            "engine.max_rounds must be <= engine.max_seq_len (decode loop bound is currently max_seq_len // max_rounds)",
        )

        engine = InferenceEngine.from_model_with_config(model=model, tokenizer=tokenizer, config=config.engine)

        base_key = jrandom.PRNGKey(base_seed)
        requests = []
        for ridx, toks in enumerate(prompt_ids):
            requests.append(
                Request(
                    prompt_tokens=list(map(int, toks)),
                    request_id=ridx,
                    decode_params=SeqDecodingParams(
                        max_num_tokens=jnp.array(len(toks) + config.max_new_tokens, dtype=jnp.int32),
                        temperature=jnp.array(config.temperature, dtype=jnp.float32),
                        stop_tokens=None,
                        key=jrandom.fold_in(base_key, ridx),
                    ),
                    n_generations=int(config.n_generations),
                )
            )

        try:
            result = engine.generate(requests)
        finally:
            if is_multihost:
                barrier_sync_with_tag("sample_lm_multihost_done")

        if is_leader:
            ...  # print/log result
```

Notes:
- `multihost_broadcast_sync` uses JSON serialization; keep the payload small and JSON-serializable (lists of ints are
  fine). If prompts are huge, broadcasting token IDs may be slow.
- The `finally` barrier helps keep hosts aligned when there are exceptions, but cannot rescue you from a hard crash.

### Step 2: Make Engine Sizing Explicit (Avoid Any Host-Dependent Shape Inference)

The single most important rule is: **do not call `_infer_max_pages_from_hbm()` in multi-host inference**.
It depends on per-host `estimated_free_device_memory()`, which can differ at runtime.

Hard requirement for multi-host determinism:
- `engine.max_pages` must be set (do not infer from HBM at runtime).

Strong recommendations (defaults are deterministic, but set these intentionally and validate them):
- `engine.max_seqs` (sequence slots)
- `engine.max_seqs_in_prefill` (how many sequences can be admitted in the single prefill flush)
- `engine.max_prefill_size` (prefill queue capacity; imputed deterministically today if None)
- `engine.max_rounds` and (if set) `engine.max_tokens_per_round`
- `engine.page_size` (page granularity)
- stop-token capacity: `engine.max_stop_seqs` / `engine.max_stop_tokens` if you plan to use stop tokens

#### The "admission must fit" requirement (important!)

Because `InferenceEngine.generate()` admits prompts only once today, you must size `max_seqs_in_prefill` and
`max_prefill_size` so that the entire request batch fits in the initial prefill.

For `n` prompts with prompt lengths `L_i` and `n_generations` clones:
- Ensure `engine.max_seqs_in_prefill >= n` for the primary prompts (clones are added in the same prefill work).
- Ensure `engine.max_seqs >= sum_i(n_generations_i)` to have enough total slots.
- Ensure `engine.max_prefill_size >= sum_i(L_i)` for the primary prompts' packed token queue.
  (Clones reuse the prompt tokens from the parent; they do not add to the packed queue.)

If you cannot guarantee this (because prompts vary), you have two directions:
- Direction A: choose conservative max sizes (higher memory usage, but deterministic).
- Direction B: change the engine to support multiple prefill batches (more complex; must keep shapes static or
  introduce fixed-size padding; also needs tests).

#### Example multi-host engine config (explicit shapes)

This example is intentionally explicit about shape-driving knobs. Adjust sizes for your hardware and prompts.

```yaml
engine:
  # Shape-driving knobs: keep explicit for multi-host determinism.
  max_seq_len: 2048
  max_pages: 128
  page_size: 128

  # Admission-fit: engine admits only one prefill batch today.
  max_seqs: 64
  max_seqs_in_prefill: 64
  max_prefill_size: 4096  # must be >= sum(prompt lengths)
  max_queued_tokens: 1024

  # Decode loop
  max_rounds: 16
  # max_tokens_per_round: 64  # optional; if set, keep identical across hosts

  # Stop tokens (only if used)
  max_stop_seqs: 4
  max_stop_tokens: 16
```

Tradeoffs for strictness:
- Requiring `max_prefill_size` explicitly avoids surprises, but the current engine already imputes a deterministic
  default when it is None. If you want to allow the default, still validate admission-fit against the imputed value.

### Step 3: Run Under the Global Mesh / Axis Mapping

- All hosts must enter the same context:
  - `with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):`
- All hosts must call `InferenceEngine.from_model_with_config(...)` and `engine.generate(reqs)` in the same order.
- Only host 0 should print/log, but avoid doing any JAX lowering/compilation work on only host 0.

### Step 4: Logging, Profiling, and Kernel Dumps (Treat as Compilation-Coupled)

Safe default: **disable** `profile` and `log_kernel_jaxprs_path` in multi-host inference runs.

If you need kernel dumps:
- Prefer doing them in a separate single-host run, or
- Have all hosts call `write_kernel_jaxprs()` with per-host output paths to avoid collisions, and gate only the
  artifact logging on leader.

Tradeoff:
- "leader-only dumps" is convenient but can reintroduce non-deterministic compilation paths.
- "all-host dumps" avoids that but can create a lot of output and slow down the run.

### Step 5: Synchronization (Be Careful About Hangs)

Use deterministic tagged barriers:
- `barrier_sync_with_tag("sample_lm_multihost_start")`
- `barrier_sync_with_tag("sample_lm_multihost_done")`

Avoid counter-based barriers unless every host is guaranteed to call them the same number of times.

If you introduce `multihost_broadcast_sync`, ensure:
- leader cannot exit before broadcast without notifying others
- non-leaders do not proceed without a valid payload

### Step 6: Validate Correctness (Not Just “It Didn’t Hang”)

At minimum:
- Validate on the host that you admitted all prompts:
  - number of outputs equals expected sequences (including `n_generations`)
  - each output token list is non-empty (unless prompt max tokens is 0)
- Optionally, have each host compute a cheap hash of output token IDs and compare across hosts.
  - This can be done via a second broadcast: leader collects and prints mismatches, or by `multihost_broadcast_sync`.

## Concrete Updates to the Plan's Config Example

Update the multi-host sampler config to be explicit about all shape-driving knobs and to ensure admission fits:

- Set `engine.max_seqs` to the total number of sequences you will generate (sum of `n_generations`).
- Set `engine.max_seqs_in_prefill` to at least the number of prompts.
- Set `engine.max_prefill_size` to at least the sum of prompt token lengths (or conservatively large).
- Set stop-token capacity if you use stop sequences.

## Known Engine Limitations (Today) and What to Watch Out For

These are codebase critiques that directly affect multi-host inference safety:

1) One-shot prefill admission can drop requests.
   - Mitigation: enforce admission-fit constraints in your entrypoint/config, or extend the engine to admit in batches.

2) Capacity check for generations is misleading.
   - `generate()` checks `max(n_generations)` against `max_seqs`, but total slots needed is closer to
     `sum(n_generations)` across requests. You can still error or silently truncate depending on batching.
   - Mitigation: validate `sum(n_generations) <= engine.max_seqs` in the entrypoint.

3) Free-slot management is internally inconsistent.
   - The code/comment claims smallest-id-first but uses `pop()` (LIFO).
   - Likely not correctness-breaking, but it is a smell when reasoning about determinism.

4) Decode iteration bound is suspicious.
   - Loop bound is `max_seq_len // max_rounds`; if `max_seq_len < max_rounds` you do zero decode iterations.
   - Mitigation: ensure config always has `max_rounds <= max_seq_len` (or fix the engine loop logic).

5) Timing instrumentation is not reliable.
   - `device_time` is computed without an explicit block, so it does not measure actual device execution.
   - Mitigation: treat logs as approximate; for real profiling use JAX profiler intentionally.

6) `InferenceEngineConfig.devices` is not an execution control.
   - Mitigation: do not use it to attempt “single-host inference” with a globally sharded model.

## Success Criteria

- All hosts enter inference together and finish without divergence/hangs.
- No `FAILED_PRECONDITION` or “unexpected peer / different launch id” errors.
- Outputs are identical across hosts given the same prompts and seeds (host 0 prints/logs).
- No silent prompt dropping: output count matches expected sequences, and all prompts are represented.

## Optional Hardening

- Broadcast stop-sequence token IDs (or broadcast the entire `Request` spec) rather than recomputing per host.
- Add a warmup generation call (with fixed shapes) before doing measured runs, but ensure all hosts do it.
- Add entrypoint-side validations:
  - sum of prompt lengths <= `engine.max_prefill_size`
  - number of prompts <= `engine.max_seqs_in_prefill`
  - sum of generations <= `engine.max_seqs`
  - `engine.max_rounds <= engine.max_seq_len`
  - if stop tokens are used, ensure required stop capacity <= configured stop capacity

## How to Run (Multi-Host)

Example launch (v5p-64):

```bash
python infra/launch.py --foreground --zone us-central1-a \
  --tpu_name sample_worker --tpu_type v5p-64 --capacity_type on-demand -- \
  python src/levanter/main/sample_lm_multihost.py \
  --config_path config/sampler/sample_llama8b_multihost.yaml
```
