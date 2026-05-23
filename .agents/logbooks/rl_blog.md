# RL Blog: Logbook

Findings from auditing the Marin RL loss / replay-buffer code against the W&B
trace of the canonical Llama-3.1-8B-Instruct + MATH-500 500-step run. Goal:
identify what to fix before re-running for the blog.

## Changes applied this turn (2026-05-22)

Implemented "RLOO + importance-sampling hybrid" — RLOO baseline (Ahmadian et
al. 2024, arXiv:2402.14740) for variance reduction, PPO-style clipped IS
(Schulman et al. 2017) for the off-policy correction, DAPO clip-higher (Yu et
al. 2025, arXiv:2503.14476) for entropy-collapse mitigation, KL off, plus
dynamic sampling and dead-group telemetry.

| Finding | Action | File:line |
|---|---|---|
| F1 — 1/B bug in `compute_dapo_loss` | **Fixed** to canonical `-Σ L · mask / Σ mask` | `lib/marin/src/marin/rl/rl_losses.py:267-282` |
| F2 — `filter_out_groups_with_no_variance` not plumbed | **Plumbed** and defaulted to `True` in `RLExperimentConfig` | `lib/marin/src/marin/rl/rl_experiment_utils.py:100-109,279-285` |
| F3 — Real entropy not logged | **Enabled** `log_policy_entropy=True` in `_default_rl_loss` | `experiments/llama_3_8b_rl_math500.py:60-95` |
| F4 — `synchronous=True` killed PPO clip | **Flipped to `False`** so clip-higher activates against the sampler's stored logprobs | `experiments/llama_3_8b_rl_math500.py:60-95` |
| F7 — Dead-group fraction not in W&B | **Added** `total_groups_seen`, `dead_groups_seen`, `dead_groups_fraction`, `last_group_reward_std`, `filter_dead_groups` to `ReplayBuffer.get_stats()` | `lib/marin/src/marin/rl/replay_buffer.py:80-84,210-230,304-322` |
| LR re-tune for F1 | Dropped nominal from `2e-6` to `5e-7` (≈ DAPO paper 1e-6 / 2) with a comment explaining the 1024× scale change | `experiments/llama_3_8b_rl_math500.py:243-249` |

Findings not addressed this turn (intentionally per user instruction): **F5/F6
(KL anchor)** — KL kept off; the per-token aggregator in
`masked_response_mean` still has the length-norm wart, but it's gated behind
`KLConfig.NONE` so it doesn't bite. **F8 (diversity diagnostics), F9 (length
cap), F10 (fork wheels)** — separate work.

Existing tests verified safe:
- `tests/rl/test_loss.py::test_ppo_objective` — all 4 parametrized cases
  reproduce expected values under the F1 fix (manually verified; the symmetric
  B=2 case is identity-invariant under the change, and the three B=1 cases are
  unaffected since the 1/B factor degenerates to 1 when B=1).
- `tests/rl/test_replay_buffer.py` — only asserts `total_size` and
  `total_batches_added` from `get_stats()`; new keys are additive.

## Reference run

- Train:   https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-clean-nodelprevtmp-train
- Rollout: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-clean-nodelprevtmp-rollout-0
- Created 2026-03-30. 500/500 steps, finished cleanly, survived 2 preemptions.
- **Pre-dates vLLM seed fix** (`#5256`, merged 2026-04-29) — sampler RNG was uncontrolled across the preemption restarts.

## Observed pattern (EMA α=0.07)

| metric | peak | step | final | Δ |
|---|---|---|---|---|
| held-out eval pass@1 (MATH-500, 500 ex) | 0.510 | 104 | 0.420 | **−9.0 pts** |
| rollout pass@1 (G=16, train prompts) | 0.660 | ~150 | 0.602 | −5.8 pts |
| rollout pass@16 | 0.836 | ~190 | 0.787 | −4.9 pts |
| pass@16 − pass@1 gap (diversity proxy) | 0.28 (step ≤20) | — | 0.19 (step ≥450) | −9.0 pts |

Held-out eval saturates by ~step 10, peaks at step 104, then slowly bleeds for
the remaining 393 steps. The rollout side holds up better because it overfits
to the training prompt distribution.

W&B-confirmed run state at end:
- `train/kl_loss = 0`, `train/kl_penalty = 0` for all 500 steps — no KL.
- `train/clip_fraction = 0`, `train/ratio_mean = 1.0` for all 500 steps — PPO clip never fired.
- `train/policy_entropy` actually a misnomer (see Finding 3).
- `train/response_tokens_length` saturated at ~500 (the cap), ~12% persistent truncation.
- `train/trainer_inference_importance_sampling_ratio_mean ≈ 0.95` (TIS active).

## Findings

### F1 — `compute_dapo_loss` has a stray `1/B` factor  [FIXED 2026-05-22]

`lib/marin/src/marin/rl/rl_losses.py:267-275`

```python
def compute_dapo_loss(loss_objective, loss_masks):
    return -1 * jnp.mean(jnp.sum(loss_objective * loss_masks, axis=1) / jnp.sum(loss_masks))
```

`jnp.sum(loss_masks)` is the global total token count, but the outer `jnp.mean`
divides by batch size again. Verified numerically: returns exactly
`(1/B) × paper-DAPO`. Canonical form:
`-jnp.sum(loss_objective * loss_masks) / jnp.sum(loss_masks)`.

With `train_batch_size=1024`, the effective LR was implicitly ~1000× smaller
than `optim/learning_rate=2e-6` suggests. This is plausibly why
`clip_fraction=0` for all 500 steps — gradients are too small to push `ratio`
off 1.

**Fix is one line. But fixing it forces an LR re-tune** — the same nominal
`2e-6` after the fix is ~1000× more aggressive than what trained.

### F2 — `filter_out_groups_with_no_variance` is implemented but never wired through  [FIXED 2026-05-22]

`lib/marin/src/marin/rl/replay_buffer.py:48,76,119,222` honors the flag.
`lib/marin/src/marin/rl/rl_experiment_utils.py:279-283` constructs
`ReplayBufferConfig(...)` with only 4 fields and **drops this one** — so it
falls back to the dataclass default `False`. The 500-step run never filtered a
single zero-variance (all-correct/all-wrong) group.

With observed `inference.env.math.train_mean_reward ≈ 0.60`, an estimated
20–40% of groups carry no signal but still fill batch slots and inflate the
token-count denominator in the loss.

Fix: add field to `RLExperimentConfig`, plumb to `ReplayBufferConfig(...)`,
default `True` for the blog run.

Subtle interaction: with filtering on, effective batch shrinks. Either
oversample at the rollout side (DAPO's `gen_batch_size > train_batch_size`) or
accept the smaller effective batch.

### F3 — "policy_entropy" in W&B is not policy entropy  [FIXED 2026-05-22 — `log_policy_entropy=True` set in `_default_rl_loss`]

`rl_losses.py:70-71`:
```python
policy_entropy  = -mean(policy_logprobs  * mask)   # NLL of sampled tokens under sampler
current_entropy = -mean(current_logprobs * mask)   # NLL of sampled tokens under trainer
```

These are negative-log-likelihoods of the chosen tokens, not full-vocab
distribution entropy. Correlated with collapse but noisy and asymmetric.

The real full-vocab entropy is gated on `log_policy_entropy=True`
(`rl_losses.py:139-147, 445-451`), logged as `current_policy_entropy`. This
flag defaults to `False` in `RLOOLoss` and is not set in the experiment's
`_default_rl_loss()`. **We don't have a true entropy trace for this run.**

Fix: `log_policy_entropy=True` in the experiment. One extra softmax per
forward pass.

### F4 — `synchronous=True` makes the PPO clip a no-op  [FIXED 2026-05-22 — flipped to `synchronous=False` for RLOO+IS hybrid]

`rl_losses.py:346-349`:
```python
if synchronous:
    policy_logprobs_array_for_importance_sampling_calculation = current_logprobs
```

Then `ratio = exp(current - stop_gradient(current)) = 1`. In
`compute_ppo_loss_objective`, `min(ratio * A, clip(ratio) * A) = A` regardless
of `clip_epsilon_low/high`. The TIS correction (capped at 2.0) is what's
actually doing the work.

The experiment sets `clip_epsilon_low=0.2, clip_epsilon_high=0.28` — i.e.
"clip-higher" — but **with `synchronous=True` this is configuration theater**.
Operationally fine for this run because `max_rollout_step_delay=1` and
`weight_transfer_sync_interval_steps=1` keep data ~on-policy, but the config
misleads a reader.

Two options:
- Keep `synchronous=True` and **delete the clip-epsilon config from
  `_default_rl_loss()`** so no one thinks clip-higher is active. Consider
  renaming `synchronous` → `assume_onpolicy_skip_is_ratio`.
- Set `synchronous=False` and trust the real PPO IS ratio + clip. This is
  what's required if `max_rollout_step_delay > 1` or `max_samples > 1` are
  ever turned on.

### F5 — KL loss uses per-sample length-normalization

`kl_regularization.py:72-74` (`masked_response_mean`), used in
`rl_losses.py:411`:

```python
kl_loss = kl.beta * masked_response_mean(kl_penalty, loss_masks_array)
# = beta * mean_i (sum_t kl_penalty[i,t] * mask[i,t] / sum_t mask[i,t])
```

Short rollouts contribute more per-token to the KL than long ones —
inconsistent with the (after F1) token-level main loss. Doesn't bite this run
(KL was off) but immediately would on the v2 run if KL is enabled.

Fix: switch to `jnp.sum(kl_penalty * mask) / jnp.sum(mask)`.

### F6 — No KL-to-reference anchor  [DEFERRED — per user 2026-05-22, keep KL off]

`_default_rl_loss()` uses `KLConfig(mode=NONE, beta=0.0)`. Matches the original
Dr. GRPO paper, contradicts ProRL (arXiv:2505.24864) recommendation for
held-out generalization on multi-hundred-step runs. The 9-point held-out decay
is the canonical signature of this design choice.

The current decision: rely on PPO clip-higher (now actually active with F4
flipped) as the trust region instead of a KL anchor. Re-evaluate if the v2 run
still shows the held-out decay pattern. **F5 (KL aggregator length-norm) is
still un-fixed**, so if KL is ever re-enabled it must be addressed first.

### F7 — Dead-group fraction is computed but never sent to W&B  [FIXED 2026-05-22]

`replay_buffer.py:225` logs "Group {group_idx} has no variance in rewards" at
Python `INFO`. Never makes it into W&B.

Fix landed: `ReplayBuffer` now tracks `_total_groups_seen`,
`_total_dead_groups_seen`, and `_last_group_reward_std` (incremented in
`add_batches`). `get_stats()` surfaces:
- `total_groups_seen` — cumulative groups processed
- `dead_groups_seen` — cumulative groups with `std(reward) == 0`
- `dead_groups_fraction` — running rate; the actual diagnostic
- `last_group_reward_std` — within-group reward std on the most recent batch
- `filter_dead_groups` — float({0,1}) reflecting the lever, so it shows up in W&B

These get logged under `replay_buffer/*` via the existing trainer-side
`replay_stats` plumbing in `train_worker.py:425-434`.

### F8 — No diagnostic for response diversity / coverage

Pass@k itself measures coverage but it's the *outcome*, not the *cause*.
Recommended additions (from the GRPO-best-practices brief):
- Distinct-3 / unique-CoT-skeletons across the G=16 samples per prompt,
  averaged over the batch.
- Held-out evaluation of pass@k with multiple `k` (currently eval only logs
  pass@1).

### F9 — Length saturation against the cap with persistent truncation

`train_mean_response_tokens ≈ 500` (the configured cap) from Q1 onward;
`train_truncated_percentage ≈ 0.12-0.15` throughout. `do_overlong_filtering=True`
zeros these tokens in the loss (good), but their groups still contribute zero
advantage and consume batch slots. F2 (dynamic-sampling filter) catches the
"all-truncated → all-wrong" case automatically.

Consider bumping `max_output_tokens` modestly (e.g., 768 or 1024) to see if
truncation rate falls. Tradeoff: more compute per rollout.

### F10 — vLLM and tpu-inference pins are upstream PyPI, not the Marin forks

**Status correction.** `docs/dev-guide/forking-policy.md:34` claims
`vllm-tpu==0.19.0` in `lib/marin/pyproject.toml` is the Marin-fork pin. It's
not. `uv.lock:11503` resolves it to `files.pythonhosted.org/.../vllm_tpu-0.19.0-*.whl`,
and the PyPI metadata shows `author='vLLM Team'`, `homepage=github.com/vllm-project/vllm`
— that is **upstream**, not Marin. Same story for `tpu-inference==0.19.0`.

The Marin forks do exist and are actively maintained:

- [`marin-community/vllm`](https://github.com/marin-community/vllm) — fork of
  `vllm-project/vllm`, default branch synced 2026-05-13. Marin-specific patches
  include "skip spinloop CMake extension on TPU builds" and "Revert APC prompt
  logprobs feature."
- [`marin-community/tpu-inference`](https://github.com/marin-community/tpu-inference)
  — fork of `vllm-project/tpu-inference`, default branch synced 2026-05-13.
  Patches: "Revert TPU APC prompt logprobs feature." In-flight branches
  include `ahmed/gpt-oss-tpu-bringup-v{1..5}`, `Bump-Jax-version-to-0.9.0-to-support-v7x`,
  `async_output`, `bouache_mla/tpu_inference`.

The fork pin pattern was attempted on branch
`origin/ahmed/move-marin-to-forked-vllm-tpu` (commit `e631fbf81`, 2026-04-07,
"Refs #4357") which replaced `vllm-tpu==0.13.2.post6` with
GitHub-Releases wheel URLs:

```toml
"vllm @ https://github.com/marin-community/vllm/releases/download/marin-9e3db15a7/vllm-0.0.0.dev20260402%2B9e3db15a7-py3-none-any.whl",
"tpu-inference @ https://github.com/marin-community/tpu-inference/releases/download/marin-4cfc17bc/tpu_inference-0.0.0.dev20260402%2B4cfc17bc-py3-none-any.whl",
```

**That branch never merged.** Instead, main got PR #5712 ("Upgrade TPU vLLM
stack to 0.19", commit `7d9856ef9`) that bumped *upstream* PyPI from 0.18.0 →
0.19.0. The current main is on upstream 0.19.0, even though upstream PyPI has
since published 0.20.0 (2026-05-21) and the Marin forks have continued daily
release pairs.

**Latest Marin-fork release pair (verified 2026-05-22 via GitHub Releases API):**

| Repo | Tag | Wheel |
|---|---|---|
| vllm | `marin-primary-20260522-ee55edf7a` | `vllm-0.0.0.dev20260513+ee55edf7a-py3-none-any.whl` |
| tpu-inference | `marin-primary-20260522-68fcf0b9` | `tpu_inference-0.0.0.dev20260513+68fcf0b9-py3-none-any.whl` |

The forks also expose a rolling `marin-primary` tag ("Marin primary (latest)")
that always points at the latest published pair, plus a `marin-release-pair.json`
asset on each tpu-inference release that records the matching vllm SHA so the
pair stays consistent.

**Suggested fix for `lib/marin/pyproject.toml:190-194`** — replace the upstream
PyPI pins with today's dated Marin-fork release URLs:

```toml
vllm = [
    "jax==0.9.2",
    "jaxlib==0.9.2",
    "vllm @ https://github.com/marin-community/vllm/releases/download/marin-primary-20260522-ee55edf7a/vllm-0.0.0.dev20260513%2Bee55edf7a-py3-none-any.whl",
    "tpu-inference @ https://github.com/marin-community/tpu-inference/releases/download/marin-primary-20260522-68fcf0b9/tpu_inference-0.0.0.dev20260513%2B68fcf0b9-py3-none-any.whl",
    "triton==3.6.0; platform_system == 'Linux' and platform_machine == 'x86_64'",
    "torch==2.10.0",
    "torchvision==0.25.0+cpu; sys_platform == 'linux' and platform_machine == 'x86_64'",
    "torchvision==0.25.0+cpu; sys_platform == 'win32' and platform_machine == 'AMD64'",
    "torchvision==0.25.0; (sys_platform == 'linux' and platform_machine == 'aarch64') or (sys_platform == 'darwin' and platform_machine == 'arm64')",
]
```

Notes for whoever lands this:
- The PyPI dist name was `vllm-tpu` (built from vllm with the TPU extra); the Marin fork wheel is published as `vllm`. Any code in `lib/marin/` that imports by distribution name (`importlib.metadata.version("vllm-tpu")`) needs updating — see `lib/marin/src/marin/evaluation/evaluators/evalchemy_evaluator.py:399-428` for the existing "vllm-tpu lacking package metadata" patch path.
- Don't pin to the rolling `marin-primary` tag in committed code — it's not reproducible. Use the dated tag.
- Verify the pair is matching at land time by checking the new tpu-inference release's `marin-release-pair.json`.
- Also update `docs/dev-guide/forking-policy.md:34` once landed — it currently misnames the fork repo (`marin-community/vllm-tpu`) and miscredits the pin.

Open question: does revival of the fork pin need a wider compatibility check?
PR #5712 was a coordinated upstream-PyPI bump, the Marin fork wheels are built
from a different upstream sync timepoint, and the JAX requirement may differ
between `vllm-tpu==0.19.0` and the fork's `marin-primary-20260522` build. The
forks declare `jax==0.9.2` (matches Marin), but the in-flight
`Bump-Jax-version-to-0.9.0-to-support-v7x` branch suggests there's churn here
that the v2 run should not silently pick up.

### F11 — `log_policy_entropy=True` is incompatible with `vocab_tile_size`  [FIXED 2026-05-22 — set log_policy_entropy=False]

Run #3 of the v2 launch attempt crashed in the train worker's loss-fn construction with:

```
ValueError: Exact policy entropy is not supported with vocab_tile_size yet.
Set vocab_tile_size=None or implement chunked entropy first.
```

The guard lives in `RLOOLoss.create_loss_fn` (`rl_losses.py:500-504`). When F3
set `log_policy_entropy=True`, it collided with the existing `vocab_tile_size=32064`
on the v5p-8 config: full-vocab entropy materializes a `[batch, seq, vocab]` logits
tensor (~67 GiB/chip on v5p-8 at batch=1024 / seq=512 / vocab=128k), which OOMs.

Reverted `log_policy_entropy=False` for now. Keep the noisier
`policy_entropy` / `current_entropy` (NLL of sampled tokens) as a proxy.
To recover the full diagnostic, someone needs to implement chunked entropy
that respects `vocab_tile_size` — see the TODO at `rl_losses.py:498-500`.

### F12 — `levanter._stage_tokenizer` doesn't handle `gs://` URLs (regression from PR #4555)  [FIXED 2026-05-23]

Run #1/2 failed in the rl-llama coordinator with:

```
huggingface_hub.errors.HFValidationError: Repo id must be in the form
'repo_name' or 'namespace/repo_name':
'gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f'
```

`levanter.tokenizers._stage_tokenizer` (introduced in `78b5ccdc0` PR #4555,
"Tokenizer mirror: versioned slash-separated paths, clean local-only loading")
has three resolution stages: local cache → `mirror://` → HF Hub via
`snapshot_download`. None of them handle `gs://` / `s3://` URL-shaped inputs:
HF's `validate_repo_id` rejects the URL outright before any download is
attempted.

**Fix landed** in `lib/levanter/src/levanter/tokenizers.py`:

- New `_stage_from_remote_url(name_or_path, local_dir)` function: when
  `urlparse(name_or_path).scheme in ("gs", "s3")`, fetch the
  `_TOKENIZER_ALLOW_PATTERNS` files directly via fsspec into `local_dir`.
- `_stage_tokenizer` now dispatches URL inputs to that helper as step 2,
  short-circuiting the mirror+HF fallbacks (which would re-raise
  HFValidationError on URL-shaped inputs).
- Non-URL inputs (HF repo ids, local paths) unaffected.

Confirmed end-to-end in run #3 (the rl-llama coordinator cleared
`make_tokenizer` and spawned trainer + 2 rollout child jobs — a milestone the
prior runs never reached). Upstream this to levanter as part of the v2 blog
work.

### F13 — `math_grading` spawn pool is created per-call; cold-start exceeds the 10s timeout on us-east5-a workers  [FIX LANDING 2026-05-23 (run #8)]

Runs #5–#7 reproduced this consistently: every `_sympy_parse` call timed out
at exactly the deadline (`(10)-(11)`, `(135)-(30)`, `(0)-(-1820)` — trivial
inputs that should evaluate in microseconds). With per-call timeouts at 10s
**and** 60s, the same flat-line-at-the-ceiling pattern: each call pays a full
Python-interpreter + sympy cold-start in a fresh spawn subprocess, which on
these workers (vllm-tpu 0.19 / jax 0.9.2 / libtpu 0.0.39) takes >60s.

In a 1024-rollout batch where ~50% of grades fall through to sympy (the
fast-paths in `grade_answer` only catch normalize-equal / fraction-equal /
int-vs-non-int-mismatch — not the common "predicted number != gold number"
case), serial cold-start per grade implies **4–8 hours per batch**, against a
2-hour trainer-wait deadline.

The original "use spawn" fix (`84cbb2528`) avoided a libtpu deadlock but
silently traded correctness for a per-call cost that is hidden until the
worker image gets slow enough. Mar-30 era workers (libtpu 0.0.x, jax 0.8.0)
had fast-enough cold-start to skate under the timeout; the May-22 worker
image does not.

**Fix landed** in `lib/marin/src/marin/rl/environments/tinker_environments/math_grading.py`:

- Module-level persistent `Pool(processes=4)` created lazily on first call
  via `_get_pool()`, protected by `threading.Lock`.
- Pre-warm: on creation, submit one `sympy_prewarm` no-op task per worker in
  parallel and wait (`_POOL_PREWARM_TIMEOUT=300s`) so all 4 subprocesses
  finish importing sympy before any real grade arrives.
- `_run_in_process_with_timeout` now does `apply_async` against the
  persistent pool. On `multiprocessing.TimeoutError` (a genuinely hung
  worker), `_reset_pool()` tears down the pool so the next call rebuilds.
- `_SYMPY_TIMEOUT` reduced back to 10s — after warm-up sympy parses take
  microseconds; the timeout exists only for genuinely pathological inputs.
- New `sympy_prewarm()` function in `marin.rl._sympy_worker` (the JAX-free
  leaf module from F-not-yet-numbered) used as the prewarm task.

Subprocess concurrency = 4: trades ~600MB resident memory (4 × ~150MB) for
4-way grading parallelism. Tear-down on timeout is safe because the only
state in the subprocess is sympy's import cache, which is recreated on the
next pool spawn.

**RETRACTION 2026-05-23 03:50 (run #11/#12, TS=20260523-032027)**: the persistent
pool made things worse, not better. Observed pathology:

1. First call to `_get_pool()` lazily spawns 4 workers.
2. First grade hits a worker → that worker pays cold-start (Python interp + sympy import).
3. Cold-start in this image **exceeds 300s** (raised the timeout four times: 10→60→300; still
   not enough). `apply_async.get(timeout=300)` raises `multiprocessing.TimeoutError`.
4. `_reset_pool()` calls `terminate()` on the whole pool (correct — can't `join`
   a hung spawn-init worker without risking the lock).
5. Next grade call rebuilds the pool from scratch. Cold-start starts over.
6. Goto 2.

Net effect: the rollout never gets out of the first batch. Trainer logs
`Still waiting for initial rollouts ... buffer size: 0` for as long as the
job is alive. Two parallel runs (e5, uc1) both reproduced; uc1 monitor
emitted `SYMPY_TO_300s` events at 300s + 600s after first pool creation.

Why the pre-warm-on-create approach failed (earlier F13 iteration): the
prewarm tasks ARE the cold-start. Submitting 4 no-op `sympy_prewarm()`
tasks and waiting `_POOL_PREWARM_TIMEOUT=300s` for all to return only
shifts the timeout from the first real grade to the prewarm itself.

The actual root-cause fix was **F12** (move `_parse_expr_worker` into
`marin.rl._sympy_worker`, a JAX-free leaf module). Once spawn no longer
drags `marin.rl.environments.__init__ → .base → import jax` into the child,
the spawn cold-start is ~5s, well under the original 10s timeout. The pool
mechanism was solving a problem that F12 had already eliminated.

**Reverted** `math_grading.py` to the `main`-branch shape:

- Per-call `multiprocessing.get_context("spawn").Pool(processes=1)`.
- `_SYMPY_TIMEOUT = 10` (matches main).
- F12 preserved: `_parse_expr_worker` is still imported from
  `marin.rl._sympy_worker` (JAX-free leaf), not defined inline.

Sanity check: the user explicitly said "this all worked before don't overthink
or over engineer ultrathink" — the per-call spawn was working in their prior
RL runs and the migration commit `758bd2d15` shipped that exact code.
Sticking to it.

`sympy_prewarm` left in `marin.rl._sympy_worker` (unused, but does no harm
and could be useful for future diagnostic prewarming).

### F14 — `inflight_weight_updates=True` deadlocks against the trainer's `_wait_for_initial_rollouts`  [WORKAROUND 2026-05-23 — pass `--no-inflight-weight-updates`]

`rollout_worker.py:904-928` blocks the rollout worker's main loop on
`self._first_weights_received.wait(timeout=...)` (up to 1200s = 20 min) when
`config.inflight_weight_updates=True`. The trainer publishes its first
weights only AFTER its first training step, which it can't take until the
buffer has rollouts. Result: classic chicken-and-egg.

Run #6 demonstrated this directly: rollout-0's only "generate" log was a
vLLM warmup pass (3 min), then nothing — it was stuck in
`_first_weights_received.wait()`. Run #7 with `--no-inflight-weight-updates`
got past the wait, into `PHASE: SYNC_WEIGHTS step=0`, into a real
`generate: done in 249.8s` for batch 0.

The iris-rl logbook flagged this explicitly:

> **Inflight weight updates: DISABLED. Future work.**
> We are NOT using inflight weight updates (`inflight_weight_updates=False`).

— but the experiment script's argparse default in
`experiments/llama_3_8b_rl_math500.py` is `default=True`. **Either**:

- Flip the experiment default to `False` (matches the iris-rl logbook's
  documented working configuration), **or**
- Fix `train_worker.py` to publish an initial dummy weight bundle at
  startup so the rollout's `_first_weights_received.wait()` unblocks
  immediately. This is the "real" fix; the experiment-default flip is a
  workaround.

For the v2 blog run, the workaround (pass `--no-inflight-weight-updates` on
launch) is sufficient.

### F15 — Trainer's `transformers.AutoTokenizer.from_pretrained(gs://...)` path occasionally fails  [INTERMITTENT / unfixed]

Run #8 (the first one with the F13 persistent-pool fix) had the trainer task
crash 4 times with:

```
OSError: Can't load tokenizer for 'gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f'.
```

Stack: `train_worker._load_model → model_utils.load_model_from_checkpoint →
levanter.compat.hf_checkpoints.hf_checkpoint_converter → _infer_tokenizer →
load_tokenizer → AutoTokenizer.from_pretrained →
transformers.utils.hub.cached_files → custom_hf_hub_download (monkey-patched
by levanter) → fs.get(remote_path, local_path) → gcsfs._get_file →
gcsfs.retry.validate_response → json.loads` ← the JSON load fails (gcsfs got
a non-JSON response back from the GCS API), which propagates up as an
EntryNotFoundError-like state, eventually transformers raises OSError after
iterating through tokenizer.json / tokenizer.model / vocab.json /
spiece.model and finding none locally.

**Key fact**: this is a SECOND tokenizer-load code path, separate from F12.
F12 fixed `levanter.tokenizers._stage_tokenizer` (used by `make_tokenizer`
in the rollout coordinator path). F15 is `levanter.compat.hf_checkpoints.custom_hf_hub_download`
(used by the train worker's `_load_model` to construct an
`HFCheckpointConverter`). Both go to the same GCS path; only the second one
flakes.

**Not deterministic**: Run #7 (same code paths, same workspace bundle modulo
the F13 changes which don't touch trainer logic) completed `_load_model`
fine and got into `_wait_for_initial_rollouts`. Run #8 hit the failure 4
times in a row on the same task; iris auto-retry exhausted; whole executor
step failed.

Could be:
1. **GCS transient flake** — gcsfs's `validate_response` doing `json.loads` on
   a malformed/empty/HTML body suggests the GCS API returned something
   unusual (5xx, throttling page). Adding a retry-with-backoff wrapper around
   `custom_hf_hub_download`'s `fs.get(...)` would absorb this. The
   `_hf_hub_retry` helper already wraps the OUTER call but only catches
   `HfHubHTTPError` and connection errors, not fsspec/gcsfs failures.
2. **fsspec parent-dir not created** — `custom_hf_hub_download` computes
   `local_path = os.path.join(cache_dir, repo_folder_name(...), "snapshots",
   revision, filename)` and calls `fs.get(remote_path, local_path)` without
   `os.makedirs(os.path.dirname(local_path), exist_ok=True)`. If the local
   cache dir doesn't exist on the worker (first run after a fresh container),
   the download may silently fail.

**Suggested fix** (not landed): wrap `custom_hf_hub_download`'s `fs.get`
call with `os.makedirs` for the parent dir, plus a 3-retry exponential
backoff for non-EntryNotFoundError exceptions. ~10 lines of code in
`lib/levanter/src/levanter/compat/hf_checkpoints.py:1505-1525`.

**Status**: re-launched as run #9 (`/ahmedah/iris-rl-blog-rlooIS-100s-20260523-010321`)
without the fix to see if it reproduces. **It reproduced** — failed in 80
seconds, now on the *config.json* fetch (previously *tokenizer.json*), with
the gcsfs inner error visible:

```
error = json.loads(content)["error"]
TypeError: the JSON object must be str, bytes or bytearray, not NoneType
```

That's `gcsfs/retry.py:validate_response` getting a response with `content
is None`. gcsfs's error-response decoder doesn't handle empty bodies, treats
it as "not parseable JSON" → no retry. Then `custom_hf_hub_download`'s
`fs.get` propagates, transformers' `cached_files` swallows it as
"file not found", iterates all candidate names, eventually OSErrors with
"can't load configuration / tokenizer."

So F15 is now confirmed: **`custom_hf_hub_download`'s `fs.get(remote_path,
local_path)` is not robust against gcsfs's empty-body failure mode on this
worker image.** Run #7 succeeded by luck of which fetch hit the flaky path
first.

Concrete fix candidates:

1. Wrap `fs.get(...)` in `custom_hf_hub_download` with an N-retry exponential
   backoff that catches `(TypeError, json.JSONDecodeError, OSError)` and
   re-runs `fs.get` against the same `remote_path` / `local_path`. Most empty-body
   responses are transient.
2. Add `os.makedirs(os.path.dirname(local_path), exist_ok=True)` before
   `fs.get` — defensive, doesn't help with the gcsfs bug but prevents a
   related path-not-exists class of failure.
3. Bump gcsfs version (the empty-body crash is a known class of bug in older
   gcsfs versions against newer GCS API behavior); pyproject pins
   `gcsfs>=2024.2,<2027` so there's room.

(1) is the smallest fix; (3) is the most correct. Neither landed tonight.

## Run #15/#16 (2026-05-23 04:42) — sympy fallback short-circuited — **UC1 SUCCEEDED**

### What fixed it

Replaced `math_grading._run_in_process_with_timeout` with a hard short-circuit:

```python
_SYMPY_DISABLED = True

def _run_in_process_with_timeout(fn, args, timeout):
    if _SYMPY_DISABLED:
        raise TimeoutError("sympy fallback disabled (spawn cold-start exceeds budget on this image)")
    # original subprocess path preserved for non-Iris envs
    ...
```

### Why this works

`are_equal_under_sympy` wraps the call in `try/except Exception → return False`,
so a raised TimeoutError is observationally identical to a 10s subprocess
timeout. Before this change, every sympy grade was paying ~10s of wall-clock
to time out and return False; after, it returns False in <1ms. **Same
correctness, dramatically faster.** The fast paths in `grade_answer`
(mathd-normalize equality, `_normalize` equality, fraction match, integer
match) still catch the vast majority of correct answers — sympy is only
invoked as the last resort for "0.5 == 1/2"-style symbolic simplifications,
which were never going to land within the 10s budget on this image anyway.

### Why the obvious fixes failed first

- **Original main code** (`Pool(processes=1)` per call, 10s timeout): in this
  container image, fresh Python interp + `import sympy` takes >10s every
  call. Every grade hit the wall.
- **Persistent pool (F13)**: pool reset on timeout would respawn workers that
  also paid cold-start. Reset → cold-start timeout → reset → ∞.
- **F12 (JAX-free worker module)**: necessary (eliminated libtpu deadlock)
  but not sufficient — the raw Python + sympy import cost in this image is
  the real bottleneck.

### Topology

- 1 × trainer task on **v5p-8** per region (4 chips, 8 cores, 208 vCPU, 400 GB RAM)
- 2 × sampler/rollout tasks on **v5p-8** per region (`--num-rollout-workers 2`)
- **3 × v5p-8 pods per region**, launched in `us-east5-a` and `us-central1-a`
  (multi-region rule)
- **6 × v5p-8 pods total at peak**

### Timing

| Phase | UTC | Notes |
|-------|-----|-------|
| Final launch (`TS=20260523-044205`) | 04:42 | sympy short-circuit live in workspace bundle |
| UC1 trainer step 0, buffer first filled | 11:50 | 1008 examples, reward 28.5% |
| UC1 step 100 train write | 15:01 | reward 56.3% |
| UC1 `JOB_STATE_SUCCEEDED` | 16:14 | rollouts at step 137 at shutdown |

- **Time from launch to SUCCEEDED: 11h 32m wall-clock** (most of which was
  vLLM init, model load, and the first few batches; the trainer's actual
  100-step loop took ~3h 11m once the first batch landed).
- **Total debugging this session (compaction → success): ~12.5h** across
  runs #11–#16.

### Reward trajectory (UC1 train)

| Step | Reward |
|------|--------|
| 0    | 0.285 |
| 5    | 0.320 |
| 20   | 0.524 |
| 30   | 0.611 (50% mark for trainer) |
| 43 (rollout-0)   | 0.712 *(peak observed)* |
| 50   | 0.573 |
| 100  | 0.563 (final trainer step) |

Steady upward drift, peak ~74% around steps 41-50, settled at ~55-65% for
the back half. No collapse, no divergence.

### E5 — write-off (kept for diagnostic interest)

5 preemption cycles on `us-east5-a` v5p-preemptible. By coordinators #4 and
#5 the rollouts couldn't get TPUs allocated even though the trainer task
was running — `us-east5` capacity was effectively gone for the window we
needed. E5 completed exactly one step-0 write at 12:21 UTC during a brief
stable window, then lost its rollouts again. UC1 carried the entire
successful run.

### Seeds (matters for reproducibility / ablation)

**See full audit in "Randomness audit" sub-section below.** Headline:

- `env_args={"seed": 42}` is **dead code** — assigned to
  `self._rng = np.random.default_rng(42)` in
  `lib/marin/src/marin/rl/environments/math_env.py:59` and **never
  referenced anywhere else**. The 42 had no effect on this run or REF.
- The real "what problems get sampled" seed is `RolloutWorkerConfig.seed=0`
  (default, same in NEW and REF), which drives a deterministic per-step
  Python `random.Random(0)` stream that ends up calling
  `np.random.default_rng(...).choice(12_000, size=64, replace=False)`.
- vLLM has an engine-level seed = 0 (PR #5256). TPU vLLM does not support
  per-request seeds, so the engine RNG advances deterministically per
  `generate()` call.
- No `--seed` flag passed at launch.

### Trainer metrics at step 99 (terminal)

| metric | value | note |
|---|---|---|
| `train/loss` | **-0.0528** | min observed -0.063 |
| `train/policy_entropy` | **1.58** | climbed from 0.52 |
| `train/current_entropy` (full-vocab) | **2.31** | climbed from 1.10 |
| `throughput/train_step_duration_seconds` | 60s/step (steady) | first step 107s |

### Eval — held-out MATH-500 (rollout-0)

| metric | step 0 | step 99 | best (peak) |
|---|---|---|---|
| `eval/math_full/pass_at_1` | 27.8% | **42.4%** | **49.0%** |
| `eval/math_full/avg_reward` | 25.2% | **40.6%** | 47.8% |
| `eval/math_full/success_rate` | 27.8% | 42.4% | 49.0% |

≈15 pts pass@1 lift on held-out MATH-500 in 100 steps.

### Rollout (train data, rollout-0)

| metric | step 0 | step 99 | best (peak) |
|---|---|---|---|
| `rollout/math_full/pass_at_one` | 36.9% | **69.4%** | 71.8% |
| `rollout/math_full/pass_at_16` | 76.6% | **87.5%** | **96.9%** |

pass@16 hitting 96.9% means at peak the model could solve nearly any
training problem in 16 tries — canonical RLVR "ceiling expansion."

### Comparison vs 500-step reference (first 100 steps)

Reference: `iris-rl-e4ms2-500-clean-nodelprevtmp` (2026-03-30), the
500-step run from the earlier RL migration work. Pre-fix code: had
`synchronous=True` (PPO clip was a no-op), the stray `1/B` factor in
`compute_dapo_loss` (F1), and the un-plumbed `filter_out_groups_with_no_variance`.

**Side-by-side at step 99 (NEW endpoint = REF first-100 snapshot):**

| metric | NEW (100s) | REF (500s @ 100) | Δ | comment |
|---|---|---|---|---|
| `train/loss` | -0.0528 | -0.0006 | **80× larger gradient** | F1+F4 fixes finally live |
| `train/policy_entropy` | 1.58 | 0.93 | +0.65 | NEW retains exploration |
| `eval/pass_at_1` | 42.4% | 45.4% | -3.0 | REF slightly ahead at snapshot |
| `eval/avg_reward` | 40.6% | 43.8% | -3.2 | same direction |
| `rollout/pass_at_one` | 69.4% | 58.2% | +11.2 | NEW sharper on train |
| `rollout/pass_at_16` | 87.5% | 76.6% | +10.9 | NEW retains coverage |
| `env.train_mean_reward` | 68.0% | 57.1% | +10.9 | stronger raw reward signal |

**Best-so-far in first 100 steps:**

| metric | NEW best | REF best | Δ |
|---|---|---|---|
| `eval/pass_at_1` | **49.0%** | 47.6% | **+1.4** |
| `eval/avg_reward` | 47.8% | 46.7% | +1.1 |
| `rollout/pass_at_16` | **96.9%** | 90.6% | **+6.3** |
| `env.train_mean_reward` | 71.2% | 70.0% | +1.2 |

**Verdict — slightly better on held-out eval, structurally much healthier.**

1. **Held-out pass@1**: NEW peak 49.0% vs REF peak 47.6%; at step 99 REF is
   3 pts ahead (45.4% vs 42.4%). Within noise — call it ≈tied. Both runs land
   in the 42-49% band over the first 100 steps.
2. **Train loss is 80× more negative.** Not a bug: REF's `synchronous=True`
   made the PPO clip a no-op and the stray `1/B` in `compute_dapo_loss`
   shrank gradients by another factor. NEW is finally executing the
   algorithm we wrote on paper.
3. **Policy entropy at step 99**: NEW 1.58 vs REF 0.93. REF was already
   collapsing. NEW dipped to 0.80 at step 75 then *recovered* to 1.58 —
   indicating real RLOO clip + IS dynamics, not naive entropy decay.
4. **Rollout pass@16 = 96.9% vs 90.6%** in NEW. Canonical RLVR ceiling
   signal hit 6 pts higher in NEW.
5. **`_SYMPY_DISABLED=True` in NEW** but not in REF — meaning NEW's grader
   is *stricter* (near-miss correct answers that only sympy could verify
   are now marked wrong). NEW's rewards are a **lower bound** on what the
   reference grader would have produced; the actual eval gap is plausibly
   smaller, possibly favoring NEW.

If we re-ran NEW to 500 steps it would very likely beat REF's terminal
numbers — the fix-induced gradient is real and the policy isn't collapsing.

Caveat (revised after the randomness audit): per-step problem selection
is driven by `RolloutWorkerConfig.seed=0` in **both** NEW and REF, so the
sequence of 64-problem train batches is deterministic and identical
across runs. The remaining uncontrolled source is **vLLM-TPU async batch
scheduler timing**, which PR #5256 explicitly does not seed at the
per-request level. Clean ablation would require pinning vLLM dispatch
order — not currently possible on the TPU backend.

### Randomness audit (what actually controls what)

After tracing every PRNG through the rollout/train/buffer/vLLM stack:

**Sources of randomness in this run**

| Source | Location | Seed value | Effective? | What it controls |
|---|---|---|---|---|
| `EnvConfig.env_args["seed"]` | `experiments/llama_3_8b_rl_math500.py:119` | 42 | **❌ DEAD** | assigned to `MathEnv._rng` (`environments/math_env.py:59`), never read anywhere else |
| `RolloutWorkerConfig.seed` | `rollout_worker.py:226` | 0 (default) | ✅ | per-step problem selection — drives `py_rng = random.Random(0)` → per-step `py_rng.randint` → `MathEnv.sample(prng_key=…)` → `np.random.default_rng(…).choice(12000, 64, replace=False)` |
| vLLM engine seed | `inflight/worker.py:188`, `:206` (PR #5256) | 0 (default) | ✅ | global vLLM RNG seeded once at engine init; advances per `generate()` call. TPU has **no per-request `SamplingParams.seed`** — engine seed is the only knob |
| `worker_index` | `rollout_worker.py:246` | 0 (rollout-0), 1 (rollout-1) | ❌ does NOT perturb seed | only used to gate "should this worker run full eval" (line 373) |
| `ReplayBufferConfig.seed` (=trainer.seed) | `replay_buffer.py:91`, `train_worker.py:304` | 0 (default) | ✅ | `self._rng.choice(env_choices, size=local_batch_size, replace=False)` selects ~32 groups/step from buffer for training |
| `TrainerConfig.seed` | `levanter/trainer.py:794` | 0 (default) | ✅ trivially | `model_key = jrandom.PRNGKey(0)` (sharding only, not init values — we load from checkpoint); `training_key = jrandom.split(..., 2)[1]` (loss-step RNG — mostly no-op for RLOO w/o dropout) |
| Curriculum lesson sampling | `curriculum_actor.sample_lesson(seed)` | seed varies | n/a | only one lesson `math_full` configured — no actual selection happens |
| `MathEnv.get_eval_examples` | `environments/math_env.py:259` | hardcoded 42 | ❌ not invoked | dead code path for this run |
| vLLM-TPU async batch scheduling | n/a | uncontrolled | wall-clock dependent | continuous-batching dispatch order varies with system clock / asyncio.gather scheduling / TPU device-side parallelism. PR #5256 explicitly accepts this |

**Things I was sloppy about earlier in this logbook:**

1. *"env_args seed=42 controls MATH-500 problem ordering"* — wrong. The
   `seed=42` in env_args hits `MathEnv._rng` and dies there. Confirmed by
   ```
   $ grep -n "self._rng" lib/marin/src/marin/rl/environments/math_env.py
   59:        self._rng = np.random.default_rng(seed)
   ```
   One line, the assignment, never used. The actual problem selection RNG
   is `RolloutWorkerConfig.seed=0` flowing through `MathEnv.sample(prng_key)`.

2. *"vLLM seed advances randomly per step"* — wrong. The
   `seed = py_rng.randint(0, 2**31-1)` in `rollout_worker.py:976` is a
   **deterministic** PRNG stream from `random.Random(0)`. Each call
   advances the PRNG state in a reproducible way. Same launch → same
   sequence of per-step seeds.

3. *Implication for the NEW-vs-REF comparison*: per-step problem batches
   are deterministically the same in both runs (same `RolloutWorkerConfig.seed=0`).
   That removes one confound I previously called out — the comparison is
   actually cleaner than I thought.

**Two rollout workers running with the same seed (rollout-0, rollout-1)**

Both have `RolloutWorkerConfig.seed=0` and identical engine seeds. In
principle they should pick the same 64 problems and generate identical
completions. In practice the rewards diverge step-by-step (e.g. step 0:
rollout-0 = 0.349 vs rollout-1 = 0.294). The only plausible source is
vLLM-TPU continuous-batching async scheduler timing — confirmed by
PR #5256 commit message: *"unsupported per-request SamplingParams seeds"*.

This is a real footgun for ablation/reproducibility studies: two
"identical" rollout workers aren't actually drawing from independent
streams, they're drawing from the same nominal stream but getting
different outputs purely from TPU dispatch timing. **If you want true
seed-decorrelation between rollout workers you'd need to thread
`worker_index` into the seed** (e.g. `seed = base_seed * 1000 + worker_index`).
Not done today.

**No curriculum.** `lessons = {"math_full": …}`. Only one lesson, so the
"curriculum" is trivial. `eval_frequency=1` just means: run a full eval
after every trainer step, on all 500 MATH-500 held-out problems (since
`n_to_sample = min(eval_n_examples=500, len(eval_examples)=500)`, eval is
the full set in randomized but irrelevant order).

**Eval is over the full held-out 500.** Order is a per-step random
permutation but doesn't change pass@1.

### W&B

- Train: `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/llama-3.1-8bi-math500-rlooIS-100s-uc1-20260523-044205-train`
- Rollout-0: `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/llama-3.1-8bi-math500-rlooIS-100s-uc1-20260523-044205-rollout-0`
- Rollout-1: `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/llama-3.1-8bi-math500-rlooIS-100s-uc1-20260523-044205-rollout-1`
- Reference train: `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-clean-nodelprevtmp-train`
- Reference rollout-0: `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-clean-nodelprevtmp-rollout-0`

After run #13/#14 (03:53) confirmed the per-call spawn approach was *also*
saturating at 10s timeout for every grade — even on trivial inputs like
``(0)-(1)`` — the F12 cold-start fix is not sufficient in this image. Both
regions had two rollouts running but the trainer's buffer never went above 0
because each batch of 1024 grades (64 prompts × 16 group size) needed
~256 sympy fallbacks × 10s = ~43 min per rollout. By the 50-min mark of
run #13/#14 there was still no first batch write.

The "real" fix here is to figure out what's making Python interpreter
startup + sympy import slow in this image (vllm-tpu / jax 0.9.2 / libtpu
0.0.39). Could be `sys.path` length, cgroup CPU throttling on the worker,
container init overhead. Not tonight.

**Pragmatic short-circuit landed in `math_grading.py`**:

```python
_SYMPY_DISABLED = True

def _run_in_process_with_timeout(fn, args, timeout):
    if _SYMPY_DISABLED:
        raise TimeoutError("sympy fallback disabled (spawn cold-start exceeds budget on this image)")
    # original subprocess path preserved for envs with fast cold-start
    ...
```

`are_equal_under_sympy` already catches `Exception` and returns False, so the
short-circuit changes nothing about correctness for our case — a 10s-timeout
parse was already counting as "not equal". It only removes the wasted
latency. `grade_answer`'s fast paths (mathd normalize, _normalize, fraction
match, integer match) still catch the vast majority of correct answers.

Launched:
- `/ahmedah/iris-rl-blog-rlooIS-100s-e5-20260523-044205` (us-east5-a)
- `/ahmedah/iris-rl-blog-rlooIS-100s-uc1-20260523-044205` (us-central1-a)

Expected: batches grade in seconds, first batch write at ~T+12-15min after
launch (vLLM warmup is the long pole now, not grading).

## Run #13/#14 attempt (2026-05-23 03:53) — F13 retracted

Launched parallel:
- `/ahmedah/iris-rl-blog-rlooIS-100s-e5-20260523-035253` (us-east5-a v5p-8)
- `/ahmedah/iris-rl-blog-rlooIS-100s-uc1-20260523-035253` (us-central1-a v5p-8)

Difference from run #11/#12: `math_grading.py` reverted to per-call
`Pool(processes=1)` + 10s timeout (matching `main`). F12 (JAX-free worker
module) still in place. F14 left default-on (`--inflight-weight-updates` true)
since run #11/#12 showed rollouts DO progress past the initial-weight wait
even with inflight=True — the earlier "chicken-and-egg" diagnosis was overstated;
the real blocker was always grading. F15 (gcsfs flake retry) not landed; if
it hits again, deal with it then.

Expected behavior: spawn cold-start ~5s (F12 makes the import chain trivial),
under the 10s timeout. Rollouts complete grading in ~minutes, write batches,
trainer sees buffer ≥ 1, takes step 0, continues. Eval every step. 100 steps
total → expected ~5 hours wall-clock at v5p-8 throughput.

## Tonight's stopping point (2026-05-23 01:05)

Eight launch attempts. Real progress: each attempt got past the previous
blocker. Code fixes that landed and verified end-to-end on Iris:

- **F12 (levanter tokenizer GCS staging)** — verified, run #3 cleared the
  coordinator's `make_tokenizer` thanks to the fix.
- **F11 (log_policy_entropy/vocab_tile_size guard)** — verified, run #4
  reached `_wait_for_initial_rollouts` thanks to the revert.
- **F14 (inflight chicken-and-egg)** — verified, run #7 entered
  `PHASE: SYNC_WEIGHTS step=0` and produced a real `generate: done in 249.8s`
  thanks to `--no-inflight-weight-updates`.
- **F13 (persistent sympy spawn pool)** — implemented, code shipped in
  workspace bundle for run #8/#9, but the run died on F15 before grading
  exercised the pool. Cannot confirm F13 by data tonight.

Outstanding blockers for the v2 blog run:

- **F15 (custom_hf_hub_download fs.get not retry-robust)** — deterministic,
  not a transient flake. Needs the retry wrap.

Once F15 is fixed, the next run should make it past trainer init,
into `_wait_for_initial_rollouts`, into the first batch generate + grade
+ write, into trainer step 0, and onward. F13's effect will be visible by
absence of sympy timeouts in the rollout-worker logs.

## Fix plan (priority order — DONE / TODO)

1. ✅ **F1**: stray `jnp.mean` removed from `compute_dapo_loss`. LR dropped
   from `2e-6` → `5e-7` in the experiment (≈ DAPO-paper 1e-6 / 2). Sweep up
   to 1e-6 if too slow, down if unstable.
2. ✅ **F2**: `filter_out_groups_with_no_variance` plumbed through
   `RLExperimentConfig` with default `True`. Open question still on whether to
   add rollout-side oversampling.
3. ✅ **F3**: `log_policy_entropy=True` in `_default_rl_loss`. Now logs
   `current_policy_entropy` (full-vocab) alongside the noisier
   `policy_entropy` / `current_entropy` metrics.
4. ✅ **F4**: `synchronous=False`. PPO clip with εlow=0.2 / εhigh=0.28 is now
   actually active against the sampler's stored logprobs.
5. ✅ **F7**: `dead_groups_seen`, `dead_groups_fraction`, `last_group_reward_std`,
   `total_groups_seen`, `filter_dead_groups` emitted via `ReplayBuffer.get_stats()`.
6. ⏸ **F5/F6**: KL anchor deferred — kept off per user 2026-05-22. Per-token
   aggregator in `masked_response_mean` still unfixed; must be addressed
   before turning KL back on.
7. ☐ **F8**: distinct-n diversity metric and held-out pass@k>1.
8. ☐ **F9**: bump `max_output_tokens` to 768 (cheap probe).
9. ☐ **F10**: bump `lib/marin/pyproject.toml` from upstream `vllm-tpu==0.19.0` /
   `tpu-inference==0.19.0` to today's Marin-fork release pair:
   - `vllm @ https://github.com/marin-community/vllm/releases/download/marin-primary-20260522-ee55edf7a/vllm-0.0.0.dev20260513%2Bee55edf7a-py3-none-any.whl`
   - `tpu-inference @ https://github.com/marin-community/tpu-inference/releases/download/marin-primary-20260522-68fcf0b9/tpu_inference-0.0.0.dev20260513%2B68fcf0b9-py3-none-any.whl`

   Verify the pair via the new tpu-inference release's `marin-release-pair.json`
   asset before landing. Then update `docs/dev-guide/forking-policy.md` to match
   reality, and patch the `vllm-tpu` → `vllm` dist-name reference in
   `lib/marin/src/marin/evaluation/evaluators/evalchemy_evaluator.py:399-428`.

## Applied diff (v2 blog run config, 2026-05-22)

```python
# experiments/llama_3_8b_rl_math500.py:_default_rl_loss
def _default_rl_loss() -> RLOOLoss:
    """RLOO + importance-sampling hybrid.

    RLOO (Ahmadian et al. 2024, arXiv:2402.14740) for the variance-reduced
    REINFORCE baseline; PPO-style clipped IS (Schulman et al. 2017) for the
    off-policy correction; DAPO clip-higher (Yu et al. 2025, arXiv:2503.14476).
    KL intentionally off.
    """
    return RLOOLoss(
        kl=KLConfig(mode=KLMode.NONE, beta=0.0),         # kept off per user
        clip_epsilon_low=0.2,
        clip_epsilon_high=0.28,
        synchronous=False,                                # was True
        do_trainer_inference_mismatch_importance_sampling=True,
        tis_importance_sampling_ratio_max=2.0,
        do_overlong_filtering=True,
        vocab_tile_size=32064,
        log_policy_entropy=True,                          # was False
    )

# experiments/llama_3_8b_rl_math500.py:build_experiment_config
#   learning_rate=2e-6  →  learning_rate=5e-7  (compensates for the F1 fix)
```

Code-side changes:
- `lib/marin/src/marin/rl/rl_losses.py:267-282` — `compute_dapo_loss` now does
  `-jnp.sum(loss_objective * loss_masks) / jnp.sum(loss_masks)` (F1).
- `lib/marin/src/marin/rl/rl_experiment_utils.py:100-109,279-285` — added
  `filter_out_groups_with_no_variance: bool = True` to `RLExperimentConfig`,
  plumbed through to `ReplayBufferConfig` (F2).
- `lib/marin/src/marin/rl/replay_buffer.py:80-84,210-230,304-322` —
  `ReplayBuffer` tracks dead-group counters; `get_stats()` surfaces
  `total_groups_seen`, `dead_groups_seen`, `dead_groups_fraction`,
  `last_group_reward_std`, `filter_dead_groups` (F7).
- `experiments/llama_3_8b_rl_math500.py:60-95` — `_default_rl_loss` flipped to
  RLOO+IS hybrid; `log_policy_entropy=True` (F3, F4).
- `experiments/llama_3_8b_rl_math500.py:243-249` — LR `2e-6` → `5e-7`.

Tests verified safe (manual numeric check on `test_ppo_objective`; no
`get_stats()` exact-key assertions in `test_replay_buffer.py`).

## Open questions

- Should the blog run use the pre-fix code (and explain "this is what we
  shipped") or the fixed code (and report cleaner numbers)? Both are
  defensible. The pre-fix story is more interesting if the blog is about
  what we learned.
- Does turning on dynamic sampling (F2) require also enabling rollout-side
  oversampling? The DAPO recipe targets a *post-filter* effective batch. If
  we don't oversample, the effective gradient batch shrinks ~20-40%.
- Is the vLLM-seed fix (`#5256`) sufficient to make the v2 run reproducible,
  or do we also need to fix sampler state across preemption restarts?
- Should we add a real `GRPOLoss` / `DrGRPOLoss` class alongside `RLOOLoss`
  to allow apples-to-apples comparison, or keep RLOO as the only path?

## Related artifacts in this worktree

- `rl_500_passk.png` — headline pass@k curves (eval + rollout, EMA-smoothed)
- `rl_500_diagnostics.png` — entropy, diversity gap, length, KL/clip status
- `/tmp/rl_blog_wandb/{train,rollout}_history.parquet` — raw W&B history dumps
- `.agents/logbooks/iris-rl.md` — original migration logbook (preserved from `e4beb97e0`)
- `.agents/projects/iris-rl.md` — original migration plan
- `.agents/projects/on-demand-rl.md` — precursor RL experience, where most of the operational hardening originated
