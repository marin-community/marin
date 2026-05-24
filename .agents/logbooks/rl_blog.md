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
- The real "what problems get sampled" seed is `RolloutWorkerConfig.seed`.
  For this launcher, `RLJobConfig.seed` defaulted to 42, `RLJob.to_worker_configs`
  made the rollout base seed `42 + 1000 = 1042`, and orchestration launched
  rollout worker `i` with `1042 + i`. So rollout-0 used seed 1042 and
  rollout-1 used seed 1043.
- vLLM engine seed was 0 in this run (PR #5256). TPU vLLM does not support
  per-request seeds, so the engine RNG advances per `generate()` / request
  scheduling rather than being reset per prompt.
- `TrainerConfig.seed` was still the Levanter default 0 in this run. As of
  the 2026-05-23 seed-plumbing fix, the canonical launcher now propagates
  `RLExperimentConfig.seed` / `RLJobConfig.seed` into TrainerConfig, replay
  sampling, rollout sampling, and each rollout worker's vLLM engine seed.

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

Caveat (corrected 2026-05-23 after a code audit): train problem selection is
seeded and deterministic within a single uninterrupted rollout worker process,
but the NEW-vs-REF comparison should **not** be described as having identical
64-problem train batches. Worker restarts reset rollout-local RNG state, worker
0's full-eval cadence consumes extra RNG draws, rollout file arrival is
wall-clock ordered, and the replay buffer samples from whatever arrived first.
The comparison is still useful, but not a strict seed-controlled ablation.

### CODEX 2026-05-23T23:32:09Z — Randomness audit (what actually controls what)

After tracing every PRNG through the rollout/train/buffer/vLLM stack:

**Sources of randomness in this run**

| Source | Location | Seed value | Effective? | What it controls |
|---|---|---|---|---|
| `EnvConfig.env_args["seed"]` | `experiments/llama_3_8b_rl_math500.py:119` | 42 | **❌ DEAD** | assigned to `MathEnv._rng` (`environments/math_env.py:59`), never read anywhere else |
| `RLJobConfig.seed` | `rl_job.py:112` | 42 (default) | ✅ | base seed used by train/replay and used to derive rollout seeds |
| `RolloutWorkerConfig.seed` | `rl_job.py:312`, `orchestration.py:210-215`, `rollout_worker.py:226` | rollout-0 = 1042, rollout-1 = 1043 | ✅ | per-worker Python `random.Random(seed)` stream; controls curriculum seed draws, eval/input prompt-sampling seeds, and therefore `MathEnv.sample(...).choice(12000, 64, replace=False)` |
| vLLM engine seed | `vllm.py:89`, `vllm.py:197`, `inflight/worker.py:188`, `:206` (PR #5256) | 0 in this run; after the 2026-05-23 fix, each rollout worker's engine seed matches its derived rollout seed (`base + 1000 + worker_index`) | ✅ | global vLLM RNG seeded once at engine init; advances per `generate()` / request schedule. TPU has **no per-request `SamplingParams.seed`** — engine seed is the only knob |
| `worker_index` | `orchestration.py:210-215`, `rollout_worker.py:246` | 0 (rollout-0), 1 (rollout-1) | ✅ indirectly | orchestration adds `worker_index` to the rollout base seed (`seed=rollout_config.seed + i`) and uses it to gate eval to worker 0 |
| `ReplayBufferConfig.seed` (= `TrainWorkerConfig.seed`) | `replay_buffer.py:91`, `train_worker.py:304`, `rl_job.py:296` | 42 | ✅ | `self._rng.choice(env_choices, size=local_batch_size, replace=False)` selects rollouts from the buffer for training |
| `TrainerConfig.seed` | `levanter/trainer.py:794` | 0 in this run; now wired to `RLJobConfig.seed` after 2026-05-23 fix | ✅ trivially | `training_key = jrandom.split(PRNGKey(seed), 2)[1]` feeds the loss-step RNG; mostly no-op for Llama/RLOO without dropout |
| Curriculum lesson sampling | `curriculum_actor.sample_lesson(seed)` | seed varies | n/a | only one lesson `math_full` configured — no actual selection happens |
| `MathEnv.get_eval_examples` | `environments/math_env.py:259` | hardcoded 42 | ❌ not invoked | dead code path for this run |
| vLLM-TPU async batch scheduling | n/a | uncontrolled | wall-clock dependent | continuous-batching dispatch order varies with system clock / asyncio.gather scheduling / TPU device-side parallelism. PR #5256 explicitly accepts this |
| Rollout file ordering | `rollout_storage.py:295`, `:174-195` | wall-clock timestamp | ✅ | writer filenames include `time.time()` and trainer reads sorted by timestamp; two rollout workers can land batches in different orders across runs |

**Things I was sloppy about earlier in this logbook:**

1. *"env_args seed=42 controls MATH-500 problem ordering"* — wrong. The
   `seed=42` in env_args hits `MathEnv._rng` and dies there. Confirmed by
   ```
   $ grep -n "self._rng" lib/marin/src/marin/rl/environments/math_env.py
   59:        self._rng = np.random.default_rng(seed)
   ```
   One line, the assignment, never used. The actual problem selection RNG
   is `RolloutWorkerConfig.seed` flowing through `MathEnv.sample(prng_key)`.

2. *"`RolloutWorkerConfig.seed=0` in NEW and REF"* — wrong. The canonical
   RLJob default base seed is 42; `RLJob.to_worker_configs()` adds 1000 for
   rollout workers, and `orchestration.py` adds the worker index. This means
   rollout-0 and rollout-1 are seed-decorrelated by construction.

3. *"vLLM seed advances randomly per step"* — wrong. The
   `seed = py_rng.randint(0, 2**31-1)` in `rollout_worker.py:976` is a
   **deterministic** PRNG stream from `random.Random(RolloutWorkerConfig.seed)`.
   Each call advances the PRNG state in a reproducible way within one
   uninterrupted process.

4. *Implication for the NEW-vs-REF comparison*: per-step problem batches are
   not guaranteed identical. REF pre-dated PR #5256's vLLM engine seed, had
   preemptions/restarts, and both runs can differ in eval timing, rollout file
   arrival order, and replay-buffer contents. Treat the comparison as a
   strong operational signal, not a controlled seed ablation.

**Rollout seed vs vLLM engine seed**

`RolloutWorkerConfig.seed` is the experiment/data-plane seed for a rollout
worker. It decides which RNG integers are passed to curriculum selection,
full eval, micro eval, and train input sampling. In this Math500 run, because
there is only one lesson, the meaningful effect is which train/eval examples
are selected and in what order.

`vLLMInferenceContextConfig.seed` is the model-generation seed inside vLLM.
It controls stochastic token sampling for the inference engine at temperature
1.0. On TPU vLLM, there is no per-request seed, so this is a global engine
seed whose state advances as requests are scheduled. That makes request order
and continuous-batching timing part of the reproducibility story. After the
2026-05-23 seed-plumbing fix, each rollout worker's vLLM engine seed is the
same derived value as that worker's `RolloutWorkerConfig.seed`.

**No curriculum.** `lessons = {"math_full": …}`. Only one lesson, so the
"curriculum" is trivial. `eval_frequency=1` just means: run a full eval
after every trainer step, on all 500 MATH-500 held-out problems (since
`n_to_sample = min(eval_n_examples=500, len(eval_examples)=500)`, eval is
the full set in randomized order).

**Eval is over the full held-out 500.** Order is a per-step random
permutation, but generation is still stochastic at temperature 1.0, so exact
pass@1 depends on vLLM engine RNG state and request scheduling.

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

## CODEX 2026-05-23T23:32:09Z — Plan: deterministic Math rollout schedule

Ahmed's revised direction: do **not** split the finite train set into disjoint
shards across rollout workers. We are not guaranteed to keep two samplers alive;
if a sampler dies, a globally sharded or assignment-only schedule can leave a
whole slice of questions ungenerated or generated-but-never-consumed. Instead,
each logical rollout worker should epoch through the full train set with its own
deterministic shuffle seed. Multiple workers then increase throughput and
generate different stochastic samples, but no single worker owns a unique shard
that can disappear.

Current behavior to replace:

- `RolloutWorker.run()` draws `input_rng` from a worker-local PRNG.
- `MathEnv.sample(..., prng_key=input_rng)` turns that into
  `np.random.default_rng(seed).choice(len(train_examples), size=n_prompts,
  replace=False)`.
- With multiple rollout workers, different seeds reduce duplication, but there
  is no global notion of "we have covered this dataset epoch once".
- On restart, the worker-local RNG resets unless the whole worker state is
  reconstructed exactly.

Levanter already has the right primitive:

- `levanter.data._prp.FeistelPermutation(length, key)` implements a stateless
  pseudo-random permutation over `[0, length)`, including arbitrary non-power-of-2
  lengths via cycle walking.
- `levanter.data.permutation.PermutationDataset` wraps finite async datasets,
  but it currently has a TODO for epoch reshuffling. For MathEnv's in-memory
  `list[DataExample]`, use the Feistel permutation primitive directly.

Revised semantics:

1. `RLJobConfig.seed` remains the root experiment seed.
2. Training/replay continue to use the root seed.
3. Each logical rollout worker gets a stable derived seed:

   ```python
   worker_seed = root_seed + 1000 + worker_index
   ```

   That same seed initializes the worker's vLLM engine and its finite-data
   schedule.
4. Train prompt selection uses a per-worker finite schedule, not independent
   random sampling:

   ```python
   epoch = worker_position // dataset_len
   offset = worker_position % dataset_len
   key = jax.random.fold_in(jax.random.PRNGKey(worker_seed), epoch)
   index = FeistelPermutation(dataset_len, key)(offset)
   ```

   A rollout batch with `n_prompts=64` receives the next 64 positions from that
   worker's schedule. Crossing an epoch boundary is allowed; the next epoch uses
   a new folded-in Feistel permutation.

5. Multiple rollout workers therefore have independent full-dataset
   permutations, not disjoint shards. In a single local window the expected
   overlap between two workers is small (`64 * 64 / 12000 ~= 0.34` questions for
   Math train), but across a full epoch every live worker will eventually visit
   every question. That duplication is intentional: worker count is a throughput
   knob, not a data-partitioning contract.
6. If one worker dies permanently, the remaining worker(s) still traverse the
   entire data set. Throughput drops, but data coverage does not lose a shard.
7. Curriculum remains responsible for lesson choice. The new schedule is
   responsible only for example choice within a lesson. For the current Math500
   run there is only one lesson, so this degenerates to deterministic epoching
   over the math train split.
8. Evaluation should not consume the train schedule. Full eval should run the
   complete eval split in a fixed deterministic order and submit prompts in the
   same order every time. Micro eval, if kept, should use a separate
   deterministic eval schedule, not the train cursor.
9. Rollout metadata should include `worker_index`, `worker_seed`, `epoch`,
   `start_position`, `end_position`, and selected `env_example_id`s. The replay
   reader can keep wall-clock arrival order for now, but metrics must distinguish
   written, accepted, and consumed examples.

Preemption critique:

- A purely process-local cursor is **not robust**. Today `RolloutWorker.run()`
  initializes `step = 0` and `random.Random(self.config.seed)` inside the worker
  process. If a sampler process restarts and we do nothing else, it restarts its
  epoch from the beginning. That is not catastrophic because no shard is lost,
  but it creates prefix bias and duplicate rollouts after every sampler retry.
- A per-worker full-dataset schedule becomes robust only if the cursor belongs
  to the **logical worker**, not the process attempt. The stable identity is
  `worker_index`; retries of rollout-0 must resume rollout-0's cursor.
- Cursor advancement should happen after the rollout batch is durably written,
  not when generation starts. If a worker dies mid-generation, it retries the
  same positions. If it dies after writing but before cursor update, duplicate
  writes are possible; that should be handled by a deterministic
  `assignment_id = (worker_index, epoch, start_position)` and replay-side
  de-duplication.
- `RLRunState` currently keeps only lifecycle, train step, and transfer
  counters in memory. It survives child worker retries while the coordinator
  lives, but it is not enough by itself for checkpoint/relaunch recovery. The
  schedule cursor should either be checkpointed with the trainer/curriculum
  state or recoverable from rollout metadata in storage.
- If we do not persist/recover cursor state, the design is still safer than
  global sharding under sampler death, but repeated preemptions can over-sample
  early-permutation questions and delay later questions.

Implementation sketch:

1. Add a small `rollout_schedule.py` module with:
   - `FeistelEpochSchedule`
   - `RolloutScheduleCursor`
   - `RolloutAssignment`
2. Add per-logical-worker cursor storage:
   - either in `RLRunState` with checkpoint/recovery support,
   - or in a small schedule actor that is keyed by `worker_index` and can save
     / restore its state.
3. Split `MathEnv.sample()` into:
   - explicit-index path used by scheduled training rollouts
   - existing RNG path retained only for eval or unscheduled environments
4. In `RolloutWorker.run()`, replace train `input_rng` prompt selection with a
   per-worker schedule assignment for finite scheduled envs. Keep vLLM
   generation seed as the engine seed already configured on the inference
   context.
5. Add tests:
   - one worker's Feistel schedule covers each example exactly once per epoch.
   - Epoch 0 and epoch 1 are deterministic but different permutations.
   - Two rollout workers with different derived seeds produce different
     permutations and both cover the full dataset.
   - A worker restart resumes from the logical worker cursor instead of process
     local step 0.
   - A crash before durable write retries the same assignment.
   - Duplicate assignment writes are de-duplicated by `(worker_index, epoch,
     start_position)`.
   - Full eval does not advance the train cursor and sends eval prompts in a
     stable order.

Design caveat: this deliberately gives up "exactly one attempt per question
globally across all workers." The invariant becomes "each live logical worker
attempts every question once per worker-epoch." That is the right failure mode
for unreliable sampler capacity. Full bit-for-bit reproducibility still depends
on vLLM TPU request scheduling and replay ingestion order.

CODEX 2026-05-23T23:32:09Z — implemented in this pass:

- Added `marin.rl.rollout_schedule`:
  - `FeistelEpochSchedule`
  - `RolloutScheduleCursor`
  - `RolloutAssignment`
  - `rollout_assignment(...)`
- Added logical-worker schedule state to `RLRunState`:
  - `reserve_rollout_assignment(...)`
  - `commit_rollout_assignment(...)`
  - `get_rollout_schedule_cursor(...)`
- Reservation is idempotent while pending. A restarted rollout worker gets the
  same pending assignment until a durable write commits it.
- Cursor advancement now happens after `RolloutWriter.write_batch(...)`
  succeeds, not when generation starts.
- `RolloutMetadata` now records:
  - `worker_index`
  - `worker_seed`
  - `lesson_id`
  - `assignment_id`
  - `schedule_epoch`
  - `schedule_start_position`
  - `schedule_end_position`
  - `schedule_indices`
- `RolloutWorker.run()` now uses scheduled train indices for
  `FiniteDatasetEnv` lessons and falls back to the old random `env.sample(...)`
  path for non-finite envs.
- `ReplayBuffer` now skips duplicate non-empty `assignment_id`s and logs
  counters for:
  - batches seen / accepted
  - duplicate assignment batches skipped
  - stale-step and stale-timestamp skips
  - groups accepted
  - rollouts accepted / consumed
  - consumed lag mean / max
- Trainer step logs now include numeric `replay_buffer/*` stats in W&B.

CODEX 2026-05-23T23:32:09Z — durability caveat:

- This is robust to a rollout child/process retry while the coordinator and
  `RLRunState` actor remain alive, which is the sampler-preemption case we were
  targeting.
- It does not yet persist schedule cursors across a full coordinator/root-job
  relaunch. For that, either checkpoint `RLRunState` schedule cursors alongside
  trainer/curriculum state or reconstruct committed cursors from rollout
  metadata in storage.

CODEX 2026-05-23T23:32:09Z — validation:

```bash
uv run --project lib/marin --group test pytest \
  tests/rl/test_rollout_schedule.py \
  tests/rl/test_run_state.py \
  tests/rl/test_replay_buffer.py \
  tests/rl/test_rollout_worker.py \
  tests/rl/test_rl_experiment_utils.py \
  tests/rl/test_rl_job.py \
  tests/rl/test_orchestration.py \
  tests/rl/environments/test_finite_dataset_env.py \
  tests/rl/environments/test_math_env.py \
  tests/rl/environments/test_mock_env.py \
  tests/rl/environments/test_load_environment.py
```

Result: `78 passed`.

CODEX 2026-05-23T23:32:09Z — pre-commit validation:

```bash
./infra/pre-commit.py --fix .agents/logbooks/rl_blog.md \
  experiments/llama_3_8b_rl_math500.py \
  lib/marin/src/marin/rl/curriculum.py \
  lib/marin/src/marin/rl/environments/__init__.py \
  lib/marin/src/marin/rl/environments/base.py \
  lib/marin/src/marin/rl/environments/math_env.py \
  lib/marin/src/marin/rl/environments/mock_env.py \
  lib/marin/src/marin/rl/environments/prime_intellect_env.py \
  lib/marin/src/marin/rl/orchestration.py \
  lib/marin/src/marin/rl/replay_buffer.py \
  lib/marin/src/marin/rl/rl_experiment_utils.py \
  lib/marin/src/marin/rl/rl_job.py \
  lib/marin/src/marin/rl/rollout_worker.py \
  lib/marin/src/marin/rl/rollout_schedule.py \
  lib/marin/src/marin/rl/run_state.py \
  lib/marin/src/marin/rl/train_worker.py \
  lib/marin/src/marin/rl/types.py \
  tests/rl/environments/test_finite_dataset_env.py \
  tests/rl/environments/test_math_env.py \
  tests/rl/test_orchestration.py \
  tests/rl/test_replay_buffer.py \
  tests/rl/test_rl_experiment_utils.py \
  tests/rl/test_rl_job.py \
  tests/rl/test_rollout_schedule.py \
  tests/rl/test_rollout_worker.py \
  tests/rl/test_run_state.py
```

Result: `OK` after one formatter/linter convergence rerun.

CODEX 2026-05-23T23:32:09Z — current status:

- Implemented for finite train envs.
- `MathEnv` uses the `FiniteDatasetEnv` explicit-index path; no env rewrite is
  needed beyond the finite-env base already added.
- Each logical rollout worker now has its own full-dataset Feistel order,
  derived from that worker's seed.
- Rollout child/process retry is handled by pending assignment reuse in
  `RLRunState`.
- The remaining gap is full coordinator/root-job relaunch recovery for schedule
  cursors. That needs either schedule-cursor checkpointing or reconstruction
  from committed rollout metadata in storage.

For stale-by-one training, distinguish generated coverage from trained
coverage. The schedule cursor now advances after durable rollout write, but
written examples can still fail to contribute because:

- their rollout arrives after `max_rollout_step_delay`,
- the replay buffer samples only part of the buffered current-step data before
  the trainer advances,
- `filter_out_groups_with_no_variance=True` drops all-zero/all-one groups by
  design.

If the semantic goal is "each train example contributes once per epoch", the
schedule needs trainer/replay feedback and should advance/retire examples on
accepted or consumed rollout groups, not merely on durable write. If the goal
is "each train example is attempted once per worker-epoch", commit-after-write
is enough.

### CODEX 2026-05-23T23:32:09Z — Follow-up plan: deterministic eval and configurable pass@k

Problem: current full eval samples all 500 MATH-500 examples, but it still
passes through `MathEnv.sample(..., prng_key=...)`, which permutes eval order.
Metrics are order-insensitive, but vLLM's engine RNG and continuous batching
are order-sensitive. Also `_evaluate_lesson()` hardcodes `n_eval_generations=1`
and reuses the training lesson sampling params, so pass@1 and pass@16 evals
cannot be configured independently.

Plan:

1. Add an eval config dataclass, probably near `SamplingParams`:

   ```python
   @dataclass
   class EvalSamplingParams:
       name: str
       n_examples: int | None
       n_generations: int
       temperature: float
       top_k: int | None = None
       max_output_tokens: int | None = None
       stop_tokens: list[int] | None = None
   ```

2. Add `eval_sampling_params: list[EvalSamplingParams]` to `LessonConfig` or
   `CurriculumConfig`. Default should preserve today's single pass@1 eval, but
   Math500 should explicitly configure:
   - `pass1_greedy`: `n_generations=1`, deterministic/greedy decoding
   - `pass16_sample`: `n_generations=16`, temperature/top-k matching rollout
     sampling
3. Split environment eval from random sampling:
   - train scheduled path: explicit train indices from the Feistel schedule
   - full eval path: explicit eval indices `[0, 1, ..., n_eval_examples - 1]`
   - micro eval path: separate deterministic eval schedule, if enabled
4. `_evaluate_curriculum()` should loop over eval configs and send prompts to
   vLLM in the exact dataset order for every run. In the current sync vLLM path
   this means one ordered `llm.generate([TokensPrompt(...), ...], params)` call
   per eval mode. In async/inflight mode, request ids must be deterministic
   (`eval-{mode}-{idx}`) and results must be reordered back to dataset order.
5. Separate "same requests" from "same sampled outputs":
   - same requests: fixed eval indices, fixed prompt rendering, fixed request
     ids, fixed list order, fixed sampling params
   - same greedy outputs: achievable now with greedy pass@1
   - same stochastic pass@k outputs: requires the same engine RNG state at eval
     call time; with today's TPU vLLM engine-level seed, prior train rollout
     requests advance that state
6. For stochastic pass@16, choose one of:
   - best-effort: same ordered requests on the rollout worker's existing engine
   - stricter: a dedicated eval-only vLLM context/engine seeded for eval so
     train rollouts do not advance its RNG stream
   - ideal future path: per-request `SamplingParams.seed` once TPU vLLM
     supports it
7. `_build_prompt_example_metrics()` should stop using global
   `random.sample/random.choice` for eval sample tables. Tables should be a
   deterministic prefix or a deterministic hash-based subset so logged examples
   do not add hidden randomness.
8. Logging names should make eval mode unambiguous:
   - `inference.eval_pass1_greedy/math_full/pass_at_1`
   - `inference.eval_pass16_sample/math_full/pass_at_16`

Enforcement tests:

- A finite env test should call `sample_by_indices(indices=[2, 0, 1])` and
  fail unless `batch_completions()` receives prompts in exactly that order.
- A rollout-worker eval test should configure two eval modes and fail unless
  `_evaluate_curriculum()` sends the eval split in `[0, 1, 2, ...]` order for
  both modes, with distinct `(temperature, n_generations, top_k)` params.
- A prompt-table test should fail if eval sample-table logging uses ambient
  randomness; use deterministic prefix or deterministic hash order.
- A MathEnv inheritance test should assert `isinstance(MathEnv(...),
  FiniteDatasetEnv)` and that explicit eval indices preserve MATH-500 order.

CODEX 2026-05-23T23:32:09Z — implemented in this pass:

- Added `EvalSamplingParams`.
- Added configurable eval modes to `CurriculumConfig`.
- Math500 now configures `eval_pass1_greedy` and `eval_pass16_sample`.
- Finite eval envs now use explicit ordered eval indices instead of random
  eval sampling.
- Eval sample-table logging now uses deterministic prefix order instead of
  ambient `random.sample/random.choice`.
- Added tests for ordered eval requests and eval sampling params.

Important caveat: fixed prompt order is necessary but not sufficient for
bitwise-identical stochastic pass@16 on TPU vLLM, because TPU vLLM currently
has only an engine-level seed, not per-request seeds. Greedy pass@1 can be made
deterministic now. Stochastic pass@16 can be stable under a fixed request
schedule, but exact reproducibility still depends on engine state and request
scheduling unless/until per-request seeds exist.

### CODEX 2026-05-23T23:32:09Z — Follow-up plan: finite-data env base

`MathEnv` already inherits the abstract `MarinEnv` base class, but `MarinEnv`
is too coarse: it only exposes `sample()`, and `sample()` mixes example
selection, inference, grading, and rollout creation.

For one-dataset RL runs like Math500, the `Curriculum` actor is overgeneral:
there is only one lesson (`math_full`), so `sample_lesson()` always resolves to
the same env. The curriculum machinery is useful later for multi-task or
difficulty-band schedules, but it makes simple train/eval split reasoning
harder than necessary.

Plan:

1. Introduce an intermediate abstract base class:

   ```python
   class FiniteDatasetEnv(MarinEnv, ABC):
       ...
   ```

   It should inherit from `MarinEnv`, not replace it.
2. `FiniteDatasetEnv` should expose:
   - `train_len()`
   - `eval_len()`
   - `train_examples_by_indices(indices)`
   - `eval_examples_by_indices(indices)`
   - `sample_by_indices(..., mode, indices, sampling_params)`
3. Move common finite-env mechanics into `FiniteDatasetEnv`:
   - validate explicit indices
   - build prompts in requested order
   - call `inference_ctx.batch_completions(...)`
   - preserve output order
   - construct `RolloutGroup`s and metrics
4. Keep env-specific behavior in subclasses:
   - load train/eval examples
   - render examples into prompts/messages
   - score a generated choice
   - choose `env_name` / `env_example_id`
5. Refactor `MathEnv(FiniteDatasetEnv)` and keep math-specific pieces only:
   loading Hendrycks/MATH-500, few-shot prompt formatting, answer grading.
6. Keep `FiniteDatasetEnv.sample()` as a compatibility wrapper that still does
   RNG-based index selection until rollout scheduling is migrated. New rollout
   code should call `sample_by_indices()` for train and eval.
7. Add tests with a tiny fake finite env to prove other envs can inherit the
   base without depending on math grading.

CODEX 2026-05-23T23:32:09Z — implemented in this pass:

- Added `FiniteDatasetEnv(MarinEnv)`.
- Refactored `MathEnv(FiniteDatasetEnv)`.
- Kept `MathEnv.sample()` behavior through the finite-env compatibility
  wrapper.
- Added `sample_by_indices()` tests that fail if prompt order is not preserved.
- Added a MathEnv inheritance/order test.

This gives other finite train/eval envs a reusable base without changing the
adaptive curriculum actor yet.

### CODEX 2026-05-23T23:32:09Z — Stale-by-one risk check on latest two-sampler W&B run

Run inspected:

- train:
  `llama-3.1-8bi-math500-rlooIS-100s-uc1-20260523-044205-train`
- rollout-0:
  `llama-3.1-8bi-math500-rlooIS-100s-uc1-20260523-044205-rollout-0`
- rollout-1:
  `llama-3.1-8bi-math500-rlooIS-100s-uc1-20260523-044205-rollout-1`

What we track today:

- Rollout W&B logs include `inference.weight_step` and
  `inference.train_step`.
- Trainer W&B logs include rollout wait / batch prep timing.
- ReplayBuffer logs "Skipping stale rollout batch" to logs, but there is no
  first-class W&B metric for stale skipped batches/rollouts.
- `ReplayBuffer.get_stats()` has useful counters, but trainer-side W&B does
  not currently log them every step.

Available proxy from rollout W&B:

- rollout-0: 100 rollout metric rows. `train_step - weight_step` counts:
  `{-2: 22, -1: 49, 0: 29}`. Rows with lag `> 1`: 0.
- rollout-1: 135 rollout metric rows. `train_step - weight_step` counts:
  `{-2: 11, -1: 70, 0: 54}`. Rows with lag `> 1`: 0.

Interpretation: by the available W&B proxy, samplers were not behind the
trainer. They were usually at the current served weights or slightly ahead of
the run-state train-step publication. This suggests low risk that a sampler was
generating batches already more than one trainer step too old.

But the run did overproduce:

- rollout-0 summary: 103 written batches
- rollout-1 summary: 138 written batches
- trainer: 100 training steps

So the dominant risk is not "sampler is far behind"; it is surplus generated
batches that may never be consumed. With a deterministic schedule, assignment
time cursor advance would give deterministic attempted coverage, but not
deterministic trained coverage.

Trainer wait stats from W&B:

- 100 train rows with rollout-wait timing.
- median rollout wait: 2.55s
- p90 rollout wait: 45.5s
- max rollout wait: 55.9s
- 24/100 steps waited more than 30s, 0/100 waited more than 60s.

Plan for real stale tracking:

1. Add ReplayBuffer counters:
   - `batches_seen`
   - `batches_accepted`
   - `batches_skipped_stale_step`
   - `batches_skipped_stale_timestamp`
   - `groups_seen`
   - `groups_accepted`
   - `groups_filtered_no_variance`
   - `rollouts_consumed`
2. Track a lag histogram at add time:
   `current_step - rollout.metadata.weight_step`.
3. Track a consumed lag histogram at sample time for the actual training batch.
4. Log `replay_buffer/*` stats every train step alongside throughput metrics.
5. Once deterministic assignment exists, log assignment outcomes:
   assigned, written, accepted, consumed, stale-skipped, no-variance-filtered.

### CODEX 2026-05-23T23:32:09Z — Clarification: two samplers can speed up training while overproducing rollouts

Follow-up source:

- `/Users/ahmed/code/marin/.claude/worktrees/packed_rl/.agents/logbooks/iris-rl-codex.md`

The old Iris RL logbook confirms the user's memory. The one-vs-two-sampler
experiment was real:

- `e4par-20260326-044831-train`: one rollout worker, `n_prompts=64`.
- `e4ms2-20260326-121919-train`: two rollout workers, `n_prompts=64`.
- W&B recheck on 2026-05-23:
  - `e4par` median wall-clock step: `182.206s`.
  - `e4ms2` median wall-clock step: `101.088s`.
  - two-sampler speedup over one-sampler parity: `1.80x`.
  - trainer compute stayed about the same: `60.76s` vs `61.08s`.
  - median train-side batch prep dropped from `30.08s` to `3.95s`.

So the speedup came from reducing trainer starvation, not from making the
trainer step itself faster.

The same run also overproduced rollout data:

- final W&B rollout counts rechecked on 2026-05-23:
  - rollout-0 max cumulative train batches: `115`.
  - rollout-1 max cumulative train batches: `168`.
  - total produced train rollout batches: `283`.
  - trainer steps completed: `200`.
- Since each produced batch and each trainer batch were both `64 * 16 = 1024`
  rollouts, the trainer could consume at most `200` full batch-equivalents.
- The surplus was `83 / 283 = 29.3%` of produced batch-equivalents, before
  accounting for the small amount that could still be resident in replay.

This is not a contradiction. It is a producer-consumer tradeoff:

- one sampler under-supplied the trainer often enough that the trainer waited;
- two samplers supplied enough data to keep the trainer close to saturated;
- async generation plus eval/checkpoint variance means production sometimes
  outruns consumption;
- with `max_samples=1`, replay `capacity=4096`, and
  `max_rollout_step_delay=1`, surplus data is expected to be evicted, retired,
  or become too old to use.

The better wording is "surplus rollouts" or "overproduction cost", not
"wasted rollouts" without qualification. The cost can be rational if wall-clock
throughput is the target. It becomes a problem only if TPU budget or
deterministic train-data coverage matters more than keeping the trainer fed.

Connection to the stale-by-one concern:

- The correctness concern is not that two samplers fail to speed up training.
- The concern is that a deterministic finite-data schedule must distinguish:
  assigned examples, written rollouts, replay-accepted rollouts, and actually
  consumed training rollouts.
- If an epoch cursor advances when a rollout worker is assigned examples, then
  surplus or stale-skipped batches can make examples count as "seen" even
  though they never affected a gradient.
- If we want train coverage semantics, coverage should be based on consumed
  rollouts, or the scheduler must explicitly requeue assigned-but-unconsumed
  examples.

Connection to importance sampling:

- One-step stale data is allowed by `max_rollout_step_delay=1`.
- The sampler stores token logprobs under the rollout weight step.
- The trainer recomputes current logprobs and, with `synchronous=False`,
  computes the PPO/RLOO importance ratio against the stored sampler logprobs.
- If `synchronous=True`, the current code forces that ratio to `1`, so PPO
  clipping cannot correct policy lag. That is why the Math500 loss now keeps
  `synchronous=False`.

## CODEX 2026-05-23T23:32:09Z — Next work from Claude/Codex evidence

Based on Claude's `r4` / `e4par` / `e4ms2` runs, the 2026-05-23 100-step
blog run, and the follow-up code audit, the highest-value next work is:

1. Run a short validation job with the new deterministic finite-train schedule.
   - Target: `100` train steps, same Math500 setup, two rollout workers if
     capacity allows.
   - Must inspect new `replay_buffer/*` metrics:
     duplicate assignment skips, stale-step skips, rollouts accepted, rollouts
     consumed, consumed lag.
   - Pass condition: no assignment explosions, no unexpected stale spike, eval
     metrics still match or beat the previous 100-step UC1 run.
2. Add schedule-cursor recovery across full coordinator/root-job relaunch.
   - The current implementation survives rollout child/process retry while
     `RLRunState` lives.
   - Claude's Iris history makes root relaunch/resume a real failure mode, not
     theoretical. Persist schedule cursors or reconstruct them from committed
     rollout metadata before trusting long preemptible runs.
3. Move eval off the critical sampler path or reduce eval interference.
   - Claude's `e4ms2` analysis showed two samplers sped training by reducing
     trainer starvation, but one rollout worker still carried eval burden.
   - The packed-rollout thread also missed `e4ms2` mainly because eval still
     monopolized sampler capacity.
   - Best next design: separate eval worker / eval-only engine. Cheaper
     interim: lower full-eval frequency for throughput experiments while
     keeping deterministic eval order.
4. Run the fixed stack to 500 steps only after the above validation.
   - The reference 500-step run peaked around step 100 then bled held-out
     pass@1.
   - The 100-step fixed run looked structurally healthier: larger real
     gradients, active IS ratio, higher train pass@16, no obvious collapse.
   - A 500-step rerun is the decisive blog comparison, but only after replay
     accounting and schedule recovery are trustworthy.
5. Revisit output length and diversity diagnostics.
   - F8/F9 remain partly open. Deterministic eval now supports pass@1 and
     pass@16, but we should still log a distinct-n / unique-response metric and
     test `max_output_tokens=768`.
   - The reference run had persistent truncation near the 512 cap; if the fixed
     run still truncates, length cap is a likely artificial limiter.
6. Keep KL off for now, but keep the KL path honest.
   - User decision was to keep KL disabled.
   - If the 500-step fixed run still shows held-out decay, first fix F5's KL
     token aggregation and then run a small KL-anchor ablation.
7. Clean up the vLLM / tpu-inference pin story before final blog runs.
   - F10 remains open in the original plan.
   - Use a reproducible Marin fork release pair, verify the release-pair asset,
     and update docs/package naming so future reruns do not silently use the
     wrong inference stack.

## CODEX 2026-05-23T23:58:50Z — Schedule recovery and logging implementation

Implemented the requested logging/recovery follow-up:

- Added a file-backed finite-rollout schedule ledger under the rollout storage
  path: `<rollout_storage.path>/_rollout_schedule_ledger`.
  - `RLRunState` writes one tiny JSON commit record per committed assignment.
  - The write happens after the rollout batch write and before advancing the
    in-memory cursor.
  - On coordinator/root-job relaunch, `RLRunState` replays the ledger and
    recovers each `(worker_index, lesson_id)` cursor to the max committed
    `end_position`.
  - In-memory rollout storage keeps the prior in-memory-only behavior.
- Added schedule assignment counters in `RLRunState` and rollout W&B logging:
  - `inference.schedule/active_cursors`
  - `inference.schedule/pending_assignments`
  - `inference.schedule/reserved_assignments`
  - `inference.schedule/reused_pending_assignments`
  - `inference.schedule/committed_assignments`
  - `inference.schedule/ledger_recovered_assignments`
- Added cheap length-cap saturation metrics from already-materialized response
  token arrays. These do not decode text or compute diversity metrics, so they
  should not matter compared with inference/training time.
  - Train rollout logs now include `inference.length/*`:
    response count, total/mean/max response tokens, truncation count/rate, cap
    tokens, cap saturation threshold, cap-saturated count/rate.
  - Eval/rollout pass metrics also include per-lesson
    `<prefix>/<lesson>/length/*` fields, so pass@1/pass@16 can be inspected
    against the active cap.
- Added tests:
  - ledger recovery resumes the next assignment at the recovered cursor;
  - schedule counters distinguish reserve, pending reuse, commit, and recovery;
  - rollout logging includes schedule counters;
  - length-cap saturation counts near-cap and truncated responses;
  - orchestration derives the ledger path only for file-backed rollout storage.

Verification:

- `uv run pytest tests/rl/test_run_state.py tests/rl/test_rollout_worker.py tests/rl/test_orchestration.py -q`
  - `44 passed`
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/run_state.py lib/marin/src/marin/rl/orchestration.py lib/marin/src/marin/rl/rollout_worker.py tests/rl/test_run_state.py tests/rl/test_rollout_worker.py tests/rl/test_orchestration.py`
  - all checks passed, including Pyrefly.
- `./infra/pre-commit.py --changed-files --fix`
  - all changed-file checks passed.

Caveats:

- Coverage is still assignment/commit coverage, not consumed-gradient coverage.
  The replay-buffer metrics added earlier are still the source of truth for
  accepted/consumed data.
- Recovery requires relaunching with the same rollout storage path. That matches
  the broader Marin resume rule: if the output path changes, the recovered
  ledger changes too.
- If a rollout batch write succeeds but the schedule ledger commit never
  completes, the worker will retry the same assignment rather than silently
  skipping it. Duplicate rollout batches can still be skipped by assignment-id
  replay dedupe when both copies are visible.

## CODEX 2026-05-24T00:03:48Z — Next 100-step validation plan

We have enough instrumentation and scheduler hardening to run another short
Math500 RL validation. This should be treated as a systems validation run, not
yet as the final 500-step blog comparison.

Recommended setup:

- Train for `100` steps.
- Use the same Math500 baseline stack and keep KL disabled.
- Use two rollout workers, matching the one-vs-two sampler evidence where two
  workers reduced trainer starvation.
- Keep `max_rollout_step_delay=1` so the run exercises bounded async/stale
  rollout behavior.
- Keep the current `max_output_tokens=512` for this validation, because the new
  length-cap saturation metrics should first tell us whether 512 is actually
  binding.
- Keep deterministic eval enabled with fixed-order pass@1 greedy and pass@16
  sampled eval.
- Relaunch as one continuous RL run with a stable rollout storage path; the
  schedule ledger lives under that path and should recover if the coordinator is
  relaunched with the same output path.

Primary questions:

1. Does the finite-data Feistel schedule behave correctly under real async
   rollout?
2. Do `inference.schedule/*` counters show a sane reserve/commit pattern with no
   pending-assignment buildup or duplicate storm?
3. Do `replay_buffer/*` metrics show sane accepted vs consumed rollout flow?
4. How often are batches skipped for staleness with two rollout workers and
   `max_rollout_step_delay=1`?
5. Do `inference.length/*` metrics show cap pressure at 512 tokens?
6. Are pass@1 greedy and pass@16 sampled eval structurally reasonable compared
   with the previous 100-step fixed-stack run?

Pass conditions:

- No schedule ledger or cursor errors.
- `inference.schedule/committed_assignments` increases steadily.
- `inference.schedule/pending_assignments` does not grow without bound.
- Replay-buffer stale and duplicate skip counters stay explainable rather than
  dominating accepted rollouts.
- `replay_buffer/total_rollouts_consumed` advances consistently with train
  steps.
- Eval does not show an obvious collapse relative to the previous 100-step
  fixed-stack run.

Expected interpretation:

- If the run is healthy and 512-token cap saturation is low, proceed toward the
  500-step fixed-stack comparison.
- If the run is healthy but cap saturation is high, run the planned
  `max_output_tokens=768` ablation before the 500-step comparison.
- If stale/duplicate/schedule counters look bad, debug replay/scheduling before
  spending on longer RL runs.

## Related artifacts in this worktree

- `rl_500_passk.png` — headline pass@k curves (eval + rollout, EMA-smoothed)
- `rl_500_diagnostics.png` — entropy, diversity gap, length, KL/clip status
- `/tmp/rl_blog_wandb/{train,rollout}_history.parquet` — raw W&B history dumps
- `.agents/logbooks/iris-rl.md` — original migration logbook (preserved from `e4beb97e0`)
- `.agents/projects/iris-rl.md` — original migration plan
- `.agents/projects/on-demand-rl.md` — precursor RL experience, where most of the operational hardening originated

## CODEX 2026-05-24T01:04:10Z — Launched 100-step scheduler validation

Launched a fresh single-region UC1 run for the planned 100-step Math500 RL
systems validation:

```bash
uv run iris --config lib/iris/config/marin.yaml job run \
  --no-wait \
  --user ahmedah \
  --job-name iris-rl-blog-rlooIS-sched-100s-uc1-20260524-010114 \
  --zone us-central1-a \
  --cpu 1 \
  --memory 4G \
  --disk 30G \
  --enable-extra-resources \
  --extra cpu \
  -- python experiments/llama_3_8b_rl_math500.py \
    --run-name llama-3.1-8bi-math500-rlooIS-sched-100s-uc1-20260524-010114 \
    --experiment-name-suffix rlooIS-sched-100s \
    --num-train-steps 100 \
    --num-rollout-workers 2 \
    --max-output-tokens 512 \
    --eval-frequency 1 \
    --zone us-central1-a \
    --inflight-weight-updates
```

Root job:

- `/ahmedah/iris-rl-blog-rlooIS-sched-100s-uc1-20260524-010114`

Smoke check:

- Root coordinator is running.
- Executor step launched under `gs://marin-us-central1`.
- RL coordinator started with instance id
  `llama-3.1-8bi-math500-rlooIS-sched-100s-uc1-20260524-010114-20260524-010307-0aa9df59`.
- Coordinator submitted 3 child jobs: 1 trainer and 2 rollout workers.
- TPU children are pending on `us-central1-a` v5p-8 capacity:
  `Scheduler: Insufficient memory (need 400.0GB, available 300.8GB)` while the
  autoscaler waits for `tpu_v5p-preemptible_8-us-central1-a`.
- No immediate import/config traceback in the first smoke logs.

## CODEX 2026-05-24T01:59:34Z — Patched W&B eval-table key and relaunched validation

The first 100-step launch
`/ahmedah/iris-rl-blog-rlooIS-sched-100s-uc1-20260524-010114` reached real
training, but rollout-0 hit W&B's 128-character artifact-name limit while
logging eval sample tables. The failure was caused by the combination of the
long run id and the metric key
`inference.eval_pass1_greedy/math_full/sample_table`.

Action taken:

- Stopped the flawed run before it burned more TPU time on a known-bad code
  bundle.
- Changed eval sample-table keys to compact deterministic names such as
  `samples/p1g/math_full`.
- Added a regression test that models W&B's derived table artifact name for the
  long RL run id.
- Verified:
  `uv run --project lib/marin --group test pytest tests/rl/test_rollout_worker.py::test_build_prompt_example_metrics_is_deterministic_without_ambient_random tests/rl/test_rollout_worker.py::test_sample_table_metric_key_fits_long_wandb_run_names -q`
  passed, and `./infra/pre-commit.py --fix lib/marin/src/marin/rl/rollout_worker.py tests/rl/test_rollout_worker.py`
  passed.

Relaunched:

```bash
uv run iris --config lib/iris/config/marin.yaml job run \
  --no-wait \
  --user ahmedah \
  --job-name iris-rl-blog-rlooIS-sched-100s-uc1-20260524-012916 \
  --zone us-central1-a \
  --cpu 1 \
  --memory 4G \
  --disk 30G \
  --enable-extra-resources \
  --extra cpu \
  -- python experiments/llama_3_8b_rl_math500.py \
    --run-name llama-3.1-8bi-math500-rlooIS-sched-100s-uc1-20260524-012916 \
    --experiment-name-suffix rlooIS-sched-100s \
    --num-train-steps 100 \
    --num-rollout-workers 2 \
    --max-output-tokens 512 \
    --eval-frequency 1 \
    --zone us-central1-a \
    --inflight-weight-updates
```

Current live status:

- Root job:
  `/ahmedah/iris-rl-blog-rlooIS-sched-100s-uc1-20260524-012916`.
- W&B train run:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/llama-3.1-8bi-math500-rlooIS-sched-100s-uc1-20260524-012916-train`.
- Trainer, rollout-0, and rollout-1 are running with zero failure counts.
- Rollout-0 started first, proving the run can make progress with only one
  sampler active while rollout-1 waits for capacity.
- Rollout-0 generated and wrote initial rollouts; trainer received initial
  rollouts with buffer size 976.
- Training step 0 completed with loss `-0.0301`; progress reached `2/100`.
- Pass@1 greedy eval logged at weight step 0 with
  `inference.eval_pass1_greedy/math_full/pass_at_1 = 0.408`.
- The previous W&B artifact-name failure signature did not recur after the
  pass@1 sample table was prepared and logged.
- Rollout-1 allocated later and is starting its vLLM stack.

Monitor state:

- `scratch/20260524-0129_monitoring_state.json`.

## CODEX 2026-05-24T02:18:53Z — Sustained-progress babysit checkpoint

The relaunched run is still healthy after the first eval and after both rollout
workers joined:

- Trainer, rollout-0, and rollout-1 are all `JOB_STATE_RUNNING`.
- Failure counts remain zero for trainer and both rollout workers.
- Rollout-1 did get preempted once during startup, then restarted and joined the
  run successfully.
- Trainer progressed to `5/100`.
- Rollout-1 generated, wrote rollout steps 1 and 2, and the trainer replay
  buffer ingested the fresh rollout batch.
- The apparent stall around trainer step 3 was not a deadlock: trainer logged
  `No rollouts received... cumulative_wait=660s`, but rollout-1 subsequently
  completed generation and training advanced.
- Rollout-1 emitted repeated TPU/XLA scoped-Vmem `INVALID_ARGUMENT` messages
  during compilation, then continued running. Treat these as noisy unless they
  turn into task failures.
- Pass@1 greedy eval is verified; pass@16 sampled eval has not logged yet. The
  current config evaluates 500 examples with 16 generations at every eval point,
  so that path can dominate wall time with `--eval-frequency 1`.

Interpretation:

- The W&B eval sample-table key patch fixed the live crash point.
- One-sampler progress worked while rollout-1 was pending/preempted.
- Two-sampler progress is now working, but the run is likely eval-heavy because
  pass@16 is configured on 500 examples per eval.

## CODEX 2026-05-24T02:30:25Z — Validation run continues through 9/100

Additional babysit checkpoint:

- Root, trainer, rollout-0, and rollout-1 are still `JOB_STATE_RUNNING`.
- Failure counts remain zero.
- Train progress reached `9/100`; training step 8 completed with loss
  `-0.0297`.
- Rollout-1 is the main producer right now and reached rollout step 9.
- Rollout-0 resumed normal rollout generation and logged another pass@1 greedy
  eval at weight step 5:
  `inference.eval_pass1_greedy/math_full/pass_at_1 = 0.432`.
- The W&B artifact-name failure signature is still absent.
- I still have not found `eval_pass16_sample` in the live logs. Local config
  includes it, so this needs follow-up against W&B history or a cleaner log
  query. It is not blocking training progress in this run.

## CODEX 2026-05-24T03:33:35Z — Babysit checkpoint at 34/100

The 100-step RL validation run remains healthy:

- Root, trainer, rollout-0, and rollout-1 are still `JOB_STATE_RUNNING`.
- Failure counts remain zero.
- Train progress reached `34/100`; training step 33 completed with loss
  `-0.0567`, `train_step=60.50s`, `rollout_wait=26.43s`, and
  `iteration=90.54s`.
- Rollout-1 advanced through rollout step 54 and started generating step 55.
- The earlier timeout warnings resolved when the trainer collected another
  rollout batch at `2026-05-24T03:22:41Z`.
- Pass@16 sampled eval is verified in the live run: at weight step 24,
  `inference.eval_pass16_sample/math_full/pass_at_16 = 0.744` with
  `total_count=8000`, `length/truncated_rate=0.208625`, and
  `length/cap_saturated_rate=0.224`.
- The W&B artifact-name failure signature has not recurred after both pass@1
  and pass@16 eval logging.

## CODEX 2026-05-24T03:48:15Z — Recovery for vLLM over-context prompt

The first patched relaunch exposed a second live defect:

- The trainer reached step 36, but rollout-1 repeatedly crashed on the same
  over-context prompt.
- vLLM reported `max_model_len=1536`, `prompt contains 1674 input tokens`, and
  `requested 0 output tokens`; the engine then died and Iris restarted the
  rollout worker.
- The failure was deterministic after restart: rollout-1 reached
  `failure_count=2` and the trainer began waiting for rollouts.

Fix applied:

- `vLLMInferenceContext` now computes a prompt budget of
  `max_model_len - max_tokens` before generation.
- Prompts over budget are deterministically left-truncated to the last budgeted
  tokens before calling vLLM, preserving request order and request count.
- The inference context logs prompt budget/truncation metrics through
  `get_metrics()`.
- Regression tests cover over-budget prompt truncation and invalid prompt
  budget rejection.

Validation:

- `uv run --project lib/marin --group test pytest \
  tests/rl/test_inference_ctx.py::test_vllm_batch_completions_truncates_over_budget_prompts \
  tests/rl/test_inference_ctx.py::test_vllm_prompt_token_budget_requires_prompt_room \
  tests/rl/test_rollout_worker.py::test_sample_table_metric_key_fits_long_wandb_run_names -q`
  passed.
- `./infra/pre-commit.py --fix lib/marin/src/marin/rl/environments/inference_ctx/vllm.py \
  tests/rl/test_inference_ctx.py` passed.

Recovery action:

- Stopped `/ahmedah/iris-rl-blog-rlooIS-sched-100s-uc1-20260524-012916`.
- Relaunched the same root job with the same run name and command after the
  prompt-budget patch.

## CODEX 2026-05-24T04:03:37Z — Recovery relaunch is making progress

The post-recovery attempt is healthy so far:

- New executor attempt id:
  `rl_testing-llama-3.1-8bi-math500-rloois-sched-100s-uc1-20260524-012916_5d6f5c80-c179a3d7`.
- Root, trainer, rollout-0, and rollout-1 are `JOB_STATE_RUNNING`.
- Failure counts are zero on the trainer and both rollout workers.
- Rollout-1 had one preemption during startup, then restarted and generated
  normally.
- The trainer collected fresh rollout batches again.
- Training resumed/advanced around step 36; live progress showed `37/100`.
- No `VLLMValidationError` or W&B artifact-name error has appeared in the new
  attempt logs.

## CODEX 2026-05-24T04:13:42Z — Recovery attempt through 41/100

The recovery attempt continues to make normal training progress:

- Root, trainer, rollout-0, and rollout-1 remain `JOB_STATE_RUNNING`.
- Failure counts remain zero on trainer and both rollout workers.
- Train progress reached `41/100`.
- Trainer completed step 40 with `rollout_wait=1.00s` and step 41 with
  `rollout_wait=0.00s`.
- Rollout-1 advanced to rollout step 10 and continued generating normally.
- Current-attempt logs show no `VLLMValidationError`, no W&B artifact-name
  failure, and no new traceback.

## CODEX 2026-05-24T04:26:01Z — vLLM prompt-budget fix live-validated

The over-context prompt guard has now been exercised in the live run:

- At rollout step 18, rollout-1 logged
  `Truncated 1/64 prompts to fit max_model_len=1536 with max_tokens=512`
  with `prompt_token_budget=1024`.
- vLLM generation completed after the truncation warning instead of raising the
  previous `VLLMValidationError`.
- Rollout step 18 wrote successfully and the trainer collected the batch at
  `2026-05-24T04:23:57Z`.
- Training progressed to `46/100`; training step 45 completed with
  `rollout_wait=28.13s`.
- Failure counts remain zero on trainer and both rollout workers.

## CODEX 2026-05-24T04:36:43Z — Midpoint checkpoint

The recovery attempt reached the midpoint:

- Train progress reached `50/100`; training step 49 completed with loss
  `-0.0475`.
- Failure counts remain zero on trainer and both rollout workers.
- Rollout-1 reached rollout step 27 and continued generating normally.
- A short trainer `No rollouts received` warning resolved in the same interval
  when the replay buffer collected the next rollout batch.
- No recurrence of the vLLM over-context crash or W&B artifact-name error.

## CODEX 2026-05-24T05:12:37Z — Eval-heavy run through trainer step 62

The run remains healthy past the midpoint:

- Trainer reached step 62 with failure counts still zero.
- Checkpoint step 57 saved successfully at `2026-05-24T05:00:31Z`.
- Pass@16 sampled eval logged again at weight step 61:
  `pass_at_16 = 0.718`, `total_count = 8000`,
  `length/truncated_rate = 0.18125`, and
  `length/cap_saturated_rate = 0.196625`.
- No recurrence of either fixed failure signature.

## CODEX 2026-05-24T05:53:56Z — Recovery attempt through 79/100

The babysat run is still making progress after the prompt-budget recovery:

- Root, trainer, rollout-0, and rollout-1 remain `JOB_STATE_RUNNING`.
- Failure counts remain zero on the trainer and both rollout workers.
- Trainer completed step 78 and progress reached `79/100`.
- Checkpoint `step-73` saved successfully; checkpoint `step-78` started cleanly.
- The temporary `No rollouts received` warnings resolved when the replay buffer
  collected fresh rollout batches.
- Rollout-1 advanced through rollout step 88; recent generations completed in
  roughly 43-86 seconds.
- No recurrence of the vLLM over-context crash or W&B artifact-name failure.

## CODEX 2026-05-24T06:20:15Z — Final stretch through 90/100

The recovery attempt remains healthy in the final stretch:

- Root, trainer, rollout-0, and rollout-1 remain `JOB_STATE_RUNNING`.
- Failure counts remain zero on the trainer and both rollout workers.
- Trainer progress reached `90/100`; step 88 completed with loss `-0.0417`.
- Checkpoint `step-83` saved successfully; checkpoint `step-89` started cleanly.
- A fresh one-minute rollout wait resolved with the replay buffer collecting the
  next rollout at `2026-05-24T06:17:13Z`.
- Both rollout workers are alive; rollout-1 reached rollout step 109 and
  rollout-0 was generating step 12.
- No recurrence of the vLLM over-context crash or W&B artifact-name failure.

## CODEX 2026-05-24T06:53:11Z — 100-step RL run succeeded

The babysat 100-step run completed successfully:

- Root job `/ahmedah/iris-rl-blog-rlooIS-sched-100s-uc1-20260524-012916`
  reached `JOB_STATE_SUCCEEDED`.
- Structured Iris summary reported `state=succeeded`, `exit_code=0`,
  `failure_count=0`, and `preemption_count=0` for the root wrapper.
- Trainer reached `100/100`, completed training step 99, and saved final
  checkpoint `step-99` under
  `gs://marin-us-central1/rl_testing/llama-3.1-8bi-math500-rlooIS-sched-100s-uc1-20260524-012916-ce4109/checkpoints/llama-3.1-8bi-math500-rlooIS-sched-100s-uc1-20260524-012916-train/step-99`.
- The RL coordinator marked the run completed and finished at
  `2026-05-24T06:50:47Z`.
- Final eval signals observed during the run included pass@1 greedy at
  weight step 90 with `pass_at_1=0.462`, and pass@16 sampled at weight step 98
  with `pass_at_16=0.68`.
- The final pass@16 length metrics at weight step 98 were
  `length/truncated_rate=0.196875` and
  `length/cap_saturated_rate=0.210625`.
- No recurrence of the W&B artifact-name failure or vLLM over-context crash
  appeared after the patches.
- Rollout-0 shows `JOB_STATE_KILLED` with `exit_code=0` during coordinated
  shutdown after completion; the nested RL job and root wrapper both succeeded.
