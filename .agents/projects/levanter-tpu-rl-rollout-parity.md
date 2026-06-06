# Levanter TPU RL Rollout Parity

## Goal

Make Levanter TPU inference a reliable RL rollout backend for dense 8B-class
models, with vLLM-TPU parity in decode-heavy regimes and correct behavior
across mixed prompt/decode workloads.

## Operational Constraint

As of 2026-06-06, Codex Desktop is unstable in this thread or a neighboring
thread. Keep background-style work narrow and explicit until the app is stable:

- Automations are usually acceptable, but use only narrow heartbeats tied to one
  known subagent or job, with concise notification criteria. Avoid broad or
  frequently mutating automations from this main thread unless explicitly
  requested.
- Do not poll Iris jobs from this thread unless explicitly requested; prior
  Iris polling left local `gcloud compute ssh ... -L ... -N` tunnels that had
  to be killed manually. Never stop or restart Iris clusters or jobs.
- Do not run long sleep loops, long foreground babysitting commands, browser
  automation, or repeated GitHub GraphQL checks from this thread.
- Prefer short local file edits, short REST `gh api` checks, and written
  handoffs. Delegate new runs to subagents when useful, then monitor them with
  tightly scoped heartbeats rather than foreground polling.
- There is no active heartbeat for the v5p follow-up. The heartbeat was deleted
  after the watched v5p job reached terminal failure before benchmark results.

## Current Evidence

- Decode-heavy Qwen3-8B on v6e-8 is production-competitive for the canonical
  sampled rollout shape: `decode_b32_i1_o128_n4`, `top_k=4096`, logprobs on,
  closed-loop serving, 60-minute soak. Levanter reached 3197.58 tok/s versus
  vLLM-TPU at 3408.09 tok/s, or 93.8%.
- Dense matrix coverage has started on v6e-8 for greedy `n=1`, sampled `n=4`,
  `o128` and `o512`, and logprobs on/off.
- Mixed long-prompt correctness is fixed by the #6185 serving/engine path:
  service batching no longer silently drops overflow prompts, normal generation
  can admit one logical request batch through multiple prefill chunks, and
  prefill-drain scheduling admits all currently possible prompts before decode.
  This avoids silent early-stop when one logical service batch exceeds one
  `max_prefill_size` admission.
- The prior weak mixed regime, `mixed_b32_i512_o512_n1`, is now correct and
  above the issue-level throughput bar after prefill-drain scheduling. Levanter
  completes the full 16384-token workload under default prefill settings at
  4084.24 decode tok/s versus vLLM-TPU at 4985.21 decode tok/s, or 81.9%.
  The remaining performance work is now a generic parity push, not a correctness
  or service-admission blocker.
- A prefill-heavy v6e-8 row has identified the next parity gap. For
  `prefill_b8_i2048_o128_n1`, backend `both`, dense matrix, TP=8,
  `--max-pages 512`, two warmups and one measured round, vLLM reached
  `1262.90` decode tok/s and `21469.31` total tok/s. Levanter reached
  `1006.83` decode tok/s and `17116.08` total tok/s, for a ratio of `0.797`;
  the generic benchmark target marked this row `fail`.
- A narrow Levanter-only v6e-8 diagnostic follow-up completed, but it exposed a
  benchmark harness artifact rather than a clean attribution split. The normal
  diagnostic row measured `907.43` decode tok/s, `15426.28` total tok/s, and a
  single `0.813`s decode iteration with `1022` iteration tokens. The
  `no_lm_head` and `lm_head_no_sampling` rows measured `562.19` and `509.11`
  decode tok/s, respectively, but changed decode token grouping to
  `510,256,256` because the diagnostic path admitted only one pending prefill
  per outer iteration instead of draining all pending prefills before decode.
  Treat those two rows as confounded until the diagnostic path mirrors normal
  prefill-drain scheduling.
- The corrected Levanter-only v6e-8 diagnostic now gives a clean attribution
  split for the same prefill-heavy shape. On #6185 head `82bb6dbfb`, Iris job
  `/dlwh/qwen3-v6e8-prefilldiag-drain-prefill-b8-i2048-o128-n1-20260606-1532`
  succeeded with all three rows using prefill chunks `4096,4096,4096,4096` and
  `1022` decode iteration tokens. Normal `levanter:auto` measured `906.62`
  decode tok/s and `15412.56` total tok/s. `no_lm_head` measured `1244.03`
  decode tok/s and `21148.47` total tok/s. `lm_head_no_sampling` measured
  `1154.22` decode tok/s and `19621.78` total tok/s. Code inspection after the
  result found the production `decode submit s` metric starts before draining
  pending prefills, unlike the diagnostic path, so the `0.184`s submit field is
  not pure decode submit overhead. The clean signal is that the production
  decode iteration itself is `1022 / 0.813s`, essentially matching the vLLM
  `1262.90` tok/s row; the remaining work is to fix benchmark attribution and
  then target any residual LM-head/sampling cost with corrected metrics.

## Authoritative Artifacts

- Dense benchmark matrix: #6176,
  `https://github.com/marin-community/marin/pull/6176`, head
  `104bf901e33c5f0b34d32d4f59edf27b82f66714`.
- Multi-prefill serving fix: #6185,
  `https://github.com/marin-community/marin/pull/6185`, current head
  `82bb6dbfba443f39997c6fcd38e43c26871a58fe`.
- Long-prompt serving issue: #6184,
  `https://github.com/marin-community/marin/issues/6184`.
- RL token rollout contracts: #6186,
  `https://github.com/marin-community/marin/pull/6186`, head
  `c4ba37bda1f429700528a442e2f18e906b96d1b9`.
- Durable handoff artifact: #6214,
  `https://github.com/marin-community/marin/pull/6214`, branch
  `codex/levanter-rl-rollout-handoff`.
- Canonical decode-heavy proof: 60-minute v6e-8
  `decode_b32_i1_o128_n4` batch-merge soak, Levanter `3197.58` decode tok/s
  versus vLLM `3408.09`, ratio `0.938`.
- Mixed long-prompt proof: Iris job
  `/dlwh/qwen3-mixed-v6e8-prefilldrain-i512-o512-20260606-0428`, Levanter
  `4084.24` decode tok/s versus vLLM `4985.21`, ratio `0.819`, with full
  `16384` completion tokens and no no-progress warnings.
- Prefill-heavy v6e-8 gap: Iris job
  `/dlwh/qwen3-v6e8-prefilldrain-prefill-b8-i2048-o128-n1-20260606-0703`,
  Levanter `1006.83` decode tok/s and `17116.08` total tok/s versus vLLM
  `1262.90` decode tok/s and `21469.31` total tok/s, ratio `0.797`.
- Prefill-heavy v6e-8 diagnostic: Iris job
  `/dlwh/qwen3-v6e8-prefilldiag-prefill-b8-i2048-o128-n1-20260606-1504`,
  Levanter-only on #6185 head `ba82f306d`, `--profile-levanter`,
  `--levanter-diagnose-without-lm-head`, and
  `--levanter-diagnose-lm-head-no-sampling`. The job succeeded with no Iris
  failures. Results: normal `levanter:auto` `907.43` decode tok/s and
  `15426.28` total tok/s, `no_lm_head` `562.19` decode tok/s and `9557.28`
  total tok/s, `lm_head_no_sampling` `509.11` decode tok/s and `8654.88` total
  tok/s. The diagnostic rows are not yet a clean LM-head/sampling attribution
  because their decode iteration tokens were `510,256,256` instead of the
  normal row's `1022`.
- Corrected prefill-heavy v6e-8 diagnostic: Iris job
  `/dlwh/qwen3-v6e8-prefilldiag-drain-prefill-b8-i2048-o128-n1-20260606-1532`,
  Levanter-only on #6185 head `82bb6dbfb`, after the diagnostic prefill-drain
  fix. The job succeeded with no Iris failures. Results: normal
  `levanter:auto` `906.62` decode tok/s and `15412.56` total tok/s,
  `no_lm_head` `1244.03` decode tok/s and `21148.47` total tok/s,
  `lm_head_no_sampling` `1154.22` decode tok/s and `19621.78` total tok/s.
  All rows used prefill chunks `4096,4096,4096,4096` and `1022` decode
  iteration tokens, so this is the clean attribution row for the prefill-heavy
  gap.

## Acceptance Criteria

1. Decode-heavy RL rollouts are production-competitive.
   - Qwen3-8B, v6e-8, sampled `n=4`, `top_k=4096`, logprobs on.
   - Levanter reaches at least 90% of vLLM-TPU steady decode throughput for the
     canonical rollout regime.

2. Correctness is non-negotiable outside the happy path.
   - No dropped requests or silent early-stop when aggregate prompt tokens exceed
     one prefill batch.
   - Mixed workloads complete expected output tokens or fail loudly with a
     structured error.
   - Tests cover aggregate prompt tokens greater than one `max_prefill_size`.

3. Performance gaps are characterized.
   - Matrix rows cover greedy `n=1`, sampled `n=4`, `o128`, `o512`, and likely
     `o2048` where runtime permits.
   - Logprobs on/off is measured separately.
   - Decode-heavy, mixed, prefill-heavy, pressure, and long-running churn cases
     are reported by regime instead of collapsed into one headline number.
   - v6e-8 is first; v5p follows because it is the more common fleet shape.

4. Long-prompt mixed performance has an engine-level follow-up.
   - Support multi-prefill admission or overlap for one logical service batch.
     The #6185 prefill-drain path satisfies the first issue-level version of
     this target on v6e-8; remaining work is broader parity characterization
     and any additional overlap/kernel work needed beyond the 0.75 issue bar.
   - `mixed_b32_i512_o512_n1` should complete all 16384 expected completion
     tokens under default prefill settings with no no-progress warnings.
   - Target at least 0.75 of vLLM decode throughput, or explain the remaining
     gap with measured kernel/device limits rather than service serialization.
   - Benchmark rows should report enough admission/chunking metrics to separate
     service serialization from device throughput.

5. RL-specific usability comes before generic OpenAI API parity.
   - Preserve tokenization identity, policy/checkpoint identity, prompt and
     completion boundaries, masks, sampled tokens, and logprobs.
   - Expose rollout data through a batched/binary API suitable for training
     rather than relying on JSON OpenAI compatibility as the primary path.
   - For MoE rollout training, plan for router replay metadata and expert-load
     accounting to become first-class outputs.

## Status by Criterion

- Decode-heavy v6e-8 parity: proved for the canonical sampled rollout soak
  (`3197.58 / 3408.09 = 0.938`). Remaining work is landing the benchmark matrix
  PR and extending coverage, not proving the headline v6e-8 decode-heavy case.
- Mixed long-prompt correctness: proved for `mixed_b32_i512_o512_n1` under
  default prefill settings. #6185 completes all 16384 expected completion
  tokens with four prefill admissions and no no-progress warnings.
- Mixed long-prompt issue-level performance: proved on v6e-8 after prefill
  drain (`4084.24 / 4985.21 = 0.819`). The old 0.40x service/decode split is a
  resolved failure mode, not the current baseline.
- Performance characterization: incomplete. v6e-8 has useful decode and mixed
  rows, plus one prefill-heavy target failure at `0.797`; v5p, churn,
  pressure, and longer-output coverage remain open.
- RL token data plane: substantially implemented in #6186 and green on the PR
  head, but still stack-dependent and not landed. Runtime integration remains a
  follow-up after the serving and benchmark PRs merge.
- Merge state verified by REST at 2026-06-06T06:39Z: #6176, #6185, and #6186
  are open, non-draft, mergeable, and not merged. #6184 remains open. Check-run
  summaries are clean on the current heads: #6176 has 30 checks with no failures
  or pending checks; #6185 has 29; #6186 has 29. The goal is not complete until
  the required code paths and benchmark artifacts are landed or otherwise
  accepted by maintainers.

## Active Work

- PR #6176 adds the dense Qwen3 benchmark matrix and the service-layer
  prefill-budget correctness fix. It is ready for review, mergeable, and all
  visible CI checks are green at commit `104bf901`. The fresh Claude review
  completed with no issues and confirmed the prior review notes were resolved.
- Issue #6184 tracks engine-level multi-prefill admission for long-prompt TPU
  serving.
- PR #6185 implements the long-prompt correctness and mixed-workload performance
  fix. The final branch head, `ba82f306d`, combines engine-level multi-prefill
  admission, OpenAI service relaxation for aggregate prompt-token budgets,
  benchmark admission/timing metrics, and prefill-drain scheduling before decode.
  Local validation for the final patch passed: `py_compile`, focused engine plus
  Qwen3 harness pytest (`61 passed`), and `./infra/pre-commit.py --all-files
  --fix`.
- PR #6185 final v6e-8 proof is
  `/dlwh/qwen3-mixed-v6e8-prefilldrain-i512-o512-20260606-0428`. The job
  succeeded with no failures or preemptions for `mixed_b32_i512_o512_n1`,
  Qwen3-8B, v6e-8, TP=8, default prefill, `--max-pages 512`, two warmups, one
  measured round, backend `both`, greedy/no logprobs. Both backends completed
  all 16384 expected completion tokens; Levanter had no no-progress warnings.
  vLLM reached `4985.21` decode tok/s and `9970.43` total tok/s. Levanter
  reached `4084.24` decode tok/s and `8168.49` total tok/s, ratio `0.819`.
  Levanter admitted four prefill chunks `4096,4096,4096,4096` before decode,
  then decoded the logical 32-request workload in one decode iteration. This
  clears the #6184 mixed-regime target of at least 0.75 vLLM ratio.
- PR #6185 prefill-heavy v6e-8 follow-up:
  `/dlwh/qwen3-v6e8-prefilldrain-prefill-b8-i2048-o128-n1-20260606-0703`.
  The job succeeded for `prefill_b8_i2048_o128_n1`, Qwen3-8B, v6e-8, TP=8,
  backend `both`, dense matrix, default prefill, `--max-pages 512`, two
  warmups and one measured round. vLLM reached `1262.90` decode tok/s and
  `21469.31` total tok/s. Levanter reached `1006.83` decode tok/s and
  `17116.08` total tok/s, for ratios `0.797`/`0.797`, so the generic target
  failed. Levanter admitted four prefill chunks `4096,4096,4096,4096`; the
  measured decode iteration was `0.802`s total, with `0.623`s device,
  `0.178`s host, `0.174`s submit, `0.003`s extract, and `1022` iteration
  tokens. This row is now the narrow v6e prefill-heavy diagnostic target.
- PR #6185 Levanter-only prefill-heavy diagnostic:
  `/dlwh/qwen3-v6e8-prefilldiag-prefill-b8-i2048-o128-n1-20260606-1504`.
  The job succeeded for `prefill_b8_i2048_o128_n1`, Qwen3-8B, v6e-8, TP=8,
  `--max-pages 512`, two warmups, one measured round, `--profile-levanter`,
  `--levanter-diagnose-without-lm-head`, and
  `--levanter-diagnose-lm-head-no-sampling`. The normal `levanter:auto`
  diagnostic row measured `907.43` decode tok/s and `15426.28` total tok/s,
  with prefill chunks `4096,4096,4096,4096` and one `0.813`s decode iteration
  carrying `1022` iteration tokens. The `no_lm_head` row measured `562.19`
  decode tok/s and `9557.28` total tok/s; the `lm_head_no_sampling` row
  measured `509.11` decode tok/s and `8654.88` total tok/s. Those two
  diagnostic rows used decode iteration token groups `510,256,256`, so they
  are confounded by diagnostic scheduling rather than a clean LM-head/sampling
  attribution. Code inspection shows normal `generate()` drains all pending
  prefill admissions before decode, while `_generate_diagnostic()` admits only
  one pending prefill per outer iteration.
- PR #6185 corrected Levanter-only prefill-heavy diagnostic:
  `/dlwh/qwen3-v6e8-prefilldiag-drain-prefill-b8-i2048-o128-n1-20260606-1532`.
  The job succeeded for `prefill_b8_i2048_o128_n1`, Qwen3-8B, v6e-8, TP=8,
  `--max-pages 512`, two warmups, one measured round, `--profile-levanter`,
  `--levanter-diagnose-without-lm-head`, and
  `--levanter-diagnose-lm-head-no-sampling`, from #6185 head `82bb6dbfb`.
  All rows used prefill chunks `4096,4096,4096,4096` and `1022` decode
  iteration tokens, validating the diagnostic prefill-drain fix. The normal
  `levanter:auto` row measured `906.62` decode tok/s and `15412.56` total
  tok/s, with decode iteration `0.813`s total, `0.623`s device, `0.190`s host,
  and `0.184`s submit. The `no_lm_head` row measured `1244.03` decode tok/s
  and `21148.47` total tok/s, with decode iteration `0.751`s total, `0.573`s
  device, `0.178`s host, and `0.003`s submit. The `lm_head_no_sampling` row
  measured `1154.22` decode tok/s and `19621.78` total tok/s, with decode
  iteration `0.815`s total, `0.637`s device, `0.178`s host, and `0.003`s
  submit. Follow-up code inspection found the production submit timer starts at
  the beginning of the outer iteration, before pending prefills are drained, so
  it includes prefill admission work and is not directly comparable to the
  diagnostic submit field. The corrected per-iteration signal says production
  transformer/cache plus greedy LM-head decode is close to the vLLM row; fix
  benchmark timing attribution before treating the prefill-heavy ratio as a
  production decode-kernel gap.
- PR #6185 current CI state as of 2026-06-06T05:21Z: all visible checks are
  green, including the rerun `marin-integration` job `79853218990`. The prior
  red lane was the external Hugging Face `429 Too Many Requests` for `gpt2`
  tokenizer staging, and the rerun passed without code changes. The PR is
  non-draft, mergeable, and ready for maintainer review/merge.
- Issue #6184 has the final proof comment:
  `https://github.com/marin-community/marin/issues/6184#issuecomment-4637412411`.
- PR #6186 adds the first RL batched-token rollout API contracts, stacked on
  `agent/20260604-fix-6184`. It is a ready-for-review, mergeable PR, opened
  from `codex/rl-token-rollout-api`. The initial patch added typed
  contracts/protocols for tokenized rollout batches and opt-in RL context
  support without wiring a runtime service or touching Iris. A follow-up commit
  `552fba57a` implements `vLLMInferenceContext.generate_token_rollouts()` using
  token-native `TokensPrompt` / `RequestOutput` conversion, preserving Marin
  request IDs, prompt/completion boundaries, logprobs, finish reasons, and
  batch-level admission metadata without going through OpenAI JSON. Local
  validation passed: full `./infra/pre-commit.py --all-files --fix` and focused
  `pytest tests/rl/test_token_rollout_types.py tests/rl/test_vllm_token_rollouts.py
  -q` with 8 passed. PR comment:
  `https://github.com/marin-community/marin/pull/6186#issuecomment-4636237004`.
  Commit `67c4c16b5` implements `LevanterInferenceContext.generate_token_rollouts()`
  against the local engine boundary under the serving model lock and Haliax
  mesh/axis mapping. It builds Levanter `Request` objects from tokenized prompts,
  preserves Marin request IDs and generation indices in the returned rollouts,
  requires logprobs for the first contract slice, and carries prefill admission
  metrics from `GenerationResult` into `TokenRolloutAdmissionMetadata`. Local
  validation passed: full `./infra/pre-commit.py --all-files --fix` and focused
  `pytest tests/rl/test_token_rollout_types.py tests/rl/test_vllm_token_rollouts.py
  tests/rl/test_levanter_token_rollouts.py -q` with 10 passed. PR comment:
  `https://github.com/marin-community/marin/pull/6186#issuecomment-4636286906`.
  Commit `019ca8e44` adds the first environment-level token rollout adapter:
  `MathEnv` now builds tokenized rollout batches when the backend supports them,
  decodes completion tokens only for reward scoring, and constructs training
  rollouts from token-native prompt/completion/logprob outputs. It fails loudly
  if a backend returns fewer generations than requested and rejects token-native
  batch construction for unsupported decoding knobs instead of silently dropping
  them. Local validation passed: full `./infra/pre-commit.py --all-files --fix`
  and focused `pytest tests/rl/test_token_rollout_types.py
  tests/rl/test_vllm_token_rollouts.py tests/rl/test_levanter_token_rollouts.py
  tests/rl/environments/test_math_env.py tests/rl/test_inference_ctx.py -q`
  with 35 passed. PR comment:
  `https://github.com/marin-community/marin/pull/6186#issuecomment-4636331406`.
  Commit `b4f897d4d` attaches rollout-worker policy identity to token-native
  batches. `RolloutWorker` now sets `PolicyIdentity` before sampling with the
  run id, checkpoint reference, current weight step, current trainer step,
  inference type, and worker index, so token batches preserve policy-version
  identity instead of relying on backend-name fallbacks. Local validation passed:
  full `./infra/pre-commit.py --all-files --fix` and focused `pytest
  tests/rl/test_token_rollout_types.py tests/rl/test_vllm_token_rollouts.py
  tests/rl/test_levanter_token_rollouts.py tests/rl/environments/test_math_env.py
  tests/rl/test_inference_ctx.py tests/rl/test_rollout_worker.py -q` with 61
  passed. PR comment:
  `https://github.com/marin-community/marin/pull/6186#issuecomment-4636354278`.
  Commit `db6a26fe5` strengthens tokenizer replay identity by including a
  stable SHA-256 hash of the tokenizer chat template and canonical special token
  IDs in `TokenizerIdentity`, in addition to name, revision, and vocab size.
  Local validation passed: full `./infra/pre-commit.py --all-files --fix` and
  focused `pytest tests/rl/test_inference_ctx.py tests/rl/environments/test_math_env.py
  tests/rl/test_token_rollout_types.py -q` with 31 passed. PR comment:
  `https://github.com/marin-community/marin/pull/6186#issuecomment-4636367582`.
  Commit `5fd3a5ee9` persists replay identities into stored rollout metadata.
  `RolloutMetadata` now has optional `tokenizer` and `policy` fields, and
  `RolloutWorker` populates them on each generated `RolloutBatch` and attached
  rollout. This keeps tokenizer/template and policy/checkpoint identity available
  after token-native batch results are converted into training rollouts. Local
  validation passed: full `./infra/pre-commit.py --all-files --fix` and focused
  `pytest tests/rl/test_rollout_worker.py tests/rl/test_replay_buffer.py
  tests/rl/test_rollout_storage.py -q` with 40 passed. PR comment:
  `https://github.com/marin-community/marin/pull/6186#issuecomment-4636384557`.
  Additional follow-ups add structured failure envelopes, backend missing-slot
  failures, stored rollout trace identity, a MockEnv token-native path, explicit
  token rollout batch identity, centralized result validation, and attached
  `response_token_ids` preservation for OpenAI-compatible Verifiers/vLLM
  results. Commit `bca6e1929` extends the shared
  `BaseInferenceContext.create_rollout_from_tokenized_rollout()` helper so
  token-native rollouts preserve tokenizer and policy replay identity in
  `RolloutMetadata` immediately, even before `RolloutWorker` attaches batch
  metadata. Focused validation passed:
  `uv run --with pytest --with pytest-timeout --with pytest-xdist pytest
  tests/rl/environments/test_math_env.py tests/rl/environments/test_mock_env.py
  tests/rl/test_inference_ctx.py -q` with 36 passed, plus
  `./infra/pre-commit.py --files
  lib/marin/src/marin/rl/environments/inference_ctx/base.py
  tests/rl/environments/test_math_env.py tests/rl/environments/test_mock_env.py
  --fix`. PR #6186 is non-draft, mergeable, and stacked on
  `agent/20260604-fix-6184`. Commit `3ea3e3a90` preserves MoE rollout replay
  metadata through the training rollout boundary: `RolloutMetadata` now has
  optional `router_replay` and `expert_load` fields, and
  `BaseInferenceContext.create_rollout_from_tokenized_rollout()` copies them
  from `TokenizedRollout`. This keeps router replay references and expert-load
  accounting available after token-native generation is converted into
  replay-buffer/training rollouts. Focused validation passed:
  `uv run --with pytest --with pytest-timeout --with pytest-xdist pytest
  tests/rl/test_inference_ctx.py tests/rl/test_token_rollout_types.py
  tests/rl/test_rollout_storage.py tests/rl/test_replay_buffer.py -q` with 45
  passed, plus `./infra/pre-commit.py --files lib/marin/src/marin/rl/types.py
  lib/marin/src/marin/rl/environments/inference_ctx/base.py
  tests/rl/test_inference_ctx.py --fix`. Commit `894139c86` hardens
  `BaseInferenceContext.rollouts_by_token_request()` so token rollout result
  validation rejects batch ID, tokenizer identity, policy identity, unknown
  request ID, and duplicate/missing generation-index mismatches before
  environments convert backend output into reward-scored training rollouts.
  Focused validation passed:
  `uv run --with pytest --with pytest-timeout --with pytest-xdist pytest
  tests/rl/test_inference_ctx.py tests/rl/environments/test_math_env.py
  tests/rl/environments/test_mock_env.py -q` with 42 passed, plus
  `./infra/pre-commit.py --files
  lib/marin/src/marin/rl/environments/inference_ctx/base.py
  tests/rl/test_inference_ctx.py --fix`. Commit `8bc2a9a59` preserves
  tokenizer and policy replay identity on OpenAI-shaped rollout construction:
  `BaseInferenceContext.create_rollout_from_choice()` now attaches the same
  `RolloutMetadata` identity as the token-native helper, and `PrimeIntellectEnv`
  attaches identity for its direct `Rollout` construction. Focused validation
  passed:
  `uv run --with pytest --with pytest-timeout --with pytest-xdist pytest
  tests/rl/test_inference_ctx.py tests/rl/environments/test_prime_intellect_env.py
  -q` with 31 passed and 1 skipped, plus `./infra/pre-commit.py --files
  lib/marin/src/marin/rl/environments/inference_ctx/base.py
  lib/marin/src/marin/rl/environments/prime_intellect_env.py
  tests/rl/test_inference_ctx.py tests/rl/environments/test_prime_intellect_env.py
  --fix`. Commit `4631a7634` makes the shared OpenAI-compatible response token
  extraction prefer attached `choice.response_token_ids` before falling back to
  BPE token-string lookup, matching the Verifiers/vLLM parser behavior and
  avoiding lossy sampled-token reconstruction when backends already expose exact
  token IDs. Focused validation passed:
  `uv run --with pytest --with pytest-timeout --with pytest-xdist pytest
  tests/rl/test_inference_ctx.py tests/rl/environments/test_process_vllm_results.py
  -q` with 37 passed, plus `./infra/pre-commit.py --files
  lib/marin/src/marin/rl/environments/inference_ctx/base.py
  tests/rl/test_inference_ctx.py --fix`. Commit `c4ba37bda` preserves token
  rollout completion masks through the training boundary: token-native
  `completion_mask` now becomes `Rollout.response_loss_mask`, and training batch
  construction uses it for both `loss_masks` and advantage-weighted
  `loss_weights`, while existing OpenAI-shaped rollouts keep the old
  all-response behavior. Focused validation passed:
  `uv run --with pytest --with pytest-timeout --with pytest-xdist pytest
  tests/rl/test_train_batch.py tests/rl/test_inference_ctx.py
  tests/rl/test_rollout_storage.py tests/rl/test_replay_buffer.py -q` with 59
  passed, plus `./infra/pre-commit.py --files lib/marin/src/marin/rl/types.py
  lib/marin/src/marin/rl/train_batch.py
  lib/marin/src/marin/rl/environments/inference_ctx/base.py
  tests/rl/test_train_batch.py tests/rl/test_inference_ctx.py --fix`. PR #6186
  is non-draft, mergeable, and stacked on `agent/20260604-fix-6184`. GitHub
  checks are green on head `c4ba37bda`, including `marin-integration`, as of
  2026-06-06T06:06Z.

## Next Sequence

1. Land PR #6176 for the dense matrix plus service-layer correctness fix once a
   maintainer is ready; CI and Claude review are clean.
2. PR #6185 now has runtime proof for correctness, service-batch coalescing,
   and prefill-drain scheduling. The `mixed_b32_i512_o512_n1` v6e-8 run clears
   the issue-level 0.75 vLLM throughput target at 0.819. CI is green after
   rerunning the external Hugging Face `marin-integration` flake, so the next
   action is maintainer review/merge.
3. The next mixed-workload performance step is raising the generic benchmark
   parity target beyond the issue bar: explain the remaining 18% gap on the
   prefill-drained row, then decide whether to optimize decode scheduling,
   cache layout, or request-side accounting. The older 0.40x failure mode is
   resolved by draining all currently admissible prefill chunks before decode.
4. Expand the matrix to v5p and the highest-value prefill-heavy/churn cases once
   mixed correctness and admission accounting are stable.
   - Subagent `019e9b3f-8ff1-79c3-9884-02f847973191` is assigned the first
     v5p runtime follow-up: same #6185 branch head `ba82f306d`, same
     `mixed_b32_i512_o512_n1` shape, backend `both`, default prefill,
     `--max-pages 512`, two warmups, one measured round. It should launch one
     Iris job with prefix `qwen3-mixed-v5p-prefilldrain-i512-o512`, babysit it,
     and only proceed to the canonical decode-heavy v5p row after clean mixed
     success.
   - The v5p job was launched as
     `/dlwh/qwen3-mixed-v5p-prefilldrain-i512-o512-20260606-0447`.
   - Terminal result: `JOB_STATE_FAILED` before benchmark results. vLLM did not
     become ready within the 3600-second startup timeout; `/v1/models` stayed
     connection-refused on the local vLLM port. The final error was
     `RuntimeError: vLLM failed to start` for
     `/app/.venv/bin/vllm serve Qwen/Qwen3-8B --trust-remote-code --host
     127.0.0.1 --port 42361 --max-model-len 4096 --tensor-parallel-size 8`.
     This is a terminal infra/backend failure for the v5p comparison, not a
     Levanter benchmark result. No follow-up decode-heavy v5p row has been
     launched.
5. Fix benchmark timing attribution for prefill-heavy rows before changing
   kernels. The corrected Levanter-only diagnostic shows `no_lm_head` reaches
   `1244.03` decode tok/s, close to the original vLLM `1262.90`, and production
   `levanter:auto` decodes `1022` tokens in `0.813`s. The production
   `decode_submit_seconds` metric currently includes prefill admission time, so
   future rows should separate prefill-drain wall time, pure decode submit,
   pure decode device, and end-to-end request latency before assigning the
   residual gap to LM-head/sampling or kernels.
6. Develop the RL batched-token API slice in parallel with the runtime wait.
   vLLM, Levanter, MathEnv, rollout-worker policy identity, tokenizer replay
   identity, and persisted rollout metadata now exercise the token-native
   contract on #6186. The next implementation step is hardening the rollout data
   plane beyond MathEnv: structured partial-failure envelopes if needed,
   token-native PrimeIntellectEnv or worker-level selection, and better
   integration coverage once the PR stack lands. Commit `75ad8f0f5` extends the
  OpenAI-compatible Verifiers/vLLM result-processing bridge to preserve attached
  `response_token_ids` when present, instead of reconstructing token IDs from
  rendered token strings. This keeps PrimeIntellect-style Verifiers flows from
  losing exact backend token IDs while the environment remains OpenAI-shaped.
  Focused validation passed:
  `uv run --with pytest --with pytest-timeout --with pytest-xdist pytest
  tests/rl/environments/test_process_vllm_results.py -q` and
  `./infra/pre-commit.py --files
  lib/marin/src/marin/rl/environments/process_vllm_results.py
  tests/rl/environments/test_process_vllm_results.py --fix`. Commit
  `bca6e1929` then preserves tokenizer/policy replay identity directly in
  token-native rollout metadata from the shared inference-context helper, with
  focused MathEnv/MockEnv/context validation passing. GitHub checks for PR
  #6186 are green on final head `c4ba37bda`.

## Coalescing Proof Decision Rule

For `/dlwh/qwen3-mixed-v6e8-prefillmulti-coalesce-i512-o512-20260605-1537`,
interpret the result against four separate questions:

1. Correctness: both backends must complete 16384 completion tokens for
   `mixed_b32_i512_o512_n1`, with no no-progress warnings. Failure here means
   #6185 is not ready as a correctness fix.
2. Service coalescing: Levanter logs should show one 32-request service batch,
   or at least fewer/larger service batches than the prior 23/9 split. If the
   request split remains similar, the batch-queue coalescing window is not
   addressing the real source of fragmentation.
3. Engine admission: a fully coalesced 32-request service batch should report
   four prefill admissions with chunks close to `4096,4096,4096,4096` under the
   default 4096-token prefill budget. Different chunks are acceptable if all
   tokens complete, but they should explain the timing.
4. Throughput: compare Levanter decode tok/s against the same-run vLLM row and
   the previous corrected baseline of 1980.17 decode tok/s / 0.402 ratio.
   - If the ratio reaches >=0.75, the mixed long-prompt target is provisionally
     met for v6e-8 and the next work is v5p plus longer churn.
   - If correctness holds and batching improves but ratio remains <0.75, inspect
     prefill/decode timing to decide whether overlap or kernel/device throughput
     is the next bottleneck.
   - If correctness holds but batching does not improve, revert or revise the
     coalescing patch rather than attributing the gap to engine internals.

## RL Rollout API Slice

The OpenAI-compatible path is useful for parity testing, but the RL rollout path
should attach below JSON serialization. Existing code already has most of the
right internal pieces:

- `levanter.inference.openai.InferenceRequest` carries `request_id`,
  `prompt_tokens`, sampling controls, `n_generations`, and logprob requests.
- `levanter.inference.engine.GenerationResult` carries generated token ids,
  per-token logprobs, total generated tokens, and prefill admission/chunk
  metrics.
- `marin.inference.types.RunningModel` currently exposes only an
  OpenAI-compatible endpoint plus optional tokenizer identity.

The next non-run implementation target should be a batched token API adjacent to
the Levanter inference engine, not an OpenAI extension. A minimal first contract:

- Inputs: batch id, policy/checkpoint identity, tokenizer identity, tokenized
  prompts, prompt lengths/boundaries, sampling controls, per-sequence max
  tokens, stop/eos policy, and optional RNG seeds.
- Outputs: generated token ids, per-token logprobs, completion lengths,
  prompt/completion masks, request ids, finish reasons, engine admission metrics,
  and enough timing to separate prefill, decode, host, and device time.
- Invariants: tokenizer identity must match the training run, prompt boundaries
  are preserved exactly, output tensors are shape-stable for trainer ingestion,
  and failures are structured per request or per batch rather than hidden in
  text responses.

For MoE/RL later, reserve explicit slots for router replay metadata, expert-load
accounting, and any auxiliary routing losses needed to make rollout tokens
replayable during training.

## Next Token-API Implementation Slice

PR #6186 now defines the token rollout contract and opt-in
`BaseInferenceContext.generate_token_rollouts()` hook, with first backend
implementations for vLLM and Levanter plus MathEnv integration. The next concrete
implementation slice should be:

1. Implement `vLLMInferenceContext.generate_token_rollouts()` first. This is
   done in #6186 commit `552fba57a`.
   - vLLM already consumes `TokensPrompt(prompt_token_ids=...)` and returns
     `RequestOutput` objects containing prompt token IDs, output token IDs,
     finish reasons, and per-token logprobs before the current OpenAI
     `ChatCompletion` conversion layer.
   - This can prove the contract without touching TPU Levanter server internals
     or the environment scoring loop.
   - Add small conversion helpers:
     `TokenSamplingParameters -> SamplingParams`,
     `RequestOutput -> tuple[TokenizedRollout, ...]`, and
     batch-level `TokenRolloutAdmissionMetadata`.
   - Tests should use lightweight fake request/output objects for conversion
     logic, not a live vLLM engine.

2. Keep the existing `batch_completions()` path intact while adding a
   token-native path.
   - Environments can continue to score text through `Choice.message.content`
     until a later slice adds token-native environment adapters.
   - The implementation should not route through OpenAI JSON, custom `Choice`
     attributes, or tokenizer BPE round-trips.

3. The Levanter backend now implements the first local engine path in #6186
   commit `67c4c16b5`.
   - It uses Levanter `Request` / `GenerationResult` directly, carrying prefill
     admission/chunk metrics into `TokenRolloutAdmissionMetadata`.
   - It preserves `request_id` and `generation_index` so RLOO grouping is
     stable.
   - A later hardening pass should add structured per-request failure envelopes
     if partial-success batch semantics become necessary; the first slice lets
     engine exceptions fail the batch loudly.

4. MathEnv now has the first environment-level adapter in #6186 commit
   `019ca8e44`.
   - Render prompts/messages once, build `TokenizedRolloutBatchRequest`, call
     `generate_token_rollouts()`, decode only for environment reward scoring,
     and construct `Rollout` from token-native outputs.
   - At that point the OpenAI path remains a compatibility/perf-test path rather
     than the primary RL rollout data plane.

5. Policy identity is now threaded from `RolloutWorker` in #6186 commit
   `b4f897d4d`.
   - `PolicyIdentity` includes run id, checkpoint ref, checkpoint step when
     non-negative, weight version, train step, inference type, and worker index.
   - This is enough for first-pass replay grouping; future work can replace the
     synthetic checkpoint ref for weight-transfer updates with a concrete
     checkpoint/object reference when the runtime exposes one.

6. Tokenizer replay identity is now stronger in #6186 commit `db6a26fe5`.
   - `TokenizerIdentity` includes tokenizer name/path, revision, vocab size,
     chat-template hash, and special token IDs.
   - This lets replay checks detect template or special-token drift even when
     the tokenizer path stays stable.

7. Stored rollout metadata now preserves replay identities in #6186 commit
   `5fd3a5ee9`.
   - `RolloutMetadata` carries optional `TokenizerIdentity` and `PolicyIdentity`.
   - `RolloutWorker` writes those identities to the batch and each rollout, so
     the information survives conversion from token-native backend results into
     persisted training data.

8. Token-native failure envelopes are now explicit in #6186 commit `015513fa0`.
   - `TokenizedRolloutBatchResult` carries structured
     `TokenizedRolloutFailure` records alongside successful rollouts.
   - MathEnv checks those failures before its missing-generation guard, so a
     backend can report request/generation-level failure identity instead of
     forcing environments to infer failure from absent rollouts.

9. Next hardening targets after #6186:
   - backends now populate `TokenizedRolloutFailure` for missing generations in
     partial-success cases in #6186 commit `2e6854214`;
   - stored rollouts now preserve token-native backend, request ID, generation
     index, finish reason, and stop token ID in #6186 commit `64f157fa5`;
   - MockEnv now uses the token-native rollout path when supported in #6186
     commit `023e0a719`, so the data plane is no longer MathEnv-specific.
   - stored rollouts now preserve token-native batch IDs explicitly in #6186
     commit `6fd760eae`.
   - token rollout result grouping, structured-failure reporting, and
     missing-generation checks are centralized in #6186 commit `3412d29bd`, so
     environments share the same correctness boundary.

10. #6185 coalescing proof result:
   `/dlwh/qwen3-mixed-v6e8-prefillmulti-coalesce-i512-o512-20260605-1537`
   succeeded on v6e-8 with `failure_count=0` and `preemption_count=0`.
   - Case `mixed_b32_i512_o512_n1`, TP=8, `max_pages=512`,
     default prefill settings, two warmups and one measured round.
   - vLLM: `4880.32` decode tok/s, `9760.64` total tok/s.
   - Levanter: `1973.12` decode tok/s, `3946.23` total tok/s,
     ratio `0.404`, target fail.
   - Levanter correctness/admission evidence: completed the full mixed workload
     with `prefill admissions=4`, chunks `4096,4096,4096,4096`, HBM
     `1207959552`, shape buckets `2`.
   - Interpretation: correctness and multi-prefill admission are working; mixed
     throughput is still the open performance target.

11. #6185 follow-up instrumentation:
   - Commit `2191864e3` on `agent/20260604-fix-6184` adds
     `prefill_seconds_per_admission` to `GenerationResult`, surfaces it through
     the OpenAI server metrics snapshot, and adds a `prefill s` column to the
     Qwen3 parity summary.
   - Focused validation: `uv run python -m py_compile ...`, focused pytest
     slice `61 passed`, and `./infra/pre-commit.py --all-files --fix` all
     passed.
   - A side-agent rerun is queued to collect timing attribution for the same
     `mixed_b32_i512_o512_n1` v6e-8 shape after the coalesced correctness proof
     showed the remaining gap is below service-batch coalescing.

12. #6185 decode timing attribution:
   `/dlwh/qwen3-mixed-v6e8-decodetiming-i512-o512-20260605-2054`
   succeeded on commit `5b20d0f0b`.
   - vLLM: `4932.63` decode tok/s, `9865.26` total tok/s.
   - Levanter: `1958.86` decode tok/s, `3917.72` total tok/s,
     ratio `0.397`.
   - Hot prefill was not the bottleneck: four admissions took roughly
     `0.060,0.057,0.058,0.058` seconds.
   - Decode timing was device dominated: iteration totals
     `2.869,2.552,2.551` seconds, device time `2.799,2.483,2.482`, host time
     about `0.07`, submit about `0.06`, extract about `0.01`, for iteration
     token counts `8184,4096,4096`.
   - Interpretation: the remaining mixed-workload gap is not service batching,
     hot prefill, host scheduling, submit latency, or extraction. It is in the
     device-side decode path.

13. Next diagnostic:
   - Launch a bounded Levanter-only v6e-8 run for `mixed_b32_i512_o512_n1`
     with `--levanter-diagnose-without-lm-head` and
     `--levanter-diagnose-lm-head-no-sampling`.
   - Goal: split the remaining device-side decode cost into transformer/cache
     decode versus LM-head/sampling cost before changing kernels or scheduling.
   - Commit `7fb385c79` on `agent/20260604-fix-6184` removes the old
     diagnostic-only aggregate prefill rejection. The diagnostic paths now admit
     pending requests in the same multi-prefill chunks as normal generation and
     report prefill/decode timing lists.
   - Validation for `7fb385c79`: `py_compile`, focused engine plus Qwen3 harness
     pytest slice (`61 passed`), and `./infra/pre-commit.py --all-files --fix`.
   - If `/dlwh/qwen3-mixed-v6e8-engine-diag-i512-o512-20260606-0408` fails on
     the old diagnostic rejection, relaunch the same shape from `7fb385c79`.
   - That old-head job did fail on the expected diagnostic rejection after
     producing only the normal Levanter row. It reconfirmed the normal
     mixed-row timing (`~1974.6` decode tok/s, chunks
     `4096,4096,4096,4096`, decode device time about
     `2.801,2.485,2.484` seconds) but produced no no-LM-head row. A
     replacement from `7fb385c79` is delegated to the side-agent.
   - Replacement job:
     `/dlwh/qwen3-mixed-v6e8-engine-diag-i512-o512-20260606-0415`,
     submitted from fixed head `7fb385c79`; initially pending only on v6e-8
     TPU capacity.
   - Interpretation caveat: the `no_lm_head` diagnostic row is the primary
     transformer/cache decode signal. The current `lm_head_no_sampling` path
     materializes LM-head logits for the full packed decode token axis, while
     production greedy serving uses the streaming greedy LM head at sample
     positions. Treat a slow `lm_head_no_sampling` row as an upper bound on
     production greedy LM-head cost unless the diagnostic is tightened.
   - Commit `0bed5a9fb` tightens that diagnostic for future runs:
     `generate_with_lm_head_no_sampling` now uses the same streaming greedy
     LM-head computation at sample positions as production greedy decode, then
     enqueues dummy tokens. Validation: `py_compile`, focused engine plus Qwen3
     harness pytest slice (`61 passed`), and full pre-commit. The running
     `/dlwh/qwen3-mixed-v6e8-engine-diag-i512-o512-20260606-0415` job is still
     from `7fb385c79`, so use its `no_lm_head` row as the primary signal and
     relaunch from `0bed5a9fb` only if precise LM-head attribution is needed.

14. #6185 engine diagnostic result and next perf hypothesis:
   `/dlwh/qwen3-mixed-v6e8-engine-diag-i512-o512-20260606-0415`
   succeeded from `7fb385c79`.
   - Normal Levanter row: `1967.05` decode tok/s with prefill chunks
     `4096,4096,4096,4096`.
   - `no_lm_head`: `2204.64` decode tok/s, so removing LM-head and sampling only
     recovers about 12%.
   - `lm_head_no_sampling`: `2119.78` decode tok/s on the old diagnostic path;
     treat this as a loose row because `0bed5a9fb` subsequently aligned it with
     production streaming greedy LM-head behavior.
   - Decode timing remained device dominated: per-iteration tokens
     `8184,4096,4096`, with decode device seconds roughly
     `2.80,2.48,2.48`.
   - Interpretation: the dominant remaining gap is not hot prefill, host
     overhead, submit latency, extraction, or LM-head/sampling. The logical
     32-request workload is still being decoded as roughly `16 + 8 + 8`
     sequences after multi-prefill admission.
   - Commit `ba82f306d` on `agent/20260604-fix-6184` drains all currently
     admissible prefill chunks before each decode call. Expected behavior for
     `mixed_b32_i512_o512_n1`: four 4096-token prefill admissions followed by a
     single decode pass over all 32 active sequences, unless slots/pages block
     further admission.
   - Validation for `ba82f306d`: `py_compile`, focused engine plus Qwen3 harness
     pytest slice (`61 passed`), and `./infra/pre-commit.py --all-files --fix`.
   - Confirmation is delegated to the #6185 side-agent with a single bounded
     v6e-8 run named with prefix
     `qwen3-mixed-v6e8-prefilldrain-i512-o512`.
   - Confirmation job:
     `/dlwh/qwen3-mixed-v6e8-prefilldrain-i512-o512-20260606-0428`,
     launched from `ba82f306d`, succeeded with no failures or preemptions.
     Same-run vLLM reached `4985.21` decode tok/s and `9970.43` total tok/s.
     Levanter reached `4084.24` decode tok/s and `8168.49` total tok/s,
     ratio `0.819`.
   - The measured Levanter row completed all 16384 expected completion tokens,
     admitted four prefill chunks `4096,4096,4096,4096`, then decoded the
     logical 32-request workload in one decode iteration. The older
     `8184,4096,4096` decode split is gone.
   - Decode attribution for the final row: prefill seconds
     `0.062,0.058,0.058,0.059`, decode iteration `3.636`s, device `3.433`s,
     host `0.203`s, submit `0.182`s, extract `0.019`s, and decode iteration
     tokens `16376`.
   - Interpretation: #6185 satisfies the mixed correctness target and the
     `>=0.75` vLLM ratio target for the known weak regime. The benchmark summary
     still marks the row as a generic target failure because that threshold is
     stricter than the issue-level acceptance bar.
   - #6185 CI state after reruns: visible checks are green on `ba82f306d`,
     including the Hugging Face-flaked `marin-integration` and `levanter-torch`
     lanes. The PR is non-draft and mergeable; the remaining action is
     maintainer review/merge.

## RL Token Data Plane Gate

The token-native rollout path should be judged by whether a trainer can consume
the rollout without reconstructing hidden state from text. Current code gives
three concrete integration points:

- `levanter.inference.engine.GenerationResult` is the right backend boundary for
  dense Levanter rollouts: it already carries generated token IDs, per-token
  logprobs, prefill admission metrics, decode timing, and completion counts.
- `levanter.inference.openai` is deliberately the wrong primary RL boundary: it
  decodes token IDs to strings, re-shapes logprobs into OpenAI response objects,
  and computes prompt echo logprobs as an extra text-API feature.
- `marin.rl.train_batch.convert_rollout_to_training_format()` already expects
  prompt tokens, response tokens, response logprobs, loss masks, and advantage
  masks, so the shortest reliable RL path is to preserve these arrays from the
  inference backend through rollout storage into this converter.

Acceptance for the next token-data-plane slice:

1. A token-native rollout created by a backend must contain enough information
   to build the existing training example fields without decoding text:
   `input_tokens`, `target_tokens`, `loss_mask`, `advantage`, `policy_logprobs`,
   temperature, top-k, and truncation/finish metadata.
2. Tokenizer and policy identity must be persisted next to each rollout before
   replay buffer or storage boundaries, not inferred later from a process-global
   tokenizer/model name.
3. Backend failures must be represented either as a batch-level exception before
   storage or as structured per-request/per-generation failures; missing
   generations cannot be silently interpreted as low reward or empty text.
4. For future MoE/Grug-style RL, leave room in the persisted rollout metadata
   for router replay: per-token selected experts or compact replay handles,
   router logits/scores if needed for loss terms, expert load/count summaries,
   and the routing policy version that produced them.
5. OpenAI JSON compatibility remains a test/debug adapter. The trainer-facing
   rollout API should stay token/binary shaped, with text decoding only in
   environment reward adapters that genuinely need text.

Code-level audit on 2026-06-06:

- #6186 worktree `codex/rl-token-rollout-api` is clean at
  `c4ba37bda1f429700528a442e2f18e906b96d1b9`.
- `RolloutMetadata` carries tokenizer identity, policy identity, token rollout
  backend/batch/request/generation identity, finish reason, stop token, router
  replay metadata, and expert-load accounting.
- `BaseInferenceContext.rollouts_by_token_request()` rejects batch ID,
  tokenizer identity, policy identity, unknown request IDs, explicit structured
  backend failures, missing generations, and duplicate/non-contiguous
  generation indexes before environment reward scoring.
- `BaseInferenceContext.create_rollout_from_tokenized_rollout()` preserves
  prompt tokens, completion tokens, selected logprobs, finish/truncation state,
  tokenizer/policy identity, token rollout identity, router replay,
  expert-load accounting, and completion masks.
- `convert_rollout_to_training_format()` maps token-native
  `response_loss_mask` into both `loss_masks` and advantage-weighted
  `loss_weights`, while preserving the existing all-response behavior for
  OpenAI-shaped rollouts without masks.
- Relevant tests cover these invariants in `tests/rl/test_inference_ctx.py`,
  `tests/rl/test_token_rollout_types.py`, and `tests/rl/test_train_batch.py`.
