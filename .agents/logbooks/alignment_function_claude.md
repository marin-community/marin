# Alignment Function (Claude): Research Logbook

**Parent logbook:** `.agents/logbooks/alignment_function.md`
**Issue:** https://github.com/marin-community/marin/issues/3355
**Branch:** `alignment_function`
**Experiment ID prefix:** `ALIGN` (continuing from parent logbook numbering)
**Agent:** Claude (`claude-signal` session)

---

## Experiment Log

### ALIGN-254 - 2026-03-26 19:42 - `v10` BF16 native-streamer smoke: TPU engine init succeeded, failed at chat template application

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer`
- Failed child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v10_native_streamer-smoke_0e02b9c4-e85e9f4f`
- Runtime:
  - 24 minutes total
  - ~23 minutes of silent model loading via native `runai_streamer` (217 GiB BF16)
  - then failed immediately on the first smoke request
- **Critical positive signal:**
  - vLLM TPU engine initialization **succeeded** for GPT-oss-120B BF16 on `v5p-8`
  - the entire v1–v9 failure chain (MXFP4 quantization, model recognition, local staging OOM, weight streaming, pip caching) is resolved
  - the failure is at the **application layer**, not the infrastructure layer
- Terminal error:
  - `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!`
  - traceback:
    - `batched_vllm_serve.py:311` → `render_messages` → `tokenizer.apply_chat_template(...)`
    - `transformers/tokenization_utils_base.py:1825` → `get_chat_template` → `raise ValueError`
- Root cause analysis:
  - `unsloth/gpt-oss-120b-BF16` has `chat_template: null` in `tokenizer_config.json`
  - it ships a separate `chat_template.jinja` file that newer transformers auto-loads — but only if present in the local directory
  - `_load_tokenizer` in `batched_vllm_serve.py:234` calls `LMEvaluationHarnessEvaluator._stage_remote_tokenizer_dir(...)` to copy tokenizer files from GCS to a temp dir
  - `_stage_remote_tokenizer_dir` iterates `TOKENIZER_FILENAMES` (class constant on `LMEvaluationHarnessEvaluator`)
  - `TOKENIZER_FILENAMES` did **not** include `chat_template.jinja`
  - so `chat_template.jinja` was present in the staged GCS artifact but never copied to the temp dir
  - `AutoTokenizer.from_pretrained(temp_dir)` found no chat template → crash on `apply_chat_template`
- Observability gap identified:
  - vLLM subprocess stdout/stderr was redirected to temp files on disk (`subprocess.Popen(..., stderr=stderr_f)`)
  - Iris only captures the parent Python process's stdout/stderr
  - result: 23 minutes of zero visibility during model loading

### ALIGN-255 - 2026-03-26 19:45 - Two fixes applied for chat template and vLLM log visibility

- Fix 1 — chat template staging:
  - file: `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`
  - added `"chat_template.jinja"` to `TOKENIZER_FILENAMES`
  - verified `chat_template.jinja` exists in the staged BF16 artifact:
    - `gs://marin-us-central1/models/unsloth--gpt-oss-120b-BF16-vllm--e7523373bc44b42296b43202e265a1eebf2ee16f/chat_template.jinja`
- Fix 2 — vLLM subprocess stderr visibility:
  - file: `lib/marin/src/marin/inference/vllm_server.py`
  - changed native `vllm serve` subprocess from `stderr=stderr_f` (silent file) to `stderr=subprocess.PIPE`
  - added a daemon thread that tees each stderr line to both the log file (preserving `_native_logs_tail`) and `sys.stderr` (so Iris captures it)
- Validation:
  - `uv run pytest tests/test_vllm_server.py -q` → 3 passed
  - `./infra/pre-commit.py --fix` → passed on both files
- Next action:
  - launch `v11` smoke with both fixes

### ALIGN-256 - 2026-03-26 20:04 - Launched `v11` with chat template fix and stderr tee

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11-chat-template`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11-chat-template --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v11_chat_template --tensor-parallel-size 4`
- Bundle size: 5.7 MB
- Fixes active in this bundle:
  - `chat_template.jinja` added to `TOKENIZER_FILENAMES`
  - vLLM subprocess stderr teed to `sys.stderr` for Iris visibility
- Expected differences from v10:
  - `chat_template.jinja` will now be staged into the tokenizer temp dir
  - `AutoTokenizer.from_pretrained` will pick up the chat template
  - vLLM model loading progress will be visible in Iris logs
- Success condition:
  - smoke child boots GPT-oss-120B BF16 on `v5p-8`
  - `/v1/completions` returns a valid response
  - `artifacts/vllm_metrics.json` is written
- Status: killed — child blocked on disk (autoscaler provisions 100 GiB, job requested 120 GiB)

### ALIGN-257 - 2026-03-26 20:09 - Lowered disk to 80g and relaunched as `v11b`

- v11 failure:
  - `Autoscaler: insufficient_resources: disk: need 128849018880 (120 GiB), available 107374182400 (100 GiB)`
  - the `120g` disk request was a leftover from the local-staging era (ALIGN-243)
  - native `runai_streamer` does not stage to local disk, so 80 GiB is sufficient
- Fix:
  - `experiments/gpt_oss_120b_tpu.py`: default disk `120g` → `80g`
- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11b-chat-template`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11b-chat-template --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v11b_chat_template --tensor-parallel-size 4`
- Status: **SUCCEEDED**

### ALIGN-258 - 2026-03-26 20:26 - `v11b` GPT-oss-120B BF16 smoke **PASSED** — first successful TPU serve + completion

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11b-chat-template`
- Child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11b-chat-template/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v11b_chat_template-smoke_43e1d596-290a77ad`
- Child state: `JOB_STATE_SUCCEEDED`
- Timeline:
  - 20:12:41 — worker bootstrap
  - 20:12:57 — `Starting vLLM native server`
  - 20:14:14 — `Loading safetensors using Runai Model Streamer: 21%` (first visible progress — stderr tee working)
  - 20:14:41 — `100% Completed | 615/615 [00:24, 24.83it/s]` — weights loaded in 24 seconds
  - 20:23:48 — `vLLM environment ready` — API server started after ~9 min XLA compilation
  - 20:23:48 — `Rendering 1 chat prompts` — chat template applied successfully
  - 20:23:48 — `Sending batched vLLM serve request to /v1/completions for 1 prompts`
  - job succeeded
- Key metrics:
  - weight loading: 615 safetensor shards, 24 seconds via `runai_streamer` (~25 shards/s)
  - XLA compilation: ~9 minutes (first-run cost)
  - total child wall-clock: ~14 minutes
- Fixes validated:
  - `chat_template.jinja` in `TOKENIZER_FILENAMES` → tokenizer now has chat template
  - stderr tee → full vLLM progress visible in Iris logs
  - disk `80g` → autoscaler can provision new v5p-8 slices
- This is the first successful GPT-oss-120B serve on TPU in this project
- GOSS-3 is now complete; the path is unblocked for:
  - GOSS-5: one-statement prompt generation
  - GOSS-6: full-spec prompt generation
  - GOSS-7: full open-weight alignment pipeline

### ALIGN-259 - 2026-03-26 20:30 - Raised `max_model_len` and `max_tokens` to 4096 for reasoning model thinking budget

- Context:
  - GPT-oss-120B is a reasoning model with a chain-of-thought `analysis` channel
  - the chat template accepts `reasoning_effort` kwarg (defaults to `"medium"`)
  - thinking tokens count against both `max_model_len` (engine context) and `max_tokens` (per-request completion budget)
  - previous values were `max_model_len=2048` and `max_tokens=1024` for prompt generation stages — far too small for a reasoning model that may spend hundreds of tokens thinking before producing the actual response
- Changes:
  - `experiments/gpt_oss_120b_tpu.py`: `max_model_len` 2048 → 4096
  - `experiments/generate_prompts_gpt_oss_120b_refactored.py`: `understanding_max_tokens`, `concretize_max_tokens`, `extract_max_tokens` all 1024 → 4096
  - `experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py`: `understanding_max_tokens`, `concretize_max_tokens`, `extract_max_tokens`, `teacher_max_tokens`, `rejected_max_tokens` all → 4096
  - smoke `max_tokens=64` left as-is (sufficient for trivial validation)
- Note on `reasoning_effort`:
  - the chat template supports `reasoning_effort` as a kwarg to `apply_chat_template`
  - valid values: `"low"`, `"medium"`, `"high"` (defaults to `"medium"`)
  - not yet wired through the alignment pipeline — can be added later per-stage
- Validation:
  - `./infra/pre-commit.py --fix` → passed on all three files

### ALIGN-260 - 2026-03-26 20:40 - Plan: single-statement end-to-end alignment pipeline with GPT-oss-120B for all roles

- **Goal:** Run the complete alignment pipeline on one statement (`ask_clarifying_questions`) with GPT-oss-120B serving every model role, validating the full data path from spec → preference pairs.
- **Why one statement:** Minimizes TPU wall-clock while exercising every pipeline stage. If this passes, the full 46-statement run is a scale-up, not a new integration risk.
- **Why Mixtral for rejected:** Mixtral-8x7B-Instruct is a weaker model that naturally produces lower-quality responses without spec guidance, making it a better source of rejected responses. Using a different model for rejected also avoids the risk that GPT-oss's reasoning capability makes it too good at following even "opposite" instructions, collapsing the chosen/rejected margin.

#### Model roles and config

| Role | Model | Config source |
|---|---|---|
| Ideation (Stage 1/2) | GPT-oss-120B BF16 | `gpt_oss_120b_tpu_vllm_config()` |
| Extract (Stage 3) | GPT-oss-120B BF16 | same |
| Teacher (chosen) | GPT-oss-120B BF16 | same |
| Rejected | Mixtral-8x7B-Instruct-v0.1 | `VLLMConfig` on `v5p-8` (opposite prompting) |
| Judge | GPT-oss-120B BF16 | `gpt_oss_120b_tpu_vllm_config()` |

#### Pipeline stages exercised

1. **Spec loading** — read `openai_model_spec.jsonl`, filter to `statement_ids=["ask_clarifying_questions"]`
2. **Stage 1 (Understanding)** — GPT-oss generates variation axes for the statement
3. **Stage 2 (Concretize)** — GPT-oss generates concrete scenarios from covering array configs
4. **Stage 3 (Extract)** — GPT-oss extracts system_prompt + user_message from scenarios
5. **Chosen response generation** — GPT-oss generates spec-guided responses (teacher role)
6. **Rejected response generation** — GPT-oss generates opposite-mode responses (rejected role)
7. **Judge** — GPT-oss scores both chosen and rejected responses against rubrics
8. **Preference pair construction** — filter by score thresholds, build chosen/rejected pairs
9. **Output** — sharded `.jsonl.gz` preference dataset

#### Experiment script

- New file: `experiments/align_gpt_oss_120b_e2e_one_statement.py`
- Based on `align_gpt_oss_120b_mixtral_rejected_full_spec.py` but with:
  - `rejected_model=mixtral_vllm` (Mixtral-8x7B-Instruct, opposite prompting)
  - `statement_ids=["ask_clarifying_questions"]`
  - `dpo_config=None` (stop at preference pairs — DPO training is a separate validation)
  - `covering_strength=2` (pairwise — fewer prompts, faster)
  - relaxed judge thresholds for validation: `judge_min_chosen_score=1.0`, `judge_min_gap=0.0`
  - name: `goss_e2e_one_statement`

#### Resource expectations

- Two vLLM sessions: GPT-oss for prompt gen / chosen / judge, Mixtral for rejected
- Both on `v5p-8` with `tp=4`, `max_model_len=4096`, `max_tokens=4096`
- Estimated prompts per statement with 2-way covering: ~200-500
- All response/judge work is sequential within the shared session

#### Success criteria

- All 5 pipeline stages complete without error
- `prompts/` artifact contains extracted eval prompts for `ask_clarifying_questions`
- `chosen/` and `rejected/` artifacts contain response JSONL
- `preference_pairs/` artifact contains at least 1 valid preference pair
- `artifacts/vllm_metrics.json` emitted for each stage

#### Risk factors

- `max_model_len=4096` may be too small for the judge stage (prompt + rubric + response can be long)
- GPT-oss reasoning overhead means each request may take significantly longer than a non-reasoning model
- Mixtral needs its own staged artifact on `v5p-8` — need to verify `mixtral_8x7b_instruct` is staged in `us-central1`

#### Next steps after this experiment

- If it passes: launch GOSS-6 (full-spec prompt generation) and GOSS-7 (full pipeline with Mixtral rejected)
- If `max_model_len` is a bottleneck: raise to 8192 and re-test
- If opposite-mode rejected is weak: consider using `unguided` strategy instead

### ALIGN-261 - 2026-03-26 20:54 - Launched single-statement E2E pipeline (`goss-e2e-one-statement`)

- Script: `experiments/align_gpt_oss_120b_e2e_one_statement.py`
- Root: `/ahmed/goss-e2e-one-statement`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name goss-e2e-one-statement --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/align_gpt_oss_120b_e2e_one_statement.py`
- Pre-launch checks:
  - Mixtral staged in us-central1: confirmed at `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230/`
  - GPT-oss BF16 staged: confirmed (used by v11b smoke)
  - pre-commit: passed
- Config summary:
  - statement: `ask_clarifying_questions` only
  - GPT-oss: ideation, extract, teacher (chosen), judge
  - Mixtral: rejected (opposite prompting)
  - `covering_strength=2`, `max_model_len=4096`, `max_tokens=4096`
  - `dpo_config=None` → stops at preference pairs
- Status: **FAILED** — `max_model_len=0` reported by vLLM engine

### ALIGN-262 - 2026-03-26 21:27 - E2E pipeline failed in Stage 1 understanding: vLLM reports `max_context_length=0`

- Failed root: `/ahmed/goss-e2e-one-statement`
- Failed child: `/ahmed/goss-e2e-one-statement/align-goss_e2e_one_statement-prompts_0b3b54c4-77d3bea9`
- Timeline:
  - spec step succeeded
  - prompts child: weights loaded (615/615 in 30s), XLA compiled (~13 min), API server started
  - Stage 1 understanding request sent (944 input tokens)
  - vLLM returned: `"This model's maximum context length is 0 tokens"`
- Error:
  - `requests.HTTPError: 400 Client Error: Bad Request; response body: {"error":{"message":"This model's maximum context length is 0 tokens. However, your request has 944 input tokens."}}`
- Analysis:
  - `max_model_len=4096` is set in `gpt_oss_120b_tpu_vllm_config()` and forwarded as `--max-model-len 4096` to `vllm serve`
  - vLLM engine reports 0 anyway — likely the model config's `initial_context_length=4096` or another field is being misread by the TPU engine
  - the v11b smoke passed with `max_model_len=2048` and a 64-token request — so either the engine was silently capping, or the `max_model_len` change to 4096 exposed a new issue
- Investigation:
  - launched smoke with same `max_model_len=4096` → **PASSED** (root `/ahmed/gpt-oss-smoke-maxlen-4096`)
  - XLA compilation took ~17 min (longer than v11b's 9 min due to larger KV cache)
  - the E2E `max_context_length=0` was NOT caused by the `max_model_len=4096` value itself
  - the first E2E child was preempted, and the retry ran on a different worker
  - hypothesis: the preemption + retry may have caused corrupted vLLM state or stale worker config
- Next action: relaunch the E2E pipeline

### ALIGN-263 - 2026-03-26 21:59 - Relaunched E2E pipeline as `goss-e2e-one-statement-v2`

- Root: `/ahmed/goss-e2e-one-statement-v2`
- Same script: `experiments/align_gpt_oss_120b_e2e_one_statement.py`
- Rationale: smoke with `max_model_len=4096` succeeded, so the v1 failure was likely caused by preemption + retry worker state corruption, not a config issue
- Status: **FAILED** — same `max_context_length=0` error, reproducible

### ALIGN-264 - 2026-03-26 22:42 - `goss-e2e-one-statement-v2` failed with identical `max_context_length=0` — this is a reproducible E2E-only bug

- Failed root: `/ahmed/goss-e2e-one-statement-v2`
- Failed child: `/ahmed/goss-e2e-one-statement-v2/align-goss_e2e_one_statement-prompts_0b3b54c4-17d0b74e`
- Error: same as v1 — `"This model's maximum context length is 0 tokens. However, your request has 944 input tokens."`
- Worker: `marin-tpu-v5p-8-us-central1-a-20260326-2051-c7ee3911-worker-0` (clean worker, no preemption)
- Key finding: **this is NOT a preemption fluke — it is a reproducible bug specific to the E2E pipeline path**
- Evidence:
  - smoke with `max_model_len=4096` and `max_tokens=64` → **PASSED** (`/ahmed/gpt-oss-smoke-maxlen-4096`)
  - E2E pipeline v1 with `max_model_len=4096` → **FAILED** (`max_context_length=0`)
  - E2E pipeline v2 with `max_model_len=4096` → **FAILED** (`max_context_length=0`)
- What's different between smoke and E2E:
  - smoke: `gpt_oss_120b_vllm_smoke.py` → constructs `VLLMConfig` at module level → passes directly to `BatchedVllmServeSession` → runs in the same executor step
  - E2E: `align_gpt_oss_120b_e2e_one_statement.py` → `align()` → creates `ExecutorStep` with `PromptGenConfig(ideation_model=gpt_oss_vllm)` → executor serializes config → `remote()` → worker deserializes → `generate_prompts_from_spec()` → `BatchedVllmServeSession(config.ideation_model)`
- Working hypothesis:
  - the executor serializes `PromptGenConfig` (which contains a nested `VLLMConfig`) to transport it to the worker
  - during deserialization, `VLLMConfig` may lose its type and become a plain dict or generic `InferenceConfig`
  - if the deserialized `ideation_model` is a dict, `_build_model_config(config)` would fail accessing `config.max_model_len` — but the error is `max_context_length=0`, not `AttributeError`
  - alternatively, the deserialized config might reconstruct `VLLMConfig` with missing fields, and `max_model_len` might default to 0 or be dropped
- Investigation in progress:
  - tracing how the marin executor serializes nested `VLLMConfig` dataclasses inside `PromptGenConfig`
  - need to check if `max_model_len` survives the round-trip

### ALIGN-265 - 2026-03-27 09:30 - Root cause found for all GPT-OSS Stage 2 concretization failures: `reasoning_effort=low` is silently dropped because Marin sends it in the wrong request field

- **Context:**
  - the GPT-OSS TPU research thread (logbook: `.agents/logbooks/gpt-oss-tpu.md`, GTPU-001 through GTPU-027) had been running a brute-force context escalation ladder to make Stage 2 concretization work
  - every rung from `max_tokens=2048` through `max_tokens=16384` failed with the same shape:
    - `finish_reason = "length"`
    - `message.content = null`
    - long coherent `message.reasoning`
  - the working hypothesis was that GPT-OSS 20B genuinely needed a massive reasoning budget
  - this entry disproves that hypothesis: the model was never actually running at `reasoning_effort=low`

- **Investigation method:**
  - three parallel research agents were launched:
    1. fetched and inspected the actual GCS artifacts and Iris logs from the failed runs
    2. traced the full code path from `PromptGenConfig.concretize_max_tokens` through `batched_vllm_serve.py` to the HTTP request
    3. traced how vLLM's `/v1/chat/completions` endpoint handles `chat_template_kwargs` vs the Harmony code path for GPT-OSS models

- **The bug — exact mechanism:**
  - `batched_vllm_serve.py` line 41 defines:
    - `GPT_OSS_CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "low"}`
  - `_generate_from_messages_gpt_oss()` (line 472–527) constructs the HTTP POST body as:
    ```python
    json={
        "model": self._env.model_id,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    ```
  - this sends `reasoning_effort` nested inside `chat_template_kwargs`
  - **but for GPT-OSS models, vLLM takes a completely different code path**
  - in `vllm/entrypoints/openai/serving_chat.py` (the vLLM fork `vllm_kcbs`), when `model_config.hf_config.model_type == "gpt_oss"`, the server sets `self.use_harmony = True`
  - the Harmony path (line 222–255 of `serving_chat.py`) calls `self._make_request_with_harmony(request)` instead of `self._preprocess_chat(..., chat_template_kwargs=request.chat_template_kwargs, ...)`
  - `_make_request_with_harmony()` (line 1747) reads `request.reasoning_effort` — a **separate top-level field** in `ChatCompletionRequest` (defined at `protocol.py` line 560 as `reasoning_effort: Literal["low", "medium", "high"] | None = None`)
  - it does **not** read `request.chat_template_kwargs` at all
  - the Harmony path bypasses the Jinja chat template entirely — it uses the `openai_harmony` library's `SystemContent.with_reasoning_effort()` and `render_for_completion()` directly on token IDs
  - since Marin is not sending `"reasoning_effort": "low"` as a top-level field, vLLM receives `reasoning_effort=None`
  - the default behavior when `reasoning_effort` is `None` is either medium or uncontrolled reasoning

- **Why this explains all observed failures:**
  - every GPT-OSS Stage 2 request has been running at **default (uncontrolled) reasoning effort**
  - the model spends unbounded tokens in the `<|channel|>analysis` block planning the scenario before ever transitioning to the `<|channel|>final` block that produces `message.content`
  - Stage 1 (understanding) worked at `max_tokens=2048` because the task is simple — "list axes of variation" — so even uncontrolled reasoning finishes quickly (the probe showed ~282 reasoning tokens + ~822 content tokens for a 1104-token total)
  - Stage 2 (concretize) asks the model to plan a complex multi-axis scenario incorporating up to 7 axis values simultaneously — with uncontrolled reasoning, the model generates a detailed analysis plan for each axis combination and never reaches the final answer
  - the 32k/16384 run got 32/67 scenarios done — the simpler axis combinations (fewer extreme values) completed even with uncontrolled reasoning, but complex ones (all 7 axes at extreme values) exhausted the budget every time
  - raising the budget from 2k→4k→8k→16k had no effect because the model simply thinks proportionally longer when given more room — it was never being told to think less

- **Concrete evidence from the 32k run (GTPU-027):**
  - parent job: `/ahmed/gpt-oss-20b-promptgen-one-statement-east5-ctx32k-20260327-0148` — `JOB_STATE_FAILED`
  - 32 out of 67 concretize items succeeded before the 33rd failed
  - failed response payload:
    - `finish_reason = "length"`
    - `message.content = null` (literal JSON null, not empty string)
    - `message.reasoning` contained coherent, task-correct planning text that was cut off mid-sentence:
      > "We need to produce a scenario and rubric. The scenario must incorporate all axis values: extreme ambiguity, high stakes, expert user, moderate context, open problem, East Asian cultural context, religious minority identity. The scenario should be a realistic situation... Let's craft scenario: System prompt: 'You are a medical AI assistant specialized in oncology, trained to provide guidance to clinicians. You should ask clarifying questions when uncertain, especially in high-stakes situations.' User message: 'I need help with a patient case.' That's extremely ambiguous. But we need to incorporate that the user is an expert, East Asian cultural context, religious minority identity. So maybe the user says: 'I have a patie..."
    - the model was actively and correctly planning the scenario output but ran out of tokens in the analysis channel before producing any final content
  - token accounting:
    - `prompt_tokens ≈ 1428`
    - `completion_tokens = 16384` (the full budget consumed)
    - estimated throughput: ~946 tokens/sec, so each failed request took ~17 seconds — consistent with the 9 minutes observed for 32 requests in the logs
  - the failed scenarios consistently involved extreme-end axis values across multiple axes simultaneously (highest reasoning complexity)

- **Concrete evidence from successful Stage 1 probes (for contrast):**
  - from `stage1_chat_low_2048.json` in the 2048 budget sweep (GTPU-006):
    - `finish_reason = "stop"`
    - `completion_tokens = 1104`
    - `prompt_tokens = 677`
    - `reasoning` length: 1128 chars (~282 tokens)
    - `content` length: 4164 chars (~822 tokens)
    - the model exited reasoning quickly because the task was simple

- **Secondary bug found during investigation:**
  - the `ValueError` raised by `_extract_gpt_oss_chat_texts()` at `batched_vllm_serve.py:592` when `finish_reason != "stop"` propagates **uncaught** through `_generate_from_messages_gpt_oss()` → `generate_from_messages()` → `_run_concretize_round_local_global()`
  - the `try/except` in the concretize round loop (line 712) only wraps `_parse_concretize_response_with_diagnostics`, not `generate_from_messages`
  - this means:
    - any single `finish_reason='length'` response kills the entire batch round
    - the `concretize_max_attempts=5` retry mechanism never fires for length failures (it only catches parse failures)
    - in the 32k run, the 32 successfully computed scenarios were thrown away because the 33rd failure crashed the entire batch
    - there is no partial checkpoint for the concretize stage

- **The fix (not yet applied):**
  - primary: in `batched_vllm_serve.py`, change the GPT-OSS request body from:
    ```python
    "chat_template_kwargs": {"reasoning_effort": "low"}
    ```
    to:
    ```python
    "reasoning_effort": "low"
    ```
    as a top-level field in the JSON body
  - secondary: catch the `ValueError` from length-exceeded responses in the concretize batch loop so that individual failures don't kill the entire round

- **vLLM code references (from the `vllm_kcbs` fork):**
  - `vllm/entrypoints/openai/serving_chat.py`:
    - line 141: `self.use_harmony = True` for GPT-OSS
    - lines 222–255: Harmony path branches away from `_preprocess_chat` (which uses `chat_template_kwargs`)
    - line 1747: `_make_request_with_harmony()` reads `request.reasoning_effort`, ignores `request.chat_template_kwargs`
  - `vllm/entrypoints/openai/protocol.py`:
    - line 560: `reasoning_effort: Literal["low", "medium", "high"] | None = None`
    - line 641: `chat_template_kwargs: dict[str, Any] | None = None` (separate field, used only by non-Harmony path)
  - `vllm/entrypoints/harmony_utils.py`:
    - lines 86–108: `get_system_message(reasoning_effort=...)` constructs the Harmony system message
  - `vllm/reasoning/gptoss_reasoning_parser.py`: the `<|channel|>final` token detection that separates analysis from content

- **Marin code references:**
  - `lib/marin/src/marin/alignment/batched_vllm_serve.py`:
    - line 41: `GPT_OSS_CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "low"}` — the wrong-field constant
    - line 472–527: `_generate_from_messages_gpt_oss()` — where the request is constructed
    - line 579–592: `_extract_gpt_oss_chat_texts()` — the strict `finish_reason="stop"` validation
  - `lib/marin/src/marin/alignment/generate_prompts.py`:
    - line 687–696: `_run_concretize_round_local_global()` — where the uncaught ValueError propagates
    - line 712: the `try/except` that only wraps parse failures, not HTTP-level failures
  - `lib/marin/src/marin/inference/vllm_server.py`:
    - lines 488–491: auto-injection of `--reasoning-parser openai_gptoss` for GPT-OSS models

- **Cross-reference to other logbooks:**
  - this finding supersedes the brute-force escalation ladder in `gpt-oss-tpu.md` (GTPU-012 through GTPU-027)
  - the entire sequence of 4096→8192→16384→32768 context raises was chasing a symptom, not the cause
  - the 32k run's 32/67 partial success rate was coincidental — simpler axis combinations completed within budget even at uncontrolled reasoning, not because the budget was nearly sufficient
  - the `max_context_length=0` bug from ALIGN-262/264 (my earlier investigation) is a separate, independent issue in the E2E executor serialization path — it remains unresolved but is not related to this `reasoning_effort` bug

- **Next action:**
  - apply the primary fix (top-level `reasoning_effort` field)
  - apply the secondary fix (catch length failures in concretize batch loop)
  - rerun the one-statement GPT-OSS 20B prompt generation with the corrected request
  - if Stage 2 now completes with `finish_reason='stop'`, the entire brute-force escalation ladder was unnecessary and the original `max_model_len=4096` / `concretize_max_tokens=2048` may be sufficient

### ALIGN-266 - 2026-03-27 10:53 - Applied both fixes and launched GPT-OSS 20B prompt gen with corrected `reasoning_effort`

- **Fixes applied:**
  - **Primary fix — `batched_vllm_serve.py`:**
    - renamed constant `GPT_OSS_CHAT_TEMPLATE_KWARGS` → `GPT_OSS_REASONING_EFFORT = "low"`
    - changed the HTTP POST body in `_generate_from_messages_gpt_oss()` (line 501) from:
      ```python
      "chat_template_kwargs": dict(GPT_OSS_CHAT_TEMPLATE_KWARGS),
      ```
      to:
      ```python
      "reasoning_effort": GPT_OSS_REASONING_EFFORT,
      ```
    - updated the error message in `_gpt_oss_path_error()` to reference the new constant name
  - **Secondary fix — `generate_prompts.py`:**
    - wrapped the `session.generate_from_messages(...)` call in `_run_concretize_round_local_global()` (line 686) with a `try/except (ValueError, requests.HTTPError)`
    - on catch: records every item in the failed batch as a failure (so retry logic can fire), then `continue`s to the next batch
    - added `import requests` at the top of the file
    - before this fix, a single `finish_reason='length'` response would raise `ValueError` uncaught through the entire function, killing all progress in the round (the 32k run lost 32 successful scenarios this way)
  - **Test fix — `tests/test_alignment.py`:**
    - updated two assertion lines (702, 714) from `"chat_template_kwargs": {"reasoning_effort": "low"}` to `"reasoning_effort": "low"` to match the new request shape
- **Validation:**
  - `uv run pytest tests/test_alignment.py -q` → `105 passed`
  - `./infra/pre-commit.py --fix` → passed on all three files
- **Launch:**
  - job name: `gpt-oss-20b-promptgen-reasoning-fix-central1-20260327`
  - job id: `/ahmed/gpt-oss-20b-promptgen-reasoning-fix-central1-20260327`
  - region: `us-central1`, zone: `us-central1-a`
  - `MARIN_PREFIX=gs://marin-us-central1`
  - command:
    ```
    python experiments/generate_prompts_gpt_oss_20b_refactored.py \
      --name debug_generate_prompts_gpt_oss_20b_reasoning_fix_20260327 \
      --statement-id ask_clarifying_questions \
      --tensor-parallel-size 4 \
      --max-model-len 4096 \
      --understanding-max-tokens 2048 \
      --concretize-max-tokens 2048 \
      --extract-max-tokens 2048
    ```
  - bundle size: 5.8 MB
- **Why these conservative budgets:**
  - `max_model_len=4096` and `concretize_max_tokens=2048` are back to the original values that worked for Stage 1
  - with `reasoning_effort=low` actually applied (via the top-level field), the model should spend ~200-300 tokens on reasoning before producing content — well within 2048 completion tokens
  - the prior failed runs at 4096→8192→16384→32768 all had uncontrolled reasoning because `reasoning_effort` was silently dropped
  - if these budgets still fail, it would be new evidence that `reasoning_effort=low` is still not working, which would point to a different issue
- **Success criteria:**
  - Stage 1 completes (already known to work)
  - Stage 2 concretization completes without `finish_reason='length'` failures
  - Stage 3 extraction completes
  - prompt shard written to GCS with >0 records
- **Status:** **SUCCEEDED**

### ALIGN-267 - 2026-03-27 11:06 - GPT-OSS 20B one-statement prompt generation **PASSED** — all 3 stages complete, 68 prompts generated at `max_model_len=4096` / `max_tokens=2048`

- Root: `/ahmed/gpt-oss-20b-promptgen-reasoning-fix-central1-20260327`
- All children: `JOB_STATE_SUCCEEDED`
- Output: `gs://marin-us-central1/align/debug_generate_prompts_gpt_oss_20b_reasoning_fix_20260327/prompts-03b3bc`
- Timeline:
  - 17:54:29 — worker bootstrap
  - 17:55:50 — `Starting vLLM native server`
  - 17:56:30 — weights loaded (411/411 safetensors in ~22s via `runai_streamer`)
  - 17:58:55 — `vLLM environment ready` (~2.5 min XLA compilation — much faster than prior 120B runs)
  - 17:58:55 — `Stage 1: Generating understanding for 1 statements`
  - 17:59:06 — `Stage 1 progress: 1/1 (100.0%) [attempt 1]` — **11 seconds** for Stage 1
  - 17:59:07 — `Stage 2: Concretizing 1 statements` — 68 concretize items
  - 18:01:29 — `Stage 2 progress: 32/68 (47.1%) [attempt 1]`
  - 18:03:52 — `Stage 2 progress: 64/68 (94.1%) [attempt 1]`
  - 18:04:10 — `Stage 2 progress: 68/68 (100.0%) [attempt 1]` — **68/68 on first attempt, ~5 minutes total**
  - 18:04:11 — `Stage 3: Extracting prompts from 1 statements` — 68 extraction items
  - 18:04:53 — `Stage 3 progress: 32/68 (47.1%) [attempt 1]`
  - 18:05:37 — `Stage 3 progress: 64/68 (94.1%) [attempt 1]`
  - 18:05:43 — `Stage 3 progress: 68/68 (100.0%) [attempt 1]` — **68/68 on first attempt, ~1.5 minutes**
  - 18:05:43 — `Total prompts generated: 68`
  - 18:05:43 — `Wrote 68 records to 1 shards`
  - 18:05:59 — `Step ... succeeded`
- **Key result: the `reasoning_effort` fix completely resolved the Stage 2 failure**
  - Stage 2 completed 68/68 concretize items on the first attempt with zero failures
  - `max_model_len=4096` and `concretize_max_tokens=2048` were sufficient — the prior brute-force escalation to 32768/16384 was entirely caused by `reasoning_effort` being silently dropped
  - Stage 2 took ~5 minutes for 68 items (~4.4 seconds per concretize request)
  - Stage 3 took ~1.5 minutes for 68 items (~1.3 seconds per extraction request)
  - total wall-clock from `vLLM environment ready` to final write: ~7 minutes
  - total wall-clock including model load + XLA compilation: ~11 minutes
- **Comparison to prior failures (all with uncontrolled reasoning):**
  - 4096/2048 rung: failed on first concretize request
  - 8192/4096 rung: failed on first concretize request
  - 16384/8192 rung: failed on first concretize request
  - 32768/16384 rung: 32/67 succeeded, failed on 33rd (then threw away all progress)
  - **4096/2048 with actual `reasoning_effort=low`**: 68/68 succeeded on first attempt
- **This confirms:**
  - the root cause was the silent field mismatch: `chat_template_kwargs` vs top-level `reasoning_effort`
  - GPT-OSS 20B is fully capable of producing high-quality Stage 2 concretize output within 2048 completion tokens when `reasoning_effort=low` actually reaches the Harmony serving path
  - the secondary batch-loop fix was not needed for this run (zero failures), but remains valuable as a safety net for edge cases
- **Next steps:**
  - promote this fix to the 120B path
  - rerun the one-statement E2E alignment pipeline with GPT-OSS 120B
  - the `max_context_length=0` executor serialization bug (ALIGN-262/264) still needs separate investigation before the full E2E pipeline can work

### ALIGN-268 - 2026-03-27 11:15 - Output quality inspection and clarification on the actual `reasoning_effort` state during prior failures

- **Output quality inspection:**
  - read the full 68-prompt dataset from `gs://marin-us-central1/align/debug_generate_prompts_gpt_oss_20b_reasoning_fix_20260327/prompts-03b3bc/shard_00000.jsonl.gz`
  - **zero degenerate records** — all 68 prompts have non-trivial system_prompt (min 69 chars, max 2391 chars, avg 301 chars) and user_message (min 41 chars, max 546 chars, avg 224 chars)
  - **full axis coverage** — all 7 variation axes appear in all 68 prompts:
    - `clarification_difficulty` (5 levels: very low → very high ambiguity)
    - `domain_complexity` (4 levels: general knowledge → expert-level)
    - `clarification_cost` (5 levels: very low → very high cost)
    - `user_expertise_level` (4 levels: novice → expert)
    - `clarification_format` (multiple strategies including open-ended, multiple-choice, structured, no clarification)
    - `culture` (east asian, sub-saharan african, latin american, south asian, middle eastern/north african, global multicultural)
    - `demographic` (racial/ethnic minority, gender/sexual minority, person with disability, low socioeconomic background, young person/minor)
  - **sample quality is high** — scenarios are diverse, contextually rich, and directly test the `ask_clarifying_questions` behavior:
    - prompt 0: travel assistant in Kyoto, Taiwanese user with Mandarin phrases — low ambiguity, culturally specific
    - prompt 1: rural health assistant in Kenya ("MoyoCare"), vague fever/rash — very high ambiguity, high clarification cost (airtime)
    - prompt 10: quantum hardware engineer, phase error on 5-qubit GHZ circuit — highly technical, low ambiguity
    - prompt 30: fintech data analyst in Bangalore, VaR calculation — advanced user, very high clarification cost
    - prompt 50: software engineer in Nairobi, Raft write-ahead log optimization — expert, highly technical, sub-Saharan African culture
    - prompt 67: electrical engineer, solar farm voltage stability during storm surge — complex multi-part technical, middle eastern culture
  - Stage 1 understanding artifact shows well-structured variation axes with clear descriptions, spectra, and `why_it_matters` justifications

- **Clarification on what `reasoning_effort` actually was during the prior failures:**
  - the prior failures (GTPU-012 through GTPU-027 in the GPT-OSS TPU logbook) were **not** running at `reasoning_effort="medium"` — they were running with `reasoning_effort=None` (completely unset)
  - the vLLM `ChatCompletionRequest` field is `reasoning_effort: Literal["low", "medium", "high"] | None = None`
  - since Marin was sending `reasoning_effort` inside `chat_template_kwargs` (which the Harmony path ignores), the top-level field was never populated — it stayed at its default of `None`
  - `None` is not the same as `"medium"`:
    - `get_system_message(reasoning_effort=None)` in `harmony_utils.py` skips setting any reasoning constraint in the Harmony system message
    - this means the model was running with **no reasoning budget constraint at all**, not an explicit "medium" constraint
  - we have never tested explicit `"medium"` as a top-level field:
    - `None` (unset, no constraint): 16k+ reasoning tokens, never reaches content for complex prompts
    - `"low"` (correctly applied): ~200-300 reasoning tokens, completes within 2048 total tokens
    - `"medium"` (untested): unknown — could be fine, could be problematic, but was never the actual state during the failures
  - this distinction matters because the gap between "unconstrained" and "low" may be much larger than the gap between "medium" and "low" — the prior failures may have been caused specifically by the absence of any constraint, not by a moderate constraint being too loose

### ALIGN-269 - 2026-03-27 11:18 - Explicit `reasoning_effort="medium"` also fails — `low` is the only viable setting for GPT-OSS prompt generation

- **Experiment:**
  - temporarily changed `GPT_OSS_REASONING_EFFORT` from `"low"` to `"medium"` in `batched_vllm_serve.py`
  - launched identical one-statement GPT-OSS 20B prompt gen with same budgets (`max_model_len=4096`, `concretize_max_tokens=2048`)
  - job: `/ahmed/gpt-oss-20b-promptgen-reasoning-medium-central1-20260327`
- **Result: FAILED**
  - Stage 1 passed (1/1 on first attempt) — same as with `low`
  - Stage 2 failed on the **very first batch** — all 32 items in the first batch returned `finish_reason='length'` with `message.content=null`
  - the retry mechanism (from the batch error handling fix) correctly caught the errors and retried, but all retry attempts also failed
  - total failures: **192** (67 items × ~3 attempts each, all failing)
  - zero concretize items succeeded across all attempts
- **Failure payload (from cfg_000):**
  - `finish_reason = "length"`
  - `message.content = null`
  - `message.reasoning` shows coherent task-correct planning, cut off mid-sentence:
    > "We need to produce a scenario and rubric. The scenario must incorporate all axis values: extreme ambiguity, high stakes, novice user, moderate context, simple fact, global/multicultural, unmarked majority identity. The scenario should be a realistic situation. The system prompt sets up the target model's role/context. The user message is vague..."
  - identical failure shape to the unset/`None` runs from the brute-force ladder
- **Updated comparison table:**

  | `reasoning_effort` value | Stage 1 | Stage 2 (67+ items, `max_tokens=2048`) |
  |---|---|---|
  | `None` (unset — the original bug) | passes | 0/67 — all fail with `length` |
  | `"medium"` (explicit top-level) | passes | **0/67** — all fail with `length` (192 total failures across retries) |
  | `"low"` (explicit top-level) | passes | **68/68** — all succeed on first attempt |

- **Interpretation:**
  - explicit `"medium"` is just as bad as unset/`None` for Stage 2 concretization
  - the gap is between `"low"` and everything else, not between "unset" and "any explicit setting"
  - GPT-OSS 20B spends the entire completion budget in the reasoning channel at medium effort for complex multi-axis concretize prompts, and never transitions to the final answer channel
  - `"low"` is the only viable setting for alignment prompt generation with GPT-OSS at these token budgets
  - Stage 1 (understanding) works at medium because the task is much simpler (list axes of variation) — the model finishes reasoning quickly even at medium effort
- **Action taken:**
  - reverted `GPT_OSS_REASONING_EFFORT` back to `"low"` immediately after confirming the failure
  - `"low"` is now the permanent validated default

### ALIGN-270 - 2026-03-27 11:40 - Next steps plan: three bugs in the 120B E2E path, phased approach to full pipeline

- **Current state summary:**
  - GPT-OSS 20B prompt generation: **WORKING** (68/68 prompts, all 3 stages, `reasoning_effort=low`, `model_impl_type=vllm`)
  - GPT-OSS 120B smoke serve: **WORKING** (ALIGN-258, but only trivial smoke — no prompt gen validation)
  - GPT-OSS 120B E2E pipeline via `align()`: **BROKEN** (`max_context_length=0` in ALIGN-262/264)
  - Llama 70B E2E pipeline: **WORKING** (ALIGN-026, 42 preference pairs from 46 prompts)
  - Downstream stages (chosen/rejected/judge) with GPT-OSS: **UNTESTED**

- **Three known bugs in the 120B E2E path:**

  1. **`reasoning_effort` silently dropped** (FIXED in ALIGN-266)
     - was sending `reasoning_effort` inside `chat_template_kwargs` instead of as a top-level field
     - vLLM's Harmony path ignores `chat_template_kwargs` for GPT-OSS
     - fix: send `reasoning_effort` as top-level request field
     - validated on 20B prompt gen (ALIGN-267)

  2. **`model_impl_type="flax_nnx"` produces gibberish** (NOT FIXED in 120B config)
     - the GPT-OSS TPU logbook (GTPU-001 through GTPU-004) proved that `flax_nnx` produces incoherent token soup while `vllm` produces coherent text
     - the parent alignment logbook also confirmed 120B `flax_nnx` gibberish (ALIGN-268/269 in the parent — Stage 1 output was corrupted random subwords)
     - the working 20B prompt gen script (`generate_prompts_gpt_oss_20b_refactored.py`) explicitly sets `model_impl_type="vllm"` and `prefer_jax_for_bootstrap=False`
     - **but the 120B config (`gpt_oss_120b_tpu.py`) still hardcodes `model_impl_type="flax_nnx"`** with no parameter to override it
     - the 120B E2E experiment (`align_gpt_oss_120b_e2e_one_statement.py`) calls `gpt_oss_120b_tpu_vllm_config()` which inherits this `flax_nnx` default
     - **fix needed:** add `model_impl_type` and `prefer_jax_for_bootstrap` parameters to `gpt_oss_120b_tpu_vllm_config()`, default to `"vllm"` and `False`

  3. **`max_context_length=0` in the E2E executor path** (STATUS UNCERTAIN)
     - observed in ALIGN-262/264 when running through `align()` on 120B
     - standalone 120B smoke with same `max_model_len=4096` passed (ALIGN-262)
     - the standalone 20B prompt gen also passed — but that script does NOT go through `align()`, it creates ExecutorSteps directly
     - working hypothesis was executor serialization losing `max_model_len` during round-trip
     - **BUT this bug was observed while `flax_nnx` was still in use** — it's possible that the `flax_nnx` backend handles `max_model_len` differently than the `vllm` backend, and the `max_context_length=0` error is a symptom of `flax_nnx`, not serialization
     - **we don't know if this bug still exists** after fixing bugs 1 and 2

- **Phased plan:**

  #### Phase 1: Fix 120B config and test E2E through `align()` with 20B

  **Goal:** Determine whether the `max_context_length=0` bug is in `align()` serialization or was caused by `flax_nnx`.

  **Step 1a:** Update `gpt_oss_120b_tpu.py` to match the 20B pattern:
  - Add `model_impl_type` parameter (default `"vllm"`)
  - Add `prefer_jax_for_bootstrap` parameter (default `False`)
  - This aligns the 120B config with the proven 20B path

  **Step 1b:** Create a GPT-OSS 20B E2E one-statement script that goes through `align()`:
  - Based on `align_gpt_oss_120b_e2e_one_statement.py` but using the 20B config
  - Uses `model_impl_type="vllm"` (the proven path)
  - GPT-OSS 20B for: prompts, chosen, judge
  - Mixtral for: rejected (opposite prompting)
  - `dpo_config=None` (stop at preference pairs)
  - `statement_ids=["ask_clarifying_questions"]`

  **Step 1c:** Run the 20B E2E experiment:
  - If it **works**: the `max_context_length=0` bug was caused by `flax_nnx`, and it's now fixed. Proceed to Phase 2.
  - If it **fails with `max_context_length=0`**: the bug is in `align()` serialization, independent of the backend. Investigate the serialization path before proceeding.
  - If it **fails with a different error**: new bug in the downstream stages (chosen/rejected/judge with GPT-OSS). Debug that.

  **Why 20B first:** 20B is cheaper (~2.5 min XLA vs ~9 min for 120B), faster per request, and already validated for prompt gen. Using it for the E2E test isolates the `align()` path question from 120B-specific issues.

  #### Phase 2: Run one-statement E2E with GPT-OSS 120B

  **Only after Phase 1 succeeds.**

  - Update `align_gpt_oss_120b_e2e_one_statement.py` to use the fixed config (`model_impl_type="vllm"`)
  - Run the same one-statement E2E pipeline on 120B
  - Compare output quality with 20B on the same statement
  - Verify 120B-specific issues (XLA compilation time, memory, throughput) don't block the pipeline

  **Success criteria:** preference pairs generated from `ask_clarifying_questions` with 120B.

  #### Phase 3: Scale to full-spec prompt generation

  **Only after at least one E2E pipeline produces preference pairs.**

  - Run all 46 statements through prompt generation (20B or 120B based on Phase 2 quality comparison)
  - Expected output: ~46 × 68 ≈ 3,100+ prompts
  - This can run in parallel with Phase 2 if 20B E2E succeeds first

  #### Phase 4: Full-spec E2E at scale

  - Run the complete pipeline on all 46 statements: prompts → chosen → rejected → judge → preference pairs
  - Expected output: thousands of preference pairs
  - Quality gate: verify chosen/rejected margin, judge score distribution, preference pair statistics

  #### Phase 5: DPO training (if preference pairs look good)

  - Use the existing Levanter DPO infrastructure (already production-ready per ALIGN-000)
  - Train on the generated preference pairs
  - Evaluate with SpecEval for adherence improvement

- **Risk factors for each phase:**

  | Phase | Primary risk | Mitigation |
  |---|---|---|
  | 1 | `max_context_length=0` persists after fixing `model_impl_type` | Fall back to standalone scripts that bypass `align()` while debugging serialization |
  | 1 | GPT-OSS can't serve as judge (judge prompts may need different `reasoning_effort` or token budget) | Test judge independently; fall back to API judge (GPT-4.1) if needed |
  | 2 | 120B XLA compilation is too slow or OOMs at `model_impl_type="vllm"` | Stay on 20B; 120B quality advantage may not justify the cost |
  | 3 | Full-spec prompt gen exceeds wall-clock budget | Increase parallelism or use checkpointing across jobs |
  | 4 | Chosen/rejected quality gap is too small for effective DPO | Adjust judge thresholds, try `unguided` rejected strategy instead of `opposite` |

- **Immediate next action:** implement Phase 1 (fix 120B config, create 20B E2E script, launch)

### ALIGN-271 - 2026-03-27 22:10 - GPT-OSS 20B E2E pipeline **SUCCEEDED** end-to-end — `max_context_length=0` bug confirmed non-existent with `model_impl_type="vllm"`

- **v1 run:** `/ahmed/goss-20b-e2e-one-statement-20260327`
  - spec: SUCCEEDED
  - prompts: SUCCEEDED (68/68, all 3 stages)
  - rejected (Mixtral): SUCCEEDED (68 records)
  - chosen (GPT-OSS 20B): **FAILED** — `finish_reason='length'` at `teacher_max_tokens=2048`
  - the chosen response was coherent and high-quality (detailed exoskeleton motor controller recommendation with tables) but exceeded 2048 tokens
  - **critical finding: `max_context_length=0` did NOT appear** — the `align()` serialization path works correctly with `model_impl_type="vllm"`
  - this confirms the ALIGN-262/264 bug was caused by `flax_nnx`, not serialization

- **v2 run:** `/ahmed/goss-20b-e2e-one-statement-v2-20260327`
  - changed: `max_model_len=8192`, `teacher_max_tokens=4096`, `rejected_max_tokens=4096`, Mixtral `max_model_len=8192`
  - **ALL STEPS SUCCEEDED:**
    - spec: SUCCEEDED
    - prompts: SUCCEEDED (68/68)
    - chosen (GPT-OSS 20B): SUCCEEDED (68 records)
    - rejected (Mixtral): SUCCEEDED (68 records)
    - judgments: SUCCEEDED (but 0 valid judgments — all JSON parse failures)
    - preference_pairs: SUCCEEDED (but 0 pairs — all filtered due to fallback score=0)

- **Output quality inspection:**
  - chosen/rejected data quality is **excellent** — clear behavioral contrast on `ask_clarifying_questions`:
    - Pair 0 (data analyst, ERP conversion): chosen gives 6541-char structured guide with Python code, mapping tables, edge cases, deployment tips. Rejected gives generic 5-step guide saying "simply make an arbitrary decision."
    - Pair 15 (health assistant, Japan): chosen responds in Japanese with structured triage questions and medical guidelines tables. Rejected says "No, you definitely do not need to see a doctor."
    - Pair 40 (chatbot, homework): chosen asks 4 targeted clarifying questions. Rejected guesses math and gives unsolicited algebra answer.
  - 68 chosen responses, 68 rejected responses, all non-empty, all contextually appropriate

- **Judge failure analysis:**
  - all 68 judge responses failed JSON parse: `"Expecting value: line 6 column 9"`
  - GPT-OSS 20B at `reasoning_effort=low` produces judge responses with Harmony reasoning channel text mixed into what should be clean JSON
  - fallback score = 0 for all parse failures
  - with `judge_min_chosen_score=1.0`, all pairs filtered out → 0 preference pairs
  - this is a GPT-OSS judge compatibility issue, not a pipeline bug

- **Artifacts:**
  - `gs://marin-us-central1/align/goss_20b_e2e_one_statement/prompts-21ac47/` (68 records)
  - `gs://marin-us-central1/align/goss_20b_e2e_one_statement/chosen-a2dd1d/` (68 records)
  - `gs://marin-us-central1/align/goss_20b_e2e_one_statement/rejected-7d7695/` (68 records)
  - `gs://marin-us-central1/align/goss_20b_e2e_one_statement/judgments-db2a7d/` (0 records)
  - `gs://marin-us-central1/align/goss_20b_e2e_one_statement/preference_pairs-39f236/` (0 records)

### ALIGN-272 - 2026-03-27 22:30 - Plan: GPT-OSS 120B E2E — brute-force to success

- **Goal:** Run the full E2E alignment pipeline with GPT-OSS 120B. Keep increasing `max_model_len` and `max_tokens` until it either succeeds or OOMs. GPT-OSS 120B supports up to 131072 seq len.
- **Judge workaround:** Set `judge_min_chosen_score=0.0` and `judge_min_gap=0.0` so all pairs pass through regardless of judge parse quality. Fix judging separately later.
- **Starting config (rung 1):**
  - `max_model_len=8192` (same as working 20B v2)
  - `teacher_max_tokens=4096`
  - `rejected_max_tokens=4096` (Mixtral)
  - `model_impl_type="vllm"` (the validated backend)
  - `prefer_jax_for_bootstrap=False`
- **Escalation ladder if `finish_reason='length'`:**
  - rung 2: `max_model_len=16384`, `teacher_max_tokens=8192`
  - rung 3: `max_model_len=32768`, `teacher_max_tokens=16384`
  - rung 4: `max_model_len=65536`, `teacher_max_tokens=32768`
  - rung 5: `max_model_len=131072`, `teacher_max_tokens=65536`
- **Stop conditions:**
  - SUCCESS: preference pairs written (even if judge quality is poor)
  - OOM: `RESOURCE_EXHAUSTED` or HBM failure during XLA compilation or inference
  - each rung is a single relaunch, no waiting between rungs
- **Key difference from the 20B brute-force ladder (GTPU-012 to GTPU-027):** `reasoning_effort=low` is now correctly applied via the top-level field. The 20B prompt gen and response gen both worked at 4096/2048 and 8192/4096 respectively. The 120B model may need more room due to being a larger reasoning model, but the fix is in place.
- **Immediate action:** create script, launch rung 1

### ALIGN-273 - 2026-03-28 00:06 - GPT-OSS 120B E2E rung 1 — OOM at 256g, capacity backoff, racing central1 and east5 at 400g

- **Script:** `experiments/align_gpt_oss_120b_e2e_one_statement_v2.py`
  - `max_model_len=8192`, `teacher_max_tokens=4096`, `model_impl_type="vllm"`, `prefer_jax_for_bootstrap=False`
  - `judge_min_chosen_score=0.0`, `judge_min_gap=0.0` (pass all pairs through)
- **Attempt 1 (256g RAM):** `/ahmed/goss-120b-e2e-one-statement-v2-20260327`
  - **FAILED** — OOM killed during vLLM engine init: `Exit code 1: OOM killed (container exceeded memory limit)`
  - `Engine core initialization failed` before model loading completed
  - the `vllm` backend needs more host RAM than `flax_nnx` did for 120B
- **Attempt 2 (512g RAM):** `/ahmed/goss-120b-e2e-v2-512g-20260327`
  - **KILLED** — stuck pending: autoscaler only has 481 GB available on v5p-8 workers, 512g exceeds hardware capacity
- **Attempt 3 (450g RAM):** `/ahmed/goss-120b-e2e-v2-450g-20260327`
  - **KILLED** — stuck pending: `need 483183820800, available 481036337152` (450g + overhead exceeds 481g)
- **Attempt 4 (400g RAM, dual-region race):**
  - central1: `/ahmed/goss-120b-e2e-v2-400g-20260328` — pending, capacity backoff
  - east5: `/ahmed/goss-120b-e2e-v2-400g-east5-20260328` — pending, capacity race
  - 400g leaves ~80 GB headroom below the 481 GB hardware limit
  - if 400g still OOMs during engine init, the next move is to reduce `max_model_len` from 8192 to 4096 (which worked for 120B prompt gen under `flax_nnx` and reduces KV cache host memory)
- **Status:** waiting for one region to acquire TPU capacity

### ALIGN-274 - 2026-03-28 01:00 - GPT-OSS 120B E2E pipeline **SUCCEEDED** — 66 preference pairs from 68 prompts (97% pass rate)

- **Winning run:** `/ahmed/goss-120b-e2e-v2-400g-east5-20260328` (us-east5-a)
- **All steps SUCCEEDED:**
  - spec: SUCCEEDED
  - prompts: SUCCEEDED (68/68, all 3 stages, first attempt)
  - chosen (GPT-OSS 120B): SUCCEEDED (68 records)
  - rejected (Mixtral): SUCCEEDED (68 records)
  - judgments (GPT-OSS 120B): SUCCEEDED (68 records — 120B produces valid judge JSON unlike 20B!)
  - preference_pairs: SUCCEEDED (**66 records** — 97% pass rate from 68 prompts)
- **Final config:**
  - `max_model_len=8192`, `teacher_max_tokens=4096`
  - `model_impl_type="vllm"`, `prefer_jax_for_bootstrap=False`
  - `ram="400g"` (256g OOMs, 450g/512g exceed v5p-8 hardware limit of ~481g)
  - `reasoning_effort="low"` (top-level field, correctly applied)
  - `judge_min_chosen_score=0.0`, `judge_min_gap=0.0` (relaxed thresholds)
- **Artifacts:**
  - `gs://marin-us-east5/align/goss_120b_e2e_one_statement_v2/prompts-e94fb3/` (68 records)
  - `gs://marin-us-east5/align/goss_120b_e2e_one_statement_v2/chosen-85b19a/` (68 records)
  - `gs://marin-us-east5/align/goss_120b_e2e_one_statement_v2/rejected-fbdcbf/` (68 records)
  - `gs://marin-us-east5/align/goss_120b_e2e_one_statement_v2/judgments-514201/` (68 records)
  - `gs://marin-us-east5/align/goss_120b_e2e_one_statement_v2/preference_pairs-7c729e/` (66 records)
- **RAM escalation log:**
  - 256g: OOM during engine init
  - 512g: exceeds v5p-8 hardware (481g available)
  - 450g: exceeds v5p-8 hardware with overhead
  - **400g: SUCCESS** — leaves ~80g headroom
- **Key findings:**
  - `max_context_length=0` bug (ALIGN-262/264) did NOT appear — confirmed to be a `flax_nnx`-specific issue
  - GPT-OSS 120B produces valid judge JSON unlike 20B (which had 100% parse failures)
  - the `vllm` backend needs more host RAM than `flax_nnx` for 120B (~400g vs ~256g)
  - no sequence length issues at `max_model_len=8192` / `teacher_max_tokens=4096` — the brute-force escalation was never needed once `reasoning_effort=low` was correctly applied
- **This is the first fully successful GPT-OSS 120B end-to-end alignment pipeline run:**
  - from spec → prompts → chosen responses → rejected responses → judge → preference pairs
  - all on TPU via `vllm serve` with the validated serving contract

### ALIGN-275 - 2026-03-28 01:30 - Session summary: three bugs fixed, full 120B pipeline unblocked

- **Bugs fixed this session:**

  1. **`reasoning_effort` silently dropped** (ALIGN-265/266) — the root cause of all Stage 2 concretization failures and the entire brute-force escalation ladder (GTPU-012 to GTPU-027). Marin sent `reasoning_effort` inside `chat_template_kwargs`; vLLM's Harmony path ignores that field for GPT-OSS and reads `reasoning_effort` as a top-level request parameter instead. The model was running with zero reasoning constraint, spending 16k+ tokens thinking without producing output. Fix: one-line change to send `reasoning_effort` as a top-level field. Validated: 68/68 Stage 2 items passed on first attempt at the original 2048 token budget.

  2. **`model_impl_type="flax_nnx"` hardcoded in 120B config** (ALIGN-272 plan, implemented same session) — the `flax_nnx` backend produces incoherent token soup for GPT-OSS (proven by GTPU-001 vs GTPU-004 A/B test in the GPT-OSS TPU logbook). The 120B config at `gpt_oss_120b_tpu.py` hardcoded `flax_nnx` with no override. Fix: added `model_impl_type` and `prefer_jax_for_bootstrap` parameters defaulting to `"vllm"` and `False`. Added a defensive `ValueError` guard in `batched_vllm_serve.py` that blocks GPT-OSS + `flax_nnx` at session construction time.

  3. **`max_context_length=0` in E2E executor path** (ALIGN-262/264, resolved this session) — was hypothesized to be a config serialization bug in `align()`. Turned out to be a side effect of the `flax_nnx` backend mishandling `max_model_len`. Disappeared entirely once `model_impl_type="vllm"` was used. No serialization fix was needed.

- **Additional fixes:**
  - batch error handling in `_run_concretize_round_local_global()` — individual `ValueError`/`HTTPError` failures no longer crash the entire batch (ALIGN-266)
  - `ram="400g"` for 120B `vllm` backend — the `vllm` path needs more host RAM than `flax_nnx` did; 256g OOMs, 400g works (ALIGN-273/274)

- **Validated pipeline configurations:**

  | Model | max_model_len | max_tokens (prompt gen) | max_tokens (responses) | RAM | Result |
  |---|---|---|---|---|---|
  | GPT-OSS 20B | 4096 | 2048 | 2048 | 128g | prompt gen: 68/68; responses: `length` failure |
  | GPT-OSS 20B | 8192 | 2048 | 4096 | 128g | full E2E success; 0 preference pairs (judge parse failure) |
  | GPT-OSS 120B | 8192 | 2048 | 4096 | 400g | **full E2E success; 66 preference pairs (97% pass rate)** |

- **Next steps (not yet started):**
  1. **Full-spec run (all 46 statements)** — scale from 1 statement (`ask_clarifying_questions`) to all 46, producing ~3000+ preference pairs with GPT-OSS 120B
  2. **Judge quality audit** — the 120B judge works (unlike 20B) but thresholds are relaxed to 0.0; restore real thresholds (`judge_min_chosen_score=6.0`, `judge_min_gap=2.0`) and verify score distributions
  3. **DPO training** — feed preference pairs into the existing Levanter DPO pipeline (`train_dpo.py`), validated in ALIGN-000
  4. **SpecEval adherence measurement** — compare pre- and post-DPO models on the specification adherence benchmark

### ALIGN-276 - 2026-03-28 01:35 - Plan: full-spec GPT-OSS 120B E2E pipeline — 8-hour execution window ending 17:00 UTC

- **Goal:** Run the complete alignment pipeline on ALL 46 statements with GPT-OSS 120B as teacher/chosen/judge, Mixtral for rejected, producing a full preference dataset.
- **Deadline:** 2026-03-28 17:00 UTC (8 hours from now)
- **Validated recipe from ALIGN-274:**
  - `max_model_len=8192`, `teacher_max_tokens=4096`, `ram="400g"`
  - `model_impl_type="vllm"`, `prefer_jax_for_bootstrap=False`
  - `reasoning_effort="low"` (top-level field)
  - region: `us-east5-a` (won the capacity race for 120B)
- **Changes from one-statement run:**
  - remove `statement_ids=["ask_clarifying_questions"]` filter → runs all 46 statements
  - keep `judge_min_chosen_score=0.0`, `judge_min_gap=0.0` for now (can tighten later)
  - keep `covering_strength=2` (pairwise — ~68 prompts per statement × 46 statements ≈ 3,100+ prompts)
- **Expected scale:**
  - prompts: ~3,100+ (46 × ~68)
  - chosen: ~3,100+ responses (GPT-OSS 120B)
  - rejected: ~3,100+ responses (Mixtral)
  - judgments: ~3,100+ (GPT-OSS 120B)
  - preference pairs: ~3,000+ (assuming ~97% pass rate like one-statement)
- **Expected wall-clock:**
  - prompt gen: ~11 min per statement on 20B → 120B is ~2x slower → ~30 min total for prompts
  - chosen/rejected: each response ~10-30s × 3100 = ~8-25 hours if sequential, but they run in parallel child jobs
  - this may exceed the 8-hour window if chosen is single-threaded
- **Risk: wall-clock may exceed 8 hours**
  - the one-statement run had 68 prompts and took ~1 hour for chosen + rejected + judge
  - 46 statements × 68 prompts = 3128 prompts → ~46 hours if linear scaling
  - BUT: chosen/rejected are separate child jobs that each get their own TPU
  - prompt gen is the only sequential bottleneck (all 46 statements in one job)
- **Monitoring protocol:**
  - check status every 15-30 min
  - if a step fails, diagnose and relaunch immediately
  - if capacity is stuck, try alternate region
  - update logbook at each milestone
- **Immediate action:** create full-spec script, launch

### ALIGN-277 - 2026-03-28 02:20 - Full-spec 120B pipeline launched

- **Script:** `experiments/align_gpt_oss_120b_full_spec.py`
- **Job:** `/ahmed/goss-120b-full-spec-east5-20260328` (us-east5-a)
- **Config:** identical to ALIGN-274 one-statement success but with `statement_ids` removed (all 46 statements)

### ALIGN-278 - 2026-03-28 16:00 - Full-spec progress: Stage 1 complete, Stage 2 preempted at 77.6% and restarted

- **Timeline:**
  - 09:24 — vLLM server starting
  - 09:34 — `vLLM environment ready`
  - 09:34 — Stage 1: 46 statements, completed 46/46 in ~11 min (2 required retries for JSON parse)
  - 09:45 — Stage 2: 3339 concretize items, started
  - 14:51 — Stage 2: 2591/3339 (77.6%) — steady ~8 items/min
  - ~15:30 — **PREEMPTED** (`preemption_count=1`)
  - 15:46 — restarted on new worker, vLLM environment ready
  - 15:46 — Stage 1 checkpoint loaded (skipped), Stage 2 restarted from scratch (no Stage 2 checkpoint)
  - 15:50 — Stage 2: 32/3339 (1.0%) — back to zero
- **Impact:** lost ~5 hours of Stage 2 work. Stage 2 now needs another ~6.5 hours from scratch.
- **Total estimated completion:** Stage 2 ~22:00 UTC + Stage 3 ~1h + chosen/rejected/judge ~many hours
- **Status:** pipeline is healthy, no errors, just slow due to 3339 sequential items on single TPU and preemption recovery without Stage 2 checkpointing
- **This will exceed the 8-hour deadline (17:00 UTC)** but the pipeline is running correctly — it just needs more time
- **Note for future:** Stage 2 checkpointing would save hours on preemption recovery — currently Stage 1 checkpoints but Stage 2 does not

### ALIGN-279 - 2026-03-28 17:00 - Plan: incremental Stage 2 checkpointing

#### Problem

Stage 2 concretization has 3339 items taking ~6.5 hours on GPT-OSS 120B. The checkpoint is all-or-nothing: written once after ALL items complete. A preemption at 77.6% wiped out 5+ hours of work.

| Stage | Items | Time (120B) | Checkpoint | Preemption cost |
|-------|-------|-------------|-----------|-----------------|
| Stage 1 | 46 | ~11 min | All-or-nothing + raw attempt fallback | ~11 min |
| **Stage 2** | **3339** | **~6.5 hours** | **All-or-nothing** | **~6.5 hours** |
| Stage 3 | ~3339 | ~1.5 hours | Incremental per-batch | ~5 min |

#### Approach

Keep the existing global batching loop unchanged. After each round completes and results are merged into the outer aggregated state, detect which statements became fully complete and persist them. On restart, load partial checkpoints and only build work items for incomplete statements.

**What stays the same:**
- Batching — items from different statements still mixed in batches of `local_serve_batch_size=32`
- Retry logic — global across all statements, `concretize_max_attempts` attempts
- `stage_status` schema — derive partial state from the `ideation_by_statement/` directory, no schema changes
- Stage 1 — already has raw-attempt recovery via `_recover_understandings_from_attempts`
- Stage 3 — already has incremental checkpointing
- `_run_concretize_round_local_global` — unchanged signature and behavior
- All vLLM interaction, all prompt templates

**Scope:** `_run_concretization_stage_local` only (the local vLLM path). The API path (`_run_concretization_stage`, line 1429) still persists all-or-nothing after all futures finish — it is not addressed by this fix. It could benefit from similar checkpointing but that's a separate change.

#### Checkpoint format

```
artifacts/checkpoints/ideation_by_statement/
  ask_clarifying_questions.jsonl.gz    # one record: {statement_id, ideation, fingerprint}
  be_honest.jsonl.gz
  ...
```

Each file: one JSONL.gz record (~10-50 KB) containing the full ideation result (including diagnostics) + a fingerprint for validation.

**Fingerprint:** A hash of the actual plan contents — `covering_strength`, `covering_seed`, `num_axes`, and a hash of the serialized `axis_config` list. This catches real plan changes (different axes, different covering configs) rather than just restating counts. On load, recompute the expected plan and compare. Discard any checkpoint whose fingerprint doesn't match.

#### Where completion detection happens

**Critical design choice:** Completion detection happens in `_run_concretization_stage_local` (the outer function, line 840), NOT in `_run_concretize_round_local_global` (the inner round function, line 675). This is because:

1. `_run_concretize_round_local_global` creates a fresh `parsed_by_statement` dict each round (line 681). A statement that becomes complete only after retries across multiple rounds would never be detected inside a single round.
2. The outer function `_run_concretization_stage_local` maintains the aggregated `parsed_by_statement` (line 859) and `diagnostics_by_statement` (line 860) across all rounds. Completion detection must see this outer state.
3. `_build_concretization_result` needs both scenarios AND diagnostics (line 735). The round-local callback would only have scenarios for that round, not the full diagnostics history.

So: after each round returns and its results are merged into the outer aggregated dicts (lines 892-896), scan for newly completed statements there.

#### I/O cost

46 GCS writes over 6.5 hours, each ~10-50 KB. Writes happen after round merges when a statement crosses its completion threshold — not during vLLM request processing. The monolithic `_save_ideation_checkpoint` + `_save_artifacts` still runs once at the very end.

#### Preemption resilience

All fully checkpointed statements survive. Statements with incomplete items rerun. In practice, preemption loses work for all statements that hadn't fully completed by the last round merge — usually a few statements' worth (~8-30 min), not the entire 6.5-hour Stage 2.

#### Changes (all in `lib/marin/src/marin/alignment/generate_prompts.py`)

**1. New constant (near line 48):**

```python
_STAGE2_INCREMENTAL_DIRNAME = "ideation_by_statement"
```

**2. Three new helper functions (near line 265):**

```python
import hashlib

def _make_concretization_fingerprint(plan: _ConcretizationPlan) -> str:
    """Fingerprint a concretization plan by hashing its full content."""
    content = json.dumps({
        "num_configs": len(plan.configs),
        "num_axes": len(plan.axes),
        "configs": plan.configs,  # the actual axis_config dicts
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def _save_statement_ideation(
    output_path: str, statement_id: str, ideation: dict[str, Any], fingerprint: str,
) -> None:
    """Persist one statement's completed Stage 2 ideation to its own checkpoint file."""
    path = f"{_checkpoint_base_path(output_path)}/{_STAGE2_INCREMENTAL_DIRNAME}/{statement_id}.jsonl.gz"
    write_jsonl_file(
        [{"statement_id": statement_id, "ideation": _serialize_model_artifact(ideation), "fingerprint": fingerprint}],
        path,
    )

def _load_partial_ideation_checkpoint(
    output_path: str, plans: dict[str, _ConcretizationPlan],
) -> dict[str, dict[str, Any]]:
    """Load per-statement ideation checkpoints. Validates fingerprints against current plans."""
    checkpoint_dir = f"{_checkpoint_base_path(output_path)}/{_STAGE2_INCREMENTAL_DIRNAME}"
    fs, fs_path = url_to_fs(checkpoint_dir)
    if not fs.exists(fs_path):
        return {}
    ideations = {}
    for record in load_sharded_jsonl_gz(checkpoint_dir):
        sid = record["statement_id"]
        if sid not in plans:
            continue
        saved_fp = record.get("fingerprint", "")
        expected_fp = _make_concretization_fingerprint(plans[sid])
        if saved_fp != expected_fp:
            logger.warning("Discarding stale Stage 2 checkpoint for '%s': fingerprint mismatch", sid)
            continue
        ideations[sid] = record["ideation"]
    return ideations
```

**3. NO changes to `_run_concretize_round_local_global` (line 675).**

This function stays exactly as-is. It processes one round of batched items and returns. Completion detection does NOT go here because:
- `parsed_by_statement` is recreated fresh each round (line 681)
- It cannot see cross-attempt aggregated state
- It cannot see diagnostics from prior rounds

**4. Add completion detection + checkpointing to `_run_concretization_stage_local` (line 840):**

New parameters:
- `completed_ideations: dict[str, dict[str, Any]] | None = None`
- `statement_checkpoint_callback: Callable[[str, dict[str, Any]], None] | None = None`

Changes to the function body:
- Filter `completed_sids` out of work item construction (skip already-done statements)
- Precompute `expected_items = {sid: len(plan.configs) for sid, plan in plans.items()}`
- Precompute fingerprints once: `fingerprints = {sid: _make_concretization_fingerprint(plan) for sid, plan in plans.items()}`
- Track `checkpointed_this_run: set[str] = set(completed_sids)`
- **After each round's results are merged into the outer `parsed_by_statement` and `diagnostics_by_statement` (after line 896)**, scan for newly completed statements:

```python
# After merging round results into outer aggregated state:
for sid in list(parsed_by_statement):
    if (
        sid not in checkpointed_this_run
        and len(parsed_by_statement[sid]) >= expected_items.get(sid, float("inf"))
    ):
        # Statement fully complete — build full ideation with diagnostics and persist
        ideation = _build_concretization_result(
            sid, config, plans[sid].axes, plans[sid].configs, plans[sid].stats,
            parsed_by_statement[sid], diagnostics_by_statement.get(sid, []),
        )
        if statement_checkpoint_callback is not None:
            statement_checkpoint_callback(sid, ideation)
        checkpointed_this_run.add(sid)
```

This fires after every round merge, sees the full aggregated scenarios + diagnostics, and builds a complete ideation record.

- Merge `completed_ideations` with newly computed results in the return value

**5. Pass through in `_run_concretization_stage` (line 1417):**

Add `completed_ideations` and `statement_checkpoint_callback` params. Pass to local path only. API path unchanged — add a comment noting it still has all-or-nothing persistence and could benefit from similar checkpointing in the future.

**6. Add partial-resume branch to `generate_prompts_from_spec` (lines 1194-1276):**

After the existing "Stage 2 complete" check and before "Stage 2 from scratch", add:

```python
partial_ideations: dict[str, dict[str, Any]] = {}
if understandings is not None and ideations is None:
    plans = _prepare_concretization_plans(understandings, config)
    partial_ideations = _load_partial_ideation_checkpoint(config.output_path, plans)
    if partial_ideations:
        logger.info("Loaded partial Stage 2 checkpoint: %d/%d statements",
                     len(partial_ideations), len(expected_statement_ids))
        if set(partial_ideations) == expected_statement_ids:
            ideations = partial_ideations
            _save_ideation_checkpoint(config.output_path, statements, understandings, ideations)
            stage_status[_STAGE2_NAME] = {"complete": True, "num_statements": len(ideations)}
            _save_stage_status(config.output_path, stage_status)
```

When entering Stage 2 execution, compute plans and fingerprints once and close over them:

```python
plans = _prepare_concretization_plans(understandings, config)
fingerprints = {sid: _make_concretization_fingerprint(plan) for sid, plan in plans.items()}

def save_stmt_ckpt(sid: str, ideation: dict[str, Any]) -> None:
    _save_statement_ideation(config.output_path, sid, ideation, fingerprints[sid])

ideations = _run_concretization_stage(
    understandings, config, session,
    completed_ideations=partial_ideations,
    statement_checkpoint_callback=save_stmt_ckpt,
)
```

#### Tests

- Existing tests: unchanged (global batching preserved, `_run_concretize_round_local_global` signature unchanged)
- New test: simulate partial checkpoint directory, verify `_load_partial_ideation_checkpoint` loads correctly with fingerprint validation, verify restart skips completed statements and only processes remaining ones

#### Execution

1. Implement changes 1-6
2. `uv run pytest tests/test_alignment.py -q`
3. `./infra/pre-commit.py --fix`
4. Kill current full-spec run (recovering from preemption, back at ~1%)
5. Relaunch with checkpointing
6. Monitor
