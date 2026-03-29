# GPT-OSS TPU: Research Logbook

**Branch:** `alignment_function`
**Experiment ID prefix:** `GTPU`
**Related logs:**
- `.agents/logbooks/alignment_function.md`
- `docs/debug-log-gpt-oss-harmony-path.md`

## Scope
- **Goal:** Make `vllm serve` on TPU produce coherent `gpt-oss-20b` outputs and unblock Marin Stage 1 prompt generation.
- **Primary metric(s):**
  - `/v1/chat/completions` returns non-null final assistant content for a deterministic probe.
  - Raw generated text or token traces are coherent on at least one supported `vllm serve` contract.
  - The one-statement Stage 1 prompt-generation run succeeds for `ask_clarifying_questions`.
- **Constraints:**
  - Must use `vllm serve`.
  - Use `unsloth/gpt-oss-20b-BF16` for debugging until the serving contract is understood.
  - Prefer one warm server with multiple probes per job to avoid repeated TPU startup cost.
  - Do not relaunch 120B until 20B passes the Stage 1 gate.
  - Do not spend time on non-serve inference paths unless the user explicitly redirects.

## Stop Criteria
- A deterministic 20B probe returns coherent assistant content through `vllm serve`.
- A single-statement Stage 1 run produces a parseable `<variation_axes>` block.
- The fix location is narrowed to one of:
  - Marin GPT-OSS client contract
  - `vllm serve` GPT-OSS request/response shaping
  - tokenizer/template staging
  - `tpu_inference` GPT-OSS TPU backend

## Baseline
- **Date:** 2026-03-26 to 2026-03-27
- **Model under test:** `unsloth/gpt-oss-20b-BF16`
- **Downloaded model artifact:** `gs://marin-us-central1/models/unsloth--gpt-oss-20b-BF16-vllm--cc89b3e7fd423253264883a80a4fa5abc619649f`
- **Known architecture path:**
  - Marin launches `vllm serve`
  - GPT-OSS TPU configs pin `model_impl_type="flax_nnx"`
  - `tpu_inference` routes GPT-OSS to JAX `GptOss`
  - `vllm serve` is therefore the server surface, while `tpu_inference` is the active TPU execution backend
- **Code refs:**
  - `experiments/gpt_oss_20b_tpu.py`
  - `experiments/gpt_oss_20b_harmony_compare_smoke.py`
  - `lib/marin/src/marin/inference/vllm_server.py`
  - `lib/marin/src/marin/alignment/batched_vllm_serve.py`
  - `tpu_inference/models/common/model_loader.py`
  - `tpu_inference/runner/tpu_runner.py`

### Existing Runs

| Run | Root | Output Path | Result |
|-----|------|-------------|--------|
| GTPU-baseline | `/ahmed/gpt-oss-20b-harmony-compare-20260326-1728` | `gs://marin-us-central1/align/debug_gpt_oss_20b_harmony_compare_20260326_1728/compare-ecc59c` | `/v1/completions` gibberish; `/v1/chat/completions` returned `message.content=null`, `reasoning_content=null`, `finish_reason="length"` |
| GTPU-parser | `/ahmed/gpt-oss-20b-harmony-compare-parser-20260326-1739` | `gs://marin-us-central1/align/debug_gpt_oss_20b_harmony_compare_parser_20260326_1739/compare-c031e0` | Adding `--reasoning-parser openai_gptoss` did not fix output |
| GTPU-low | `/ahmed/gpt-oss-20b-harmony-compare-low-20260326-1750` | `gs://marin-us-central1/align/debug_gpt_oss_20b_harmony_compare_low_20260326_1750/compare-81c174` | Lowering `reasoning_effort` to `low` did not fix output |

### Baseline Conclusion
- Basic TPU bring-up is no longer the blocker.
- The model boots, the server starts, and requests run.
- The unresolved bug is in the serving contract above basic startup:
  - prompt rendering
  - endpoint choice
  - response shaping / parsing
  - tokenizer/template pairing
  - or backend-specific corruption in the `flax_nnx` GPT-OSS path

## Hard Rules For The Next Agent
- Keep `vllm serve` as a hard requirement.
- Stay on 20B until one deterministic probe returns coherent content.
- Do not rerun the full alignment pipeline before a small serve probe passes.
- Use one warm server per experiment job and run a probe matrix inside that job.
- Persist raw request and raw response payloads for every probe.
- Persist enough token-level evidence to tell apart:
  - bad text decoding
  - bad OpenAI payload shaping
  - bad token generation

## Working Hypotheses

### H1: OpenAI-compatible response shaping is broken on this TPU path
- The model may be generating sane tokens, but `/v1/chat/completions` is failing to surface them as `message.content` or `reasoning_content`.

### H2: Marin is still using the wrong GPT-OSS serve contract
- One of these may be the only valid contract:
  - raw messages -> `/v1/chat/completions`
  - Harmony-rendered prompt -> `/v1/completions`
  - HF chat-template-rendered prompt -> `/v1/completions`

### H3: The corruption is specific to `MODEL_IMPL_TYPE=flax_nnx`
- `vllm serve` may be fine, but the JAX GPT-OSS backend beneath it may be returning corrupted generations.

### H4: Tokenizer/template staging is mismatched
- GPT-OSS may boot from the right checkpoint while serving with the wrong tokenizer or template packaging.

### H5: Termination semantics are wrong
- The model may be generating Harmony channel tokens but never reaching a final assistant message under current stop or budget settings.

## Experiment Plan

| Run | Config Change | Hypothesis |
|-----|---------------|------------|
| GTPU-001 | Build a warm-server probe matrix job for `gpt-oss-20b` and persist raw request/response artifacts plus token traces | We need a cheaper, denser discriminator than repeated one-off jobs |
| GTPU-002 | Compare `/v1/chat/completions` and `/v1/completions` with `temperature=0`, short budgets, and `logprobs` enabled | If token traces are coherent but payload fields are wrong, the bug is response shaping |
| GTPU-003 | Compare three serve contracts in one job: raw messages, HF-rendered prompt, Harmony-rendered prompt | One GPT-OSS contract may be valid while the others are not |
| GTPU-004 | Hold `vllm serve` fixed and compare `MODEL_IMPL_TYPE=flax_nnx` vs `MODEL_IMPL_TYPE=vllm` or `auto` if it boots | If only `flax_nnx` is broken, the bug is in the TPU backend path rather than `vllm serve` itself |
| GTPU-005 | Hold server config fixed and vary tokenizer source / staging path | If one tokenizer pairing fixes coherence, the bug is packaging not generation |
| GTPU-006 | Sweep `reasoning_effort`, `max_tokens`, and any supported stop settings on a tiny prompt | If final content appears only under some settings, the issue is termination semantics |
| GTPU-007 | Promote the winning serve contract into the one-statement Stage 1 run | The only real gate is parseable Stage 1 output |

## Required Artifact Contract

Every future probe experiment must write at least:

```text
artifacts/
  model_metadata.json
  server_start.json
  logs_tail.txt
  prompts/
    raw_messages.json
    hf_rendered_prompt.txt
    harmony_rendered_prompt.txt
  requests/
    *.json
  responses/
    *.json
  traces/
    *.json
  summary.json
```

Minimum contents:
- `model_metadata.json`
  - model path
  - tokenizer path
  - TPU type
  - tensor parallel size
  - `MODEL_IMPL_TYPE`
  - `--reasoning-parser` setting
- `requests/*.json`
  - exact request body sent to the server
- `responses/*.json`
  - exact raw server response
- `traces/*.json`
  - decoded text
  - logprobs if available
  - token ids if available
  - finish reason
- `summary.json`
  - one-line verdict per probe
  - next-branch recommendation

## Concrete Run Order

### GTPU-001: Warm Server Probe Matrix
- **Hypothesis:** The fastest path is a single experiment that launches one warm `vllm serve` instance and runs many tiny probes.
- **Implementation task:**
  - Create `experiments/gpt_oss_20b_vllm_probe_matrix.py`.
  - Start from `experiments/gpt_oss_20b_harmony_compare_smoke.py`.
  - Reuse `experiments/gpt_oss_20b_tpu.py` for the base model config.
- **Probe matrix to run in one job:**
  - `chat_low_64`
  - `chat_low_128`
  - `chat_low_256`
  - `completions_hf_template_64`
  - `completions_hf_template_128`
  - `completions_harmony_template_64`
  - `completions_harmony_template_128`
- **Fixed settings:**
  - `temperature=0`
  - `n=1`
  - `tpu_type=v5p-8`
  - `tensor_parallel_size=4`
  - `reasoning_effort=low`
  - `logprobs=True` wherever accepted
- **Required output:** full artifact contract above
- **Expected discriminator:**
  - if any probe returns coherent content, stop broad debugging and standardize on that contract
  - if all payload fields are empty but token traces are coherent, focus on server-side response shaping
  - if token traces are already gibberish, focus on backend or tokenizer mismatch

### GTPU-002: Response-Shape Probe
- **Only run if:** GTPU-001 shows plausible tokens but unusable `message.content`
- **Hypothesis:** The OpenAI-compatible response object is being assembled incorrectly.
- **Config change:** no change to model or TPU backend; only increase instrumentation on response parsing.
- **What to inspect:**
  - raw JSON payload
  - finish reason
  - token ids
  - logprobs content
  - any hidden reasoning fields
- **Success criterion:** identify whether the generated tokens are sane before payload shaping.

### GTPU-003: Serve-Contract A/B/C
- **Only run if:** GTPU-001 does not immediately identify a winning contract
- **Hypothesis:** GPT-OSS works on one contract but not the others.
- **Contracts to compare in the same job:**
  - raw messages -> `/v1/chat/completions`
  - HF `tokenizer.apply_chat_template(...)` -> `/v1/completions`
  - Harmony-rendered prompt -> `/v1/completions`
- **Decision rule:**
  - if raw messages win, move Marin GPT-OSS integration toward `/v1/chat/completions`
  - if Harmony-rendered prompt wins, keep `/v1/completions` but replace HF template rendering
  - if none win, continue to backend or tokenizer isolation

### GTPU-004: Backend A/B
- **Only run if:** GTPU-001 through GTPU-003 still show incoherent generations
- **Hypothesis:** `MODEL_IMPL_TYPE=flax_nnx` is the failing layer.
- **Config change:**
  - run the same minimal probe job with:
    - `MODEL_IMPL_TYPE=flax_nnx`
    - `MODEL_IMPL_TYPE=vllm`
    - `MODEL_IMPL_TYPE=auto` if boot succeeds
- **Decision rule:**
  - if `vllm` or `auto` produces coherent output and `flax_nnx` does not, file the issue against the `tpu_inference` GPT-OSS JAX path
  - if all variants fail similarly, focus on tokenizer or serve contract

### GTPU-005: Tokenizer A/B
- **Only run if:** backend A/B is inconclusive
- **Hypothesis:** the checkpoint is fine but tokenizer/template staging is wrong.
- **Tokenizer variants to compare:**
  - tokenizer from model artifact path
  - explicit tokenizer repo path if different
  - fully staged local tokenizer snapshot
- **Decision rule:**
  - if one tokenizer source fixes coherence, patch Marin staging and stop backend debugging

### GTPU-006: Termination Sweep
- **Only run if:** outputs look structurally plausible but no final assistant content appears
- **Hypothesis:** GPT-OSS is generating, but final message termination is never reached.
- **Sweep:**
  - `reasoning_effort in {low, medium}`
  - `max_tokens in {32, 64, 128, 256}`
  - any supported stop-token or stop-string knobs for GPT-OSS on `vllm serve`
- **Decision rule:**
  - if final content appears at some budgets or stop settings, codify those defaults in Marin

### GTPU-007: Stage 1 Promotion Gate
- **Only run after:** one serve contract is clearly coherent on 20B
- **Hypothesis:** once the serve contract is fixed, the one-statement Stage 1 job should pass without further parser changes.
- **Target:** rerun only the one-statement prompt generation for `ask_clarifying_questions`
- **Success criterion:** parseable `<variation_axes>` and saved understanding artifact

## First Command The Next Agent Should Prepare

Do not rerun the old compare script first. Build the probe matrix script and launch that.

Planned command after implementing `experiments/gpt_oss_20b_vllm_probe_matrix.py`:

```bash
uv run experiments/gpt_oss_20b_vllm_probe_matrix.py -- \
  --name debug_gpt_oss_20b_vllm_probe_matrix_$(date +%Y%m%d_%H%M%S) \
  --tpu-type v5p-8 \
  --tensor-parallel-size 4 \
  --temperature 0 \
  --max-tokens 128 \
  --reasoning-effort low
```

## Prewritten First Entry

### 2026-03-26 01:30 - GTPU-001: Warm-server probe matrix for `vllm serve` on `gpt-oss-20b`
- **Hypothesis:** repeated one-off jobs are too sparse; one warm server with a dense probe matrix will reveal whether the bug is in response shaping, serve contract, tokenizer pairing, or the `flax_nnx` backend.
- **Exact implementation target:** `experiments/gpt_oss_20b_vllm_probe_matrix.py`
- **Base files to copy from:**
  - `experiments/gpt_oss_20b_harmony_compare_smoke.py`
  - `experiments/gpt_oss_20b_tpu.py`
- **Must capture:**
  - raw messages
  - HF-rendered prompt
  - Harmony-rendered prompt
  - raw request JSON
  - raw response JSON
  - token traces
  - server logs tail
  - summary verdict for each probe
- **Expected outcome:** at least one of the following becomes true:
  - a supported `vllm serve` contract is identified
  - a payload-shaping bug is isolated
  - backend corruption is strongly implicated
  - tokenizer mismatch is strongly implicated
- **Immediate next action after the run:** choose only one branch among GTPU-002 through GTPU-006 based on the evidence; do not broaden the search again.

### 2026-03-26 18:39 - GTPU-001: Implemented the warm-server probe matrix experiment and prepared the first real launch
- **Action:**
  - added:
    - `experiments/gpt_oss_20b_vllm_probe_matrix.py`
  - validated it locally with:
    - `python3 -m py_compile experiments/gpt_oss_20b_vllm_probe_matrix.py`
    - `./infra/pre-commit.py --fix experiments/gpt_oss_20b_vllm_probe_matrix.py`
- **What the new experiment does:**
  - launches one warm `vllm serve` session on `gpt-oss-20b`
  - runs a probe matrix across:
    - prompt family:
      - `tiny`
      - `stage1`
    - endpoint:
      - `/v1/chat/completions`
      - `/v1/completions`
    - prompt contract:
      - raw messages
      - HF `apply_chat_template(...)`
      - `openai-harmony` rendered completion prompt
  - persists:
    - raw request payloads
    - raw response payloads
    - rendered prompts
    - Harmony stop-token metadata
    - per-probe traces
    - server logs tail
    - summary branch recommendation
- **Important implementation details:**
  - all probes are deterministic:
    - `temperature=0`
    - `n=1`
    - default `reasoning_effort=low`
  - Harmony-rendered `/completions` probes now use:
    - `openai-harmony`
    - plus explicit `stop_token_ids`
  - `/completions` probes also set:
    - `skip_special_tokens=false`
  - the remote job explicitly installs:
    - `openai-harmony`
- **Launch command prepared:**
  - `uv run experiments/gpt_oss_20b_vllm_probe_matrix.py -- --name debug_gpt_oss_20b_vllm_probe_matrix_20260326_1839 --tpu-type v5p-8 --tensor-parallel-size 4 --temperature 0 --reasoning-effort low`
- **Immediate next action:**
  - launch the probe matrix
  - inspect `artifacts/summary.json`, `artifacts/traces/*.json`, and `artifacts/logs_tail.txt`
  - then choose exactly one next branch among:
    - GTPU-002
    - GTPU-003
    - GTPU-004

### 2026-03-26 18:42 - GTPU-001: The first launch path is region-misaligned before TPU serve even starts
- **Finding:**
  - the current shell exports:
    - `MARIN_PREFIX=gs://marin-us-central2`
  - `iris.marin_fs.marin_prefix()` resolves storage prefix from `MARIN_PREFIX` first
  - so this launch is defaulting:
    - experiment metadata
    - downloaded model artifacts
    - and step outputs
    - into `marin-us-central2`
- **Why this matters:**
  - in the checked-in Iris config, `v5p` capacity is configured in:
    - `us-central1-a`
    - `us-east5-a`
  - not in:
    - `us-central2`
  - so the current run is on a path that can later fail region-overlap checks even though it has not reached TPU allocation yet
- **Observed symptom:**
  - the probe launch started re-materializing the 20B model into:
    - `gs://marin-us-central2/...`
  - which is unnecessary for the intended `v5p-8` experiment path
- **Conclusion:**
  - this is not evidence that `v5p` exists in `us-central2`
  - it is evidence that the current environment prefix is wrong for this experiment
- **Corrective action for the next launch:**
  - set `MARIN_PREFIX` to a `v5p`-compatible region before launching:
    - `gs://marin-us-central1`
    - or `gs://marin-us-east5`
  - then relaunch GTPU-001 so model download, experiment metadata, and TPU region constraints all agree

### 2026-03-26 18:45 - GTPU-001: Live run is still downloading into the wrong bucket and should not be used as evidence about TPU placement
- **Live status:**
  - the launched probe is still in the download/materialization stage
  - it has not yet reached TPU allocation or `vllm serve`
  - current observed state:
    - `stage0-Map → Write`
    - `3/16 complete`
    - repeatedly streaming `unsloth/gpt-oss-20b-BF16` shards into `gs://marin-us-central2/...`
- **Clarification for the next agent:**
  - the wrong part is the storage/output prefix, not the checked-in `v5p` zone map
  - the checked-in config still says:
    - `tpu_v5p_8 -> [us-central1-a, us-east5-a]`
    - `tpu_v4_8 -> [us-central2-b]`
- **Operational guidance:**
  - do not keep debugging model behavior from this run
  - first stop it with user approval
  - then relaunch the exact same probe command under:
    - `MARIN_PREFIX=gs://marin-us-central1`

### 2026-03-26 18:55 - GTPU-001: The probe must be launched inside an Iris parent job, not from a plain local shell
- **New finding:**
  - launching `uv run experiments/gpt_oss_20b_vllm_probe_matrix.py ...` directly from the local shell does not give the step an Iris/Fray backend
  - the executor logs show:
    - `fray.v2.client current_client: using LocalClient (fallback)`
  - that means the nested `remote(...)` probe step runs on the local machine instead of a TPU worker
- **Observed failure from the bad local launch:**
  - `FileNotFoundError: [Errno 2] No such file or directory: 'vllm'`
  - this happened because the local Mac process tried to execute the native TPU `vllm serve` path directly
- **Conclusion:**
  - the successful GPT-OSS 20B compare jobs were not launched the same way as this bad local probe attempt
  - the correct contract for this probe is:
    - outer envelope: `iris job run`
    - inner TPU work: nested executor `remote(...)` job
- **Corrected launch command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-vllm-probe-matrix-central1-20260326-1856 --cpu 4 --memory 16GB --disk 10GB --region us-central1 --zone us-central1-a -e MARIN_PREFIX gs://marin-us-central1 -- python experiments/gpt_oss_20b_vllm_probe_matrix.py --name debug_gpt_oss_20b_vllm_probe_matrix_central1_20260326_1856 --tpu-type v5p-8 --tensor-parallel-size 4 --temperature 0 --reasoning-effort low`

### 2026-03-26 18:57 - GTPU-001: Corrected central1 launch is now running end-to-end through the nested TPU child
- **Autoscaler/placement check before relaunch:**
  - `tpu_v5p_8-us-central1-a` is currently idle
  - `tpu_v5p_8-us-east5-a` is currently blocked on repeated no-capacity failures
  - for this session, central1 is the right choice
- **Submitted parent job:**
  - `/ahmed/gpt-oss-20b-vllm-probe-matrix-central1-20260326-1856`
  - parent constraints:
    - `region = us-central1`
    - `zone = us-central1-a`
    - `MARIN_PREFIX = gs://marin-us-central1`
- **Nested TPU child job:**
  - `/ahmed/gpt-oss-20b-vllm-probe-matrix-central1-20260326-1856/align-debug_gpt_oss_20b_vllm_probe_matrix_central1_20260326_1856-probe_matrix_3becb8be-8d785358`
  - child resources:
    - `v5p-8`
    - `32 CPU`
    - `128 GB RAM`
    - `60 GB disk`
  - child worker:
    - `marin-tpu-v5p-8-us-central1-a-20260326-1801-39c8210e-worker-0`
- **Important confirmation:**
  - inside the parent task logs, the executor now says:
    - `fray.v2.client current_client: using Iris backend (auto-detected)`
  - this confirms the nested `remote(...)` step is no longer falling back to LocalClient
- **Current child logs:**
  - `marin.inference.vllm_server Starting vLLM environment`
  - `marin.inference.vllm_server Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
  - then repeated warning:
    - `Tensorflow library not found, tensorflow.io.gfile operations will use native shim calls. GCS paths (i.e. 'gs://...') cannot be accessed.`
- **Interpretation:**
  - the corrected launch has reached real TPU-side native `vllm` startup
  - the next likely blocker is now inside the TPU child runtime itself, around GCS access from the native server path
  - keep watching for either:
    - later successful `vLLM environment ready`
    - or a concrete model-load failure tied to the missing TensorFlow/GCS path support

### 2026-03-26 19:03 - GTPU-001 result: `vllm serve` boots cleanly in `us-central1-a`, but both chat and manual Harmony completions still generate unusable text
- **Completed jobs:**
  - parent:
    - `/ahmed/gpt-oss-20b-vllm-probe-matrix-central1-20260326-1856`
  - TPU child:
    - `/ahmed/gpt-oss-20b-vllm-probe-matrix-central1-20260326-1856/align-debug_gpt_oss_20b_vllm_probe_matrix_central1_20260326_1856-probe_matrix_3becb8be-8d785358`
- **Artifact root:**
  - `gs://marin-us-central1/align/debug_gpt_oss_20b_vllm_probe_matrix_central1_20260326_1856/probe_matrix-7700b3`
- **What succeeded:**
  - native TPU `vllm serve` reached:
    - `vLLM environment ready`
  - all probe requests returned:
    - `200 OK`
  - the child persisted:
    - raw requests
    - raw responses
    - rendered prompts
    - diagnostics
    - summary
- **What failed semantically:**
  - `/v1/chat/completions` with raw messages does not produce a usable assistant message
    - `message.content = null`
    - `message.reasoning_content = null`
    - `finish_reason = "length"`
    - but logprobs reveal visible generated tokens like:
      - `crashes Exceptional ‑ p form dimensions wow interrupted zone502 ...`
  - `/v1/completions` with HF-rendered prompt is also incoherent
  - `/v1/completions` with `openai-harmony` rendered prompt is also incoherent
- **Important interpretation:**
  - this is not just a Stage 1 XML-parser problem
  - this is not just a `/chat/completions` response-shaping problem
  - this is not just a Harmony prompt-rendering mistake in Marin
  - the current TPU-native decode path is producing junk tokens even on a tiny one-word prompt
- **Representative raw evidence:**
  - tiny chat request:
    - system: `You are a concise assistant. Return exactly one uppercase word and nothing else.`
    - user: `Reply with exactly HELLO.`
  - tiny chat raw response still generated tokens like:
    - `crashes Exceptional ‑ p form dimensions wow interrupted zone502 max design CLI parameters gamma CLI`
  - tiny Harmony `/completions` raw response text:
    - `UPS tagSelectedSurfacePlayer JSs vitShort cejq shortest managementData α values`
- **Most likely next branch:**
  - GTPU-004
  - backend A/B under the same `vllm serve` surface
  - current evidence points more strongly at the `flax_nnx` execution path than at the prompt contract

### 2026-03-26 19:06 - GTPU-004 prep: add explicit backend override knobs so the next agent can A/B `flax_nnx` against the native vLLM wrapper
- **Code changes made:**
  - `experiments/gpt_oss_20b_tpu.py`
    - added explicit parameters:
      - `model_impl_type`
      - `prefer_jax_for_bootstrap`
    - stopped hard-coding `flax_nnx`
  - `experiments/gpt_oss_20b_vllm_probe_matrix.py`
    - added CLI flags:
      - `--model-impl-type {auto,flax_nnx,vllm}`
      - `--prefer-jax-for-bootstrap`
      - `--no-prefer-jax-for-bootstrap`
    - threaded those flags into the 20B TPU config builder
- **Why this matters:**
  - `tpu_inference/models/common/model_loader.py` marks:
    - `GptOssForCausalLM`
    - as a `_VLLM_PREFERRED_ARCHITECTURES` model
  - but the current 20B experiment builder was forcing:
    - `model_impl_type="flax_nnx"`
    - and `prefer_jax_for_bootstrap=True`
  - so the existing probe never actually tested the vLLM wrapper path
- **Validation:**
  - `python3 -m py_compile experiments/gpt_oss_20b_tpu.py experiments/gpt_oss_20b_vllm_probe_matrix.py`
  - `./infra/pre-commit.py --fix experiments/gpt_oss_20b_tpu.py experiments/gpt_oss_20b_vllm_probe_matrix.py`
  - both passed
- **Exact next launch command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-vllm-probe-matrix-vllm-central1-20260326-1906 --cpu 4 --memory 16GB --disk 10GB --region us-central1 --zone us-central1-a -e MARIN_PREFIX gs://marin-us-central1 -- python experiments/gpt_oss_20b_vllm_probe_matrix.py --name debug_gpt_oss_20b_vllm_probe_matrix_vllm_central1_20260326_1906 --tpu-type v5p-8 --tensor-parallel-size 4 --temperature 0 --reasoning-effort low --model-impl-type vllm --no-prefer-jax-for-bootstrap`
- **Decision rule for this run:**
  - if `vllm` backend produces coherent text:
    - standardize GPT-OSS TPU serve on the vLLM wrapper path
    - stop debugging Harmony prompt format
  - if `vllm` backend is still incoherent:
    - the next branch is checkpoint/tokenizer packaging rather than backend routing

### 2026-03-26 19:16 - GTPU-004 result: the true TPU vLLM-wrapper path fixes the junk-token failure
- **Completed jobs in allowed region/zone:**
  - parent:
    - `/ahmed/gpt-oss-20b-vllm-probe-matrix-vllm-central1-20260326-1906`
  - TPU child:
    - `/ahmed/gpt-oss-20b-vllm-probe-matrix-vllm-central1-20260326-1906/align-debug_gpt_oss_20b_vllm_probe_matrix_vllm_central1_20260326_1906-probe_matrix_80188743-fc2d4353`
- **Artifact root:**
  - `gs://marin-us-central1/align/debug_gpt_oss_20b_vllm_probe_matrix_vllm_central1_20260326_1906/probe_matrix-5b9820`
- **Critical backend confirmation from native server logs:**
  - this run really did route through the requested backend:
    - `Loading model with MODEL_IMPL_TYPE=vllm`
    - `[phase] get_vllm_model: ...`
  - so the previous confusion was timing-related, not a silent fallback to `flax_nnx`
- **What changed semantically:**
  - the outputs are no longer gibberish token soup
  - tiny `/v1/completions` now returns valid Harmony analysis text, e.g.:
    - `<|channel|>analysis<|message|>The user says: "Reply with exactly HELLO." ...`
  - stage1 `/v1/chat/completions` with `max_tokens=256` now returns:
    - coherent `message.reasoning_content`
    - and partial `message.content` beginning with:
      - `<behavior_understanding>\nAsk clar`
- **What is still not working:**
  - the chat path is still `finish_reason="length"` on the tested budgets
  - tiny chat at `16` tokens only surfaces reasoning, not final content
  - stage1 chat at `256` tokens starts the final channel but does not finish it
- **Interpretation:**
  - the core corruption bug was in the old `flax_nnx` GPT-OSS path
  - the TPU vLLM-wrapper path is viable and should remain the primary branch
  - the current blocker is now Harmony budget/channel handling:
    - the model spends tokens in `analysis`
    - the `final` channel needs a larger budget to complete
  - this also means raw `/v1/completions` is the wrong contract for alignment:
    - it exposes Harmony channel text directly
    - `/v1/chat/completions` is the better path because it already separates `reasoning_content` and `content`

### 2026-03-26 19:17 - GTPU-005 prep: run a focused chat-budget sweep on the working `MODEL_IMPL_TYPE=vllm` path
- **Immediate goal:**
  - find the smallest `max_tokens` that yields a complete `final` answer on `/v1/chat/completions`
- **Code changes queued before relaunch:**
  - extend `experiments/gpt_oss_20b_vllm_probe_matrix.py` with:
    - `--probe-profile {default,chat_budget_sweep}`
    - explicit trace capture for `reasoning_content`
- **Focused sweep profile:**
  - tiny chat:
    - `16`
    - `32`
    - `64`
  - stage1 chat:
    - `256`
    - `512`
    - `768`
    - `1024`
  - no completions in this sweep
- **Decision rule for GTPU-005:**
  - if some chat budget yields a complete final answer:
    - promote GPT-OSS Stage 1 to `/v1/chat/completions` on the TPU vLLM-wrapper path
    - then adapt Marin prompt-generation parsing to consume `message.content` and optionally log `reasoning_content`
  - if even `1024` only yields partial final content:
    - the next branch is server-side stopping / Harmony termination rather than backend selection

### 2026-03-26 19:18 - GTPU-005 launch setup: make the sweep concrete and keep it in an allowed zone
- **Code changes actually made before launch:**
  - `experiments/gpt_oss_20b_vllm_probe_matrix.py`
    - added:
      - `--probe-profile {default,chat_budget_sweep}`
      - `reasoning_text` extraction from chat responses
      - `reasoning_text_preview` in per-case traces and summary
    - added a `chat_budget_sweep` profile containing:
      - `tiny_chat_low_16`
      - `tiny_chat_low_32`
      - `tiny_chat_low_64`
      - `stage1_chat_low_256`
      - `stage1_chat_low_512`
      - `stage1_chat_low_768`
      - `stage1_chat_low_1024`
- **Validation completed before launch:**
  - `python3 -m py_compile experiments/gpt_oss_20b_vllm_probe_matrix.py`
  - `./infra/pre-commit.py --fix experiments/gpt_oss_20b_vllm_probe_matrix.py`
  - both passed
- **Exact launch command used:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-vllm-chat-budget-sweep-central1-20260326-1918 --cpu 4 --memory 16GB --disk 10GB --region us-central1 --zone us-central1-a -e MARIN_PREFIX gs://marin-us-central1 -- python experiments/gpt_oss_20b_vllm_probe_matrix.py --name debug_gpt_oss_20b_vllm_chat_budget_sweep_central1_20260326_1918 --probe-profile chat_budget_sweep --tpu-type v5p-8 --tensor-parallel-size 4 --temperature 0 --reasoning-effort low --model-impl-type vllm --no-prefer-jax-for-bootstrap`
- **Why these exact settings were chosen:**
  - `us-central1-a` is an allowed zone for `v5p-8`
  - `MARIN_PREFIX=gs://marin-us-central1` keeps storage aligned with the TPU region
  - `model_impl_type=vllm` and `prefer_jax_for_bootstrap=false` deliberately stay on the path that fixed the corruption bug

### 2026-03-26 19:21 - Contract check while TPU was warming: the docs support the new hypothesis
- **Primary-source reasoning gathered during the run:**
  - vLLM's GPT-OSS guidance and Harmony docs point in the same direction:
    - GPT-OSS is a Harmony-format reasoning model
    - `/v1/chat/completions` and `/v1/responses` are the reasoning-aware surfaces
    - `/v1/completions` is the simple raw-completion surface and does not give the structured final/reasoning split that GPT-OSS wants
- **Operational interpretation:**
  - Marin's current local alignment client is structurally suspicious for GPT-OSS because it:
    - renders messages locally with `tokenizer.apply_chat_template(...)`
    - then calls `/v1/completions`
  - that behavior was probably acceptable for ordinary instruct models
  - for GPT-OSS it is likely the wrong contract even when the backend is healthy
- **Why this mattered before the sweep finished:**
  - it changed the question from:
    - "is the model still corrupt?"
  - to:
    - "does true chat inference work if we give the final channel enough budget?"

### 2026-03-26 19:25 - GTPU-005 result: `/v1/chat/completions` is correct, but `1024` is still not enough for Stage 1
- **Completed jobs in allowed region/zone:**
  - parent:
    - `/ahmed/gpt-oss-20b-vllm-chat-budget-sweep-central1-20260326-1918`
  - TPU child:
    - `/ahmed/gpt-oss-20b-vllm-chat-budget-sweep-central1-20260326-1918/align-debug_gpt_oss_20b_vllm_chat_budget_sweep_central1_20260326_1918-probe_matrix_e8fcdd5f-584a2060`
- **Artifact root:**
  - `gs://marin-us-central1/align/debug_gpt_oss_20b_vllm_chat_budget_sweep_central1_20260326_1918/probe_matrix-d48dff`
- **What the sweep established:**
  - tiny prompt:
    - `16`: reasoning only, no final content
    - `32`: reasoning only, no final content
    - `64`: success; `message.content = "HELLO"`, `finish_reason = "stop"`
  - Stage 1 prompt:
    - `256`: reasoning present, final content still null
    - `512`: coherent final content begins and includes completed `<behavior_understanding>` plus part of `<scientific_motivation>`, but truncates
    - `768`: still truncates
    - `1024`: still truncates, but by then the response contains:
      - full `<behavior_understanding>`
      - full `<scientific_motivation>`
      - most of `<variation_axes>`
      - truncation occurs in the fifth axis (`clarification_depth`)
- **Key raw evidence from `stage1_chat_low_1024`:**
  - `finish_reason = "length"`
  - `message.content` is fully coherent prose and valid partial XML/JSON structure, not junk
  - `message.reasoning_content` is also coherent and about `1128` characters long
  - the final content is about `3777` characters long before truncation
- **Token accounting that changed the next hypothesis:**
  - `response_json.usage.prompt_tokens = 677`
  - `response_json.usage.completion_tokens = 1024`
  - total used = `1701`
  - with `max_model_len = 4096`, a `1536` or `2048` completion budget should still fit comfortably
- **Important nuance discovered during inspection:**
  - the probe trace field `rendered_prompt_token_count` is `0` for chat cases
  - that is only a probe-harness limitation because chat requests are not locally re-rendered there
  - the server's `usage.prompt_tokens` is the authoritative number for chat requests
- **What this means:**
  - the serving path is now good enough to trust semantically
  - the remaining failure is not "bad decoding"
  - it is "not enough chat completion budget once Harmony spends tokens on reasoning"

### 2026-03-26 19:27 - Client-path implication: switching Marin to GPT-OSS chat is necessary, but not a trivial one-line swap
- **Current local alignment implementation facts:**
  - `lib/marin/src/marin/alignment/batched_vllm_serve.py`
    - `render_messages(...)` locally calls `tokenizer.apply_chat_template(...)`
    - `generate_from_prompt_texts(...)` posts batched prompt strings to `/v1/completions`
    - `generate_from_messages(...)` is just:
      - render messages
      - then call batched completions
- **Intermediate reasoning from reading that code after the sweep:**
  - for GPT-OSS, this path is likely wrong because it bypasses the reasoning-aware chat surface
  - but there is also a design constraint:
    - Marin's current local path relies on `/v1/completions` accepting multiple prompts in one request
    - `/v1/chat/completions` does not obviously provide the same multi-conversation batched request shape
  - so the migration is probably:
    - necessary
    - but not just "change the URL string"
- **Working conclusion at end of this pass:**
  - the immediate next experiment should be a tiny follow-up confirmation at:
    - `stage1_chat_low_1536`
    - `stage1_chat_low_2048`
  - if either one completes the XML cleanly, the next code task is:
    - add a GPT-OSS chat-completions path to `BatchedVllmServeSession`
    - keep `/v1/completions` for models that benefit from multi-prompt batching
    - thread GPT-OSS-specific chat settings explicitly, likely including:
      - `reasoning_effort = low`
  - I did **not** patch `batched_vllm_serve.py` in this pass because the batching implication needed to be written down first before changing shared alignment infrastructure

### 2026-03-26 23:13 - GTPU-006 prep: extend the focused chat sweep to `2048` before touching shared client code
- **Why this is the next move:**
  - the previous sweep already established that:
    - `1024` is coherent but truncates
    - prompt tokens are only `677`
    - `max_model_len` remains `4096`
  - so increasing request `max_tokens` is the cleanest way to test the current best hypothesis:
    - the remaining blocker is completion budget exhaustion, not bad decoding
- **Change made before launch:**
  - `experiments/gpt_oss_20b_vllm_probe_matrix.py`
    - added:
      - `stage1_chat_low_2048`
    - to the `chat_budget_sweep` profile
- **Planned launch shape:**
  - keep everything else fixed:
    - `MODEL_IMPL_TYPE=vllm`
    - `prefer_jax_for_bootstrap=false`
    - `reasoning_effort=low`
    - `tpu_type=v5p-8`
    - region/zone restricted to:
      - `us-central1-a`
- **Decision rule:**
  - if `2048` still truncates:
    - do not patch Marin client code yet
    - investigate chat termination or GPT-OSS reasoning/final budgeting more directly
  - if `2048` completes the XML cleanly:
    - proceed to implement a GPT-OSS-specific chat path in `batched_vllm_serve.py`

### 2026-03-26 23:27 - GTPU-006 result: `stage1_chat_low_2048` completes cleanly, so budget was the last inference blocker
- **Completed jobs in allowed region/zone:**
  - parent:
    - `/ahmed/gpt-oss-20b-vllm-chat-budget-sweep-2048-central1-20260326-2317`
  - TPU child:
    - `/ahmed/gpt-oss-20b-vllm-chat-budget-sweep-2048-central1-20260326-2317/align-debug_gpt_oss_20b_vllm_chat_budget_sweep_2048_central1_20260326_2317-probe_matrix_cfa943b6-2aaf10f9`
- **Artifact root:**
  - `gs://marin-us-central1/align/debug_gpt_oss_20b_vllm_chat_budget_sweep_2048_central1_20260326_2317/probe_matrix-1fe0bc`
- **High-signal result:**
  - `stage1_chat_low_2048` returned:
    - `finish_reason = "stop"`
    - coherent full `message.content`
    - coherent `message.reasoning_content`
    - a complete `<variation_axes>` block with a closing `</variation_axes>` tag
- **Concrete evidence from the finished artifacts:**
  - `artifacts/summary.json`
    - `stage1_chat_low_2048`:
      - `ok = true`
      - `finish_reason = "stop"`
      - `coherence_guess = true`
  - `artifacts/responses/stage1_chat_low_2048.json`
    - contains a complete response with:
      - `<behavior_understanding>`
      - `<scientific_motivation>`
      - `<variation_axes>`
      - five axes
      - clean closing tags
  - compact trace summary:
    - `primary_len = 4164`
    - `reasoning_len = 1128`
- **Token accounting that closes the budget question:**
  - `prompt_tokens = 677`
  - `completion_tokens = 1104`
  - `total_tokens = 1781`
  - this is important because it shows:
    - `2048` was sufficient
    - the model did **not** need the full `2048`
    - the prior `1024` failure was simply cutting off a response that needed slightly more than `1024`
- **Interpretation:**
  - the inference path is now proven for GPT-OSS 20B on:
    - `vllm serve`
    - `MODEL_IMPL_TYPE=vllm`
    - `/v1/chat/completions`
    - `reasoning_effort=low`
    - `max_model_len=4096`
    - `max_tokens=2048`
  - this means the remaining work is no longer about inference correctness
  - the next step should move to client integration:
    - add a GPT-OSS chat-completions path in `lib/marin/src/marin/alignment/batched_vllm_serve.py`
    - preserve `/v1/completions` for the existing batched-prompt path used by non-Harmony models
- **What I would do next, based on this result:**
  - stop running budget sweeps
  - patch the local alignment client so GPT-OSS uses the winning contract explicitly
  - then rerun a one-statement prompt-generation job on 20B before promoting anything to 120B

### 2026-03-26 23:45 - GTPU-007 implementation: hard-gate GPT-OSS onto the validated chat-completions contract inside `batched_vllm_serve.py`
- **Code changed:**
  - `lib/marin/src/marin/alignment/batched_vllm_serve.py`
  - `tests/test_alignment.py`
- **Implementation summary:**
  - imported `_looks_like_gpt_oss_model` from `marin.inference.vllm_server`
  - added an explicit GPT-OSS contract warning at the top of `batched_vllm_serve.py`
  - added:
    - `GPT_OSS_CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "low"}`
    - `GPT_OSS_REQUIRED_FINISH_REASON = "stop"`
  - added GPT-OSS detection helpers so the session can recognize GPT-OSS from:
    - `config.model`
    - `config.tokenizer`
    - `config.hf_overrides`
    - the server-reported model id
  - changed session behavior so GPT-OSS now:
    - **refuses** `render_messages(...)`
    - **refuses** `generate_from_prompt_texts(...)`
    - **routes only** through `generate_from_messages(...) -> /v1/chat/completions`
- **Exact guardrail reasoning written down here so future edits do not undo it casually:**
  - the old local path was:
    - local `apply_chat_template(...)`
    - then `/v1/completions`
  - that path is now intentionally disabled for GPT-OSS because the TPU bring-up work already showed:
    - bad backend path:
      - gibberish token soup
    - good backend path:
      - coherent chat responses only when using the chat surface
  - the risk is not just "slightly worse quality"
  - the risk is:
    - silent contract drift back to a path that already failed empirically
  - that is why the new code raises `ValueError` loudly instead of trying to be permissive
- **New GPT-OSS request path in the shared client:**
  - one conversation per HTTP request to:
    - `/v1/chat/completions`
  - request payload now includes:
    - `model`
    - `messages`
    - `temperature`
    - `max_tokens`
    - `n`
    - `chat_template_kwargs={"reasoning_effort":"low"}`
  - metrics now use server-reported:
    - `usage.prompt_tokens`
    - `usage.completion_tokens`
- **Strict response validation added for GPT-OSS:**
  - fail if `choices` is missing
  - fail if any choice is not a dict
  - fail if `finish_reason != "stop"`
  - fail if `message.content` is missing or empty
  - intermediate reasoning for keeping this strict:
    - earlier GPT-OSS failures looked superficially like parser bugs
    - in reality they were serving-contract bugs
    - so this client should reject partial or reasoning-only payloads immediately instead of letting later XML parsing hide the true failure mode
- **Tests added in `tests/test_alignment.py`:**
  - GPT-OSS rejects `render_messages(...)`
  - GPT-OSS rejects `generate_from_prompt_texts(...)`
  - GPT-OSS uses `/v1/chat/completions` with:
    - one request per conversation
    - fixed `chat_template_kwargs={"reasoning_effort":"low"}`
  - GPT-OSS fails loudly if the chat response comes back with:
    - `finish_reason = "length"`
- **Validation run after patch:**
  - `python3 -m py_compile lib/marin/src/marin/alignment/batched_vllm_serve.py tests/test_alignment.py`
    - passed
  - `uv run pytest tests/test_alignment.py -q`
    - `105 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/batched_vllm_serve.py tests/test_alignment.py`
    - passed
- **Current conclusion after implementation:**
  - the shared local alignment client now encodes the winning GPT-OSS TPU serving contract directly
  - the next experiment should stop being a probe and become a product-path check:
    - rerun a one-statement GPT-OSS 20B prompt-generation job through the shared client
  - only after that succeeds should this contract be promoted back to the 120B path

### 2026-03-26 23:49 - GTPU-007 sync note: logbook updated before any product-path rerun
- **Reason for this append:**
  - keep the thread append-only and make it explicit that the code patch and validation are recorded before the next TPU launch
- **State at this exact handoff point:**
  - shared client patch:
    - complete
  - unit/integration-style local validation:
    - complete
  - one-statement GPT-OSS 20B prompt-generation rerun through the shared path:
    - not launched yet
- **Immediate next action from here:**
  - launch a one-statement GPT-OSS 20B prompt-generation job in:
    - `us-central1-a`
    - or `us-east5-a`
  - and verify the shared client now produces a valid Stage 1 artifact through the real alignment path

### 2026-03-26 23:52 - GTPU-008 prep: east5 one-statement race is blocked on missing east5 model materialization, so download that first
- **New user direction:**
  - instead of launching the one-statement race immediately, first materialize the 20B vLLM checkpoint in:
    - `us-east5-a`
- **Why this is necessary:**
  - current bucket state:
    - `gs://marin-us-central1/models/unsloth--gpt-oss-20b-BF16-vllm--cc89b3e7fd423253264883a80a4fa5abc619649f/`
      - exists
    - `gs://marin-us-east5/models/...gpt-oss-20b...`
      - does not exist yet
  - earlier Iris validation already showed that east5 TPU DAGs reject a central1-only model artifact because the GCS region and TPU-capable DAG region must overlap
- **Operational consequence:**
  - an `us-east5-a` prompt-generation launch is not real until the 20B vLLM-serving checkpoint subset exists under:
    - `gs://marin-us-east5/models/...`
- **Planned command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name download-gpt-oss-20b-vllm-east5-20260326-2352 --cpu 4 --memory 16GB --disk 10GB --region us-east5 --zone us-east5-a -- python experiments/download_gpt_oss_20b_vllm.py --prefix gs://marin-us-east5`
- **Immediate next step after submit:**
  - confirm the parent job reaches running/building cleanly
  - then inspect the Zephyr child stages to ensure the east5 download actually starts writing shards

### 2026-03-26 23:53 - GTPU-008 launch result: east5 20B vLLM download job submitted successfully and is waiting on east5 worker readiness
- **Submitted command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name download-gpt-oss-20b-vllm-east5-20260326-2352 --cpu 4 --memory 16GB --disk 10GB --region us-east5 --zone us-east5-a -- python experiments/download_gpt_oss_20b_vllm.py --prefix gs://marin-us-east5`
- **Returned parent job id:**
  - `/ahmed/download-gpt-oss-20b-vllm-east5-20260326-2352`
- **Early controller state after submit:**
  - `JOB_STATE_PENDING`
  - no child logs yet
  - no DAG-validation or region-overlap failure
- **Current pending reason:**
  - `Scheduler: No worker matches constraints and has sufficient resources (need cpu=4 cores, memory=17179869184, constraints=['region', 'zone', 'reservation-job'])`
  - `Autoscaler: Waiting for workers in scale group 'tpu_v5p_16-us-east5-a' to become ready (selected: demand-routed)`
- **Interpretation:**
  - the command itself is valid
  - east5 is the current bottleneck, not repo code or bucket wiring
  - this is a real wait-for-capacity state, so the next useful check is whether the parent transitions into active Zephyr download work and starts writing shards under:
    - `gs://marin-us-east5/models/unsloth--gpt-oss-20b-BF16-vllm--cc89b3e7fd423253264883a80a4fa5abc619649f/`

### 2026-03-26 23:57 - GTPU-008 progress: east5 download acquired workers and is actively streaming shards into `marin-us-east5`
- **Current live state:**
  - parent:
    - `/ahmed/download-gpt-oss-20b-vllm-east5-20260326-2352`
    - `JOB_STATE_RUNNING`
  - Zephyr coordinator child:
    - `/ahmed/download-gpt-oss-20b-vllm-east5-20260326-2352/zephyr-download-hf-efb41bf2-p0-a0`
    - `JOB_STATE_RUNNING`
- **Experiment metadata path:**
  - `gs://marin-us-east5/experiments/download_gpt_oss_20b_vllm-5f9b62.json`
- **What the logs now prove:**
  - parent executor reached the download step
  - Hugging Face file enumeration succeeded:
    - `16` files
    - `38.98 GB`
  - Zephyr coordinator started with:
    - `4` workers
    - `16` shards
  - stage is now:
    - `stage0-Map → Write`
- **First concrete write evidence in east5 bucket:**
  - `chat_template.jinja`
    - streamed successfully to `gs://marin-us-east5/models/unsloth--gpt-oss-20b-BF16-vllm--cc89b3e7fd423253264883a80a4fa5abc619649f/chat_template.jinja`
  - `config.json`
    - streamed successfully
  - `generation_config.json`
    - streamed successfully
- **Live progress snapshot from logs:**
  - `3/16 complete`
  - `4 in-flight`
  - `9 queued`
  - `4/4 workers alive`
- **Interpretation:**
  - the east5 prerequisite is now genuinely underway
  - this is no longer blocked on scheduler capacity
  - once this completes, an `us-east5-a` one-statement GPT-OSS 20B prompt-generation launch becomes region-valid

### 2026-03-27 00:02 - GTPU-009 prep: launch the real one-statement GPT-OSS 20B prompt-generation job in `us-central1-a` while east5 keeps downloading
- **Why this launch can happen now:**
  - the central1 20B vLLM subset already exists under:
    - `gs://marin-us-central1/models/unsloth--gpt-oss-20b-BF16-vllm--cc89b3e7fd423253264883a80a4fa5abc619649f/`
  - east5 download is still useful, but it is no longer on the critical path for the first product-path validation
- **Code prep completed before submit:**
  - added:
    - `experiments/generate_prompts_gpt_oss_20b_refactored.py`
  - this mirrors the 120B refactored prompt-generation entrypoint but pins the validated 20B inference settings:
    - `model_impl_type = vllm`
    - `prefer_jax_for_bootstrap = false`
    - `max_tokens = 2048`
    - shared local alignment client path through GPT-OSS chat completions
  - quick validation:
    - `python3 -m py_compile experiments/generate_prompts_gpt_oss_20b_refactored.py`
      - passed
- **Planned launch command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-promptgen-one-statement-central1-20260327-0002 --cpu 4 --memory 16GB --disk 10GB --region us-central1 --zone us-central1-a -e MARIN_PREFIX gs://marin-us-central1 -- python experiments/generate_prompts_gpt_oss_20b_refactored.py --name debug_generate_prompts_gpt_oss_20b_one_statement_central1_20260327_0002 --statement-id ask_clarifying_questions --tensor-parallel-size 4`
- **Immediate monitoring goal after submit:**
  - confirm the prompt-generation parent and TPU child reach running state
  - then watch whether Stage 1 now emits a valid understanding artifact through the shared GPT-OSS chat path

### 2026-03-27 00:03 - GTPU-009 launch result: central1 one-statement prompt-generation job submitted and entered execution immediately
- **Submitted command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-promptgen-one-statement-central1-20260327-0002 --cpu 4 --memory 16GB --disk 10GB --region us-central1 --zone us-central1-a -e MARIN_PREFIX gs://marin-us-central1 -- python experiments/generate_prompts_gpt_oss_20b_refactored.py --name debug_generate_prompts_gpt_oss_20b_one_statement_central1_20260327_0002 --statement-id ask_clarifying_questions --tensor-parallel-size 4`
- **Returned parent job id:**
  - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002`
- **Immediate controller state after submit:**
  - parent:
    - `JOB_STATE_RUNNING`
  - child:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002/align-debug_generate_prompts_gpt_oss_20b_one_statement_central1_20260327_0002-spec_9082c112-20bdba8c`
    - `JOB_STATE_RUNNING`
    - `task_state_counts.building = 1`
- **First log signal:**
  - the parent worker is already past environment setup and running the user command
  - this launch did **not** stall in scheduler capacity wait
- **Concurrent state worth preserving in the same entry:**
  - east5 model download remains healthy and fully running with:
    - parent running
    - Zephyr coordinator running
    - Zephyr worker pool running with `4` tasks
- **Immediate next check:**
  - wait through the startup interval and inspect whether the central1 run advances from `spec` into the TPU `prompts` child

### 2026-03-27 00:05 - GTPU-009 status: `spec` succeeded; TPU `prompts` child is now queued on `v5p-8` capacity in `us-central1-a`
- **Current central1 job tree:**
  - parent:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002`
    - `JOB_STATE_RUNNING`
  - `spec` child:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002/align-debug_generate_prompts_gpt_oss_20b_one_statement_central1_20260327_0002-spec_9082c112-20bdba8c`
    - `JOB_STATE_SUCCEEDED`
  - `prompts` child:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002/align-debug_generate_prompts_gpt_oss_20b_one_statement_central1_20260327_0002-prompts_9227c740-f81de044`
    - `JOB_STATE_PENDING`
- **Concrete scheduler signal for the TPU child:**
  - `Scheduler: Insufficient TPUs (need 4, available 0) - 9 worker(s)`
  - `Autoscaler: Waiting for workers in scale group 'tpu_v5p_8-us-central1-a' to become ready (selected: demand-routed)`
- **Interpretation:**
  - the product-path job itself is valid
  - the only blocker at this moment is central1 TPU capacity
  - this is still the right launch to keep alive while east5 continues downloading

### 2026-03-27 00:12 - GTPU-010 status refresh: east5 finished successfully, and central1 has now acquired the TPU
- **Correction to the user-visible intuition that east5 was "down":**
  - the east5 download is actually complete
  - parent:
    - `/ahmed/download-gpt-oss-20b-vllm-east5-20260326-2352`
    - `JOB_STATE_SUCCEEDED`
  - Zephyr coordinator:
    - `JOB_STATE_SUCCEEDED`
  - Zephyr worker pool:
    - `JOB_STATE_KILLED`
    - error text:
      - `Terminated by user`
  - interpretation:
    - this appears to be normal worker-pool teardown after coordinator completion, not a failed download
- **More important live result:**
  - the central1 prompt-generation TPU child has now moved from pending to running
  - parent:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002`
    - `JOB_STATE_RUNNING`
  - prompts child:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002/align-debug_generate_prompts_gpt_oss_20b_one_statement_central1_20260327_0002-prompts_9227c740-f81de044`
    - `JOB_STATE_RUNNING`
    - `pending_reason = ""`
- **Interpretation:**
  - central1 is now the primary thread
  - the next useful evidence is no longer scheduler state
  - it is:
    - `vllm` startup
    - Stage 1 understanding progress
    - or the first concrete runtime failure, if any

### 2026-03-27 00:14 - GTPU-011 milestone: the real central1 product-path run reached `vLLM environment ready`, completed Stage 1, and entered Stage 2
- **High-signal log lines:**
  - `marin.inference.vllm_server vLLM environment ready`
  - `marin.alignment.generate_prompts Stage 1: Generating understanding for 1 statements`
  - `marin.alignment.batched_vllm_serve Sending GPT-OSS vLLM serve requests to /v1/chat/completions one conversation at a time`
  - `marin.alignment.generate_prompts Stage 1 progress: 1/1 (100.0%) [attempt 1]`
  - `marin.alignment.generate_prompts Saved artifacts for 1 statements ...`
  - `marin.alignment.generate_prompts Stage 2: Concretizing 1 statements`
- **Interpretation:**
  - the shared-client GPT-OSS chat-path patch is now validated on the real prompt-generation product path, not just probe jobs
  - Stage 1 is no longer blocked on:
    - gibberish outputs
    - wrong endpoint contract
    - `max_context_length=0`
  - the next remaining question is higher in the pipeline:
    - does Stage 2/3 finish cleanly
    - or is there another GPT-OSS-specific failure later in prompt generation
- **Current live position of the run:**
  - parent:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002`
  - prompts child:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002/align-debug_generate_prompts_gpt_oss_20b_one_statement_central1_20260327_0002-prompts_9227c740-f81de044`
  - output root:
    - `gs://marin-us-central1/align/debug_generate_prompts_gpt_oss_20b_one_statement_central1_20260327_0002/prompts-4e9a94`

### 2026-03-27 00:19 - GTPU-012 failure analysis and retry plan: Stage 2 exhausted the chat budget before emitting final assistant content
- **Final result of the first central1 run:**
  - parent:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-20260327-0002`
    - `JOB_STATE_FAILED`
  - prompts child:
    - `JOB_STATE_FAILED`
- **Important nuance from the failure:**
  - this is **not** a Stage 1 regression
  - Stage 1 succeeded on attempt 1
  - the failure happened in Stage 2 concretization
- **Exact failure signature:**
  - `ValueError: GPT-OSS chat response did not finish cleanly. Expected finish_reason='stop', got 'length'.`
  - response excerpt shows:
    - `message.content = null`
    - a long `message.reasoning` trace
  - interpretation:
    - GPT-OSS spent the full completion budget in reasoning and never reached the final assistant channel for the scenario/rubric payload
- **Code change made immediately after this failure:**
  - `experiments/generate_prompts_gpt_oss_20b_refactored.py`
    - added explicit CLI knobs for:
      - `--max-model-len`
      - `--understanding-max-tokens`
      - `--concretize-max-tokens`
      - `--extract-max-tokens`
    - set the new defaults for the retry direction to:
      - `max_model_len = 8192`
      - `understanding_max_tokens = 2048`
      - `concretize_max_tokens = 4096`
      - `extract_max_tokens = 4096`
- **Why this retry direction is justified:**
  - the first run already proved the product-path contract itself is correct:
    - Stage 1 reached `vLLM environment ready`
    - Stage 1 completed successfully
  - the new blocker is simply that later stages need more room for GPT-OSS reasoning plus final output
- **Planned retry command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-promptgen-one-statement-central1-lenfix-20260327-0019 --cpu 4 --memory 16GB --disk 10GB --region us-central1 --zone us-central1-a -e MARIN_PREFIX gs://marin-us-central1 -- python experiments/generate_prompts_gpt_oss_20b_refactored.py --name debug_generate_prompts_gpt_oss_20b_one_statement_central1_lenfix_20260327_0019 --statement-id ask_clarifying_questions --tensor-parallel-size 4 --max-model-len 8192 --understanding-max-tokens 2048 --concretize-max-tokens 4096 --extract-max-tokens 4096`

### 2026-03-27 00:23 - GTPU-013 launch result: the central1 `lenfix` retry is running and past the cheap setup stages
- **Submitted retry job:**
  - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-lenfix-20260327-0019`
- **Experiment metadata path:**
  - `gs://marin-us-central1/experiments/generate_prompts_gpt_oss_20b_refactored-b38e1d.json`
- **Current job tree snapshot:**
  - parent:
    - `JOB_STATE_RUNNING`
  - `spec` child:
    - `JOB_STATE_SUCCEEDED`
  - `prompts` child:
    - `JOB_STATE_RUNNING`
- **Live runtime signal at the moment of this append:**
  - `prompts` child has already:
    - loaded the spec
    - started the native TPU `vllm` server
    - begun Runai streamer weight loading
  - latest visible streamer progress:
    - `372/411` safetensor chunks
    - `91% Completed`
- **Interpretation:**
  - the retry is healthy so far
  - the next checkpoint to watch is:
    - `vLLM environment ready`
    - then Stage 1/2 progression under the larger `8192/4096` budget

### 2026-03-27 00:28 - GTPU-014 progress: the `lenfix` retry reached `vLLM environment ready`, finished Stage 1 again, and re-entered Stage 2 under the larger budget
- **High-signal log lines:**
  - `marin.inference.vllm_server vLLM environment ready`
  - `Stage 1 progress: 1/1 (100.0%) [attempt 1]`
  - `Saved artifacts for 1 statements ...`
  - `Stage 2: Concretizing 1 statements`
  - `Stage 2 local work queue: 67 concretize items across 1 statements`
- **Why this matters:**
  - the retry did not regress startup or Stage 1
  - the only remaining question is the original one:
    - does Stage 2 now complete with `finish_reason='stop'`
    - or does GPT-OSS still exhaust the larger `4096` concretization budget
- **Current live output root for the retry:**
  - `gs://marin-us-central1/align/debug_generate_prompts_gpt_oss_20b_one_statement_central1_lenfix_20260327_0019/prompts-8140d7`

### 2026-03-27 00:30 - GTPU-015 terminal result: the larger `8192/4096` budget did not fix Stage 2; concretization still dies on `finish_reason='length'`
- **Final result of the retry:**
  - parent:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-lenfix-20260327-0019`
    - `JOB_STATE_FAILED`
  - prompts child:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-lenfix-20260327_0019-prompts_fad08a5c-351409cc`
    - `JOB_STATE_FAILED`
- **What the retry proved:**
  - Stage 1 still succeeds under the larger context/budget
  - startup still succeeds
  - the failure remains specifically in Stage 2 concretization
  - raising:
    - `max_model_len` from `4096` to `8192`
    - `concretize_max_tokens` from `2048` to `4096`
    - `extract_max_tokens` from `2048` to `4096`
    - was not sufficient
- **Exact repeated failure signature:**
  - `ValueError: GPT-OSS chat response did not finish cleanly. Expected finish_reason='stop', got 'length'.`
  - response excerpt again shows:
    - `message.content = null`
    - long `message.reasoning`
  - interpretation:
    - GPT-OSS is still spending its generation budget in reasoning and not reaching the final assistant answer for this concretization prompt
- **Why I am not auto-retrying a third time here:**
  - this is now a repeated, stable failure mode rather than a flaky launch or transient TPU issue
  - the next move should be a design/debug decision, not another blind rerun
- **Most likely next branches from this evidence:**
  - lower GPT-OSS reasoning effort specifically for Stage 2
  - add GPT-OSS-specific request controls for later stages rather than one fixed global contract
  - or switch only Stage 2 concretization away from GPT-OSS if the reasoning budget remains pathological

### 2026-03-27 00:34 - GTPU-016 reasoning update: we *can* increase Stage 2 length further, but the evidence now says length alone is probably not the real fix
- **Direct answer to the current question:**
  - yes, it is technically possible to increase length again
  - the next brute-force point would be something like:
    - `max_model_len = 16384`
    - `concretize_max_tokens = 8192`
- **Why I am not treating that as the default next step:**
  - the first central1 run failed with:
    - `concretize_max_tokens = 2048`
    - `message.content = null`
    - long `message.reasoning`
    - `finish_reason = "length"`
  - the retry failed with the same exact shape after raising to:
    - `max_model_len = 8192`
    - `concretize_max_tokens = 4096`
    - `extract_max_tokens = 4096`
  - that is not the pattern of a near-miss where the model clearly starts emitting the final answer and simply needs a bit more room
  - instead, the model is still spending the budget in the reasoning channel and never reaching the final assistant content that Stage 2 actually needs
- **Important consequence for debugging:**
  - simply loosening the current GPT-OSS chat-path guard would not help
  - the problem is not just:
    - `finish_reason != "stop"`
  - it is also:
    - `message.content = null`
  - so there is nothing parseable for the concretization stage to consume even if the transport guard were relaxed
- **Updated best diagnosis:**
  - the Stage 1 contract is now proven
  - the remaining failure is Stage-2-specific behavior:
    - either the concretization prompt is encouraging too much reasoning before final output
    - or GPT-OSS needs stage-specific request controls rather than the single global `reasoning_effort="low"` contract currently hardcoded in the shared client
- **Highest-signal next steps from here:**
  - persist the full raw Stage 2 chat payloads on `length` failures for inspection
  - add stage-specific GPT-OSS controls for concretization rather than one global setting
  - only after that, if needed, run one explicit brute-force test at `16384/8192`

### 2026-03-27 01:01 - GTPU-017 new user direction and concrete escalation policy: keep raising GPT-OSS 20B context and generation budgets until the path either succeeds or becomes physically impossible
- **New explicit instruction from the user:**
  - stop treating higher budgets as "not worthwhile"
  - keep increasing:
    - `max_model_len`
    - Stage 2 / Stage 3 generation budgets
  - continue until one of these becomes true:
    - the one-statement run succeeds
    - the run fails with a real memory/resource ceiling
    - or another hard infrastructure cap makes a larger setting literally impossible
- **Current pre-launch state before the next run:**
  - there are no live `gpt-oss-20b` prompt-generation jobs left running
  - the two prior central1 runs are both terminal failures
  - the east5 20B download is complete, so either:
    - `us-central1-a`
    - or `us-east5-a`
    - is now region-valid for future retries
- **How I am operationalizing the new instruction:**
  - I will run a monotonic escalation ladder rather than stopping at the first repeated `length` failure
  - first brute-force rung:
    - `max_model_len = 16384`
    - `understanding_max_tokens = 2048`
    - `concretize_max_tokens = 8192`
    - `extract_max_tokens = 8192`
  - if that still fails without OOM:
    - raise again rather than declaring the approach exhausted
  - likely next rung after that:
    - `max_model_len = 32768`
    - `concretize_max_tokens = 16384`
    - `extract_max_tokens = 16384`
- **Why this ladder is internally consistent:**
  - the shared client already enforces:
    - `prompt_tokens + max_tokens <= max_model_len`
  - so higher `max_tokens` requires a correspondingly larger `max_model_len`
  - the current builder does not impose an additional code-side hard cap beyond what is passed into the config
- **Exact next launch to run immediately after this append:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-promptgen-one-statement-central1-ctx16k-20260327-0101 --cpu 4 --memory 16GB --disk 10GB --region us-central1 --zone us-central1-a -e MARIN_PREFIX gs://marin-us-central1 -- python experiments/generate_prompts_gpt_oss_20b_refactored.py --name debug_generate_prompts_gpt_oss_20b_one_statement_central1_ctx16k_20260327_0101 --statement-id ask_clarifying_questions --tensor-parallel-size 4 --max-model-len 16384 --understanding-max-tokens 2048 --concretize-max-tokens 8192 --extract-max-tokens 8192`

### 2026-03-27 01:02 - GTPU-018 launch result and capacity branch: the central1 `16k/8k` run submitted cleanly, but the TPU child is waiting in autoscaler backoff rather than failing
- **Submitted parent job:**
  - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-ctx16k-20260327-0101`
- **Current job tree snapshot right after startup stabilization:**
  - parent:
    - `JOB_STATE_RUNNING`
  - `spec` child:
    - `JOB_STATE_SUCCEEDED`
  - `prompts` child:
    - `JOB_STATE_PENDING`
- **Current TPU child pending reason:**
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - `Autoscaler: Unsatisfied autoscaler demand: no_capacity: tpu_v5p_8-us-central1-a=backoff`
- **Important interpretation:**
  - this is not yet evidence about:
    - OOM
    - `max_model_len=16384`
    - or Stage 2 behavior
  - it is only evidence about temporary central1 TPU supply
- **Operational branch chosen from this state:**
  - do not discard the central1 run yet
  - immediately launch the same `16k/8k` rung in:
    - `us-east5-a`
  - keep both alive until one of them actually acquires TPU capacity first
  - once one region gets the TPU and starts real model work:
    - stop the other queued duplicate

### 2026-03-27 01:03 - GTPU-019 east5 mirror launch: the same `16k/8k` rung is now submitted in `us-east5-a` so capacity no longer depends on one zone
- **Submitted parent job:**
  - `/ahmed/gpt-oss-20b-promptgen-one-statement-east5-ctx16k-20260327-0103`
- **Exact mirror command used:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-promptgen-one-statement-east5-ctx16k-20260327-0103 --cpu 4 --memory 16GB --disk 10GB --region us-east5 --zone us-east5-a -e MARIN_PREFIX gs://marin-us-east5 -- python experiments/generate_prompts_gpt_oss_20b_refactored.py --name debug_generate_prompts_gpt_oss_20b_one_statement_east5_ctx16k_20260327_0103 --statement-id ask_clarifying_questions --tensor-parallel-size 4 --max-model-len 16384 --understanding-max-tokens 2048 --concretize-max-tokens 8192 --extract-max-tokens 8192`
- **Why this is safe and useful:**
  - the 20B vLLM-serving checkpoint subset was already materialized in:
    - `gs://marin-us-east5/models/...`
  - so this adds:
    - TPU capacity diversity
  - without introducing:
    - new cross-region artifact movement
- **Immediate monitoring rule now in force:**
  - keep both the central1 and east5 `16k/8k` runs alive only until one of them enters real TPU execution
  - the moment one region reaches:
    - active `prompts` execution
    - or `vLLM environment ready`
  - stop the duplicate in the other region to avoid paying for two full runs of the same rung

### 2026-03-27 01:10 - GTPU-020 current live state: both `16k/8k` runs are now structurally valid and sitting at the same TPU-capacity gate
- **Central1 snapshot:**
  - parent:
    - `JOB_STATE_RUNNING`
  - `spec` child:
    - `JOB_STATE_SUCCEEDED`
  - `prompts` child:
    - `JOB_STATE_PENDING`
  - current pending reason:
    - `Autoscaler: Unsatisfied autoscaler demand: no_capacity: tpu_v5p_8-us-central1-a=backoff`
- **East5 snapshot:**
  - parent:
    - `JOB_STATE_RUNNING`
  - `spec` child:
    - `JOB_STATE_SUCCEEDED`
  - `prompts` child:
    - `JOB_STATE_PENDING`
  - current pending reason:
    - `Autoscaler: Unsatisfied autoscaler demand: no_capacity: tpu_v5p_8-us-east5-a=backoff`
- **What this does and does not prove:**
  - it proves the `16k/8k` rung is accepted by:
    - the experiment code
    - Iris
    - the alignment step graph
    - both region-valid model artifact paths
  - it does **not** yet prove anything about:
    - OOM
    - startup success at `max_model_len=16384`
    - or whether Stage 2 still ends on `finish_reason='length'`
- **Operational consequence:**
  - do not launch an even larger rung yet
  - keep monitoring until one `prompts` child actually acquires TPU capacity
  - only then can this brute-force escalation produce a real result:
    - success
    - length exhaustion
    - or OOM

### 2026-03-27 01:19 - GTPU-021 monitoring ownership note: a dedicated 8-hour babysitter is now the standing instruction for this thread
- **Explicit user instruction being carried forward:**
  - keep babysitting these jobs for `8` hours
  - keep reminding future agents that this monitoring obligation remains active
- **Exact monitoring window:**
  - started:
    - `2026-03-27 01:19:28 PDT`
  - stop no earlier than:
    - `2026-03-27 09:19:28 PDT`
    - `2026-03-27 16:19:28 UTC`
- **Persistent monitor artifacts created for this obligation:**
  - script:
    - `scratch/gpt_oss_ctx16k_race_monitor.py`
  - state file:
    - `scratch/20260327-0119_gpt_oss_ctx16k_race_state.json`
  - event log:
    - `scratch/20260327-0119_gpt_oss_ctx16k_race_events.jsonl`
  - pid file:
    - `scratch/20260327-0119_gpt_oss_ctx16k_race_monitor.pid`
- **What the babysitter is responsible for:**
  - poll both `16k/8k` parents:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-ctx16k-20260327-0101`
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-east5-ctx16k-20260327-0103`
  - persist snapshots and relevant log lines during the 8-hour window
  - if one region actually acquires TPU execution first:
    - stop the duplicate loser parent job
- **Instruction to future agents if context switches before the deadline:**
  - do **not** quietly abandon monitoring
  - first inspect the state file and event log above
  - continue honoring the monitoring window through:
    - `2026-03-27 09:19:28 PDT`
  - only after that deadline, or after explicit user redirection, should the thread stop babysitting the ctx16k race

### 2026-03-27 01:24 - GTPU-022 first babysit outcome: `east5` won the ctx16k race, `central1` was stopped, and the persistent monitor is now running under `tmux`
- **What the first successful monitor poll observed:**
  - `east` was the first region whose `prompts` child entered real TPU work:
    - `JOB_STATE_RUNNING`
    - `started_at_ms = 1774599359584`
  - relevant live signal already captured in the state file:
    - `marin.inference.vllm_server Starting vLLM environment`
    - `marin.inference.vllm_server Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
- **Duplicate-cost action taken by the babysitter:**
  - stopped:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-central1-ctx16k-20260327-0101`
  - this matches the earlier explicit race rule:
    - once one region acquires TPU execution, kill the duplicate loser
- **Current live monitor owner:**
  - detached `tmux` session:
    - `gpt_oss_ctx16k_race_20260327`
  - current monitor pid file points at:
    - `scratch/20260327-0119_gpt_oss_ctx16k_race_monitor.pid`
- **Important standing instruction for future agents after this point:**
  - the thread is no longer watching a two-region race
  - it is now watching the single surviving east5 parent:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-east5-ctx16k-20260327-0103`
  - keep honoring the 8-hour monitoring window through:
    - `2026-03-27 09:19:28 PDT`
  - inspect:
    - `scratch/20260327-0119_gpt_oss_ctx16k_race_state.json`
    - `scratch/20260327-0119_gpt_oss_ctx16k_race_events.jsonl`
    - and the `tmux` session above
  - before making any new escalation decision

### 2026-03-27 01:33 - GTPU-023 correction after user interruption: no background monitor should remain; both ctx16k runs are now killed, and the east5 run had already reached Stage 2 before termination
- **New explicit user correction:**
  - do **not** rely on background monitoring
  - kill any detached/background babysitter
  - continue foreground monitoring in the active turn only
- **Actions taken immediately:**
  - killed the detached `tmux` babysitter session
  - verified no `gpt_oss_ctx16k_race_monitor.py` process remains
- **Current job state after that correction:**
  - central1 parent:
    - `JOB_STATE_KILLED`
  - east5 parent:
    - `JOB_STATE_KILLED`
  - both show:
    - `Terminated by user`
- **Critical recovery detail from the east5 logs before it was killed:**
  - this was **not** just a startup-only run
  - east5 had already reached:
    - `vLLM environment ready`
    - `Stage 1: Generating understanding for 1 statements`
    - `Stage 1 progress: 1/1 (100.0%) [attempt 1]`
    - `Saved artifacts for 1 statements ...`
    - `Stage 2: Concretizing 1 statements`
    - `Stage 2 local work queue: 67 concretize items across 1 statements`
- **Interpretation:**
  - the `16k/8k` rung is now strongly justified as the next foreground rerun
  - we still do **not** know whether it ends in:
    - success
    - `finish_reason='length'`
    - or OOM
  - because the promising east5 run was interrupted mid-Stage-2 before reaching a natural terminal outcome
- **Immediate next step from this exact point:**
  - relaunch the same east5 `16k/8k` rung manually
  - babysit it directly in the active turn
  - and only escalate to a larger rung after this rerun yields a real terminal result

### 2026-03-27 01:40 - GTPU-024 foreground-only east5 rerun: the `16k/8k` rung is live again and has already reproduced the good path through Stage 2
- **Relaunched parent job:**
  - `/ahmed/gpt-oss-20b-promptgen-one-statement-east5-ctx16k-rerun-20260327-0134`
- **Exact rerun command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name gpt-oss-20b-promptgen-one-statement-east5-ctx16k-rerun-20260327-0134 --cpu 4 --memory 16GB --disk 10GB --region us-east5 --zone us-east5-a -e MARIN_PREFIX gs://marin-us-east5 -- python experiments/generate_prompts_gpt_oss_20b_refactored.py --name debug_generate_prompts_gpt_oss_20b_one_statement_east5_ctx16k_rerun_20260327_0134 --statement-id ask_clarifying_questions --tensor-parallel-size 4 --max-model-len 16384 --understanding-max-tokens 2048 --concretize-max-tokens 8192 --extract-max-tokens 8192`
- **Current job tree state:**
  - parent:
    - `JOB_STATE_RUNNING`
  - `spec` child:
    - `JOB_STATE_SUCCEEDED`
  - `prompts` child:
    - `JOB_STATE_RUNNING`
- **High-signal logs already reproduced on the rerun:**
  - `marin.inference.vllm_server Starting vLLM environment`
  - `marin.inference.vllm_server Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
  - `marin.inference.vllm_server vLLM environment ready`
  - `marin.alignment.generate_prompts Stage 1: Generating understanding for 1 statements`
  - `Stage 1 progress: 1/1 (100.0%) [attempt 1]`
  - `Saved artifacts for 1 statements ...`
  - `Stage 2: Concretizing 1 statements`
  - `Stage 2 local work queue: 67 concretize items across 1 statements`
- **What this rerun proves so far:**
  - the previous interrupted east5 run was not a fluke
  - the `16k/8k` rung is reproducibly able to reach:
    - `vLLM environment ready`
    - Stage 1 completion
    - Stage 2 entry
- **Current unanswered question, still on the critical path:**
  - how this rung ends naturally:
    - success
    - `finish_reason='length'`
    - or OOM / HBM failure

### 2026-03-27 01:46 - GTPU-025 result: the `16k/8k` east5 rerun naturally fails in Stage 2 with the same `finish_reason='length'` signature
- **Terminal job result:**
  - parent:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-east5-ctx16k-rerun-20260327-0134`
    - `JOB_STATE_FAILED`
  - `prompts` child:
    - `JOB_STATE_FAILED`
- **What this run conclusively reproduced:**
  - `vLLM environment ready`
  - Stage 1 success
  - Stage 2 entry
  - then the same Stage-2 concretization failure as before, but now on the larger rung:
    - `ValueError: GPT-OSS chat response did not finish cleanly. Expected finish_reason='stop', got 'length'.`
    - `message.content = null`
    - long `message.reasoning`
- **Why this entry matters:**
  - the previous east5 `16k/8k` run was interrupted, so it was not stable evidence
  - this rerun removes that ambiguity
  - `max_model_len = 16384` and `concretize_max_tokens = 8192` are still **not** enough
- **Current interpretation under the user's brute-force directive:**
  - this is still not OOM
  - this is still not a startup failure
  - therefore the next valid move is to raise the rung again rather than stop the line of attack
- **Immediate next rung to launch:**
  - `max_model_len = 32768`
  - `understanding_max_tokens = 2048`
  - `concretize_max_tokens = 16384`
  - `extract_max_tokens = 16384`

### 2026-03-27 02:07 - GTPU-026 live checkpoint: the `32k/16k` rung does not fail quickly like `16k/8k`; it is still running and has already re-entered Stage 2 after a preemption
- **Launched parent job:**
  - `/ahmed/gpt-oss-20b-promptgen-one-statement-east5-ctx32k-20260327-0148`
- **Exact launch config:**
  - `max_model_len = 32768`
  - `understanding_max_tokens = 2048`
  - `concretize_max_tokens = 16384`
  - `extract_max_tokens = 16384`
- **What this rung has already proven:**
  - it clears startup
  - it reaches:
    - `vLLM environment ready`
    - Stage 1 completion
    - Stage 2 entry
  - unlike `16k/8k`, it has **not** yet ended in the quick `finish_reason='length'` failure during the same observation window
- **Important new behavior on this rung:**
  - the `prompts` child shows:
    - `preemption_count = 1`
  - the logs show:
    - `Loaded Stage 1 checkpoint for 1 statements`
    - followed by a fresh `Starting vLLM environment`
    - then another `vLLM environment ready`
    - then another `Stage 2: Concretizing 1 statements`
- **Interpretation of that new behavior:**
  - the larger rung is materially different, not just a slower copy of the 16k run
  - it is surviving long enough that:
    - checkpoint resume
    - preemption recovery
    - and second-pass Stage 2 execution
    - are now part of the observed path
- **Current state at the time of this append:**
  - parent:
    - `JOB_STATE_RUNNING`
  - `prompts` child:
    - `JOB_STATE_RUNNING`
  - no terminal OOM or terminal `length` failure has been observed yet on this rung
- **Immediate next action from this point:**
  - continue foreground babysitting on the active east5 `32k/16k` run
  - do not launch `64k/32k` yet unless this rung reaches a natural terminal outcome

### 2026-03-27 09:11 - GTPU-027 result: the `32k/16k` rung also fails naturally with the same Stage-2 `finish_reason='length'` signature, despite surviving much longer and recovering from preemption
- **Terminal job result:**
  - parent:
    - `/ahmed/gpt-oss-20b-promptgen-one-statement-east5-ctx32k-20260327-0148`
    - `JOB_STATE_FAILED`
  - `prompts` child:
    - `JOB_STATE_FAILED`
- **Critical new evidence from this rung:**
  - it did **not** OOM during startup or Stage 2
  - it survived long enough to:
    - reach Stage 2
    - get preempted once
    - reload the Stage 1 checkpoint
    - restart `vLLM`
    - re-enter Stage 2
  - only after that longer path did it finally fail
- **But the terminal failure shape is still the same:**
  - `ValueError: GPT-OSS chat response did not finish cleanly. Expected finish_reason='stop', got 'length'.`
  - `message.content = null`
  - long `message.reasoning`
- **Interpretation after the 32k rung:**
  - increasing context and completion budgets is changing runtime behavior
  - but it is **not** yet changing the fundamental failure mode
  - GPT-OSS is still spending the budget in reasoning and not reaching final assistant content for Stage 2 concretization
  - we still have **not** hit the requested stopping condition of:
    - OOM
    - HBM exhaustion
    - or another hard resource ceiling
- **Immediate next rung if we continue the brute-force line exactly as instructed:**
  - `max_model_len = 65536`
  - `understanding_max_tokens = 2048`
  - `concretize_max_tokens = 32768`
  - `extract_max_tokens = 32768`

### 2026-03-28 10:03 - GTPU-028 design: robust Stage-2 incremental checkpointing for the full-spec local GPT-OSS path

- **Why this entry exists:**
  - I read the ALIGN-279 plan in `.agents/logbooks/alignment_function_claude.md`.
  - The core direction is right:
    - keep global Stage 2 batching
    - add per-statement persistence
    - add a real partial-resume path in the orchestrator
  - But the plan still leaves three correctness gaps:
    - completion detection must run against the **outer accumulated state**, not the round-local state
    - a checkpointed statement must include the full ideation payload, including **diagnostics / concretization_attempts**
    - the fingerprint must identify the **actual concretization plan**, not just counts

## Problem

- **Observed failure mode from the real 120B run:**
  - ALIGN-278 in the alignment logbook showed the full-spec east5 run lose:
    - `2591 / 3339` concretize items
    - about `5+` hours of Stage 2 work
  - trigger:
    - TPU worker preemption during the local `vllm serve` prompt-generation path
  - recovery today:
    - Stage 1 resumes from checkpoint
    - Stage 2 restarts from zero

- **Current code shape causing that loss:**
  - `lib/marin/src/marin/alignment/generate_prompts.py:840`
    - `_run_concretization_stage_local(...)`
    - builds one flat list of work items across all statements
    - accumulates results in memory only
  - `lib/marin/src/marin/alignment/generate_prompts.py:675`
    - `_run_concretize_round_local_global(...)`
    - processes batches but has no persistence hook tied to completed statements
  - `lib/marin/src/marin/alignment/generate_prompts.py:1272`
    - monolithic `_save_ideation_checkpoint(...)` only runs after the entire stage returns
  - `lib/marin/src/marin/alignment/generate_prompts.py:1194`
    - orchestrator only loads Stage 2 checkpoints when `stage_status["concretize"]["complete"]` is `True`

- **Concrete consequence:**
  - the local GPT-OSS Stage 2 path has all-or-nothing persistence even though the natural recovery unit is a single completed statement

## Goals

- Preserve the existing local Stage 2 execution semantics:
  - same global batching across statements
  - same retry behavior
  - same `local_serve_batch_size`
- Bound preemption loss to only statements that were still incomplete at interruption time.
- Persist a **full** per-statement ideation artifact:
  - variations
  - coverage metadata
  - concretization diagnostics / attempts
- Avoid `stage_status` schema churn.
- Avoid heavy GCS rewrite behavior during Stage 2.
- Keep the existing monolithic Stage 2 checkpoint as the final fast path.

**Non-goals**
- No Stage 1 redesign.
- No Stage 3 redesign.
- No GPT-OSS prompting or token-budget changes here.
- No change to request shape or batching behavior at the `vllm serve` boundary.
- No attempt to make the API path perfectly resumable in this pass.

## Proposed Solution

### 1. Add a lightweight incremental checkpoint directory for Stage 2

- Add:
  - `artifacts/checkpoints/ideation_by_statement/`
- Store one file per completed statement.
- Use plain JSON artifacts rather than one-record JSONL.GZ shards:
  - each file is a single structured object
  - this reuses `_write_json_artifact(...)`
  - it avoids awkward "directory of one-row shard files" semantics

- Proposed helpers in `generate_prompts.py`:
  - `_stage2_incremental_checkpoint_dir(output_path: str) -> str`
  - `_stage2_incremental_checkpoint_path(output_path: str, statement_id: str) -> str`
  - `_save_statement_ideation(output_path: str, statement_id: str, ideation: dict[str, Any], fingerprint: dict[str, Any])`
  - `_load_partial_ideation_checkpoint(output_path: str, plans: dict[str, _ConcretizationPlan]) -> dict[str, dict[str, Any]]`
  - `_make_concretization_fingerprint(plan: _ConcretizationPlan) -> dict[str, Any]`

### 2. Fingerprint the actual plan, not just counts

- The fingerprint should validate the real statement-level concretization plan.
- It should include enough information to detect:
  - changed variation axes
  - changed covering behavior
  - changed concretization configs even when `num_configs` stays the same

- Recommended fingerprint payload:
  - `statement_id`
  - `num_axes`
  - `num_configs`
  - `plan_sha256`

- `plan_sha256` should be computed from a stable JSON encoding of:
  - `plan.axes`
  - `plan.configs`

- Reasoning:
  - `config_ids = cfg_000..cfg_n` are not enough because they mostly restate the count
  - hashing the real `axis_config` list catches actual plan drift

### 3. Move completion detection onto the outer accumulated state

- This is the most important fix to Claude's remaining gap.
- The callback cannot rely on the round-local `parsed_by_statement` inside `_run_concretize_round_local_global(...)` because that map is rebuilt on every retry round.
- A statement can become complete only after combining:
  - earlier successful items
  - later retry successes

- Therefore the local Stage 2 design should be:
  - `_run_concretization_stage_local(...)` owns the authoritative mutable accumulators:
    - `parsed_by_statement`
    - `diagnostics_by_statement`
    - `checkpointed_statement_ids`
  - `_run_concretize_round_local_global(...)` updates those caller-owned accumulators directly
  - after each successful item merge, `_run_concretize_round_local_global(...)` checks whether that statement has now reached completion against the **outer** accumulated scenario count
  - when it does, it calls a callback with enough material to build the final ideation record for that statement

- Do **not** build completion detection off a per-round temporary map.

### 4. Persist the full per-statement ideation record, including diagnostics

- The checkpoint callback must receive enough data to build the same per-statement object produced by `_build_concretization_result(...)`.
- That means the checkpoint boundary must preserve:
  - `parsed scenarios`
  - `diagnostics / concretization_attempts`
  - `axes`
  - `configs`
  - `coverage stats`

- Minimal safe callback contract:

```python
def statement_completed_callback(
    statement_id: str,
    parsed_scenarios: dict[str, dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> None: ...
```

- Inside `_run_concretization_stage_local(...)`, that callback should build the **full** ideation via `_build_concretization_result(...)` and write it immediately.

### 5. Compute plans once and thread them through

- Do not recompute `_prepare_concretization_plans(...)` inside the checkpoint callback.
- Compute plans once at the orchestrator boundary after Stage 1 is available.
- Reuse that same plan map for:
  - loading incremental checkpoints
  - fingerprint validation
  - Stage 2 execution
  - statement checkpoint writes

- This removes redundant work and guarantees that validation and runtime execution are talking about the same plan.

### 6. Add a true partial-resume branch to the orchestrator

- In `generate_prompts_from_spec(...)`:
  - keep the existing complete-checkpoint path unchanged
  - add a second Stage 2 branch for partial incremental checkpoints

- Flow:
  1. Stage 1 loads or recovers as today.
  2. If local ideation path is active and `stage_status["concretize"]["complete"]` is `False`, compute plans once.
  3. Load `ideation_by_statement/` using those plans and discard any stale records with fingerprint mismatches.
  4. If partial checkpoints cover all expected statements:
     - materialize the monolithic `_save_ideation_checkpoint(...)`
     - mark Stage 2 complete
     - continue on the existing fast path
  5. Otherwise:
     - run Stage 2 only for missing statements
     - merge newly completed statements with previously loaded ideations
     - write the monolithic Stage 2 checkpoint once at the end

- Important:
  - `stage_status` remains unchanged structurally
  - the partial state lives in `ideation_by_statement/`, not in `stage_status`

## Core Sketch

```python
plans = _prepare_concretization_plans(understandings, config)
partial_ideations = _load_partial_ideation_checkpoint(config.output_path, plans)

def save_completed_statement(statement_id: str, parsed_scenarios: dict[str, dict[str, Any]], diagnostics: list[dict[str, Any]]) -> None:
    plan = plans[statement_id]
    ideation = _build_concretization_result(
        statement_id,
        config,
        plan.axes,
        plan.configs,
        plan.stats,
        parsed_scenarios,
        diagnostics,
    )
    _save_statement_ideation(
        config.output_path,
        statement_id,
        ideation,
        _make_concretization_fingerprint(plan),
    )

ideations = _run_concretization_stage(
    understandings,
    config,
    active_ideation_session,
    plans=plans,
    completed_ideations=partial_ideations,
    statement_checkpoint_callback=save_completed_statement,
)
```

## Implementation Outline

1. Add Stage 2 incremental checkpoint path helpers and a robust plan fingerprint helper in `generate_prompts.py`.
2. Refactor local Stage 2 so `_run_concretize_round_local_global(...)` mutates caller-owned accumulators and can fire a statement-completed callback against the true aggregated state.
3. Extend `_run_concretization_stage_local(...)` and `_run_concretization_stage(...)` to accept precomputed plans, preloaded completed ideations, and the checkpoint callback for the local path only.
4. Add a partial-resume branch in `generate_prompts_from_spec(...)` between the current "complete Stage 2" and "run from scratch" branches.
5. Test:
   - resume from partial incremental checkpoints
   - stale fingerprint rejection
   - preservation of `concretization_attempts` after resume
   - existing global batching behavior remains unchanged

## Notes

- **Why local-only right now:**
  - the real failing production path is the local GPT-OSS `vllm serve` path
  - the API branch at `generate_prompts.py:1429` is still less resilient than ideal, but that is not the current bottleneck

- **Why no Stage 1 work:**
  - Stage 1 already has raw-attempt recovery and is not the expensive preemption sink

- **Why plain JSON per statement:**
  - one file maps cleanly to one completed statement
  - existing JSON artifact helpers already exist
  - this keeps the incremental checkpoint format obvious when browsing GCS

- **Expected resilience after this fix:**
  - all fully checkpointed statements survive
  - only statements still incomplete at interruption time rerun
  - practical loss should usually be on the order of the currently active statement set, not the full 3339-item stage

- **Expected I/O cost:**
  - about one small object write per completed statement
  - no repeated `_save_artifacts(...)` calls
  - no change to `vllm serve` request volume

## Future Work

- Extend the same incremental persistence idea to the API Stage 2 path for parity.
- If preemptions still make 1-2 statements too expensive, consider per-config checkpointing, but only after the statement-level version proves insufficient.
- Add optional cleanup for stale `ideation_by_statement/` files after monolithic Stage 2 success if storage clutter becomes an issue.

## Decision

- I would implement this version, not the earlier per-statement-loop design and not the weaker callback-only v2.
- The critical invariants are:
  - preserve global batching semantics
  - checkpoint from outer accumulated state
  - persist full ideation records with diagnostics
  - validate incremental checkpoints against the actual statement plan

### 2026-03-28 10:13 - GTPU-029 implementation: shipped the GTPU-028 / ALIGN-279 hybrid Stage-2 checkpointing fix

- **Files changed:**
  - `lib/marin/src/marin/alignment/generate_prompts.py`
  - `tests/test_alignment.py`

- **What was actually implemented:**
  - added Stage 2 incremental checkpoint helpers:
    - `_stage2_incremental_checkpoint_dir(...)`
    - `_stage2_incremental_checkpoint_path(...)`
    - `_make_concretization_fingerprint(...)`
    - `_save_statement_ideation(...)`
    - `_load_partial_ideation_checkpoint(...)`
  - added a partial-resume path in `generate_prompts_from_spec(...)`
  - extended `_run_concretization_stage_local(...)` and `_run_concretization_stage(...)` to accept:
    - precomputed `plans`
    - `completed_ideations`
    - `statement_checkpoint_callback`
  - changed `_run_concretize_round_local_global(...)` so it now receives the **outer accumulated**
    - `parsed_by_statement`
    - `diagnostics_by_statement`
    - expected item counts
    - checkpointed statement ids
    - and can fire a completion callback against the true cross-round state

- **Why this is the final hybrid and not Claude's latest text verbatim:**
  - Claude's latest plan correctly moved completion detection out of the round-local temporary maps.
  - I kept that insight.
  - But I still implemented the callback at the item-processing boundary inside `_run_concretize_round_local_global(...)`, using the caller-owned outer accumulators.
  - That preserves the stronger invariant from GTPU-028:
    - completion detection happens as soon as the outer aggregated state reaches the expected count
    - not only after a whole retry round finishes

- **Important implementation choice:**
  - per-statement Stage 2 checkpoints are stored as:
    - `artifacts/checkpoints/ideation_by_statement/<statement_id>.json`
  - not:
    - one-record `.jsonl.gz` files
  - Reason:
    - one file really is one object here
    - `_write_json_artifact(...)` already exists
    - the format is easier to browse and debug on local disk or GCS

- **Fingerprint shape actually implemented:**
  - fingerprint is a dict containing:
    - `statement_id`
    - `num_axes`
    - `num_configs`
    - `plan_sha256`
  - `plan_sha256` is computed from a stable JSON encoding of:
    - `plan.axes`
    - `plan.configs`
  - this is stronger than count-only validation and catches real plan drift

- **Bug found during implementation and fixed immediately:**
  - the first draft of the code was computing Stage 2 plans too early when Stage 1 had only been partially recovered from raw attempts
  - consequence:
    - Stage 2 plan cache could be built from incomplete `understandings`
    - then reused later after Stage 1 finished
  - fix:
    - only load partial Stage 2 checkpoints once **all** understandings are present

- **Behavior after this change:**
  - Stage 1 resume behavior is unchanged
  - Stage 3 checkpointing is unchanged
  - local Stage 2 now:
    - skips already checkpointed statements
    - writes full ideation records, including `concretization_attempts`, when a statement becomes complete
    - writes the monolithic Stage 2 checkpoint only once at the end, as before
  - API Stage 2 remains unchanged and still has all-or-nothing persistence

- **Tests added:**
  - integration test:
    - Stage 2 failure after one statement completes
    - verifies incremental checkpoint file is written
    - verifies rerun skips the completed statement and only concretizes the missing one
  - integration test:
    - tampers with the saved fingerprint
    - verifies the stale incremental checkpoint is discarded and the statement is recomputed

- **Validation:**
  - `python3 -m py_compile lib/marin/src/marin/alignment/generate_prompts.py tests/test_alignment.py`
    - passed
  - `uv run pytest tests/test_alignment.py -q`
    - `107 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py tests/test_alignment.py`
    - `OK`

- **Net result:**
  - the repo now has the checkpointing design GTPU-028 argued for, with the implementation details tightened where the design was still too abstract
  - this should convert the current Stage 2 preemption loss from:
    - "restart the whole 3339-item stage"
  - to:
    - "rerun only statements that were still incomplete when the worker died"

### 2026-03-28 23:56 - GTPU-030 failure: Harmony token protocol violation crashed the vLLM server at 98.7% Stage 2 completion on the full-spec 120B run

- **Job:** `/ahmed/goss-120b-full-spec-ckpt-east5-20260328`
- **Terminal state:** `JOB_STATE_FAILED` (prompts child), `failure_count=1`, `preemption_count=0`
- **Checkpointing status:** 44/46 statements saved to `ideation_by_statement/` before crash — checkpointing worked correctly

#### Timeline

- 17:17 — vLLM server started, weights loading (615 shards)
- 17:27 — `vLLM environment ready`
- 17:27 — Stage 1: 46 statements, completed 46/46
- 17:27 — Stage 2: 3339 concretize items started
- 19:42–23:54 — Stage 2 progressing steadily at ~8 items/min, checkpointing statements as they complete
- 23:50:08 — Stage 2 progress: 3262/3339 (97.7%)
- 23:54:25 — Stage 2 progress: 3294/3339 (98.7%) — last successful progress log
- 23:55:55 — `Stage2 attempt 1 left 2 concretize item(s) pending retry` — first sign of trouble
- 23:55:55 — retry batch sent to `/v1/chat/completions`
- 23:56:15 — `RuntimeError: Stage 2 failed` — all remaining items returned 500

#### The error

Every failing request returned the same vLLM server-side error:

```json
{
  "error": {
    "message": "Unexpected token 200002 while expecting start token 200006",
    "type": "Internal Server Error",
    "param": null,
    "code": 500
  }
}
```

- Token `200002` and `200006` are Harmony special tokens (channel control tokens in the GPT-OSS vocabulary)
- `200006` is likely the `<|start|>` conversation/turn start token
- `200002` is likely the `<|channel|>analysis` or another channel start token
- The error means: the model generated a channel control token in a position where vLLM's Harmony state machine expected a conversation start token

#### Failure pattern

- The error first appeared on `avoid_errors` cfg_056 — this is partway through the 45th statement (out of 46)
- Items cfg_000 through cfg_055 of `avoid_errors` completed successfully before the crash
- After the first 500 error, **every subsequent request** also returned the same 500 error:
  - `avoid_errors` cfg_056 through cfg_075 (20 items)
  - `avoid_extremist_content` cfg_000 through cfg_011 (12 items)
- The retry mechanism fired (`concretize_max_attempts`), but retries also failed because the server was permanently broken
- Total: 32 items lost across 2 statements

#### Root cause analysis

- **This is a vLLM Harmony protocol bug, not a Marin bug.**
- The vLLM server's Harmony token state machine entered an unrecoverable state after the model produced an unexpected token sequence.
- Possible triggers:
  - A specific prompt content in `avoid_errors` cfg_056 caused the model to generate tokens in an order that violated Harmony protocol expectations
  - The server's internal state (KV cache, token tracking) accumulated corruption over ~3294 requests / ~6 hours of continuous inference
  - An edge case in the `openai_gptoss` reasoning parser's state tracking
- **The server cannot self-recover** — once the Harmony state is corrupted, every new request fails with the same error because the parser rejects the conversation before inference even runs

#### Visibility gap

- We only see the HTTP response body (`{"error":{"message":"Unexpected token 200002..."}}`)
- We do NOT have:
  - The vLLM server-side Python traceback that produced this 500
  - The exact request payload for cfg_056 of `avoid_errors` (the triggering request)
  - The vLLM EngineCore logs showing what the model actually generated
  - Whether the Harmony parser state was corrupted before or during that specific request
- The stderr tee (ALIGN-255) captures vLLM startup logs but may not capture all runtime errors — the 500 is generated inside the FastAPI handler, which logs to its own stream

#### What the next agent should investigate

1. **Find where this error is raised in the vLLM codebase:**
   - Search `vllm_kcbs` for `"Unexpected token"` and `"expecting start token"`
   - This is likely in `vllm/entrypoints/harmony_utils.py` or `vllm/reasoning/gptoss_reasoning_parser.py`
   - Understand the Harmony state machine: what states expect token 200006, and what token 200002 means

2. **Determine if this is per-request or server-global corruption:**
   - If the error is raised during response parsing (after generation), the model output bad tokens and the parser is correct to reject — but the server should recover for the next request
   - If the error is raised during request construction (before generation), something in the server's persistent state is broken — this explains why all subsequent requests fail
   - Key question: does vLLM maintain per-connection Harmony state that survives across requests?

3. **Reproduce with the triggering prompt:**
   - The `avoid_errors` statement's Stage 1 understanding and Stage 2 covering configs are checkpointed in `gs://marin-us-east5/align/goss_120b_full_spec/prompts-c623f9/artifacts/checkpoints/`
   - Reconstruct the exact cfg_056 prompt and send it to a fresh vLLM server to see if it's prompt-content-triggered or accumulated-state-triggered
   - If a fresh server handles cfg_056 fine, the bug is in state accumulation over thousands of requests

4. **Check if this is a known issue:**
   - Search the `vllm_kcbs` or upstream vLLM issue tracker for Harmony token validation failures
   - Check if there are similar reports for GPT-OSS on long-running inference sessions

5. **Defensive improvements in Marin (regardless of root cause):**
   - Add a consecutive-500 circuit breaker: if N consecutive requests return 500, treat the server as dead and abort early instead of burning through all remaining items
   - Consider restarting the vLLM server subprocess after a server-side 500 (not just client-side errors)
   - The per-statement checkpointing already limits blast radius — 44/46 statements survived this crash

#### Checkpointed artifacts that survived

```
gs://marin-us-east5/align/goss_120b_full_spec/prompts-c623f9/artifacts/checkpoints/ideation_by_statement/
  ask_clarifying_questions.json
  assume_objective_pov.json
  avoid_abuse.json
  avoid_being_condescending.json
  avoid_hateful_content.json
  avoid_info_hazards.json
  avoid_overstepping.json
  avoid_regulated_advice.json
  avoid_sycophancy.json
  avoid_targeted_political_manipulation.json
  be_clear.json
  be_creative.json
  be_empathetic.json
  be_engaging.json
  be_kind.json
  ... (44 total)
```

Missing (need rerun): `avoid_errors`, `avoid_extremist_content`

#### Recovery plan

A simple relaunch will:
1. Load Stage 1 checkpoint (skip Stage 1 entirely)
2. Load 44/46 per-statement Stage 2 checkpoints
3. Only process `avoid_errors` and `avoid_extremist_content` (~156 items)
4. Get a fresh vLLM server that isn't in a corrupted Harmony state
5. Complete Stage 2 in ~20 minutes instead of ~6.5 hours

### 2026-03-28 15:35 PDT - GTPU-031 TODO if the Harmony 500 comes back and becomes a real time sink

This is the short resumption note for the next person who has to touch this.

#### Current best understanding

- The most likely failure mode is no longer "Marin built the wrong request."
- Marin is already on the validated GPT-OSS path:
  - `/v1/chat/completions`
  - top-level `reasoning_effort="low"`
  - `MODEL_IMPL_TYPE=vllm`
- The exact exception text comes from the upstream `openai/harmony` parser, not from Marin.
- Token mapping correction:
  - `200006` = `<|start|>`
  - `200002` = `<|return|>`
- So the failing parse means:
  - Harmony was waiting for the start of a new assistant message
  - but instead saw a return/stop token
- That is much more consistent with malformed GPT-OSS Harmony output than with a bad client request.

#### Strongest external clue

- OpenAI Harmony issue `#80` reports a known GPT-OSS 120B failure mode on refusal-style outputs:
  - the model emits structurally invalid Harmony text/tokens
  - strict parsing then fails
- That matches the remaining failed statements unusually well:
  - `avoid_errors`
  - `avoid_extremist_content`

#### Most important thing we still do NOT have

We still do not have the one artifact that would settle this quickly:

- the exact request payload and raw generated token ids for `avoid_errors` `cfg_056`

Without that, it is too easy to keep hand-waving between:

- prompt-triggered malformed refusal output
- long-run server state corruption after thousands of requests

#### TODO: minimal reproduction sequence

If this failure comes back and starts wasting time, do these in order.

1. Reconstruct the exact failed request offline.
   - Use:
     - `gs://marin-us-east5/align/goss_120b_full_spec/prompts-c623f9/artifacts/checkpoints/understandings.jsonl.gz`
     - `artifacts/avoid_errors/understanding.json`
   - Recreate the concretization plan with the same covering config.
   - Select `avoid_errors` `cfg_056`.
   - Build the request with the same repo helpers used in production:
     - `_prepare_concretization_plans(...)`
     - `_build_concretize_messages(...)`
     - `_ConcretizeConfig(index=56, ...)`
   - Write out the final JSON payload exactly as Marin would POST to `/v1/chat/completions`.

2. Run that one request against a completely fresh GPT-OSS 120B vLLM server.
   - Same validated stack:
     - GPT-OSS
     - `MODEL_IMPL_TYPE=vllm`
     - Harmony path enabled
   - Turn on diagnostics first:
     - `VLLM_SERVER_DEV_MODE=1`
     - `VLLM_LOGGING_LEVEL=DEBUG`
     - `--enable-log-requests`
     - optionally `--enable-log-outputs`

3. Run immediate neighbors as controls.
   - `avoid_errors cfg_055`
   - `avoid_errors cfg_056`
   - `avoid_errors cfg_057`

4. Interpret the result conservatively.
   - If `cfg_056` fails immediately on a fresh server:
     - treat this as prompt-triggered malformed GPT-OSS Harmony output first
   - If all three succeed on a fresh server:
     - only then escalate the "long-run server/engine state" hypothesis

5. If the fresh repro still only gives a generic 500, patch vLLM once to log token ids before re-raising.
   - Patch the non-streaming Harmony parse site around:
     - `parse_chat_output(token_ids)`
     - in `vllm/entrypoints/openai/chat_completion/serving.py`
   - Log:
     - request id
     - token ids
     - decoded text
     - finish reason
   - Re-raise unchanged.

#### Important caveat about logging

- `--enable-log-outputs` alone is not enough.
- In non-streaming chat mode, vLLM logs outputs after successful response assembly.
- If Harmony parsing throws first, you will NOT get the raw failing completion from normal output logging.

#### Cheap follow-up only if fresh repro succeeds

Do not jump straight to another 6-hour run.

Instead:

1. replay a small set of known-safe Stage 2 prompts
2. then send `avoid_errors cfg_056`
3. gradually increase the warmup count if needed

That is the cheapest way to test whether prior request history matters.

#### Where the fuller notes live

- See `docs/debug-log-gpt-oss-harmony-failure.md` for the evidence chain and the full minimal reproduction plan.

### 2026-03-28 16:20 PDT - GTPU-032 live progress / ETA / token-throughput visibility plan

#### Problem

We can now resume long prompt-generation runs correctly, but the live operator visibility is still weak:

- Stage 1/2/3 progress logs only show `completed/total (%)`
- there is no dynamic ETA like `tqdm`
- local vLLM token throughput is collected internally but only written to `artifacts/vllm_metrics.json` at the end
- response generation via local vLLM currently sends the full prompt set in one `generate_from_messages(...)` call, so there is no intermediate hook for live progress at all

This is why a live job can look "stuck" even when the worker is healthy and pushing tokens.

#### Existing state

- `BatchedVllmServeSession` already accumulates per-stage metrics:
  - request count
  - prompt count
  - input token count
  - output token count
  - request seconds
- Those metrics are exposed via `metrics_snapshot()`
- Prompt generation already has natural progress boundaries:
  - Stage 1 local: one batch of statements
  - Stage 2 local: one concretize microbatch
  - Stage 3 local: one extraction microbatch
  - API paths also have per-future / per-batch boundaries
- Response generation does **not** have those boundaries yet on the local path because it batches the entire prompt set in one call
- Judge already has an outer batch loop, so it is easy to instrument

#### Design

Add one shared live progress reporter for alignment inference stages.

The reporter should:

- track elapsed wall-clock time
- maintain a smoothed recent throughput estimate instead of a naive whole-run average
- log:
  - `completed/total`
  - percent
  - smoothed `items/s`
  - ETA
  - optional contextual fields like `attempt=...` or `retries pending=...`
  - optional local vLLM `prompt tok/s` and `completion tok/s`

Important implementation detail:

- token throughput should come from the live `BatchedVllmServeSession.metrics_snapshot()` for the specific logical stage
- this should be surfaced in logs during the run, not only in the final artifact

#### Scope

Apply the reporter to:

1. prompt generation
   - Stage 1 understanding
   - Stage 2 concretization
   - Stage 3 extraction
   - both local vLLM and API paths

2. response generation
   - local vLLM chosen generation
   - local vLLM rejected generation
   - shared-session chosen+rejected generation

3. judging
   - local vLLM judge
   - API judge can reuse the same ETA reporter without token stats

#### Required structural change

To get live progress for local response generation, stop sending the entire prompt set in one monolithic `generate_from_messages(...)` call.

Instead:

- add an explicit local microbatch size to the response-generation config
- iterate prompts in batches
- log progress after each batch
- preserve output order exactly

This is the only substantive behavioral change required for visibility.

#### Non-goals

- do **not** redesign checkpointing
- do **not** change vLLM request semantics
- do **not** try to estimate cost from model internals; use observed recent throughput only

#### Success criteria

Healthy long-running jobs should emit lines shaped roughly like:

```text
Stage 3 progress: 767/3339 (23.0%) [attempt 1, retries pending=0, 0.41 items/s, ETA 1h 43m, prompt 2.8k tok/s, completion 380 tok/s]
Rejected progress: 512/3339 (15.3%) [0.35 items/s, ETA 2h 15m, prompt 3.1k tok/s, completion 420 tok/s]
Judge progress: 900/3339 (27.0%) [0.58 items/s, ETA 1h 13m, prompt 2.4k tok/s, completion 210 tok/s]
```

The final structured metrics artifact should remain unchanged; this work is about live visibility while the job is running.
