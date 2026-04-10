# SWE-ZERO MVP: Research Logbook

Experiment issue: https://github.com/marin-community/marin/issues/4561
Branch: `kevin/swe-zero-mvp`
Parent issue: https://github.com/marin-community/marin/issues/4435

## Scope
- **Goal**: Generate diverse execution-free agentic traces for pretraining in SWE-ZERO style, using Gemma 4 E2B/E4B on TPU via vLLM-tpu, against SWE-rebench V2 PRs.
- **Primary metric(s)**: (1) Can Gemma 4 produce sensible multi-turn tool calls? (2) Rollout diversity measured by MinHash-based pairwise Jaccard similarity.
- **Constraints**: Truncate rollouts at 8192 tokens. Execution-free only (no code execution). Use whatever TPU type is most available.
- **Stop criteria**: Complete 1000 diverse rollouts (10 repos x 10 PRs x 10 rollouts) with diversity report, or determine Gemma 4 cannot produce sensible tool calls and escalate.

## Baseline
- Date: 2026-04-09
- Code refs: `experiments/swe_zero/` (scaffold, data loader, rollout generator, diversity analyzer)
- Baseline numbers: N/A (no prior runs)

## Key References
- [SWE-ZERO paper](https://arxiv.org/abs/2604.01496): Execution-free rollout generation using str_replace_editor, think, finish tools
- [Code World Model paper](https://arxiv.org/abs/2510.02387): MinHash-based trajectory near-deduplication (Jaccard < 0.5)
- [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2): 32K instances, 20 languages
- Models: `google/gemma-4-E2B-it` (5B), `google/gemma-4-E4B-it` (8B)

## Experiment Log

### 2026-04-09 — SZ-001: Kickoff and code scaffold
- **Hypothesis**: MVP code can be written to orchestrate execution-free rollouts with simulated tool responses.
- **Result**: Created 8 files in `experiments/swe_zero/`:
  - `data_loader.py` — SWE-rebench V2 HuggingFace loader
  - `scaffold.py` — 5 tools (str_replace_editor, find_file, search, think, finish), system prompt, simulated env
  - `rollout_generator.py` — Multi-turn conversation orchestration via OpenAI-compatible API
  - `diversity.py` — MinHash/Jaccard diversity measurement
  - `prototype_tool_calling.py` — Step 1 test script
  - `serve_vllm_tpu.sh` — vLLM TPU server launch script
  - `run_swe_zero_mvp.py` — Steps 3-6 pipeline
  - `submit_iris_job.py` — Iris cluster job helper
- All modules pass lint (ruff) and basic integration tests.
- **Interpretation**: Code scaffold is ready. Next: allocate dev TPU and run Step 1.
- **Next action**: Commit, push, create experiment issue, allocate dev TPU.

### 2026-04-09 — SZ-002: Step 1 — Gemma 4 E2B tool calling prototype
- **Hypothesis**: Gemma 4 E2B (5B) can produce reasonable multi-turn tool calls on a SWE-bench-style task.
- **Command**: `python3 ~/test_gemma4_tool_calling.py` and `python3 ~/test_gemma4_multiturn.py` on `kevin-swe-zero-v6e4` (v6e-4 TPU VM in us-east1-d)
- **Config**: `google/gemma-4-E2B-it`, bfloat16, temperature=0.7, top_p=0.9, max_new_tokens=512
- **Result** (confidence: replicated):
  - Gemma 4 E2B chat template natively supports tool calling via `<|tool_call>call:name{args}<tool_call|>` format
  - Single-turn test: Model correctly called `think` tool, planned to view file. 98 tokens in 9.5s.
  - Multi-turn test (5 turns): Model correctly executed a complete SWE workflow:
    1. `str_replace_editor(view, /src/calc.py)` — read the file
    2. `str_replace_editor(str_replace, ...)` — correctly replaced `return None` with `return price`
    3. `think(...)` — reflected on the fix
    4. Natural language summary of the fix
    5. End of conversation
  - The fix was semantically correct.
  - Generation speed on CPU: ~10 tok/s (need TPU acceleration via vLLM for scale)
- **Interpretation**: Gemma 4 E2B works perfectly for SWE-ZERO. E2B is the right choice (smaller = faster).
- **Decision**: Use E2B going forward (per issue plan: "If E2B works, use it instead of E4B to speed things up").
- **Infra note**: Iris controller was timing out (overloaded managing v6e-128 scaling ops). Created TPU VM directly via gcloud. vLLM on TPU requires `tpu_inference` package (not in pip vLLM). For Steps 3-6, need to either use vLLM with tpu_inference Docker image, or use transformers+torch_xla directly.
- **Next action**: Set up vLLM-tpu serving infra for scaled rollout generation (Step 2).

### 2026-04-09 — SZ-003: Switch to mini-swe-agent v2 format
- **Change**: Rewrote scaffold to use mini-swe-agent v2 bash-only format instead of custom tool-calling.
- **Result**: Simulated env handles cat, find, grep, ls, sed, head, heredocs. Step 3 ran successfully.

### 2026-04-09 — SZ-004: Step 4 — 10 rollouts diversity measurement
- **Hypothesis**: At temperature=1.0, rollouts from the same PR show meaningful diversity.
- **Command**: `uv run python -c "..." ` (Step 4 script against `core-gatech-group/serpent-tools` PR #21)
- **Config**: Gemma 4 E2B, temperature=1.0, max_total_tokens=8192, 10 rollouts
- **Result** (confidence: exploratory):
  - 10 rollouts in 3866s (~6.4 min/rollout on CPU)
  - 9/10 finished cleanly, 1 hit token budget (24 steps)
  - Diversity (MinHash Jaccard):
    - **Unique (Jaccard < 0.5): 9/10**
    - Mean pairwise: 0.2102
    - Median pairwise: 0.2031
    - Min pairwise: 0.0703
    - Max pairwise: 0.5859
    - Std: 0.1247
- **Interpretation**: Good diversity at temperature=1.0. Most rollouts are substantially different (median Jaccard ~0.2). Only one pair is near-duplicate (0.59). This is promising for generating diverse pretraining data.
- **Next action**: Step 5 (100 rollouts from 10 PRs) and Step 6 (1000 rollouts from 10 repos).
- **Blocker**: CPU inference is slow (~6 min/rollout). Steps 5-6 would take 10+ hours and 100+ hours respectively at this speed. Need TPU acceleration via vLLM-tpu or to reduce rollout count for MVP.

### 2026-04-09 — SZ-005: Steps 5 & 6 submitted to Iris cluster with TPU inference
- **Change**: Refactored to use `ray_run.py` for cluster submission with Docker vLLM sidecar on TPU. `VLLM_TPU_SKIP_PRECOMPILE=1`.
- **Command**:
  ```
  uv run lib/marin/src/marin/run/ray_run.py \
    --cluster marin-us-east5 --extra vllm --tpu v6e-8 \
    --env_vars VLLM_TPU_SKIP_PRECOMPILE 1 \
    --env_vars MARIN_VLLM_MODE docker \
    --no_wait \
    -- python experiments/swe_zero/run_swe_zero_mvp.py --local --model google/gemma-4-E2B-it --step {5,6} \
       --output_dir gs://marin-us-central2/experiments/swe_zero_mvp
  ```
- **Jobs**:
  - Step 5: `ray-run-kevin-run_swe_zero_mvp-20260409-180301` (PENDING)
  - Step 6: `ray-run-kevin-run_swe_zero_mvp-20260409-180313` (PENDING)
- **Output**: `gs://marin-us-central2/experiments/swe_zero_mvp/step{5,6}/`
- **Next**: Monitor jobs, report results when complete.

### 2026-04-09 — SZ-006: Switched to Gemma 3 4B IT for TPU inference
- **Blocker**: Gemma 4 E2B-it is multimodal (audio_tower) — vllm-tpu (both pip 0.10.1 and 0.13.0) can't load it:
  - vllm-tpu Docker images have transformers 4.57 (pre-gemma4)
  - Even with upgraded transformers, weight loading fails on `model.audio_tower.*` params
- **Decision**: Use Gemma 3 4B IT instead. Verified serving on v6e-4 with `VLLM_TPU_SKIP_PRECOMPILE=1` (~290s startup).
- **New jobs**: Resubmitted Steps 5+6 with `google/gemma-3-4b-it` on `marin-us-east5` v6e-4.
  - Step 5: `ray-run-kevin-run_swe_zero_mvp-20260409-203933`
  - Step 6: `ray-run-kevin-run_swe_zero_mvp-20260409-203945`
- **Negative result**: Gemma 4 E2B is not yet usable via vllm-tpu on Iris cluster. Need vllm-tpu Docker image update or custom image with latest transformers + Gemma 4 text-only support.

### 2026-04-10 — SZ-007: Switch to mini-swe-agent v1 + ricdomolm/mini-coder-1.7b
- **Change**: Switched scaffold to mini-swe-agent v1.17.5 format and the
  `ricdomolm/mini-coder-1.7b` model (Qwen3-1.7B fine-tuned on 400k mini-swe-agent
  trajectories — purpose-built for this scaffold).
- **Format details (v1)**:
  - Action regex: `r"```bash\s*\n(.*?)\n```"` (NOT `mswea_bash_command`)
  - System prompt requires THOUGHT + a single ```bash block
  - Submission: first line of bash *output* must be `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  - Observation format: `Observation: {{output}}`
- **Why this should work better**:
  - Qwen3-1.7B is fully supported by vllm-tpu (no multimodal issues)
  - 1.7B is much smaller than Gemma 3 4B → faster startup, fits in v6e-4 trivially
  - Trained specifically on mini-swe-agent trajectories → high prior on producing valid format
- **Jobs submitted** (cluster: marin-us-east5, TPU: v6e-4, model: ricdomolm/mini-coder-1.7b):
  - Step 5: `ray-run-kevin-run_swe_zero_mvp-20260410-032016`
  - Step 6: `ray-run-kevin-run_swe_zero_mvp-20260410-032022`
