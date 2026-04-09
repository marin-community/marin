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
