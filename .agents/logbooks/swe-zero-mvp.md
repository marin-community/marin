# SWE-ZERO MVP: Research Logbook

Experiment issue: (TBD — will be linked after creation)
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
