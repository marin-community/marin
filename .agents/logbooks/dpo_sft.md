# DPO + SFT Mixture Training: Research Logbook

**Project doc:** `.agents/projects/dpo_sft.md` (high-level goals + scope; this file holds everything tactical)
**Branch:** `dpo_sft` (forked from `origin/dpo-lora-clean`, no upstream — first push creates `origin/dpo_sft`)
**Started:** 2026-04-26

---

## Status snapshot (last updated 2026-04-27 — third session)

**Done this project so far:**
- Project + branch setup.
- **Verifier port** (`experiments/ifbench/verifiers/`): vendored IFBench@cb932e352a50 + IFEvalG@881b18458b48; scorer reproduces upstream byte-for-byte; 14 tests.
- **Stage 1 data prep** (`experiments/ifbench/data/prepare.py`): 95,373-row load + stratified 2k val by `num_constraints` + zero-overlap contamination check vs IFBench_test; 8 tests.
- **RolloutBackend protocol** (`experiments/ifbench/rollout/backend.py`): unified Protocol; 8 tests.
- **Together + Gemini batch clients** (`together_backend.py`, `gemini_backend.py`) — implemented all 3 ops (submit_batch / poll / download). Uncovered + worked around schema quirks (Gemini inline ≠ file-based; Together CREATE wraps in `{"job":...}`; Together upload is 3-step S3-redirect — delegated to SDK).
- **Sync fallback** (`sync_runner.py`) — bounded-concurrency asyncio runners; was needed because Gemini + 8B-Lite batches stalled in queue for 40+ min on 2026-04-27.
- **Stage 4 extraction** (`experiments/ifbench/rollout/extract.py`): pair construction + SFT cache + per-num_constraints stats; 7 tests.
- **Stage 2 smoke driver** (`experiments/ifbench/rollout/smoke.py`): submit / poll / run subcommands, state-file resumable.
- **100-prompt smoke ran end-to-end ✅** (pair yield 34%, $0.34 cost, see Final smoke result below).
- **Verifier deps added to root `pyproject.toml`**.
- **D-001 closed:** mixture trainer should extend `SimpleDPOConfig` with optional `sft_data` + `alpha_sft` fields. SFT lives at marin layer (`experiments.defaults.default_sft`).
- Decisions D-001 through D-013 captured.

**37 tests passing** across `experiments/ifbench/` (run with `PYTHONPATH=. /tmp/ifbench_port/port_venv/bin/python -m pytest experiments/ifbench/ -o addopts=""`).

**Smoke headline numbers (2026-04-27):**
- Pair yield: **34/100 (34%)** — clears the 30% gate.
- Per-model pass-all rate (strict): Llama-3.3-70B-Turbo **26.0%**, Gemini-Flash thinking=high **22.0%**, Llama-3-8B-Lite **10.0%**.
- Surprise: Gemini-Flash thinking=high underperformed 70B-Turbo on strict — see "Final smoke result" entry. Loose accuracy not yet measured; could close the gap.
- Gemini thinking-token volume: 95% of "output" billing. Confirms cost concern.
- Smoke cost: $0.34 actual (70B batch $0.029 + 8B-Lite sync $0.005 + Gemini sync $0.31).

**Open / next:**
1. **Cost decision (Task #5):** full 95k × 3 run is $170 if Gemini batch ever works, $310 if forced to sync. Need explicit "yes go" before launch.
2. **Validate Gemini batch on a different day (open question 1 in Final smoke entry).** 10-prompt smoke first; if it completes in < 30 min, Gemini batch is usable and the cheap path is real.
3. **Loose-accuracy re-score** of the existing 300 smoke rollouts. ~5-min local job; tells us whether Gemini's underperformance is a strict-mode artifact.
4. **Mixture trainer config** (Task #2) — extend `SimpleDPOConfig`. Independent of any rollout work.
5. **Mixture trainer test** (Task #6) — synthetic, depends on #2.
6. Stage-1 + Stage-2 Marin StepSpec wrappers (Tasks #9, #10) — the pure-Python code is done; just plumbing into Marin's executor model.
7. Eval arms (Task #7) — depends on a trained model.

**Files dropped on disk this session that aren't in git** (so future agent doesn't confuse them):
- `/tmp/ifbench_port/port_venv/` — ephemeral venv with verifier deps; `pip install nltk langdetect immutabledict emoji syllapy unicodedata2 absl-py "setuptools<81" pytest httpx together datasets pyarrow` to rebuild.
- `/tmp/ifbench_port/upstream_venv/` + `/tmp/ifbench_port/ifbench_upstream/` — used for the parity comparison; can delete.
- `/tmp/ifbench_port/if_train.parquet` — 72MB cached parquet, can re-download from HF.
- `/tmp/ifbench_port/prepared_v2/{train.jsonl, val.jsonl}` — the 93k+2k split.
- `/tmp/ifbench_port/smoke/{rollouts_70b.jsonl, rollouts_8b_sync.jsonl, rollouts_gemini_sync.jsonl, handles.json, outputs/}` — smoke artefacts.
- `/tmp/ifbench_port/diag_batches.py`, `/tmp/ifbench_port/peek_batches.py`, `/tmp/ifbench_port/test_keys.py` — small one-off diagnostics, can delete.

**Use `TaskList` in a session to see live state** — logbook is the durable record, not the live tracker.

**To re-run the verifier parity test (HARD GATE) from scratch:** see "How to re-run the verifier parity test" section below.

---

## Scope

- **Goal:** Train an 8B-class LLM that follows multi-constraint instructions, using DPO preference pairs + verifier-quality-filtered SFT data in a single mixture trainer (per-batch DPO loss + α·CE loss, single scalar per step).
- **Primary metrics:** IFEval prompt-level strict + loose accuracy (in-domain), IFBench_test prompt-level strict accuracy (OOD generalization). Secondary: at least one general-capability eval (MMLU/AlpacaEval) to catch regression.
- **Constraints:**
  - Marin compute is preemptible — checkpoint frequently, accept multi-region launches, never assume single-shot completion.
  - All real-money pipelines (anything > a few dollars) get a smoke run first and explicit dollar sign-off before scaling.
  - No reasoning or low-tier reasoning on any LM-in-the-loop call (per memory rule for alignment work; not load-bearing here yet but applies if we ever add LM-as-judge).
  - Don't chase Bug-1 pathology in mixture experiments.

## Baseline

- **Date:** Not yet established.
- **Code refs:** Existing DPO stack on this branch (see "Existing primitives" below).
- **Baseline numbers:** TODO. Plan: run `Tülu-3-8B-SFT` (or whatever SFT init we land on) zero-shot through IFEval and IFBench_test before any training. That floor is what every arm has to beat to claim progress.

---

## Existing primitives we reuse

| Piece | Path | Notes |
|---|---|---|
| Preference example | `lib/levanter/src/levanter/data/text/preference.py::DpoExample` | `chosen: LmExample`, `rejected: LmExample` |
| Preference dataset | `lib/levanter/src/levanter/data/text/preference.py::PreferencePairDataset` | Greedy-packed, applies `mask_user_turns` via `assistant_masks` |
| DPO model wrapper | `lib/levanter/src/levanter/dpo.py::DpoModel` | `policy` (trainable) + `reference` (frozen) |
| DPO loss | `lib/levanter/src/levanter/dpo.py::dpo_loss_from_logps` | `-log_sigmoid(beta * (delta_pi - delta_ref))` |
| Cached reference logprobs | `lib/levanter/src/levanter/dpo.py::CachedReferenceDataset` | Lets us drop the frozen ref model from the step |
| LoRA-DPO entrypoint | `lib/levanter/src/levanter/main/lora_dpo.py` | LoRA-only path; merge to full HF on export |
| SFT loss masking | `data/text/preference.py` already emits `assistant_masks`; `data/text/formats.py` parses chat templates | Same masking machinery should work for plain SFT |
| vLLM inference | `lib/marin/src/marin/inference/vllm_server.py` | Used for rollout generation |
| Marin pipeline orchestration | `marin.execution.step_runner.StepRunner`, `StepSpec` | Wraps each pipeline stage |

**Open question on existing infra:** `lib/levanter/src/levanter/main/sft.py` is currently empty. Either SFT training lives elsewhere (e.g., inside `train_lm.py` with chat-tokenized data) or it hasn't been written. Resolving this gates the mixture trainer design — see Decision D-001 below.

---

## Decision Log

Append-only. Each entry: decision, evidence, date, status.

### D-001 — SFT entrypoint location *(closed 2026-04-26)*
- **Resolution:** SFT is implemented at the **marin layer**, not in levanter. The empty `lib/levanter/src/levanter/main/sft.py` is misleading but not load-bearing.
- **Where it actually lives:**
  - Entry function: `experiments.defaults.default_sft(name, tokenized, model_config, sft_config)` at `experiments/defaults.py:601`.
  - Config: `experiments.simple_sft_config.SimpleSFTConfig` at `experiments/simple_sft_config.py:55`.
  - Data format: `levanter.data.text.ChatLmDatasetFormat` (used by callers like `experiments/exp808_sft_mixture.py`).
  - Under the hood, `default_sft` shells out to `default_train` (the same path as plain LM training) with `use_default_validation=False`. So **SFT and pretraining share the same Levanter trainer**; the only differences are the dataset format and the validation-loop wiring.
- **What this means for the mixture trainer:**
  - DPO has its own entrypoint (`experiments.defaults.default_dpo` at line 676; uses `levanter.main.train_dpo`) because of the frozen reference model.
  - SFT just goes through `default_train` with a chat-formatted dataset — no special trainer.
  - **Cleanest path for D-002:** extend `SimpleDPOConfig` with optional `sft_data` + `alpha_sft` fields. When `sft_data` is `None`, behavior is unchanged (pure DPO); when set, the trainer samples a parallel SFT batch and adds α·CE to the loss. No new entrypoint needed.
  - The actual loss combination has to land in `levanter.main.train_dpo` (or a new `train_dpo_sft`) since `default_train` doesn't know about chosen/rejected.

### D-002 — Mixture example representation *(tentative)*
- **Decision (tentative):** Two parallel batches (sample DPO and SFT batches independently each step, compute both losses, weighted sum). Revisit if we need fine-grained per-example mixing.
- **Alternatives considered:**
  - Tagged union (`kind: Literal["dpo", "sft"]` on a single dataclass) — harder to jit because shapes differ; per-example dispatch in a jit'd loss is ugly.
- **Evidence:** Parallel batches keep DPO shapes (2× sequence length per example) and SFT shapes (1× sequence length) cleanly separated.
- **Status:** Tentative pending implementation sketch in Task #2.
- **Date:** 2026-04-26

### D-003 — Loss combination shape *(tentative)*
- **Decision (tentative):** `loss = dpo_loss + α · sft_loss`, with explicit token-count normalization on the SFT side (so α is interpretable as a per-example weight).
- **Alternatives:** scale-aware dynamic balancing; logit-level mixing.
- **Status:** Tentative pending experiment ablation (α ∈ {0.25, 1.0, 4.0}).
- **Date:** 2026-04-26

### D-004 — Reference model handling for SFT examples *(open)*
- **Question:** SFT loss does not need the reference model. With cached reference logprobs already supported, the frozen ref is only loaded for DPO batches. Need to verify the SFT path doesn't accidentally trigger ref-model materialization.
- **Status:** Verify at implementation time.
- **Date opened:** 2026-04-26

### D-005 — Don't use THU-KEG/IFBench as DPO training data *(closed)*
- **Decision:** Use AI2's IFBench rollout pipeline (we generate the pairs); do NOT use the 444-example THU-KEG/IFBench as training data.
- **Evidence (full analysis in "Datasets considered" below):**
  - 444 pairs is ~70× too small for DPO to do anything but memorize.
  - Both chosen and rejected come from a single model (GPT-4o), removing the cross-model diversity that makes verifier-pair construction work.
  - Half the constraints (Style, Content) are LM-judged, not programmatic. Defeats the entire reason we wanted verifier data.
  - It's an *evaluation* benchmark for reward models by design — using it as training data is a contamination foot-gun.
  - The DPO training data the THU-KEG paper actually used is unreleased UltraFeedback/on-policy pairs scored by their RewardAgent; not this dataset.
- **Mitigation:** Score our trained model on the 444 chosen/rejected pairs as one more eval signal. No training use.
- **Date:** 2026-04-26

### D-006 — v1 prompt source: `allenai/IF_multi_constraints_upto5` *(tentative)*
- **Decision:** Use AI2's IF-RLVR training set as the v1 source, restricted to single-turn for v1.
- **Confirmed exact size:** 95,373 rows. Paper says they "create about 60k–100k prompts" — 95,373 is the specific canonical release.
- **Construction (per AI2 paper §3):** TÜLU-3-SFT prompts + 1–5 constraints sampled from IFTrain (29 new) ∪ IFEval taxonomy. Prompt-constraint contradictions filtered via a constraint-conflict dictionary.
- **Rationale:** Largest open IF training set with structured constraints + open verifier code + a real held-out OOD eval.
- **Status:** Tentative pending smoke run yield.
- **Date:** 2026-04-26

### D-013 — Train/val/test split design *(tentative)*
- **No published validation set.** AI2's IFBench collection has only train (`IF_multi_constraints_upto5`, 95k) + test (`IFBench_test`, 300; `IFBench_multi-turn`, 3.16k). We carve our own val.
- **Decision:**
  | Split | Source | Size | Use |
  |---|---|---|---|
  | Train | `IF_multi_constraints_upto5` − val slice | ~93,373 | Rollout pool input → DPO + SFT data |
  | Val | Stratified hold-out from `IF_multi_constraints_upto5` (by `constraint_type`) | ~2,000 | Per-constraint yield monitoring, verifier reproducibility, eventual DPO eval-loss. Never trained on. |
  | Test (in-domain) | IFEval | ~500 | Final report; do not tune to. |
  | Test (OOD, primary) | `allenai/IFBench_test` | 300 | **Headline generalization number. NEVER load during training or data prep — only at eval time.** |
  | Test (OOD, multi-turn) | `allenai/IFBench_multi-turn` | 3,161 | Out of v1 scope; reserved untouched. |
- **Contamination guarantees (per AI2 paper §2):**
  1. **Different prompt corpora**: train from TÜLU-3-SFT; test from held-out WildChat ("held out from release" = not in any public SFT mixture).
  2. **Disjoint constraint sets**: 29 IFTrain constraints vs 58 IFBench constraints; explicit paper guarantee.
- **Programmatic verification:** before any rollout, compare exact prompt strings between our 95k train and the 300 IFBench_test prompts. Report overlap. Expect 0 by construction; if not, surface as a contamination bug.
- **Hard implementation rule:** `allenai/IFBench_test` is referenced only in the eval harness. It must not be importable from the rollout pipeline, verifier validation, trainer, or data-prep code. Wire as a config-level guard.
- **Date:** 2026-04-26

### D-007 — v1 rollout pool: 3-model strong/mid/weak spread *(tentative; see "Rollout pool plan" below)*
- **Decision (tentative):** 3 models, one per tier, no diversity-for-its-own-sake slot.
  - **Strong:** `gemini-3-flash-preview` with `thinkingLevel="high"` (Gemini batch backend)
  - **Mid:** `meta-llama/Llama-3.3-70B-Instruct-Turbo` (Together batch backend)
  - **Weak:** `meta-llama/Meta-Llama-3-8B-Instruct-Lite` (Together batch backend)
- **Sampling:** T=0.7, top_p=0.95, max_new_tokens=1024. Deterministic seeds per (prompt_id, model_id, rollout_idx) where the backend supports them (Together yes, Gemini no — recorded in metadata).
- **Rationale (pass-rate spread, not stylistic diversity):**
  - User explicitly wants strong/mid/weak spread — wide gaps between tiers, not stylistic variation.
  - AI2 Table 4 baseline: Qwen-72B 26%, Llama-3.1-405B 21%, Tülu-3-70B 15%, Yi-34B-Chat 10%, Llama3-8B 6% on multi-constraint prompts.
  - **Strong (~30–40% expected):** Gemini-Flash with reasoning should cleanly beat AI2's Qwen-72B on multi-constraint prompts since reasoning matters disproportionately for planning 3–5 simultaneous constraints.
  - **Mid (~15–25% expected):** Llama-3.3-70B-Turbo. AI2's Tülu-3-70B got 15%; Llama-3.3 is slightly newer/better. Solid mid that doesn't blur into the strong tier (no reasoning).
  - **Weak (~5–10% expected):** Llama-3-8B-Lite is the AI2 paper's exact 6%-pass-rate model; cheapest reliable rejected-producer.
- **Dropped:** the diversity slot (DeepSeek-V3.1), Qwen2.5-7B-Turbo (redundant given the spread). Mistral never reappeared on Together's serverless catalog.
- **4-model fallback:** if smoke pair yield < 30%, insert `Qwen/Qwen2.5-7B-Instruct-Turbo` as a second mid. Adds ~$10. Don't pre-commit.
- **Status:** Tentative — confirm `gemini-3-flash-preview` is on user's GEMINI_API_KEY (Task #14), confirm Llama-3.3-70B-Turbo + Llama-3-8B-Lite are on user's TOGETHER_API_KEY (Task #12).
- **Date:** 2026-04-26

### D-008 — Drop Llama-3.1-405B from v1 *(closed)*
- **Decision:** No 405B in v1. Revisit only if pair yield from the 4-model pool < 30%.
- **Evidence:** AI2 Table 4 shows 405B at 21% all-correct, behind Qwen-72B's 26%. Serving cost would dominate the budget for a marginal-pass-rate win.
- **Date:** 2026-04-26

### D-009 — Use Together AI Batch API for rollouts (instead of vLLM-on-TPU) *(tentative)*
- **Decision:** Generate rollouts via Together's hosted serverless inference, using the Batch API (50% discount, 24-hour async). User has an existing Together API key.
- **Why pivot:** Removes all vLLM-on-TPU compatibility risk; removes the "is this model available on TPU" question; cuts cost roughly in half via the batch discount; fully async fits our workload.
- **Hard exclusion:** DeepSeek-R1 / R1-0528 family (reasoning model where reasoning is inseparable from the model — risk profile differs from the targeted Gemini override in D-010).
- **Open dependency:** Confirm model availability via `curl https://api.together.xyz/v1/models`. Mistral may not be on serverless any more (pricing page showed none).
- **Status:** Tentative pending the /v1/models response.
- **Date:** 2026-04-26

### D-010 — Use Gemini 3 Flash + `thinkingLevel="high"` as the strong slot *(tentative)*
- **Decision:** `gemini-3-flash-preview` with `thinkingLevel="high"`, issued via the Gemini Batch API (50% discount).
- **Why Flash, not Pro:** Pro is ~5× more expensive; Flash is the price-performance sweet spot Google explicitly markets for high-volume reasoning workloads.
- **Why thinking=high:** Reasoning matters disproportionately on multi-constraint prompts (model has to plan to satisfy 3–5 simultaneous constraints). This is the strong slot of the strong/mid/weak spread; we want it to actually be strong.
- **Memory note:** `feedback_no_reasoning_for_alignment_project.md` is correctly scoped to the spec-driven alignment / executable-specifications project only — does **not** apply here. No override needed.
- **Pricing (confirmed from docs):** Gemini 3 Flash batch tier — $0.25/M input, $1.50/M output (thinking tokens billed as output).
- **Cost share:** Gemini will dominate the rollout budget (~91% of full-run cost) because thinking=high inflates output token counts to ~2500 avg vs ~500 for non-reasoning models.
- **Status:** Tentative pending smoke run (Task #4) cost validation.
- **Date:** 2026-04-26

### D-011 — Unified `RolloutBackend` protocol over Together + Gemini *(tentative)*
- **Decision:** Abstract the per-provider batch flow behind a `RolloutBackend` protocol with methods like `submit_batch(jsonl_path) -> batch_id`, `poll(batch_id) -> Status`, `download(batch_id) -> Iterable[Rollout]`. One backend per provider. No mixing of providers in a single batch.
- **Schema mapping:**
  - Together: `{"custom_id": "...", "body": <chat-completions>}` (max 64 char custom_id)
  - Gemini: `{"key": "...", "request": <GenerateContentRequest>}`
- **Custom-id schema (uniform):** `f"{prompt_id}:{model_id_short}:{seed}"` truncated/hashed to fit Together's 64-char cap if needed.
- **Wait-on-all merge step:** the verifier scoring stage requires all 5 batches to be downloadable. If any batch fails terminally, we proceed with the partial rollout set and record the missing model in metadata; partial pair yield is acceptable.
- **Date:** 2026-04-26

### D-012 — Cost target: ~$393 for full 95k × 3-model run *(tentative)*
- **Decision:** Realistic full-run cost: ~$393. Smoke run cost: <$0.50.
- **Breakdown (batch pricing, full 95k run):**
  - Gemini 3 Flash + thinking=high: ~$360 (~91% of total)
  - Llama-3.3-70B-Turbo: ~$30
  - Llama-3-8B-Lite: ~$3
- **Why no DeepSeek / Qwen-7B:** dropped per D-007 — user explicitly wants strong/mid/weak spread, not stylistic diversity.
- **Hard gate:** I do not launch the full run without re-quoting the actual figure post-smoke and getting an explicit "yes go". Per `feedback_ambiguous_consent` memory rule.
- **Date:** 2026-04-26

---

## Open Questions

1. **(D-001)** Where does SFT live in this repo? Need to read `experiments/exp808_sft_mixture.py`, `experiments/exp1880_sft_baseline.py`, and `tests/test_sft.py` to figure out if there's a de facto SFT entrypoint we missed.
2. **(D-004)** Does the cached-reference-logprobs path correctly skip ref-model materialization on SFT-only batches?
3. **Mixture ratio at the dataset level vs the loss level.** If we use parallel batches, do we want the DPO and SFT batches to be the same size, or independently sized? (Probably independently sized, with α controlling the loss-weighting and dataset sizes controlling the *amount* of each signal seen per epoch.)
4. **Token budget under packing.** DPO examples are 2× sequence length per example. With greedy packing, how do we keep wall-clock per step comparable across DPO-only / SFT-only / mixture configurations?
5. **Verifier port reproducibility.** What exact published number do we target on IFBench_test for "verifier port is correct" (±1pt)? Need to pick a model whose number we trust.

---

## Datasets considered

### AI2 `allenai/IF_multi_constraints_upto5` (95k) — *adopted as v1 prompt source*
- 95k single-turn prompts with 1–5 sampled constraints each.
- Columns: `messages`, `constraint`, `constraint_type`, `ground_truth`.
- 29 training-time constraints in `allenai/open-instruct/open_instruct/IFEvalG/`.
- 58 held-out OOD constraints in `allenai/IFBench_test` (300 prompts, used as eval).
- All verifiers are open Python — no LM-as-judge.

### AI2 `allenai/IFBench_test` (300) — *adopted as primary OOD eval*
- 300 prompts with the held-out constraints. Headline generalization metric.
- Multi-turn variant (`allenai/IFBench_multi-turn`, 3.16k) deferred — single-turn first.

### Google IFEval (~500) — *adopted as in-domain eval*
- Original 25 verifiable-instruction categories, prompt-level strict + loose accuracy.

### THU-KEG `THU-KEG/IFBench` (444) — *rejected as training data; usable as RM eval*
- **Source paper:** "Agentic Reward Modeling" (Peng et al., Tsinghua, Feb 2025, arXiv:2502.19328). Naming collides with AI2's IFBench despite being a completely different benchmark.
- **What it actually is:** A 444-example reward-model eval benchmark, not a training set. Each row has chosen/rejected GPT-4o samples with constraint annotations.
- **Why rejected as training data:**
  1. **Way too small.** 444 pairs vs the ~31k AI2 used or the ~95k prompt source we're targeting.
  2. **Single-model contamination.** Both chosen and rejected come from GPT-4o at T=1.0. Same-model preference pairs are notoriously low-signal.
  3. **Half the constraints are LM-judged.** `llm_constraints_used` (Style, Content) ship with no `check_following` function — they were verified during construction by GPT-4o judging itself. The whole point of verifier-based data is to escape LM-as-judge; this defeats it.
  4. **Pipeline fully GPT-4o-driven:** GPT-4o generates constraints → paraphrases instructions → filters contradictions → samples 8 candidates → judges them.
  5. **Designed as an *eval*.** Using it as training data while reporting on it is contamination.
- **Did the paper run SFT or DPO?** DPO only (Section 5.2). They DPO'd `zephyr-7b-sft-full` on UltraFeedback prompts and zephyr-on-policy rollouts re-scored by their RewardAgent. **The DPO training pairs are NOT released.** Best DPO IFEval = 58.2 (RewardAgent-OP) vs 43.3 baseline; modest gains on a small backbone. No SFT-on-passers experiment.
- **Useful for:** Scoring our trained mixture model on these 444 pairs as one more eval signal (does the model prefer chosen over rejected? bonus diagnostic, not a primary metric).

### THU-KEG `IF-Verifier-Data` (131k) — *rejected, wrong shape*
- 131k single instruction-response pairs with QwQ-32B step-by-step verification annotations.
- For training a *generative verifier model*, not for DPO/SFT of a policy. Mixed English/Chinese.

### UltraFeedback (~64k) — *holding pattern*
- Backup if AI2 IFBench yield is too low. Standard preference dataset, GPT-4 annotated.

---

## Rollout pool plan (v1)

### Evidence from the AI2 paper (arXiv:2507.02833, Table 4)

Per-model "All Correct" rate on their multi-constraint prompts:

| Model | All Correct | Notes |
|---|---|---|
| Qwen-72B | **26%** | Strongest. Likely Qwen2-72B-Instruct era; we substitute Qwen2.5-72B-Instruct for v1. |
| Llama-3.1-405B | 21% | Marginal gain over Qwen-72B at much higher cost. **Dropped.** |
| Tülu-3-70B | 15% | Solid mid-tier strong model. Dropped (Qwen-72B is enough). |
| Yi-34B-Chat | 10% | Older arch; redundant w/ our 8B and 72B. **Dropped.** |
| Llama3-8B | **6%** | **Cheapest reliable fail-generator** — we keep this slot. |

Paper also notes "most LLMs get the same easy instances right and the same hard instances wrong" — confirms the yield-collapse risk. The cross-model diversity helps but doesn't eliminate it; some prompts will yield no pairs.

### v1 pool — see "v1 pool (3 models — strong/mid/weak spread)" subsection below for the current decision

(An earlier 4-model pool was superseded by D-007. The strong/mid/weak spread is the canonical version.)

**Notably absent (still true):** our own SFT init / policy model. Cross-distribution preference signal is the point — including the policy model in the rollout pool is iterative DPO, deferred to v2.

### Inference backends: Together + Gemini Batch APIs

**Pivoted from vLLM-on-TPU.** See D-009 (Together) and D-010 (Gemini). Rollouts go to two providers; both have batch APIs at 50% discount with 24-hour async SLAs. We keep a `RolloutBackend` protocol (D-011) so the rest of the pipeline doesn't care which provider produced a given rollout.

| | Together | Gemini (Google AI Studio) |
|---|---|---|
| Auth | `Authorization: Bearer $TOGETHER_API_KEY` | `x-goog-api-key: $GEMINI_API_KEY` |
| Batch endpoint | `POST /v1/batches` (after Files API upload) | `:batchGenerateContent` (or Files API for >20MB) |
| Input format | `.jsonl`: `{"custom_id": "...", "body": <chat-completions>}` | `.jsonl`: `{"key": "...", "request": <GenerateContentRequest>}` |
| Custom-id field | `custom_id` (max 64 chars) | `key` |
| Per-batch limits | 50k requests, 100MB file, 30B tokens enqueued per model | 2GB input file |
| SLA | "Best-effort" 24 hours | "Target 24 hours, often much quicker" |
| Discount | 50% vs sync (Llama-3.3-70B, Qwen2.5-7B confirmed) | 50% vs sync (all batch-supported Gemini models) |

### v1 pool (3 models — strong/mid/weak spread)

| Slot | Model ID | Backend | Reasoning | Batch input $/M | Batch output $/M | Expected pass-rate |
|---|---|---|---|---|---|---|
| **Strong** | `gemini-3-flash-preview` | Gemini | `thinkingLevel="high"` | $0.25 | $1.50 (incl. thinking) | ~30–40% |
| **Mid** | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Together | — | ~$0.44 | ~$0.44 | ~15–25% |
| **Weak** | `meta-llama/Meta-Llama-3-8B-Instruct-Lite` | Together | — | ~$0.05 | ~$0.05 | ~5–10% |

**4-model fallback** *(only if smoke yields pair rate < 30%)*: insert `Qwen/Qwen2.5-7B-Instruct-Turbo` as a second mid (~$0.15/M batch). Adds ~$10 to the full-run budget.

**Pre-commit gate (per backend):**
- **Together:** confirm `meta-llama/Llama-3.3-70B-Instruct-Turbo` and `meta-llama/Meta-Llama-3-8B-Instruct-Lite` are listed in `/v1/models` for the user's key. Verify Llama-3-8B-Lite's 8K context covers prompt + 1024-token output (it should, but check on a real prompt).
- **Gemini:** confirm `gemini-3-flash-preview` is enabled on user's `GEMINI_API_KEY` and that the key has Batch API access. 1-prompt sync test before any batch submission.

### Sampling settings (uniform across backends)

| Param | Value | Notes |
|---|---|---|
| Temperature | 0.7 | Need diversity so hard constraints sometimes pass and easy ones sometimes fail. |
| top_p | 0.95 | Standard. |
| max_new_tokens | 1024 | IFBench prompts often want long-form responses; Length constraints sometimes ask for word counts in the hundreds. |
| K rollouts per (prompt, model) | 1 | Cross-model diversity does the work; same-model resampling is mostly redundant. |
| Seed scheme | `hash((prompt_id, model_id, rollout_idx))` | Deterministic; reruns produce identical output for Together. Gemini does not currently expose seeded sampling, so its reruns are non-identical — note this in metadata. |
| **Gemini extra:** `thinkingLevel` | `"high"` | See D-010. Memory rule `feedback_no_reasoning_for_alignment_project` is correctly scoped to the executable-specifications project and does not apply here. |

### Caching scheme

- Output of Stage 2 (rollouts) keyed by `(prompt_set_version, model_id, sampling_config_hash, seed)`. Marin's StepRunner content-addresses naturally; no extra work needed.
- Stage 3 (verifier scoring) keyed on rollout cache + verifier_version. So a verifier bug-fix doesn't re-trigger generation.
- Both DPO and SFT caches in Stage 4 are derived from Stage 3 output — both are downstream of one expensive rollout run.

### Smoke run spec (Task #4)

| | |
|---|---|
| **Prompts** | First 100 single-turn rows from `allenai/IF_multi_constraints_upto5` |
| **Models** | All 3 v1 pool models (1 Gemini + 2 Together). Each gets its own batch submission. |
| **Sampling** | T=0.7, top_p=0.95, max=1024, K=1, seeded; Gemini gets `thinkingLevel="high"` |
| **Total generations** | 300 |
| **Estimated cost** | <$0.50 (~$0.40 Gemini + ~$0.05 Together) |
| **Stages exercised** | 1, 2 (both backends), 3, 4a, 4b — full end-to-end |
| **Eval** | Skip; we don't train anything from smoke output |

### Smoke success criteria

1. **Pipeline runs end-to-end.** Output: a non-empty DPO cache + non-empty SFT cache.
2. **Pair yield ≥ 30%.** At least 30 of 100 prompts produce a usable (chosen ∈ passes_all, rejected ∈ fails_≥1) pair. Below this we have a constraint-mix problem — fix before scaling.
3. **Verifier port self-check passes.** ✅ DONE (2026-04-26). Verifier port reproduces upstream IFBench's `run_eval.py` numbers byte-for-byte against `sample_output.jsonl`. See "How to re-run the verifier parity test" + Experiment Log "FINISHED" entry for re-run procedure.
4. **No model produces >90% empty / refusal responses.** If a model is misconfigured (wrong chat template, etc.), catch it here.

### Cost estimation (pre-smoke, batch pricing)

Per-prompt assumptions: ~200 input tokens; ~500 output tokens (non-reasoning) or ~2500 output tokens (Gemini Flash with thinking=high, since thinking tokens are billed as output).

| Model | Cost / 95k prompts |
|---|---|
| `gemini-3-flash-preview` + thinking=high | **~$360** (~91% of total) |
| `Llama-3.3-70B-Turbo` | ~$30 |
| `Meta-Llama-3-8B-Lite` | ~$3 |
| **Full run total (rough)** | **~$393** |
| **Smoke total (100 × 3 models)** | **<$0.50** |

If the 4-model fallback triggers (Qwen-7B added as second mid), full-run becomes ~$403.

**Hard gate:** Smoke run measures actual tokens/output for Gemini under thinking=high (this is the biggest unknown — the rough estimate could easily be off by 1.5–2× either way). Re-quote the full-run cost from real smoke data and get explicit "yes go" before the full run. Per `feedback_ambiguous_consent`.

---

## Data pipeline plan (current)

```
                      ┌──────── pass-only rollouts ──────► SFT cache
prompts + constraints │                                    (chat tokenized)
  ───────────────────▶│ rollout pool ─► verifier scoring ─┤
  (IF_multi_*)        │                                    │
                      └──── pair (pass, fail) ────────────► DPO cache
                                                            (PreferencePairDataset)
```

### Stage 1 — Prompt set assembly
- Source: `allenai/IF_multi_constraints_upto5` (95,373 rows confirmed). Single-turn only for v1.
- **Carve val split** (per D-013): stratified 2,000-row hold-out by `constraint_type`. Train pool becomes ~93,373.
- **Contamination check** (per D-013): before producing any output, load `IFBench_test`'s 300 prompts and assert zero exact-string overlap with our train+val. Fail loudly if any overlap.
- Wrap as a Marin `StepSpec` under `experiments/ifbench/`.
- Output: two jsonl artifacts (`train.jsonl`, `val.jsonl`), each with `{prompt_id, messages, constraint, constraint_type, ground_truth}`.

### Stage 2 — Rollout pool
- 3-model pool per D-007 (Gemini batch + Together batch).
- T=0.7, K=1 rollout per (prompt, model). Deterministic seeds where supported.
- Smoke target: 100 prompts (drawn from train, not val) × 3 models. Measure tokens/sec, real cost, pair yield.

### Stage 3 — Verifier scoring
See "Verifier vendor plan" section below for the full file-by-file port spec. Summary:
- Vendor IFBench (58 OOD constraint classes + scorer) + IFEvalG (25 IFEval + 29 IFTrain). Skip Google's original IFEval (IFEvalG is a superset). 112 verifier classes total.
- Strict-only for pair construction; strict + loose for final eval.
- **Validation gate:** reproduce IFBench's `sample_output.jsonl` reference numbers exactly using our ported scorer. Hard gate; nothing else proceeds until this passes.

### Stage 4a — DPO pair construction
- For each prompt: pick one (chosen, rejected) from passes_all × fails_≥1.
- Skip prompts with empty pass-set or fail-set; track skip rate.
- Tokenize via `PreferencePairDataset`.

### Stage 4b — SFT cache construction
- Every passing rollout becomes one SFT example (prompt, response).
- Tokenize through the chat-template path used in `data/text/formats.py`.
- Auto-quality-filtered: every example provably satisfies its constraint.

### Stage 5 — Wire into mixture trainer
- Both caches as `TreeCache` artifacts.
- Mixture trainer consumes both, applies per-batch `dpo_loss + α · sft_loss`.
- Metadata records verifier version + rollout pool snapshot for reruns.

### Pipeline-level risks
- **Yield collapse on easy/hard prompts.** Constraints like `english_lowercase` will pass on every model; obscure ones may fail on every model. Skip rate measured at smoke time.
- **Verifier-port drift.** Pin verifier version + checksum a fixed test set in CI.
- **Chat-template drift across the rollout pool.** Each model has its own chat template; assistant-mask computation must use each model's template at rollout time, then re-tokenize under the *training* tokenizer (`marin_tokenizer`) when building caches.
- **Cost.** 93k × 3 = ~280k generations across hosted APIs; rough $393 with batch pricing per D-012. Smoke first, sign-off before scaling.
- **No published validation set.** Per D-013 we carve a 2k stratified val out of the 95k train ourselves; treat IFBench_test as never-loadable except in the eval harness.

---

## Verifier vendor plan

> **Status: IMPLEMENTED 2026-04-26.** The plan below is preserved as the design record. For what actually shipped + the live state, see the FINISHED entry in the Experiment Log. For re-running the parity gate, see "How to re-run the verifier parity test" further down.

### Inventory (what's where, confirmed by reading source)

Three relevant upstream repos:

| Repo | Files we'd touch | What's in the registry | Used for |
|---|---|---|---|
| `allenai/IFBench` | `instructions.py` (78KB), `instructions_registry.py` (4.5KB), `instructions_util.py` (24KB), `evaluation_lib.py` (7KB) | **58 OOD test constraints** (e.g. `count:word_count_range`, `format:options`, `custom:csv_city`) | Scoring `IFBench_test` (300) and `IFBench_multi-turn` (3.16k, deferred) |
| `allenai/open-instruct` → `open_instruct/IFEvalG/` | `instructions.py` (97KB), `instructions_registry.py` (16KB), `instructions_util.py` (26KB) | **25 IFEval + 29 IFTrain = 54 train-eligible constraints** (e.g. `keywords:existence`, `length_constraints:number_words`, `last_word:last_word_answer`) | Verifying our 95k training rollouts AND scoring Google's IFEval (~500) |
| `google-research/google-research` → `instruction_following_eval/` | `instructions_registry.py` etc. | 25 original IFEval constraints — **strict subset of IFEvalG** | **Skip — IFEvalG already contains all 25** |

**Net constraint coverage we need: 58 + 54 = 112 classes.** No overlap between IFBench and IFEvalG namespaces (prefixes differ — IFBench uses `count:`, `ratio:`, `words:`, `custom:`, etc.; IFEvalG uses `keywords:`, `length_constraints:`, `detectable_format:`, `paragraphs:`, `last_word:`, etc.).

### Train-data ground_truth schema (verified)

Each row of `IF_multi_constraints_upto5` has a `ground_truth` field that's a stringified Python list, e.g.:

```python
[{'instruction_id': ['detectable_format:sentence_hyphens', 'last_word:last_word_answer'],
  'kwargs': [None, {'last_word': 'brief'}]}]
```

So the `(instruction_id_list, kwargs)` tuple expected by the verifier registry comes for free — parse with `ast.literal_eval` (note: Python repr style, not JSON, so `json.loads` won't work). Then it's fed to the verifier the same way as `IFBench_test` rows.

**Important:** the training data uses **only IFEvalG IDs** (54 constraint types). It does NOT use IFBench's 58 test constraints — those are held out from training by paper construction. We could in principle skip IFBench's verifiers in the rollout-scoring path, but we still need them for IFBench_test eval.

### Files we vendor (and why)

```
experiments/ifbench/verifiers/
├── __init__.py             # exports unified INSTRUCTION_DICT (union of both)
├── _versions.txt           # pinned upstream commit SHAs + checksums
├── ifbench/                # vendored from allenai/IFBench@<sha>
│   ├── __init__.py
│   ├── instructions.py     #  ←  58 IFBench OOD checker classes
│   ├── instructions_registry.py
│   └── instructions_util.py
├── ifevalg/                # vendored from allenai/open-instruct@<sha>:open_instruct/IFEvalG/
│   ├── __init__.py
│   ├── instructions.py     #  ←  25 IFEval + 29 IFTrain checker classes
│   ├── instructions_registry.py
│   └── instructions_util.py
├── scoring.py              # ←  PORTED from IFBench/evaluation_lib.py (strict + loose)
├── parse.py                # ←  NEW: ground_truth (Python repr) → (instruction_id_list, kwargs)
└── tests/
    └── test_reference_parity.py  # ←  validates port against IFBench's sample_output.jsonl
```

**Vendor and not modify** the upstream files except for the import-path fixes below. All bug fixes go upstream; we re-pull when they're released.

### Files we do NOT vendor

| Upstream file | Why skip |
|---|---|
| `IFBench/run_eval.py` | CLI wrapper; we'll write our own that integrates with our pipeline. |
| `IFBench/generate_responses.py` | Their inference code; we use Together/Gemini batch APIs instead. |
| `IFBench/instructions_test.py` (66KB) | Their unit tests; we write our own narrow validation against `sample_output.jsonl`. |
| `IFBench/config.py` | Their inference config; not relevant to scoring. |
| `IFBench/data/IFBench_test.jsonl` | We get this from `allenai/IFBench_test` HF dataset, with version pinning. |
| `IFBench/data/sample_output.jsonl` | Used at port-validation time only — not vendored, downloaded once into a test fixtures dir. |

### Import-path rewrites (the only modifications to vendored code)

Both upstream files use ambient/non-relative imports that won't work in our package layout:

| File | Current import | Rewrite to |
|---|---|---|
| `IFBench/instructions.py` | `import instructions_util` | `from . import instructions_util` |
| `IFBench/instructions_registry.py` | `import instructions` | `from . import instructions` |
| `IFBench/evaluation_lib.py` | `import instructions_registry` | (we don't vendor this verbatim — see scoring.py below) |
| `IFEvalG/instructions_registry.py` | `from open_instruct.IFEvalG import instructions` | `from . import instructions` |
| `IFEvalG/instructions.py` | (similar pattern, check) | `from . import instructions_util` |

These are mechanical sed edits applied at vendor-time. We script the import-rewrite so re-pulling from upstream is one command.

### `scoring.py` — the ported scorer

`evaluation_lib.py` is small (7KB) and calls into the registry; we re-implement it with a small twist — it accepts the registry as a parameter, so the same scorer works for both IFBench and IFEvalG inputs:

```python
def test_instruction_following_strict(inp, response, registry: dict[str, type]) -> OutputExample: ...
def test_instruction_following_loose(inp, response, registry: dict[str, type]) -> OutputExample: ...
```

The unified `__init__.py` exports `INSTRUCTION_DICT_ALL = {**ifbench.INSTRUCTION_DICT, **ifevalg.INSTRUCTION_DICT}` so by default we just use the union. (Disjoint keys; no collision.)

### Dependency surface

Vendored verifiers need:

| Dep | Used by | Notes |
|---|---|---|
| `nltk` | sentence/word tokenization | IFBench autodownloads to local `.nltk_data` dir; we keep that pattern |
| `langdetect` | `language:response_language` checker | |
| `immutabledict` | various | |
| `emoji` | `format:emoji` checker | |
| `syllapy` | `words:odd_even_syllables` | |
| `unicodedata2` | non-ASCII handling | stdlib `unicodedata` may suffice on Python 3.11+; check |
| `absl-py` | only for the original CLI | **drop — we don't vendor `run_eval.py`** |

`pyproject.toml` of IFBench also lists `spacy` but I confirmed by inspecting the imports that `instructions.py` does NOT actually import spaCy — likely a leftover dep. **Drop it.** Big win — spaCy + a model would be ~500MB.

Add to our project's `pyproject.toml`: `nltk`, `langdetect`, `immutabledict`, `emoji`, `syllapy`. ~25MB total install.

### Validation gate (this is the real test)

IFBench ships `data/sample_output.jsonl` (624KB) — an example output file plus `IFBench_test.jsonl` (421KB). Their `run_eval.py` produces a known accuracy number on this pair.

**Port validation procedure:**

1. Download both files from `allenai/IFBench@<pinned-sha>:data/`.
2. Run `python -m experiments.ifbench.verifiers.scoring --input IFBench_test.jsonl --responses sample_output.jsonl` (our port).
3. Independently run their `run_eval.py` on the same files (in a temp venv with their package installed) — captures the canonical numbers.
4. Compare prompt-level strict, prompt-level loose, instruction-level strict, instruction-level loose, and per-constraint-type breakdown. **All numbers must match exactly.**
5. Same procedure for IFEval prompts: download `google/IFEval` HF dataset → use our scorer with `INSTRUCTION_DICT_ALL` → compare to a published number.

If any number disagrees, we don't proceed — the disagreement points to a bug in the import-rewrite or a vendored-but-broken file. **Hard gate. No rollouts before this passes.**

For IFEval specifically (since the upstream IFBench repo doesn't ship IFEval scoring), we cross-check against either:
- Google's `instruction_following_eval/run.sh` on `google/IFEval`, or
- A published model's IFEval score (e.g. Llama-3.3-70B IFEval ≈ 91; if we score Llama-3.3-70B on `google/IFEval` and get 91 ± 1, port is good).

### Pinning + re-pull strategy

- `_versions.txt` records the upstream commit SHA + a SHA256 of each vendored file at vendor time.
- A `tools/refresh_ifbench_verifiers.sh` script re-pulls from upstream, re-applies the import rewrites, and updates `_versions.txt`. Diff is reviewable.
- If upstream releases a verifier bug fix, we re-run the validation gate after the refresh — same procedure as initial port.

### Risks specific to the port

1. **Constraint-type rarity in our 95k.** Some IFEvalG constraint types may be near-zero in our training data; we still need to support all 54 because verification can fail on rare types. Don't lazy-load.
2. **NLTK data version drift.** NLTK changes default tokenizers between minor versions. Pin `nltk` version exactly in `pyproject.toml` and freeze the downloaded data archive (NLTK's downloader caches by version).
3. **`syllapy` and `emoji` quirks.** These are small libs maintained by individuals; pin exact versions. Periodic re-validate against `sample_output.jsonl`.
4. **Loose-accuracy heuristics changing.** The loose mode strips markdown / leading lines / trailing lines. Our port copies this exactly; any divergence in the slicing breaks reproducibility.
5. **License compliance.** Both upstream are Apache 2.0; we preserve the copyright headers in vendored files. Vendor commit message records source SHA.

---

## How to re-run the verifier parity test

The hard-gate test isn't yet wired into the main marin venv (deps not added to `pyproject.toml`). To validate the port from a clean state today:

```bash
cd /Users/ahmed/code/marin/.claude/worktrees/dpo_sft

# 1. Set up an ephemeral venv with the verifier deps
uv venv /tmp/ifbench_port/port_venv --python 3.11
uv pip install --python /tmp/ifbench_port/port_venv/bin/python \
    nltk langdetect immutabledict emoji syllapy unicodedata2 absl-py "setuptools<81" pytest

# 2. Run our port's parity test (overrides repo's pytest addopts)
PYTHONPATH=. /tmp/ifbench_port/port_venv/bin/python -m pytest \
    experiments/ifbench/verifiers/tests/test_reference_parity.py -v -o addopts=""
# expect: 14 passed
```

To re-pull upstream files (e.g. when bumping the pinned SHA):

```bash
# Edit IFBENCH_SHA / OI_SHA in tools/refresh_ifbench_verifiers.sh, then:
bash tools/refresh_ifbench_verifiers.sh
git diff experiments/ifbench/verifiers   # review upstream changes
# Then re-run the parity test above
```

To do an independent cross-check against upstream's own `run_eval.py` (used during the initial port to establish the reference numbers):

```bash
uv venv /tmp/ifbench_port/upstream_venv --python 3.11
git clone --quiet https://github.com/allenai/IFBench.git /tmp/ifbench_port/ifbench_upstream
cd /tmp/ifbench_port/ifbench_upstream && git checkout cb932e352a505306ad0115272211df14bb8f628f
uv pip install --python /tmp/ifbench_port/upstream_venv/bin/python \
    "setuptools<81" absl-py langdetect nltk immutabledict emoji syllapy unicodedata2

# Patch upstream's evaluation_lib.py to add the rstrip-fallback (it crashes on its
# own fixtures otherwise — see "Real-world finding" in Experiment Log):
python3 -c "
import pathlib
p = pathlib.Path('evaluation_lib.py')
src = p.read_text()
new = src.replace(
    'return_dict[example[\"prompt\"]] = example[\"response\"]',
    'return_dict[example[\"prompt\"]] = example[\"response\"]\n      return_dict[example[\"prompt\"].rstrip()] = example[\"response\"]'
).replace(
    'response = prompt_to_response[inp.prompt]',
    'response = prompt_to_response.get(inp.prompt) or prompt_to_response.get(inp.prompt.rstrip(), \"\")'
)
p.write_text(new)
"

mkdir -p eval_out
/tmp/ifbench_port/upstream_venv/bin/python -m run_eval \
    --input_data=data/IFBench_test.jsonl \
    --input_response_data=data/sample_output.jsonl \
    --output_dir=eval_out
# Then compare upstream's printout against our pytest result.
```

**Reference numbers (must match exactly):**

| Metric | Value |
|---|---|
| STRICT prompt-level | 0.25333333333333335 |
| STRICT instruction-level | 0.26744186046511625 |
| LOOSE prompt-level | 0.2866666666666667 |
| LOOSE instruction-level | 0.311046511627907 |

If `tools/refresh_ifbench_verifiers.sh` updated the pinned SHA, these reference numbers may shift — re-establish them via the upstream cross-check, then update `tests/test_reference_parity.py::REFERENCE`.

---

## Eval plan

| Eval | Role | Metric | Notes |
|---|---|---|---|
| Our 2k val (carved from train) | Training-time monitoring | per-constraint pair yield, DPO eval-loss | Train on 93,373; val on 2,000 |
| IFEval (~500) | In-domain | prompt-level strict + loose | T=0; do not tune to |
| IFBench_test (300) | OOD headline | prompt-level strict | T=0; 58 held-out constraints; **never load outside eval harness** |
| IFBench_multi-turn (3,161) | Out of v1 scope | — | Reserved untouched |
| MMLU or AlpacaEval | Capability floor | benchmark default | Detect regression from mixture training |
| THU-KEG IFBench (444) | Bonus RM signal | chosen-preference rate | Optional; not a primary metric |

Arms (all from same SFT init):
1. SFT-on-passers only
2. DPO-on-pairs only
3. Mixture (α ∈ {0.25, 1.0, 4.0})
4. Starting SFT checkpoint as floor

---

## Experiment Log

*(append-only; newest at top)*

### 2026-04-26 — Project setup
- **What:** Created branch `dpo_sft` from `origin/dpo-lora-clean`, unset upstream, wrote project doc + this logbook.
- **Result:** Branch ready; project scope agreed; v1 dataset path = AI2 IFBench rollout pipeline; THU-KEG IFBench rejected for training, usable as RM eval signal.
- **Next:** Resolve D-001 (where does SFT live), then sketch `MixtureDpoSftConfig`.

### 2026-04-26 — Verifier vendor implementation: STARTING
- **What:** Implementing the full verifier port per the "Verifier vendor plan" section: vendor `IFBench` (58 OOD constraints) + `IFEvalG` (54 train-eligible constraints), write `scoring.py` + `parse.py`, add deps, download reference data, validate parity locally vs upstream `run_eval.py`.
- **Scope of validation:** all local execution only — no cluster jobs, no Together/Gemini API spend. The IFEval-via-Together cross-check is deferred until cost sign-off; for now we validate IFEvalG with synthetic unit tests.
- **Hard gate:** IFBench scorer parity against `sample_output.jsonl` must match upstream `run_eval.py` numbers exactly before this entry gets a "FINISHED" sibling.

### 2026-04-26 — Verifier vendor implementation: FINISHED ✅
- **Outcome:** Hard gate passed. Verifier port reproduces upstream IFBench scoring byte-for-byte at full precision, on both top-level metrics and the per-instruction-id breakdown across all 58 IFBench constraints.
- **Files landed** under `experiments/ifbench/verifiers/`:
  - `ifbench/{instructions,instructions_registry,instructions_util}.py` — vendored from `allenai/IFBench@cb932e352a50` (Apr 11). 58 OOD checker classes.
  - `ifevalg/{instructions,instructions_registry,instructions_util}.py` — vendored from `allenai/open-instruct@881b18458b48` (Apr 24), `open_instruct/IFEvalG/` path. 25 IFEval + 29 IFTrain = 54 checker classes.
  - `__init__.py` exports `IFBENCH_DICT` (58), `IFEVALG_DICT` (54), `INSTRUCTION_DICT_ALL` (112, asserted disjoint).
  - `scoring.py` — port of upstream `evaluation_lib.py`. Same strict + loose semantics; registry passed as parameter so one scorer covers both registries. Adds rstrip-fallback on prompt key lookup to work around the trailing-whitespace mismatch in upstream's own fixtures (see "Real-world finding" below).
  - `parse.py` — `parse_ground_truth(raw)` for the IF_multi_constraints_upto5 training-set `ground_truth` field. Uses `ast.literal_eval` (Python repr-style strings, not JSON).
  - `_versions.txt` — pinned upstream SHAs + per-file SHA256.
  - `tests/test_reference_parity.py` — 14 tests (all passing): registry sizes/disjointness, full-precision parity vs upstream, 10 hand-crafted constraint-classification spot-checks across IFBench + IFEvalG, parser roundtrip, parser malformed-input handling.
  - `tests/fixtures/{IFBench_test,sample_output}.jsonl` — reference fixtures vendored at the same SHA.
  - `.gitignore` — excludes the runtime-populated `.nltk_data/` cache.
- **`tools/refresh_ifbench_verifiers.sh`** — idempotent re-pull script (verified clean re-run produces no diff).
- **Reference parity numbers (ours vs upstream-with-rstrip-patch, identical to 9 decimals):**

  | Metric | Value |
  |---|---|
  | STRICT prompt-level | 0.25333333333333335 |
  | STRICT instruction-level | 0.26744186046511625 |
  | LOOSE prompt-level | 0.2866666666666667 |
  | LOOSE instruction-level | 0.311046511627907 |

  All 58 per-instruction-id strict accuracies agree (`tier1` dict, byte-for-byte). Same for loose.
- **Real-world finding (worth recording):** Upstream IFBench's `run_eval.py` *crashes on its own published fixtures* — `sample_output.jsonl` and `IFBench_test.jsonl` have 7 prompt rows that differ in trailing whitespace, and upstream's `prompt_to_response[inp.prompt]` raises KeyError on the mismatch. We added a rstrip fallback in our scorer; we patched upstream the same way locally for the parity comparison. The rstrip fix is conservative — it only kicks in when an exact match misses — so byte-equal cases score identically.
- **Real-data smoke (additional):** Pulled `IF_multi_constraints_upto5` parquet directly (95,373 rows confirmed). Sampled 200 rows → `parse.parse_ground_truth` succeeds on every row → all 54 IFEvalG constraint classes successfully `build_description(**clean_kwargs)` from the parsed payload. Zero unknown IDs, zero verifier-build failures. Confirms registry coverage for the training data.
- **Unused dep dropped:** verified `spacy` is in upstream IFBench's `pyproject.toml` but never imported anywhere in the vendored modules — left out of our deps. Saves ~500MB of transitive install.
- **Dep set we'll need in marin's `pyproject.toml`** (for any code path that imports the verifiers): `nltk`, `langdetect`, `immutabledict`, `emoji`, `syllapy`, `unicodedata2`, `absl-py`, `setuptools<81` (syllapy uses deprecated `pkg_resources`).
- **Validation venv:** `/tmp/ifbench_port/{port_venv,upstream_venv}` — kept around for re-runs but not committed.
- **Tasks closed:** #3 (vendor), #18 (scoring), #19 (parse), #20 (HARD GATE — passed), #21 (refresh script).
- **Next:** add the deps to `pyproject.toml` of the marin project so the verifiers are usable from the main venv (was deliberately deferred to avoid noise during validation). Then unblock task #11 (Stage 4 pair extraction) and #4 (smoke rollout).

### 2026-04-26 — Second push: data prep + protocol + extraction ✅
- **What landed:**
  - `pyproject.toml`: added the 8 verifier deps to root `dependencies`. Verifiers are now first-class importable from the main venv (after `uv sync`).
  - `experiments/ifbench/data/prepare.py` (~190 lines) + `test_prepare.py` (8 tests): loads `IF_multi_constraints_upto5` (95,373 rows confirmed), stratified val carve, contamination check.
  - `experiments/ifbench/rollout/backend.py` (~95 lines) + `test_backend.py` (8 tests): `RolloutBackend` Protocol + supporting dataclasses (`SamplingConfig`, `RolloutRequest`, `Rollout`, `BatchHandle`, `BatchStatus`). `@runtime_checkable` so duck-typed implementations can be `isinstance()`-checked.
  - `experiments/ifbench/rollout/extract.py` (~190 lines) + `test_extract.py` (7 tests): `verify_rollouts`, `extract_pairs_and_sft`, `ExtractionStats` (tracks pair_yield + skip strata by num_constraints).
  - D-001 resolution: closed with concrete recommendation.
- **Real-data validation:**
  - 95,373 rows confirmed exact (matches dataset card).
  - 93,373 train + 2,000 val carved with stratified split.
  - **Stratification key changed from `constraint_type` → `num_constraints`** after discovering every row has `constraint_type == "multi"`. The new key has actual signal — counts range from 1 (24% of rows) to 5 (7% of rows). Val percentages match overall to ±0.1pp at every count.
  - Contamination check: 0 overlap with the 300 IFBench_test prompts (per AI2's by-construction guarantee).
- **37 tests passing** total across `experiments/ifbench/` (8 backend + 7 extract + 14 verifier parity + 8 prepare).
- **Tasks closed:** #1 (SFT entrypoint located), #16 (RolloutBackend protocol), #17 (val carve + contamination), #11 (Stage 4 extraction).
- **Did NOT do** in this push (deliberate):
  - Together / Gemini batch client wrappers (#13, #15) — would have to write without being able to test against the real APIs. Better to do these in a session where API keys are confirmed up-front.
  - Stage-1 Marin StepSpec wrapper (#9) — `prepare.py` is done; the StepSpec is just plumbing that depends on the rest of the pipeline.
  - `MixtureDpoSftConfig` sketch (#2) — would benefit from a fresh look at the existing `SimpleDPOConfig` field-by-field; better as its own focused session.
- **One real bug fixed:** the `loose` scorer in `evaluation_lib.py` upstream relies on `strict` mutating `inp.kwargs` in-place to strip None values; my port did not mutate, so loose blew up on the first `KeyError: N`. Fixed by applying the None-strip independently in both paths.
- **Next:** confirm API keys → write Together + Gemini client wrappers + smoke them with 1-prompt sync calls → wire Stage-2 fanout StepSpec → run smoke (#4) → cost-quote → full run (#5).

### 2026-04-27 — Smoke run: 100 prompts × 3 models — DATA COLLECTED ✅ (extraction pending)

**What:** Submitted 3 batch jobs (50%-discount tier). 70B-Turbo completed in 6 min. 8B-Lite + Gemini stalled in queue for 40+ min with **zero progress** (Gemini's `updateTime == createTime` — Google never started it; 8B-Lite Together batch sat at `progress=0`). **Pivoted to sync API** for the two stalled models — completed in **70.7s** wall (parallel asyncio, concurrency=15 Together / 10 Gemini).

**Sync-mode unblock:** New file `experiments/ifbench/rollout/sync_runner.py` (~140 lines) with bounded-concurrency async runners for both backends. Saves rollouts in the same jsonl schema as the batch path so Stage 4 consumes either source identically. Extracted-out 70B rollouts from the completed batch into `/tmp/ifbench_port/smoke/rollouts_70b.jsonl` (also no-recompute).

**Per-model sanity:**

| Model | Path | Rollouts | Empty | Wall time | Notes |
|---|---|---:|---:|---|---|
| `Llama-3.3-70B-Instruct-Turbo` | Together batch | 100 | 0 | ~6 min | 26/100 passed all constraints (matches AI2 paper's Qwen-72B at 26%) |
| `Meta-Llama-3-8B-Instruct-Lite` | Together sync | 100 | 0 | ~70 s | Cheap, was stuck in batch queue at `progress=0` for 40+ min |
| `gemini-3-flash-preview` (thinking=high) | Gemini sync | 100 | 0 | ~9 s | Stuck in batch queue at `BATCH_STATE_PENDING` for 40+ min — Google never started processing |

**Token counts (Gemini Flash with `thinkingLevel="high"`):**
- 100 prompts: **19,833 input + 4,995 output + 95,060 thinking** = 119,888 total
- Avg per prompt: 198 input, 50 response, **951 thinking**
- **Thinking is 95% of "output" billing.** Confirms the pre-smoke worst-case estimate was right — Gemini thinking dominates.
- Sync price (no batch discount): 95k thinking @ $3/M output ≈ $0.29 for the Gemini 100-prompt sync call.

**Cost so far (smoke):**
| Component | Cost |
|---|---|
| 70B-Turbo batch (100 prompts) | $0.029 |
| 8B-Lite sync (100 prompts) | ~$0.005 |
| Gemini Flash sync, thinking=high (100 prompts) | ~$0.31 |
| **Total smoke** | **~$0.34** |

**Cost projection update for full 95k×3 run:** Extrapolating Gemini sync token volume × 950 to 95k prompts → ~90M thinking tokens. At batch pricing ($1.50/M output) that's $135. **Full-run estimate revised down from $393 → $170 IF Gemini batch ever works for us; ~$300 if forced to sync.** Need to retry batch on a different day to see if the queue clears.

**API quirks discovered:**
- **Gemini inline batch** uses `{"request": ..., "metadata": ...}`, NOT `{"key": ..., "request": ...}` (latter is file-based-only). Correlation id goes in `metadata.correlation_id`.
- **Together file upload** is a 3-step protocol (POST metadata → PUT to signed S3 redirect → POST callback), not single-shot multipart. Delegated to the `together` SDK.
- **Together batch CREATE** wraps in `{"job": {...}}`; **GET** returns flat. `together_backend.py` handles both.
- **Gemini batch SLA is "best-effort 24h, often quicker"** — but on 2026-04-27 the queue was effectively dead for 40+ min on a 100-prompt thinking=high job. Sync was 100× faster.

**Outputs on disk:**
- `/tmp/ifbench_port/smoke/rollouts_70b.jsonl` (195KB, 100 rollouts)
- `/tmp/ifbench_port/smoke/rollouts_8b_sync.jsonl` (100 rollouts)
- `/tmp/ifbench_port/smoke/rollouts_gemini_sync.jsonl` (100 rollouts)
- `/tmp/ifbench_port/smoke/outputs/{dpo_pairs.jsonl, sft_examples.jsonl, summary.json}`

### Final smoke result — PASSED ✅

**Pair yield: 34/100 (34%)** — clears the 30% gate. SFT cache: 58 examples.

| Model | Pass-all rate | Notes |
|---|---:|---|
| Llama-3.3-70B-Instruct-Turbo | **26.0%** | Spot-on match of AI2 paper's Qwen-72B baseline (26%) |
| Gemini Flash, thinking=high | **22.0%** | Underperformed 70B! See "Surprise" below. |
| Meta-Llama-3-8B-Instruct-Lite | **10.0%** | Slightly above AI2 paper's Llama3-8B (6%) |

**Skip breakdown:**
- 62 prompts had **no passing model** (all 3 failed) → hard prompts that need a stronger pool
- 4 prompts had **no failing model** (all 3 passed) → easy prompts
- 34 prompts had at least one passer + one failer → DPO pairs

**Yield by `num_constraints`:**
| #constraints | pairs | skips |
|---:|---:|---:|
| 1 | 13 | 6 |
| 2 |  7 | 14 |
| 3 |  7 | 17 |
| 4 |  6 | 22 |
| 5 |  1 |  7 |

**Surprise: Gemini-Flash with thinking=high passes fewer constraints than Llama-3.3-70B-Turbo (22% vs 26%).** Possible causes (not yet investigated):
- Gemini may produce responses with markdown / formatting habits that fail strict verifiers (e.g., adding bold to a "no_whitespace" output).
- thinking=high may encourage stylistic "explanation" content that breaks a Length or Format constraint.
- Strict accuracy is brutal — we should check loose accuracy too. Loose mode strips first/last lines + markdown bold; could lift Gemini meaningfully.
- The 70B has the implicit advantage of being trained heavily on IFEval-style instruction-following.

This **doesn't break the smoke** — we still get pair yield > 30% from the spread. But it does suggest:
- Pre-smoke expectation that Gemini=strong / 70B=mid was wrong on this dataset. They're both "strong" with the 70B slightly ahead.
- We may want to revisit the pool: keep 70B as the strong slot and use Gemini as a *complementary* strong model (different failure modes), not the "stronger of the two".
- Loose accuracy will likely tell a different story; worth measuring before the full run.

**Cost:** $0.34 actual smoke (70B batch $0.029 + 8B sync $0.005 + Gemini sync $0.31). Gemini sync at thinking=high is the budget driver.

**Smoke success criteria recap:**
1. ✅ Pipeline runs end-to-end — 3 rollout sources → verifier → pairs + SFT.
2. ✅ Pair yield ≥ 30% — got 34%.
3. ✅ Verifier port self-check — passed earlier (37 tests green).
4. ✅ No model >90% empty/refusal — all 3 models 100/100 non-empty.

**Cost projection update for full 95k×3:**

| Path | Estimate |
|---|---|
| All-batch (assuming Gemini batch ever works) | ~$170 |
| Gemini batch broken → forced sync | ~$310 |
| Mixed (70B batch + 8B batch + Gemini sync) | ~$310 (Gemini dominates regardless) |

**Open questions to resolve before the full run:**
1. Will Gemini batch work later, or is it always slow on Flash thinking=high? Cheapest check: submit a 10-prompt batch tomorrow morning, see if it completes in <30 min.
2. How does loose accuracy compare to strict on Gemini? Run our scoring loose on the 100 smoke responses, see if Gemini lifts to ≥ 70B.
3. Should the pool stay 3-model or extend to 4? If pair yield stays ~34% on the full run, that's ~32k pairs from 95k — fine, but more models would push it higher.

**Tasks closed:** #4 (smoke rollout). Next blocking: cost approval for full run (#5). Tasks remaining for full pipeline: #2 (mixture trainer config), #6 (mixture test), #7 (eval arms), #9 (Stage-1 StepSpec wrapper), #10 (Stage-2 StepSpec wrapper).

<!--
Template for future entries:

### YYYY-MM-DD HH:MM UTC — <short label>
- **Hypothesis:**
- **Command:**
- **Config:**
- **Result:**
- **Interpretation:**
- **Next action:**
-->

---

## References

- AI2 IFBench paper: arXiv:2507.02833 — "Generalizing Verifiable Instruction Following"
- AI2 IFBench repo: https://github.com/allenai/IFBench
- AI2 IF_multi_constraints_upto5: https://huggingface.co/datasets/allenai/IF_multi_constraints_upto5
- IFEval paper: arXiv:2311.07911 — Zhou et al., Google
- IFEval verifiers: https://github.com/google-research/google-research/tree/master/instruction_following_eval
- THU-KEG Agentic Reward Modeling paper: arXiv:2502.19328
- THU-KEG/IFBench (eval, not training): https://huggingface.co/datasets/THU-KEG/IFBench
- Project sibling design notes: `.agents/projects/dpo_levanter.md` (Haliax/Levanter DPO infra analysis)
