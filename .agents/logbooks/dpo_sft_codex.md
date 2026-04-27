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

### 2026-04-27 07:20 UTC — Codex overnight execution plan

**User goal:** keep the IFBench rollout thread moving for the next ~8 hours while the user sleeps. Finish the 100-prompt smoke screen, launch provider batch jobs and Iris vLLM jobs for the Llama models, sweep Gemini 3 Flash thinking on the same 100 prompts, and launch Gemini 3 Flash/Pro file-batch jobs.

**Docs read before execution:**
- Together Batch docs: batch input is JSONL with `custom_id` + `body`, uploaded with Files API `purpose=batch-api`; create with `/v1/batches`, endpoint `/v1/chat/completions`; poll every 30-60s; output rows are not ordered; max 50k requests / 100MB per batch / 30B tokens per model; 70B Turbo has 50% discount.
- Gemini Batch docs: file-batch JSONL rows use `{"key": ..., "request": <GenerateContentRequest>}`; upload via Files API; create with `models/{model}:batchGenerateContent` using `input_config.file_name`; poll job state; download `dest.fileName` on success; batch is 50% standard cost.
- Gemini Thinking docs: Gemini 3 should use `generationConfig.thinkingConfig.thinkingLevel`, not `thinkingBudget`. Flash supports `minimal`, `low`, `medium`, `high`; default is dynamic/high, and `minimal` is the closest available "no thinking" mode but not a hard off switch.

**Constraints / guardrails:**
- Do not restart or mutate an Iris cluster. Launch jobs only.
- Avoid large cross-region data movement. The 100-prompt smoke artifacts are local and small.
- Do not launch the full 95k rollout without explicit cost approval. This plan is only 100-prompt smoke/batch comparison work.
- Source `.env` before provider calls. The inherited shell `TOGETHER_API_KEY` is stale; `.env` has the valid Together key.
- Use the same prepared prompt set and output schema everywhere so `extract_pairs_and_sft` can score all sources identically.

**Execution queue:**
1. **State sync and smoke finalization.** Verify `/tmp/ifbench_port/smoke/outputs/{dpo_pairs.jsonl,sft_examples.jsonl,summary.json}` exists; if not, re-run extraction from the three existing 100-prompt rollout JSONLs. Record exact summary in this logbook.
2. **Dependency/code unblock.** Make the local provider clients runnable:
   - add/install `together` for the existing Together batch upload path, or replace upload with direct HTTP if faster;
   - add a Gemini file-batch path beside the current inline-batch path, preserving `BatchHandle`/`Rollout` output.
3. **Together 100-prompt batch launches.** Submit fresh 100-prompt batches for:
   - `meta-llama/Llama-3.3-70B-Instruct-Turbo`
   - `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
   Save handles under `/tmp/ifbench_port/overnight/together_batch/handles.json`, poll in the background, download and score when complete.
4. **Gemini 3 Flash thinking sweep.** On the same 100 prompts, run `minimal`, `low`, `medium`, `high` for `gemini-3-flash-preview`.
   - Prefer sync for quick measurement if file-batch queue stalls, because the sweep is only 400 calls.
   - Score strict and loose, record pass rate, pair yield contribution, output tokens, thinking tokens, wall time, and cost.
   - Pick the lowest thinking level whose strict pass rate does not regress materially from high on this sample. If strict ties are noisy, use loose accuracy + failure inspection as tie-breakers.
5. **Gemini file-batch launches.** Using the file-batch API, submit 100-prompt batches for:
   - `gemini-3-flash-preview` at the selected thinking level and also `high` if the selected level differs;
   - Gemini 3 Pro using the lowest supported thinking level to start. If the API rejects the exact Pro model id, list models and retry with the available Gemini 3 Pro id.
   Save handles under `/tmp/ifbench_port/overnight/gemini_file_batch/handles.json`.
6. **Iris vLLM race jobs.** Launch 100-prompt inference jobs on Iris for both Llama models, each on `v6e-8` with vLLM concurrency 256:
   - `meta-llama/Llama-3.3-70B-Instruct-Turbo`
   - `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
   Use existing Marin/Iris inference patterns if present; otherwise create a small script/job wrapper that consumes the same prepared JSONL and writes Rollout JSONL. Record job ids and monitor via `babysit-job` cadence. No cluster restart actions.
7. **Race monitor loop.** Poll provider batches and Iris jobs until terminal or until the 8-hour window ends. Every completion triggers download, verifier scoring, extraction, and logbook update. If one path fails due to a small code bug, fix and resubmit once. If it fails due to capacity or provider queueing, keep monitoring and proceed with other queued work.
8. **End-of-window summary.** Produce a compact table: method, model, status, wall time, pass-all strict/loose, empty count, tokens, estimated cost, output path. Recommend the next full-run path based on observed reliability and cost.

**Immediate next action:** verify smoke outputs, then implement the Gemini file-batch gap and ensure Together batch can import its SDK.

### 2026-04-27 07:30 UTC — Revised overnight plan for sign-off

**Revision from user:** cap tonight's full-scale provider rollout at **20,000 prompts**, drop Gemini 3 Pro entirely, use Iris **interactive priority**, try `v6e-4` first for Iris vLLM jobs and fall back to `v5p-8` if `v6e-4` is unavailable.

**Cost envelope for 20k prompts, no Gemini Pro:**
- Expected from 100-prompt smoke extrapolation: about **$38** for Together 70B + Together 8B + Gemini 3 Flash batch.
- Conservative 1024-token cap estimate: about **$46**.
- If Together's 70B discount or Gemini batch behavior differs, stop before projected API spend exceeds **$75** unless the user gives a higher cap.
- Iris TPU usage is separate cluster spend; use only the requested small 20k race jobs, interactive priority, and no cluster restart/mutation.

**Execution order after sign-off:**
1. **Preserve completed smoke.** Confirm `/tmp/ifbench_port/smoke/outputs/summary.json` remains the canonical 100-prompt result: 34 pairs, 58 SFT examples, 34% pair yield. Log exact UTC timestamp and paths.
2. **Patch provider runners.** Finish local runner support before any launch:
   - Together batch import must work from `uv run`.
   - Gemini file-batch support must create docs-shaped JSONL rows: `{"key": ..., "request": ...}`.
   - All provider calls source `.env` so stale inherited keys are ignored.
3. **Gemini 3 Flash 100-prompt thinking sweep.**
   - Run or batch `minimal`, `low`, `medium`, `high` on the same 100 prompts.
   - Score strict and loose.
   - Pick the lowest thinking level whose strict pass-all is not meaningfully worse than `high`; if tied, use loose pass rate, empty count, and thinking-token cost.
   - Log one UTC entry per level with path, token totals, pass-all, loose pass-all, cost estimate.
4. **20k prompt provider batches.**
   - Use the first 20,000 train prompts from `/tmp/ifbench_port/prepared_v2/train.jsonl`.
   - Submit Together batch for `meta-llama/Llama-3.3-70B-Instruct-Turbo`.
   - Submit Together batch for `meta-llama/Meta-Llama-3-8B-Instruct-Lite`.
   - Submit Gemini 3 Flash file-batch at the selected thinking level.
   - Save handles under `/tmp/ifbench_port/overnight_20k/{together,gemini}/handles.json`.
   - Poll every 30-60s early, then 5-10 min cadence once stable. Log UTC status entries at every state change and at least hourly.
5. **Iris vLLM race jobs for Llama models.**
   - Launch inference jobs for the same 20,000 prompt slice.
   - Requested resources: try `v6e-4`; if unavailable/unschedulable, relaunch on `v5p-8`.
   - Use Iris interactive priority.
   - vLLM concurrency: 256.
   - Models:
     - `meta-llama/Llama-3.3-70B-Instruct-Turbo`
     - `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
   - Do not stop/restart/bounce any Iris cluster. Only submit jobs and monitor job state.
   - Record job ids, exact commands, output paths, and resource fallback decisions with UTC timestamps.
6. **Download, score, and extract as each source finishes.**
   - Convert all outputs into the common `Rollout` JSONL schema.
   - Score strict and loose verifier outcomes.
   - Run `extract_pairs_and_sft` on completed 20k sources; if partial sources finish first, record partial yield but do not treat it as final.
   - Log pair yield, SFT count, no-passer/no-failer counts, per-`num_constraints` yield, per-model pass rates, token totals, and cost estimates.
7. **Nightly stop/continue rules.**
   - Stop before API projected spend exceeds **$75**.
   - Stop before submitting anything beyond the 20,000-prompt cap.
   - Continue monitoring already-submitted provider/Iris jobs until terminal or until the user wakes.
   - If a provider batch stalls, do not auto-scale beyond 20k or switch to sync for the 20k without a new cost check.
   - If a small code bug blocks download/scoring, fix it once and continue; if the same failure recurs, log and move to the next runnable path.
8. **Wake-up summary.**
   - Final UTC-stamped logbook entry with a table: backend, model, prompt count, status, wall time, strict/loose pass-all, empty count, pair yield contribution, cost, output path.
   - Recommendation for training data construction and whether to continue from 20k to 95k.

### 2026-04-27 07:32 UTC — Overnight execution STARTED

**Authorization interpreted:** execute the revised 20k-capped overnight plan, update this Codex logbook continuously, no Gemini Pro, no >20k prompt submissions, stop before projected API spend exceeds $75.

**Immediate tasks:**
1. Validate local changes and provider dependencies.
2. Finish/test Gemini file-batch and Together batch paths on tiny requests.
3. Run the Gemini Flash 100-prompt thinking sweep.
4. Submit the 20k provider batches and Iris interactive vLLM race jobs only after the small paths pass.

**State note:** no new provider/Iris jobs have been launched yet in this entry.

### 2026-04-27 07:34 UTC — Local validation checkpoint

**Status:**
- `together` dependency added to `pyproject.toml` and `uv.lock` so `TogetherBackend` imports under `uv run`.
- `GeminiBackend(use_file_batch=True)` import path works.
- Targeted rollout tests pass with ephemeral pytest:
  - `experiments/ifbench/rollout/test_backend.py`
  - `experiments/ifbench/rollout/test_extract.py`
  - Result: 15 passed, 2 warnings.

**No provider or Iris jobs launched yet.**

**Next:** add a reproducible overnight CLI for sweep/submit/poll/download so the 20k run is not a pile of one-off shell commands.

### 2026-04-27 07:35 UTC — Overnight CLI smoke-tested locally

**Added:** `experiments/ifbench/rollout/overnight.py`

**Local checks:**
- `uv run python -m experiments.ifbench.rollout.overnight --help` works.
- `py_compile` passes for `overnight.py` and `gemini_backend.py`.
- Rescored existing 100-prompt smoke rollout paths through the new CLI.

**Rescore output:** `/tmp/ifbench_port/overnight_20k/smoke_rescore_summary.json`

**Strict / loose pass-all from rescore:**
| Model | Strict | Loose |
|---|---:|---:|
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | 26/100 | 33/100 |
| `gemini-3-flash-preview` | 22/100 | 24/100 |
| `meta-llama/Meta-Llama-3-8B-Instruct-Lite` | 10/100 | 13/100 |

**Pair extraction reproduced:** 34 DPO pairs, 58 SFT examples, 34% pair yield.

**Next:** submit tiny one-request provider batch probes for Together and Gemini file-batch; if they both reach a valid queued/completed state, proceed to the Gemini 100-prompt thinking sweep.

### 2026-04-27 07:38 UTC — Provider batch probe finding

**Together:** one-request probe submitted far enough to appear in `/v1/batches` as `d5db2f4b-6c97-4499-8924-12250cfee9da` (`VALIDATING` at first list call). This was a tiny probe, not the 20k launch.

**Gemini file-batch:** first probe failed with `400 INVALID_ARGUMENT` when row-level `generationConfig.thinkingConfig.thinkingLevel` was present.

**Manual Gemini file-batch variants:**
- Simple row with only `contents`: accepted, returned batch names.
- Row with `generation_config` but no thinking: accepted.
- Row with `generationConfig` but no thinking: accepted.
- Any row with `thinkingConfig`: rejected.

**Implementation adjustment:** file-batch rows now omit `thinkingConfig`. For Gemini 3 Flash file-batch, this means default dynamic/high thinking. The 100-prompt thinking sweep still uses sync API, where `thinkingLevel` is accepted, to decide whether lower thinking is promising for future non-file runs.

**No 20k provider/Iris launches yet.**

### 2026-04-27 07:38 UTC — Tiny batch probes passed

**Together:** probe batch `d5db2f4b-6c97-4499-8924-12250cfee9da` is now `completed`.

**Gemini file-batch:** patched file-batch probe accepted:
- batch id: `batches/y77t3m9jf2779dxen55y8yiqsz15i5393ztr`
- status after three quick polls: `pending`

**Interpretation:** provider submission paths are valid. Gemini file-batch queues can still be slow, but the request shape is no longer rejected.

**Next:** run the 100-prompt Gemini 3 Flash thinking sweep via sync API.

### 2026-04-27 07:43 UTC — Gemini 3 Flash thinking sweep finished

**Command:** `uv run python -m experiments.ifbench.rollout.overnight --work-dir /tmp/ifbench_port/overnight_20k sweep-gemini --concurrency 10 --force`

**Outputs:**
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/rollouts_flash_minimal.jsonl`
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/rollouts_flash_low.jsonl`
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/rollouts_flash_medium.jsonl`
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/rollouts_flash_high.jsonl`
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/decision.json`

**Results on the same 100 prompts:**
| Thinking | Strict pass-all | Loose pass-all | Input tok | Output tok | Thinking tok |
|---|---:|---:|---:|---:|---:|
| `minimal` | **40/100** | **49/100** | 19,833 | 29,808 | 0 |
| `low` | 26/100 | 28/100 | 20,813 | 8,831 | 88,681 |
| `medium` | 19/100 | 20/100 | 19,833 | 5,416 | 94,020 |
| `high` | 20/100 | 21/100 | 19,833 | 4,826 | 95,726 |

**Decision:** selected `minimal` by the prewritten rule: lowest thinking level with strict pass-all >= high. It also wins outright on strict, loose, and cost.

**Important batch limitation:** Gemini file-batch rejects row-level `thinkingConfig`, so the 20k Gemini file-batch can only use default dynamic/high thinking unless we switch to sync API. I will still submit the file-batch path because the user asked to use Gemini Batch API, but we should interpret it as "Gemini Flash default/high batch", not "minimal batch".

**No 20k provider/Iris launches yet. Next:** submit the 20k provider batches.

### 2026-04-27 07:44 UTC — 20k provider batches submitted

**Prompt slice:** first 20,000 rows from `/tmp/ifbench_port/prepared_v2/train.jsonl`.

**Handle file:** `/tmp/ifbench_port/overnight_20k/handles.json`

| Backend | Model | Batch id | Mode | Requests |
|---|---|---|---|---:|
| Together | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `9afbd565-a6be-48c8-beaf-29be61d2b81d` | batch | 20,000 |
| Together | `meta-llama/Meta-Llama-3-8B-Instruct-Lite` | `9b109f48-a95a-4fb6-a658-7d16c52053d3` | batch | 20,000 |
| Gemini | `gemini-3-flash-preview` | `batches/zqp38vk6tqywpekvf9zcbr3v4o72m7n7bo1b` | file batch | 20,000 |

**Gemini caveat:** the file batch omits `thinkingConfig` because file-batch rejected it in probes. This is Gemini Flash default dynamic/high thinking, not the selected `minimal` sync setting.

**Projected API spend still under cap:** expected provider spend roughly $38-$46 for these three batches; hard stop remains $75 projected API spend.

**Next:** poll provider status, then launch Iris interactive vLLM race jobs.

### 2026-04-27 07:48 UTC — 20k prompt slice staged for Iris

**Local slice:** `/tmp/ifbench_port/overnight_20k/iris/prompts_20k.jsonl` — 20,000 rows.

**GCS input for Iris vLLM jobs:** `gs://marin-us-central2/scratch/ifbench/overnight_20k/prompts_20k.jsonl`

**Verified object:** 34.18 MiB, 35,841,328 bytes, GCS mtime `2026-04-27T07:46:34Z`.

**Iris launch constraints:** interactive priority, TPU `v6e-4,v5p-8` fallback list, vLLM concurrency 256, no Iris cluster restarts.

**Next:** launch two Iris vLLM jobs:
- `meta-llama/Llama-3.3-70B-Instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct`

### 2026-04-27 07:49 UTC — Iris vLLM worker pre-launch fix

**Issue caught before launch:** `VllmEnvironment` forwards `max_model_len` from `ModelConfig.engine_kwargs`, but it does not forward `tensor_parallel_size` or `max_num_seqs`.

**Fix:** `experiments/ifbench/rollout/vllm_iris_infer.py` now passes these as explicit vLLM CLI args:
- `--tensor-parallel-size 4`
- `--max-num-seqs 256`

**Validation:** `uv run python -m py_compile experiments/ifbench/rollout/vllm_iris_infer.py` passed.

**Next:** submit the two Iris interactive jobs.

### 2026-04-27 07:50 UTC — Iris vLLM jobs running + provider poll

**Iris vLLM jobs submitted:**

| Job | Model | TPU request | Scheduled | State | Output |
|---|---|---|---|---|---|
| `/ahmed/ifbench-vllm-70b-20k` | `meta-llama/Llama-3.3-70B-Instruct` | `v6e-4,v5p-8` | `v6e-4` | `JOB_STATE_RUNNING` | `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b.jsonl` |
| `/ahmed/ifbench-vllm-8b-20k` | `meta-llama/Meta-Llama-3-8B-Instruct` | `v6e-4,v5p-8` | `v6e-4` | `JOB_STATE_RUNNING` | `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_8b.jsonl` |

**Iris monitor state files:**
- `scratch/20260427T074936Z_ifbench_vllm_70b_monitoring_state.json`
- `scratch/20260427T074936Z_ifbench_vllm_8b_monitoring_state.json`

**Provider poll at 2026-04-27T07:49:38Z:**
- Together 70B batch `9afbd565-a6be-48c8-beaf-29be61d2b81d`: `IN_PROGRESS` / pending
- Together 8B batch `9b109f48-a95a-4fb6-a658-7d16c52053d3`: `IN_PROGRESS` / pending
- Gemini Flash file-batch `batches/zqp38vk6tqywpekvf9zcbr3v4o72m7n7bo1b`: `BATCH_STATE_RUNNING` / pending

**Next:** watch Iris logs until vLLM is ready or fails, poll provider batches periodically, and download/score/extract whichever side finishes first.

### 2026-04-27 07:53 UTC — Iris vLLM internal startup logs inspected

Used `iris task exec` to inspect the native vLLM temp logs on both running workers.

**70B job:** `/ahmed/ifbench-vllm-70b-20k`
- vLLM received the intended args: `tensor_parallel_size=4`, `max_num_seqs=256`, `max_model_len=8192`.
- TPU: `v6e-4`, 4 chips, ~31.25 GiB HBM/chip.
- State: model loader initialized; weight loading still in progress at log check.
- Risk: 70B BF16 on v6e-4 may be memory-tight or impossible. Watch for OOM; if it fails, relaunch 70B on explicit `v5p-8` with tensor parallel size 8.

**8B job:** `/ahmed/ifbench-vllm-8b-20k`
- vLLM received the intended args: `tensor_parallel_size=4`, `max_num_seqs=256`, `max_model_len=8192`.
- Weights downloaded in ~6.4s and loaded in ~1.3s.
- It is compiling TPU graphs for request batches up to 256; no failure yet.
- KV cache log shows maximum concurrency for 8192-token requests is ~97.5x, but the server still accepted `max_num_seqs=256`; actual scheduler throughput may be below the requested client concurrency.

**Note:** vLLM logs warn TensorFlow is absent for `tf.io.gfile`, so a GCS-backed compilation cache may not work. Prompt input already loaded through `fsspec`; this warning is likely compile-cache only unless later logs say otherwise.

**Next:** continue watching 8B for server-ready/progress logs and 70B for OOM or successful load.

### 2026-04-27 07:57 UTC — Iris vLLM relaunch required

**8B root cause:** JAX vLLM rejects per-request `seed`. The first 8B job reached serving but every request returned:

`JAX does not support per-request seed.`

**Action:** patched `experiments/ifbench/rollout/vllm_iris_infer.py` to omit `seed` from the OpenAI-compatible HTTP payload while still recording the run seed in the output `Rollout` metadata. `py_compile` passed.

**Invalid first 8B job:** `/ahmed/ifbench-vllm-8b-20k`
- Killed at ~2026-04-27T07:56:30Z because outputs would have been all HTTP 400 empties.

**70B first job:** `/ahmed/ifbench-vllm-70b-20k`
- Failed on `v6e-4` after ~6m12s with exit code 1 before server readiness.
- This matches the expected memory risk for 70B BF16 on 4x v6e chips.

**Next relaunches:**
- 8B: relaunch on `v6e-4,v5p-8`, tensor parallel 4, same 20k prompts.
- 70B: relaunch explicitly on `v5p-8`, tensor parallel 8, same 20k prompts.

### 2026-04-27 07:58 UTC — Iris relaunches submitted

**New Iris jobs:**

| Job | Model | TPU | TP | State at submit check | Output |
|---|---|---|---:|---|---|
| `/ahmed/ifbench-vllm-8b-20k-r1` | `meta-llama/Meta-Llama-3-8B-Instruct` | `v6e-4` from `v6e-4,v5p-8` | 4 | `JOB_STATE_RUNNING` | `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_8b.jsonl` |
| `/ahmed/ifbench-vllm-70b-20k-v5p8-r1` | `meta-llama/Llama-3.3-70B-Instruct` | `v5p-8` | 8 | `JOB_STATE_RUNNING` / task building | `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b.jsonl` |

**Relaunch monitor state files:**
- `scratch/20260427T075804Z_ifbench_vllm_8b_r1_monitoring_state.json`
- `scratch/20260427T075804Z_ifbench_vllm_70b_v5p8_r1_monitoring_state.json`

**Watch item:** Iris reports `v5p-8` with device count `4`; if vLLM sees only four JAX devices and rejects `tensor_parallel_size=8`, relaunch 70B on the same `v5p-8` worker shape with TP=4.

### 2026-04-27 08:01 UTC — 70B v5p-8 TP fixed, 8B still starting

**70B v5p-8 r1 failure confirmed:**
- Job: `/ahmed/ifbench-vllm-70b-20k-v5p8-r1`
- vLLM saw `tpu_type=v5p-8`, `num_chips=4`, `num_cores_per_chip=2`.
- Root cause: `ValueError: Number of devices 4 must be >= the product of mesh_shape (1, 8)`.
- Interpretation: for this vLLM/JAX path, v5p-8 exposes four usable JAX devices, so tensor parallel size must be 4.

**70B v5p-8 r2 submitted:**
- Job: `/ahmed/ifbench-vllm-70b-20k-v5p8-r2`
- TPU: `v5p-8`
- Tensor parallel: 4
- Monitor state: `scratch/20260427T080034Z_ifbench_vllm_70b_v5p8_r2_monitoring_state.json`

**8B r1:** `/ahmed/ifbench-vllm-8b-20k-r1` remains running; latest top-level logs show prompt load + vLLM startup, no seed-related 400s yet.

**Provider poll at 2026-04-27T08:00:00Z:** Together 70B + 8B still `IN_PROGRESS`; Gemini Flash file-batch still `BATCH_STATE_RUNNING`.

### 2026-04-27 08:04 UTC — Iris r2/r1 health check

**8B r1:** `/ahmed/ifbench-vllm-8b-20k-r1`
- vLLM server is ready.
- Confirmed many `200 OK` chat completions after removing the per-request `seed` field.
- Some `400 Bad Request` responses remain for prompts above the model context window (example: 36,356 input tokens vs 8,192 max). These are expected dataset outliers; the worker records them as empty rollouts.

**70B v5p-8 r2:** `/ahmed/ifbench-vllm-70b-20k-v5p8-r2`
- TP=4 mesh initialized successfully.
- vLLM sees `v5p-8`, `num_chips=4`, `num_cores_per_chip=2`, HBM ~95.74 GiB per visible device.
- State: loading model weights; server not ready yet.

**Next:** monitor 8B rollout progress/writer completion; continue watching 70B through weight load and precompile.

### 2026-04-27 08:04 UTC — 8B rollout in progress

**8B r1:** `/ahmed/ifbench-vllm-8b-20k-r1`
- Rollout progress log: `1800/20000` at ~17 req/s.
- HTTP status is dominated by `200 OK`; over-context prompts still show as expected 400s and will be empty rollouts.
- No GCS output object yet; worker writes the jsonl after all 20k requests finish.

**70B r2:** still running and loading/precompiling; no output object yet.

### 2026-04-27 08:08 UTC — Active race status

**8B Iris vLLM:** `/ahmed/ifbench-vllm-8b-20k-r1`
- Progress: `5800/20000`
- Throughput: ~18.2 req/s
- Status: running, mostly `200 OK`, expected occasional context-window `400`s.

**70B Iris vLLM:** `/ahmed/ifbench-vllm-70b-20k-v5p8-r2`
- Status: running.
- Last confirmed internal state: TP=4 mesh valid on v5p-8, model load/precompile path underway.
- `iris task exec` timed out while trying to tail temp logs; top-level job remains healthy with no pending reason/failure.

**Provider batches:** still pending/in-progress at the 08:06 UTC poll.

### 2026-04-27 08:14 UTC — 70B r3 launched with lower server max sequences

**70B v5p-8 r2:** failed after ~9m54s, exit code 1. Iris bug report did not preserve the vLLM temp-log traceback; last top-level log was native vLLM startup.

**Best current hypothesis:** TP=4 was correct, but 70B startup still hit HBM/KV-cache pressure with `max_num_seqs=256`.

**70B v5p-8 r3 submitted:**
- Job: `/ahmed/ifbench-vllm-70b-20k-v5p8-r3`
- TPU: `v5p-8`
- Tensor parallel: 4
- Client concurrency: 256
- Server max sequences: 64 (reduced from 256 to lower startup/KV memory pressure)
- Monitor state: `scratch/20260427T081358Z_ifbench_vllm_70b_v5p8_r3_monitoring_state.json`

**Fallback if r3 fails:** stop spending Iris time on 70B tonight and rely on the Together 70B batch for the 70B slot.

### 2026-04-27 08:15 UTC — 8B nearing completion, 70B r3 alive

**8B Iris vLLM:** `/ahmed/ifbench-vllm-8b-20k-r1`
- Progress: `13900/20000`
- Throughput: ~18.4 req/s
- Output object not present yet; expected because the worker writes after all requests finish.

**70B Iris vLLM:** `/ahmed/ifbench-vllm-70b-20k-v5p8-r3`
- State: `JOB_STATE_RUNNING`
- This r3 job has not failed immediately after submission; continue watching through startup.

### 2026-04-27 08:24 UTC — 8B Iris completed; 70B Iris fallback exhausted

**8B Iris vLLM:** `/ahmed/ifbench-vllm-8b-20k-r1`
- Completed successfully and wrote `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_8b.jsonl`.
- Local copy: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_8b.jsonl`.
- Scored 20,000 rollouts:
  - empty: 31
  - strict pass-all: 3,554 / 20,000 = 17.77%
  - loose pass-all: 4,172 / 20,000 = 20.86%
  - SFT passers: 3,556 prompt-level winners from this single-model view
  - DPO pairs: 0, expected because one model alone cannot provide chosen/rejected contrast.

**70B Iris vLLM:** `/ahmed/ifbench-vllm-70b-20k-v5p8-r3`
- Failed after ~9m54s with exit 1, same broad shape as r2.
- r3 was already the reduced-memory fallback (`v5p-8`, TP=4, server `max_num_seqs=64`, client concurrency 256).
- Decision for tonight: stop spending Iris retries on 70B unless the postmortem shows a trivial command-line bug. Use Together 70B batch as the primary 70B source.

**Provider poll at 2026-04-27T08:24:00Z:**
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

**Next:** inspect 70B r3 logs once for a simple fix/no-fix classification, then continue polling provider batches and score/extract any completed provider outputs.

### 2026-04-27 08:26 UTC — 70B Iris r4 launched with startup heartbeat

**r3 postmortem:**
- `job summary` shows task duration ~9m49s and peak container memory only ~15.5GB, so this was not a container OOM.
- Iris logs contain only wrapper setup, prompt load, and `Starting vLLM native server`; no Python traceback or vLLM stderr surfaced.
- The failure timing is suspiciously close to a 10-minute no-output window while the 70B server is loading/precompiling.

**Code fix before one final retry:**
- Added a 60s startup heartbeat to `experiments/ifbench/rollout/vllm_iris_infer.py` while `VllmEnvironment.__enter__` blocks.
- `uv run python -m py_compile experiments/ifbench/rollout/vllm_iris_infer.py` passes.

**70B r4 submitted:**
- Job: `/ahmed/ifbench-vllm-70b-20k-v5p8-r4`
- TPU: `v5p-8`
- Priority: `interactive`
- Tensor parallel: 4
- Server max sequences: 64
- Client concurrency: 256
- Output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b.jsonl`
- Monitor state: `scratch/20260427T082625Z_ifbench_vllm_70b_v5p8_r4_monitoring_state.json`

**Rule for r4:** if it fails with real vLLM/model/TPU diagnostics, stop Iris 70B attempts tonight and rely on Together 70B batch.

### 2026-04-27 08:28 UTC — r4 heartbeat surfaced real first-attempt failure

**70B r4 state:** still `JOB_STATE_RUNNING`, but with `preemption_count=1`.

**First r4 attempt outcome:**
- The startup-log patch worked: vLLM stdout/stderr is now visible in Iris logs.
- First attempt failed with a bad-node/device-busy TPU signature:
  - `TPU initialization failed: open(/dev/vfio/1): Device or resource busy`
  - `Couldn't open iommu group /dev/vfio/1`
  - Iris reported: `TPU bad-node signature detected ("Couldn't open iommu group"); reporting as worker failure`
- Iris automatically restarted the job on a new worker/attempt. No manual cluster mutation taken.

**Provider poll at 2026-04-27T08:28:15Z:**
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

**Next:** watch the second r4 attempt. If it reaches vLLM readiness, let it race; if it fails with a non-bad-node model/runtime issue, stop Iris 70B attempts and rely on Together.

### 2026-04-27 08:30 UTC — r4 second attempt alive with heartbeat

**70B r4 state:** `JOB_STATE_RUNNING`, `preemption_count=1`.

**Second attempt:**
- Started at ~08:27:30 UTC after the first bad-node attempt.
- vLLM startup heartbeat is working:
  - `2026-04-27 08:28:47 INFO ... Still waiting for vLLM startup for meta-llama/Llama-3.3-70B-Instruct`
  - `2026-04-27 08:29:47 INFO ... Still waiting for vLLM startup for meta-llama/Llama-3.3-70B-Instruct`
- No 70B rollout object exists yet at `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b.jsonl`.

**Interpretation:** the second r4 attempt is past wrapper setup and into vLLM startup/precompile. The heartbeat should prevent silent-startup watchdog issues while we learn whether 70B actually starts on v5p-8.

### 2026-04-27 08:35 UTC — r4 still in vLLM startup; providers still pending

**70B r4 state:** `JOB_STATE_RUNNING`, `preemption_count=1`.

**Second attempt heartbeat:**
- Latest visible heartbeat: `2026-04-27 08:32:47 INFO ... Still waiting for vLLM startup...`
- This attempt has now survived beyond the old silent-window failure point because the wrapper is emitting heartbeat logs.
- No 70B output object yet.

**Provider poll at 2026-04-27T08:34:51Z:**
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

**Next:** continue r4/provider polling. First finished 70B/Gemini source unlocks partial cross-model extraction with the completed Iris 8B rollouts.

### 2026-04-27 08:41 UTC — Iris 70B closed for tonight

**70B r4 final state:** `JOB_STATE_FAILED`.

**What happened:**
- Attempt 0 failed on bad TPU node/device-busy (`Couldn't open iommu group`); Iris counted it as worker failure/preemption.
- Attempt 1 ran on the replacement attempt for ~9m43s, emitted startup heartbeats through 08:32:47 UTC, then exited 1 with no rollout object.
- `job summary`: peak container memory ~15.6GB, so not a host-memory/cgroup OOM.
- No `gs://.../rollouts_llama_3_3_70b.jsonl` object was written.

**Decision:** close Iris 70B attempts for tonight. The requested Iris race produced a useful 8B result but did not get 70B running on available `v5p-8`; use Together 70B batch as the 70B source.

**Provider poll:**
- 08:40 UTC retry recovered from a transient Gemini 503.
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

**Next:** continue provider polling. When any provider batch completes, download → score → extract against the completed Iris 8B rollouts.

### 2026-04-27 08:42 UTC — provider poll path hardened

**Small code hardening while provider batches run:**
- Patched `experiments/ifbench/rollout/overnight.py` so `poll` records transient provider poll errors as `poll_error` and exits 2 instead of crashing the monitor path.
- Patched `download` to skip a handle on transient poll error and continue checking the other handles.
- `uv run python -m py_compile experiments/ifbench/rollout/overnight.py` passes.

**Provider poll at 2026-04-27T08:41:28Z:**
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

### 2026-04-27 08:42 UTC — local rollout tests still green

**Validation while provider batches run:**
- `uv run --with pytest python -m pytest experiments/ifbench/rollout/test_backend.py experiments/ifbench/rollout/test_extract.py -o addopts=''`
- Result: 15 passed, 2 warnings.

**Git status note:** this branch already has broad untracked IFBench/project files and tracked changes to `pyproject.toml` / `uv.lock`; do not infer all dirtiness came from this polling hardening.

### 2026-04-27 08:47 UTC — provider poll unchanged

**Provider poll at 2026-04-27T08:47:09Z:**
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

**Finished 20k local artifacts remain unchanged:**
- Iris 8B rollouts: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_8b.jsonl`.
- Iris 8B score: `/tmp/ifbench_port/overnight_20k/iris/score_llama_3_8b.json`.

### 2026-04-27 08:52 UTC — provider poll unchanged

**Provider poll at 2026-04-27T08:52:27Z:**
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

**Action:** keep polling; no local/provider output to download yet.

### 2026-04-27 09:02 UTC — provider poll unchanged

**Provider poll at 2026-04-27T09:01:51Z:**
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

**Elapsed since submission:** ~1h18m. Continue stable-cadence polling.

### 2026-04-27 09:14 UTC — provider poll unchanged

**Provider poll at 2026-04-27T09:13:36Z:**
- Together 70B batch: `IN_PROGRESS`.
- Together 8B batch: `IN_PROGRESS`.
- Gemini Flash file batch: `BATCH_STATE_RUNNING`.

**Action:** continue stable-cadence polling. No provider output is downloadable yet.

### 2026-04-27 16:02 UTC — 70B + Gemini provider outputs downloaded and partially extracted

**Provider state changed at 2026-04-27T15:59:50Z:**
- Together 70B batch `9afbd565-a6be-48c8-beaf-29be61d2b81d`: `COMPLETED`, but under-delivered.
- Together 8B batch `9b109f48-a95a-4fb6-a658-7d16c52053d3`: still `IN_PROGRESS`.
- Gemini Flash file batch `batches/zqp38vk6tqywpekvf9zcbr3v4o72m7n7bo1b`: `BATCH_STATE_SUCCEEDED`.

**Downloaded outputs:**
- `/tmp/ifbench_port/overnight_20k/rollouts/together_meta-llama_Llama-3.3-70B-Instruct-Turbo.jsonl`
  - 2,299 rollouts only; Together reported completed but output contained 2,299/20,000 rows.
- `/tmp/ifbench_port/overnight_20k/rollouts/gemini_gemini-3-flash-preview.jsonl`
  - 20,000 rollouts.
- Existing Iris 8B:
  - `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_8b.jsonl`
  - 20,000 rollouts.

**Per-model scoring on partial 3-source set:**

| Model/source | n | empty | strict pass-all | loose pass-all | notes |
|---|---:|---:|---:|---:|---|
| Iris `meta-llama/Meta-Llama-3-8B-Instruct` | 20,000 | 31 | 3,556 (17.78%) | 4,173 (20.87%) | local vLLM |
| Together `Llama-3.3-70B-Instruct-Turbo` | 2,299 | 0 | 747 (32.49%) | 844 (36.71%) | partial provider output only |
| Gemini `gemini-3-flash-preview` file batch | 20,000 | 74 | 3,675 (18.38%) | 4,030 (20.15%) | default/dynamic thinking; 18,874,833 thinking tokens |

**Written extraction artifacts:**
- `/tmp/ifbench_port/overnight_20k/outputs_partial_iris8b_together70b_gemini/dpo_pairs.jsonl`
- `/tmp/ifbench_port/overnight_20k/outputs_partial_iris8b_together70b_gemini/sft_examples.jsonl`
- `/tmp/ifbench_port/overnight_20k/outputs_partial_iris8b_together70b_gemini/summary.json`

**Partial extraction result (use the written artifact counts):**
- DPO pairs: 3,979 / 20,000 = 19.895%
- SFT examples: 7,978
- no passers: 14,251
- no failers: 1,770
- yield by constraints: `{1: 1751, 2: 1248, 3: 655, 4: 275, 5: 50}`

**Note:** the score-only summary and extract summary differ by 1 pair/SFT example, likely from verifier edge-case behavior across repeated scoring. The on-disk extraction artifacts are the source of truth for dataset counts.

**Next:** keep polling Together 8B-Lite. Also inspect the Together 70B batch metadata/error surface to understand why the completed job produced only 2,299 rows.

### 2026-04-27 16:04 UTC — Together 70B under-delivery explained

**Together 70B batch metadata:**
- Batch id: `9afbd565-a6be-48c8-beaf-29be61d2b81d`
- Status: `COMPLETED`
- Progress: `100`
- Output file: `file-e124ff53-1c95-495c-89a4-6b4c60038145`
- Error file: `file-c48ef989-6965-4009-a8e6-9be26d90fbe8`
- Completed at provider timestamp: `2026-04-27T09:18:27.995414Z`

**Error file:**
- Saved locally: `/tmp/ifbench_port/overnight_20k/rollouts/together_70b_error_file.jsonl`
- Error rows: 17,701
- Successful rows: 2,299
- Sampled error shape: `{"code": "batch_client_error", "message": "Internal Server Error"}`

**Interpretation:** this is provider internal under-delivery, not model refusal/empty output. Keep these failed rows out of model-quality scoring rather than converting them to model failures.

**Together 8B-Lite metadata:** still `IN_PROGRESS`, progress `55.89%`.

**Code hardening:** patched `TogetherBackend.poll` to log the provider `progress` field when present; `py_compile` passes.

### 2026-04-27 16:25 UTC — start focused Iris/vLLM 70B debug

**User correction:** 70B inference should work on `v5p-8`; do not accept the overnight failure as final.

**Debug objective:** isolate why `meta-llama/Llama-3.3-70B-Instruct` did not become ready under the IFBench Iris vLLM runner.

**Known evidence from overnight:**
- TP=8 was invalid on this `v5p-8` host because JAX exposed 4 devices.
- TP=4 got past the mesh issue.
- One r4 attempt hit a real bad-node/device-busy TPU error (`Couldn't open iommu group /dev/vfio/1`) and Iris retried.
- The replacement attempt emitted heartbeats but no durable vLLM stderr on final exit.
- Host cgroup memory was not the issue (peak ~15.6GB).

**Plan now:**
1. Add a tiny startup probe that preserves native vLLM stdout/stderr to GCS on every failure.
2. Run one `v5p-8`, TP=4, 70B startup probe with interactive priority and one prompt only.
3. If the probe starts, diff flags against the 20k runner. If it fails, use durable vLLM logs to fix the actual launch issue.

**Debug log:** `docs/debug-log-ifbench-vllm-70b-v5p8.md`.

### 2026-04-27 16:27 UTC — 70B startup probe submitted

**Added probe script:** `experiments/ifbench/rollout/vllm_70b_startup_probe.py`
- Runs `vllm serve` directly instead of through `VllmEnvironment`.
- Removes stale `/tmp/libtpu_lockfile` before startup.
- Polls `/v1/models`, sends one `pong` completion if ready.
- Always writes `status.json`, `stdout.log`, `stderr.log`, and `tail.txt` to GCS.

**Validation:** `uv run python -m py_compile experiments/ifbench/rollout/vllm_70b_startup_probe.py` passes.

**Iris job submitted:**
- Job: `/ahmed/ifbench-vllm-70b-probe-lock-r1`
- TPU: `v5p-8`
- Priority: `interactive`
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Flags: TP=4, `max_model_len=4096`, `max_num_seqs=16`, timeout 1800s
- Durable debug output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_20260427T162702Z`
- Monitor state: `scratch/20260427T162721Z_ifbench_vllm_70b_probe_lock_r1_monitoring_state.json`

### 2026-04-27 16:37 UTC — 70B v5p-8 failure root cause: HF cache on root overlay

**Probe r1 result:** vLLM got past the TPU/JAX part on `v5p-8`:
- JAX exposed 4 devices on the host, so TP=4 is the right tensor-parallel size.
- vLLM initialized the TP=4 mesh and started loading `meta-llama/Llama-3.3-70B-Instruct`.
- Failure happened during HF model shard download, not TPU serving.

**Root cause:** Hugging Face cache was under `/root/.cache/huggingface`, which lives on the small root overlay. Even though the Iris job requested `--disk 750GB`, HF was not writing to a disk-backed Iris workdir/cache mount. Durable stderr ended with `No space left on device (os error 28)`.

**Code fix now:**
- `experiments/ifbench/rollout/vllm_70b_startup_probe.py` now forces `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`, and `VLLM_ASSETS_CACHE` under `/app/.hf_cache`, and writes `df`, `mount`, and cache `du` into `status.json`.
- `experiments/ifbench/rollout/vllm_iris_infer.py` now applies the same model-cache placement before starting `VllmEnvironment`.
- Validation: `uv run python -m py_compile experiments/ifbench/rollout/vllm_70b_startup_probe.py experiments/ifbench/rollout/vllm_iris_infer.py` passes.

**Next:** submit cache-fixed r2 probe on `v5p-8`, TP=4, one prompt, interactive priority. If it serves, relaunch the 20k 70B Iris runner with the same cache fix.

### 2026-04-27 16:38 UTC — cache-fixed 70B probe submitted

**Iris job submitted:**
- Job: `/ahmed/ifbench-vllm-70b-probe-cache-r2`
- TPU: `v5p-8`
- Priority: `interactive`
- Flags: TP=4, `max_model_len=4096`, `max_num_seqs=16`, timeout 2400s
- Cache: `/app/.hf_cache`
- Durable debug output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_cache_20260427T163738Z`
- Monitor state: `scratch/20260427T163753Z_ifbench_vllm_70b_probe_cache_r2_monitoring_state.json`

**Watch condition:** if `/v1/models` becomes ready and the one-prompt completion succeeds, relaunch the 20k 70B Iris inference job with the production runner's cache fix.

### 2026-04-27 16:43 UTC — v6e-8 question answered

**What happened last night:** I did not actually request `v6e-8`. The first draft plan said `v6e-8`, but the final revised plan said to try `v6e-4` first and fall back to `v5p-8` if unavailable. The submitted Iris jobs used `--tpu v6e-4,v5p-8`, and Iris scheduled both initial jobs on `v6e-4`.

**Consequence:** because `v6e-4` was available, the fallback list never tried `v5p-8` for the first launch. 8B succeeded on `v6e-4`; 70B failed before server readiness on `v6e-4`, likely because 70B BF16 was too tight for the 4-chip v6e host. I then moved 70B to explicit `v5p-8`.

**If we want `v6e-8`:** submit it explicitly as `--tpu v6e-8`. Do not mix it in the same fallback list with `v6e-4`/`v5p-8`; Iris validates that fallback alternatives have the same single-VM chip shape, and `v6e-8` is an indivisible 8-chip VM while `v6e-4`/`v5p-8` expose 4 chips per VM in this scheduler path.

### 2026-04-27 16:44 UTC — explicit v6e-8 70B probe submitted

**Iris job submitted:**
- Job: `/ahmed/ifbench-vllm-70b-probe-v6e8-r1`
- TPU: `v6e-8`
- Priority: `interactive`
- Flags: TP=8, `max_model_len=4096`, `max_num_seqs=16`, timeout 2400s
- Cache: `/app/.hf_cache`
- Memory/disk request: 360GB RAM, 750GB disk
- Durable debug output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_v6e8_20260427T164345Z`
- Monitor state: `scratch/20260427T164421Z_ifbench_vllm_70b_probe_v6e8_r1_monitoring_state.json`

**Parallel watch:** keep monitoring existing `v5p-8` job `/ahmed/ifbench-vllm-70b-probe-cache-r2` and new `v6e-8` job together. Whichever gets `/v1/models` + one-prompt completion first becomes the candidate for full 20k 70B Iris inference.

### 2026-04-27 16:46 UTC — v6e-8 r1 disk request fixed, r2 submitted

**v6e-8 r1:** `/ahmed/ifbench-vllm-70b-probe-v6e8-r1` accepted the `v6e-8` topology but stayed pending. Iris pending reason:
- requested disk: 750GB
- v6e scale-group available disk: 100GB
- `insufficient_resources: disk`

**Action:** stopped pending r1 and submitted r2 with the same model/TP/cache settings but `--disk 100GB`.

**Iris job submitted:**
- Job: `/ahmed/ifbench-vllm-70b-probe-v6e8-r2`
- TPU: `v6e-8`
- Priority: `interactive`
- Flags: TP=8, `max_model_len=4096`, `max_num_seqs=16`, timeout 2400s
- Cache: `/app/.hf_cache`
- Memory/disk request: 360GB RAM, 100GB disk
- Durable debug output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_v6e8_20260427T164600Z`
- Monitor state: `scratch/20260427T164551Z_ifbench_vllm_70b_probe_v6e8_r2_monitoring_state.json`

**v5p-8 status at resubmit:** `/ahmed/ifbench-vllm-70b-probe-cache-r2` still running, last heartbeat 360s waiting for `/v1/models`; no error yet.

### 2026-04-27 16:47 UTC — parallel probe monitor update

**v6e-8 r2:** `/ahmed/ifbench-vllm-70b-probe-v6e8-r2`
- State: pending
- Pending reason: waiting for workers in scale group `tpu_v6e-preemptible_8-europe-west4-a` to become ready.
- Interpretation: request is now schedulable; this is capacity/startup wait, not a malformed resource request.

**v5p-8 r2:** `/ahmed/ifbench-vllm-70b-probe-cache-r2`
- State: running
- Latest heartbeat: 540s waiting for `http://127.0.0.1:8000/v1/models`.
- Peak memory: ~145GB
- Disk/resource usage reported by Iris: ~139GB
- No stderr error beyond TensorFlow/Transformers warnings; last native vLLM line remains `Using Pallas V1 backend`.

### 2026-04-27 16:49 UTC — v6e-8 worker allocated

**v6e-8 r2:** `/ahmed/ifbench-vllm-70b-probe-v6e8-r2`
- State: running
- Worker: `marin-tpu-v6e-preemptible-8-europe-west4-20260427-1645-2dd38d0e-worker-0`
- Logs: deps synced, venv active, user command entered.

**v5p-8 r2:** `/ahmed/ifbench-vllm-70b-probe-cache-r2`
- State: running
- Latest heartbeat: 660s waiting for `/v1/models`.
- Still no new vLLM stderr; current native startup tail remains at model load/Pallas backend.

### 2026-04-27 16:53 UTC — v6e-8 passes the previous v5p stall point

**v6e-8 r2 native vLLM status:**
- TPU info: `v6e-8`, `num_chips=8`, `num_cores_per_chip=1`.
- Cache placement works: `/app` tmpfs is ~709GB; `/app/.hf_cache` is 132GB.
- TP=8 mesh initialized: `Mesh('data': 1, 'model': 8)`.
- Weights downloaded/loaded:
  - download time: 66.6s
  - safetensor load: 15.4s
  - all 30 shards loaded.
- Model init HBM: ~16.46 / 31.25 GiB per chip.
- vLLM reports KV capacity: 305,664 tokens, max concurrency 74.62x at 4,096 tokens/request.
- Precompile has started and completed at least one sample shape.

**v5p-8 contrast:** still alive, but native stdout remains stuck after `Using Pallas V1 backend`; it has not emitted the `weight_utils.py` download/load lines that `v6e-8` reached quickly.

### 2026-04-27 16:54 UTC — monitor update

**v6e-8 r2:** `/ahmed/ifbench-vllm-70b-probe-v6e8-r2`
- State: running
- Latest heartbeat: 240s waiting for `/v1/models`.
- Current memory: ~149GB; peak memory: ~281GB under the 360GB request.
- Still inside startup/precompile; no failure.

**v5p-8 r2:** `/ahmed/ifbench-vllm-70b-probe-cache-r2`
- State: running
- Latest heartbeat: 900s waiting for `/v1/models`.
- Still no native stdout past `Using Pallas V1 backend`.

### 2026-04-27 16:56 UTC — v6e-8 probe succeeded; v5p-8 failed by host-memory limit

**v6e-8 r2:** `/ahmed/ifbench-vllm-70b-probe-v6e8-r2`
- Final state: succeeded
- Startup duration: ~6m44s
- Ready: yes
- One-prompt chat completion: HTTP 200, response `pong`
- Peak host memory: ~281GB under 360GB request
- Durable status: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_v6e8_20260427T164600Z/status.json`

**v5p-8 r2:** `/ahmed/ifbench-vllm-70b-probe-cache-r2`
- Final state: failed
- Failure: OOM killed by container memory limit
- Peak host memory: ~237.6GB against 240GB request
- Important interpretation: this was host/container memory pressure after placing the 132GB HF cache under `/app` tmpfs, not evidence that v5p HBM cannot run 70B.
- Durable status: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_cache_20260427T163738Z/status.json`

**Next actions now:**
1. Launch real 20k 70B Iris inference on explicit `v6e-8`, interactive priority, TP=8.
2. Relaunch v5p probe with a fair host-memory request (400GB) to test Ahmed's assertion that 70B inference works on `v5p-8`.

### 2026-04-27 16:58 UTC — full 20k v6e-8 run + fair v5p retry submitted

**Full 20k 70B Iris inference:**
- Job: `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- TPU: `v6e-8`
- Priority: `interactive`
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Flags: TP=8, client concurrency=256, server `max_num_seqs=64`, `max_model_len=4096`
- Memory/disk request: 500GB RAM, 100GB disk
- Output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`
- Monitor state: `scratch/20260427T165724Z_ifbench_vllm_70b_20k_v6e8_r1_monitoring_state.json`

**Fair v5p-8 retry:**
- Job: `/ahmed/ifbench-vllm-70b-probe-v5p8-400gb-r3`
- TPU: `v5p-8`
- Priority: `interactive`
- Flags: TP=4, `max_model_len=4096`, `max_num_seqs=16`, timeout 2400s
- Memory/disk request: 400GB RAM, 100GB disk
- Durable debug output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_v5p8_400gb_20260427T165651Z`
- Monitor state: `scratch/20260427T165740Z_ifbench_vllm_70b_probe_v5p8_400gb_r3_monitoring_state.json`

**Startup check:**
- Full v6e 20k job is running, loaded all 20,000 prompts, and started the vLLM native server.
- v5p 400GB retry is running on `marin-tpu-v5p-preemptible-8-us-east5-a-20260427-1648-6b16264e-worker-0` and started `vllm serve`.

### 2026-04-27 17:01 UTC — v5p confirmed past previous stall

**Full v6e 20k job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- State: running
- Native vLLM logs show TP=8 mesh initialized with `max_num_seqs=64`.
- Weight download/load completed:
  - download time: ~66s
  - load time: ~21s
- Still waiting for `/v1/models`; likely in model init/precompile.

**v5p 400GB probe:** `/ahmed/ifbench-vllm-70b-probe-v5p8-400gb-r3`
- State: running
- Native vLLM logs now show TP=4 mesh initialized and all 30 shards loaded.
- Weight download/load completed:
  - download time: ~80s
  - load time: ~29s
- This passes the previous r2 stall point and supports Ahmed's claim: 70B can get through model load on `v5p-8`; r2 was host-memory/cache pressure, not a v5p impossibility.

### 2026-04-27 17:03 UTC — monitor update

**Full v6e 20k job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- State: running
- Latest heartbeat: still waiting for vLLM startup.
- Native vLLM logs: weights downloaded and loaded; likely in model init/precompile.
- Peak host memory: ~284GB under 500GB request.

**v5p 400GB probe:** `/ahmed/ifbench-vllm-70b-probe-v5p8-400gb-r3`
- State: running
- Latest heartbeat: 240s waiting for `/v1/models`.
- Peak host memory: ~281GB under 400GB request.
- No OOM; fair-memory retry remains viable.

### 2026-04-27 17:05 UTC — native precompile status

**Full v6e 20k job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- Native logs show full `max_num_seqs=64` precompile is underway.
- KV cache initialized at ~28.14 / 31.25 GiB per v6e chip.
- `num_reqs` 8/16/32/64 sampling + logprob shapes compiled; backbone token shapes through 2048 compiled; select/logits shapes compiling.
- No failure; still waiting for `/v1/models`.

**v5p 400GB probe:** `/ahmed/ifbench-vllm-70b-probe-v5p8-400gb-r3`
- Native logs show model init succeeded with ~32.89 / 95.74 GiB per v5p chip before KV allocation.
- KV cache initialized at ~86.18 / 95.74 GiB per chip.
- Reported max concurrency at 4096 tokens/request: 170.50x.
- Backbones are compiling; no OOM under 400GB host memory.

### 2026-04-27 17:06 UTC — v5p probe succeeded; v6e full rollout serving

**Full v6e 20k job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- State: running on `v6e-8` interactive.
- vLLM is ready and serving requests.
- Latest observed progress: `200/20000` at ~3.1 req/s.
- One prompt hit a vLLM 400 because `max_tokens=1024` exceeded the remaining context window for a 3212-token input under `max_model_len=4096`; runner logged the warning and continued.
- Resource state: peak host memory ~284GB under the 500GB request.

**v5p 400GB probe:** `/ahmed/ifbench-vllm-70b-probe-v5p8-400gb-r3`
- State: succeeded.
- Worker: `marin-tpu-v5p-preemptible-8-us-east5-a-20260427-1648-6b16264e-worker-0`.
- vLLM became ready after ~465s, served the one-prompt `pong` probe, and wrote diagnostics to:
  `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_v5p8_400gb_20260427T165651Z`
- Peak host memory: ~281GB under the 400GB request.
- Conclusion: 70B inference does work on `v5p-8`. The earlier failures were cache/root-disk pressure plus too-low container memory, not a v5p capability issue.

### 2026-04-27 17:08 UTC — v6e rollout quality watch

**Full v6e 20k job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- Latest observed progress: `400/20000` at ~3.7 req/s.
- Context-window 400s observed so far: 2 in the first 400 prompts.
- Cause: fixed `max_tokens=1024` with `max_model_len=4096`; long inputs with ~3200-3300 tokens leave less than 1024 output tokens.
- Current action: keep running and monitor the error rate. If this becomes material, relaunch with a larger context window and/or adaptive per-request `max_tokens`.

### 2026-04-27 17:10 UTC — decision: do not kill v6e r1 for long-tail prompts

**Full v6e 20k job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- Latest observed progress: `900/20000` at ~4.1 req/s.
- Context-window 400s observed so far: 6 in the first 900 prompts (~0.7%).
- Prompt length check on the 20k prepared set:
  - median: 642 chars
  - p95: 2107 chars
  - p99: 6810 chars
  - p99.5: 13192 chars
  - p99.9: 41720 chars
  - max: 313553 chars
  - `>12000` chars: 114 prompts; `>20000` chars: 49 prompts; `>40000` chars: 21 prompts
- Decision: keep the fast `max_model_len=4096` v6e run alive. Relaunching the entire 20k at 16k/32k context would slow the common case for a small long-tail.
- Follow-up after r1 completes: identify `finish_reason=http_400` rows, rerun only those prompt IDs with a long-context low-server-concurrency vLLM job, then merge replacements into the rollout file before scoring/pair extraction.

### 2026-04-27 17:12 UTC — v6e r1 remains stable

**Full v6e 20k job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- Latest observed progress: `1600/20000` at ~4.2 req/s.
- Context-window 400s observed in the log sample: 12 by `1600/20000` (~0.75%).
- Action unchanged: let r1 finish, then repair only `http_400` rows with a long-context rerun. This avoids throwing away the fast path for the 99%+ short/medium prompts.

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

### 2026-04-27 17:13 UTC — detached monitor: v6e rollout checkpoint

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- State: `running`
- Latest sampled progress: `2000/20000` at ~4.2 req/s.
- Context-window warnings in last sampled log window: 13.
- Peak host memory: 283648 MB.

### 2026-04-27 17:15 UTC — persistent monitor moved to tmux

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- `nohup` was reaped by the tool environment after the first checkpoint, so the detached monitor is now running inside tmux session `ifbench_v6e_monitor`.
- Monitor script: `scratch/monitor_ifbench_v6e_r1.py`.
- State file: `scratch/monitor_ifbench_v6e_r1_state.json`.
- Behavior: append UTC checkpoints to this logbook every ~1000 prompts, and append a terminal entry when Iris reports non-running state.

### 2026-04-27 17:21 UTC — current wait state

**Iris 70B v6e full rollout:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- State: `JOB_STATE_RUNNING`, task state `running`.
- Preemption count: 1.
- Previous worker `marin-tpu-v6e-preemptible-8-europe-west4-20260427-1645-2dd38d0e-worker-0` failed with `worker ping threshold exceeded` after reaching ~`2000/20000`.
- Iris automatically moved the task to new worker `marin-tpu-v6e-preemptible-8-europe-west4-20260427-1716-2c14b3a6-worker-0`.
- New task restarted from the top: dependency sync completed, cache configured under `/app/.hf_cache`, 20k prompts loaded, and native vLLM server startup began at `2026-04-27 17:20:54 UTC`.
- No full 70B output object exists yet under `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/`; the runner writes only at job completion, so the preempted ~2k completions were not durable.

**Completed / not waiting:**
- `v5p-8` 70B probe succeeded with 400GB memory. This resolves the "does 70B work on v5p-8?" question.
- Iris 8B 20k rollout exists at `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_8b.jsonl`.
- Gemini 20k file-batch and Together 70B partial output were already downloaded/scored earlier.

**Provider polling caveat:**
- A poll attempt from this shell at `17:21 UTC` returned Together `401 Unauthorized` for both batch ids and Gemini `404 Not Found` for the batch endpoint.
- Local env has `TOGETHER_API_KEY` and `GEMINI_API_KEY` set, but Together rejects the current key. Treat this as a polling/auth issue in the current session, not new evidence about provider job state.
- Last trustworthy provider state remains the earlier downloaded/scored state: Gemini 20k complete; Together 70B complete but under-delivered with 2,299/20,000 successes; Together 8B batch was still `IN_PROGRESS` at the last successful poll.

### 2026-04-27 17:25 UTC — COMPREHENSIVE HANDOFF SUMMARY FOR NEXT AGENT

This section is the short path for the next agent. Read this first, then drill into the timestamped entries above only if needed.

## Goal

Build an empirical IFBench-derived training dataset for DPO/SFT:
- Generate rollouts on a capped 20k prompt slice from `IF_multi_constraints_upto5`.
- Use verifier scores to create:
  - DPO pairs: chosen = passing / higher verifier score; rejected = failing / lower verifier score.
  - SFT examples: chosen/pass examples.
- Keep tonight's provider spend under the agreed cap; no Gemini Pro.
- Compare provider batch inference vs Iris local vLLM inference, especially for Llama 3.3 70B and Llama 3 8B.

## Inherited Baseline Before Tonight

The 100-prompt smoke test was already completed before the overnight run:
- 3 models: Together 70B Turbo, Together 8B Lite, Gemini 3 Flash.
- End-to-end pipeline worked: rollouts -> verifier -> DPO/SFT extraction.
- Pair yield: 34/100 = 34%, clearing the 30% smoke gate.
- SFT examples: 58.
- Strict pass-all:
  - 70B Turbo: 26%.
  - Gemini Flash thinking=high: 22%.
  - 8B Lite: 10%.
- Loose scoring lifted all models slightly; later Codex rescore reproduced the smoke extraction exactly.

## What Codex Did Overnight

### 1. Added a reproducible overnight orchestration CLI

Main file:
- `experiments/ifbench/rollout/overnight.py`

Capabilities added/used:
- `sweep-gemini`: run 100-prompt Gemini Flash thinking sweep by sync API.
- `submit`: submit 20k provider batches.
- `poll`: poll provider batch handles and record `poll_error` instead of crashing on transient errors.
- `download`: download completed provider batch outputs while skipping still-pending or poll-error handles.
- `score` / extraction helpers: score rollouts and build partial artifacts.

Validation run:
- `uv run python -m experiments.ifbench.rollout.overnight --help`
- `uv run python -m py_compile experiments/ifbench/rollout/overnight.py experiments/ifbench/rollout/gemini_backend.py`
- `uv run --with pytest python -m pytest experiments/ifbench/rollout/test_backend.py experiments/ifbench/rollout/test_extract.py -o addopts=''`
- Result: 15 passed, 2 warnings.

Dependency update:
- Added `together` SDK dependency in `pyproject.toml` / `uv.lock` so `TogetherBackend` imports under `uv run`.

### 2. Gemini 3 Flash thinking sweep

Command:
- `uv run python -m experiments.ifbench.rollout.overnight --work-dir /tmp/ifbench_port/overnight_20k sweep-gemini --concurrency 10 --force`

Outputs:
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/rollouts_flash_minimal.jsonl`
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/rollouts_flash_low.jsonl`
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/rollouts_flash_medium.jsonl`
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/rollouts_flash_high.jsonl`
- `/tmp/ifbench_port/overnight_20k/gemini_flash_sweep/decision.json`

Result on same 100 prompts:

| Thinking | Strict pass-all | Loose pass-all | Input tok | Output tok | Thinking tok |
|---|---:|---:|---:|---:|---:|
| `minimal` | 40/100 | 49/100 | 19,833 | 29,808 | 0 |
| `low` | 26/100 | 28/100 | 20,813 | 8,831 | 88,681 |
| `medium` | 19/100 | 20/100 | 19,833 | 5,416 | 94,020 |
| `high` | 20/100 | 21/100 | 19,833 | 4,826 | 95,726 |

Decision:
- `minimal` wins outright for sync API use.
- But Gemini file batch rejects row-level `thinkingConfig`, so the 20k Gemini batch cannot use `minimal`; it uses Gemini Flash default/dynamic thinking.

### 3. Submitted 20k provider batches

Prompt source:
- Local source slice: first 20,000 rows from `/tmp/ifbench_port/prepared_v2/train.jsonl`.
- GCS staged prompt file for Iris: `gs://marin-us-central2/scratch/ifbench/overnight_20k/prompts_20k.jsonl`.
- Local/GCS size: ~34.18 MiB.

Handle file:
- `/tmp/ifbench_port/overnight_20k/handles.json`

Provider handles:

| Backend | Model | Batch id | Requests | Current trustworthy state |
|---|---|---|---:|---|
| Together | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `9afbd565-a6be-48c8-beaf-29be61d2b81d` | 20,000 | `COMPLETED`, but under-delivered |
| Together | `meta-llama/Meta-Llama-3-8B-Instruct-Lite` | `9b109f48-a95a-4fb6-a658-7d16c52053d3` | 20,000 | last trustworthy poll: `IN_PROGRESS`, progress ~55.89% |
| Gemini | `gemini-3-flash-preview` | `batches/zqp38vk6tqywpekvf9zcbr3v4o72m7n7bo1b` | 20,000 | `BATCH_STATE_SUCCEEDED`, downloaded |

Important provider caveat:
- At `2026-04-27 17:21 UTC`, polling from the current shell returned Together `401 Unauthorized` and Gemini `404 Not Found`.
- Local env still has `TOGETHER_API_KEY` and `GEMINI_API_KEY` set, but Together rejects the current key. Treat this as current-session auth/polling trouble, not as evidence the provider jobs changed.

### 4. Provider outputs downloaded/scored so far

Downloaded:
- Gemini Flash 20k:
  - `/tmp/ifbench_port/overnight_20k/rollouts/gemini_gemini-3-flash-preview.jsonl`
  - 20,000 rollouts.
- Together 70B partial:
  - `/tmp/ifbench_port/overnight_20k/rollouts/together_meta-llama_Llama-3.3-70B-Instruct-Turbo.jsonl`
  - 2,299 rollouts only.
- Together 70B error file:
  - `/tmp/ifbench_port/overnight_20k/rollouts/together_70b_error_file.jsonl`
  - 17,701 error rows, all provider-side `batch_client_error` / `Internal Server Error`.
- Iris 8B:
  - GCS: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_8b.jsonl`
  - Local: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_8b.jsonl`
  - 20,000 rollouts.

Partial scoring/extraction:
- Output dir: `/tmp/ifbench_port/overnight_20k/outputs_partial_iris8b_together70b_gemini/`
- Files:
  - `dpo_pairs.jsonl`
  - `sft_examples.jsonl`
  - `summary.json`

Per-model partial scoring:

| Model/source | n | empty | strict pass-all | loose pass-all | notes |
|---|---:|---:|---:|---:|---|
| Iris `meta-llama/Meta-Llama-3-8B-Instruct` | 20,000 | 31 | 3,556 (17.78%) | 4,173 (20.87%) | local vLLM |
| Together `Llama-3.3-70B-Instruct-Turbo` | 2,299 | 0 | 747 (32.49%) | 844 (36.71%) | partial provider output only |
| Gemini `gemini-3-flash-preview` file batch | 20,000 | 74 | 3,675 (18.38%) | 4,030 (20.15%) | default/dynamic thinking; 18,874,833 thinking tokens |

Partial extraction artifact counts:
- DPO pairs: 3,979 / 20,000 = 19.895%.
- SFT examples: 7,978.
- no passers: 14,251.
- no failers: 1,770.
- yield by constraints: `{1: 1751, 2: 1248, 3: 655, 4: 275, 5: 50}`.

Do not treat Together 70B's 17,701 provider-internal errors as model failures. Keep them out of model-quality scoring.

### 5. Iris 8B local vLLM run

Final successful job:
- `/ahmed/ifbench-vllm-8b-20k-r1`

Config:
- Model: `meta-llama/Meta-Llama-3-8B-Instruct`.
- TPU: `v6e-4`.
- TP=4.
- Client concurrency 256.
- Output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_8b.jsonl`.

Important bug fixed before successful run:
- JAX vLLM rejects per-request `seed`.
- Patched `experiments/ifbench/rollout/vllm_iris_infer.py` to omit `seed` in the HTTP request while keeping seed in rollout metadata.

Expected issue:
- Some prompts exceed context and produce HTTP 400 empty rollouts. For 8B, this was 31/20,000 empties.

### 6. Iris 70B overnight failures and later debug

Initial overnight 70B attempts:
- First `v6e-4` attempt failed before server readiness; 70B BF16 is too tight for 4 v6e chips.
- First `v5p-8` TP=8 attempt failed because JAX exposed only 4 devices, so TP=8 is invalid on this scheduler path.
- Later `v5p-8` TP=4 attempts got past mesh setup but failed before serving. Original logs did not preserve native vLLM stderr well enough.
- Added 60s startup heartbeat to `vllm_iris_infer.py` so Iris logs do not look silent during long vLLM startup.

Focused debug after user correction:
- Added `experiments/ifbench/rollout/vllm_70b_startup_probe.py`.
- Added `docs/debug-log-ifbench-vllm-70b-v5p8.md`.
- Probe writes `status.json`, native `stdout.log`, `stderr.log`, and `tail.txt` to GCS.

Debug findings:
- `v5p-8` exposes 4 JAX devices; use TP=4.
- First v5p probe failure was HF cache filling `/root/.cache/huggingface` on root overlay: `No space left on device`.
- Fixed cache placement in both:
  - `experiments/ifbench/rollout/vllm_70b_startup_probe.py`
  - `experiments/ifbench/rollout/vllm_iris_infer.py`
- Cache envs are now forced under `/app/.hf_cache`:
  - `HF_HOME`
  - `HUGGINGFACE_HUB_CACHE`
  - `HF_HUB_CACHE`
  - `TRANSFORMERS_CACHE`
  - `XDG_CACHE_HOME`
  - `VLLM_ASSETS_CACHE`
- Cache-fixed v5p run with only 240GB memory failed by host/container memory OOM because `/app` is tmpfs and the model cache is ~132GB.
- v5p fair retry with 400GB memory succeeded:
  - Job: `/ahmed/ifbench-vllm-70b-probe-v5p8-400gb-r3`
  - TP=4, `max_model_len=4096`, `max_num_seqs=16`
  - Ready after ~465s.
  - Served one `pong` completion.
  - Peak host memory ~281GB under 400GB.
  - Durable diagnostics: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_v5p8_400gb_20260427T165651Z`

Conclusion:
- 70B does work on `v5p-8` for this runner when cache placement is fixed and host memory is raised to ~400GB.
- Earlier failures were cache/root-disk pressure, too-low host memory, and one bad-node/device-busy attempt, not a fundamental v5p incompatibility.

### 7. Explicit v6e-8 70B probe and full run

Why there was no v6e-8 overnight:
- The final overnight launch used fallback `--tpu v6e-4,v5p-8`, not explicit `v6e-8`.
- Iris scheduled the first jobs on `v6e-4`.
- To get v6e-8, request `--tpu v6e-8` explicitly.

v6e-8 probe:
- r1 requested 750GB disk and was unschedulable because v6e scale group offered 100GB disk.
- r2 with 100GB disk succeeded:
  - Job: `/ahmed/ifbench-vllm-70b-probe-v6e8-r2`
  - TP=8, `max_model_len=4096`, `max_num_seqs=16`
  - Peak host memory ~281GB under 360GB.
  - Ready in ~6m44s.
  - One-prompt completion returned `pong`.
  - Durable status: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/debug/vllm_70b_probe_v6e8_20260427T164600Z/status.json`

Full v6e-8 run currently active:
- Job: `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- TPU: `v6e-8`
- Priority: `interactive`
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- TP=8
- Client concurrency=256
- Server `max_num_seqs=64`
- `max_model_len=4096`
- Memory/disk: 500GB RAM, 100GB disk
- Output target: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`
- Monitor state: `scratch/20260427T165724Z_ifbench_vllm_70b_20k_v6e8_r1_monitoring_state.json`

First full v6e attempt:
- Reached vLLM serving.
- Reached ~2,000/20,000 at ~4.2 req/s.
- Had expected long-context HTTP 400s at ~0.7-0.8%.
- Then preemptible worker failed with `worker ping threshold exceeded`.
- Iris automatically restarted the task on a new v6e-8 worker.
- No durable 70B output was written because `vllm_iris_infer.py` writes only after all 20k are complete; the ~2k completions are lost.

Current full v6e state as of `2026-04-27 17:25 UTC`:
- Job: `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- Iris state: `running`
- `preemption_count=1`
- Worker: `marin-tpu-v6e-preemptible-8-europe-west4-20260427-1716-2c14b3a6-worker-0`
- Task has restarted from scratch.
- Latest summary: duration ~300s, current memory ~147GB, peak memory ~281GB.
- Logs show:
  - deps installed,
  - cache configured under `/app/.hf_cache`,
  - 20k prompts loaded,
  - native vLLM server started at `2026-04-27 17:20:54 UTC`,
  - startup heartbeats through `17:24:55 UTC`.
- No new `/v1/models` ready/progress line yet after restart.

Detached monitor:
- tmux session: `ifbench_v6e_monitor`
- Script: `scratch/monitor_ifbench_v6e_r1.py`
- State file: `scratch/monitor_ifbench_v6e_r1_state.json`
- It appends checkpoints every ~1k prompts once progress resumes.
- Caveat: the monitor is simple; it did not explicitly log the preemption until I manually checked and wrote this handoff. Future agents should still manually check `job summary` and recent logs.

### 8. Context-window issue and planned repair

The 20k prompt slice has a long tail:
- median: 642 chars
- p95: 2107 chars
- p99: 6810 chars
- p99.5: 13192 chars
- p99.9: 41720 chars
- max: 313553 chars
- `>12000` chars: 114 prompts
- `>20000` chars: 49 prompts
- `>40000` chars: 21 prompts

For the full 70B v6e run, `max_model_len=4096` and fixed `max_tokens=1024` cause occasional HTTP 400s:
- too little output room for ~3200-4000-token inputs, or
- input alone exceeds 4096 tokens.

Decision:
- Do not kill the fast 4k-context full run for a <1% long-tail.
- Let it complete.
- Then identify `finish_reason=http_400` rows and rerun only those prompt IDs with a long-context, low-concurrency vLLM job.
- Merge replacements before final verifier scoring and DPO/SFT extraction.

### 9. Files Changed / Added By Codex

Core rollout/provider code:
- `experiments/ifbench/rollout/overnight.py`
- `experiments/ifbench/rollout/vllm_iris_infer.py`
- `experiments/ifbench/rollout/vllm_70b_startup_probe.py`
- `experiments/ifbench/rollout/together_backend.py` (progress logging / poll hardening)
- `experiments/ifbench/rollout/gemini_backend.py` (file-batch behavior already in branch; validated and used)
- `pyproject.toml`
- `uv.lock`

Monitoring / docs:
- `.agents/logbooks/dpo_sft_codex.md`
- `docs/debug-log-ifbench-vllm-70b-v5p8.md`
- `scratch/monitor_ifbench_v6e_r1.py`
- multiple `scratch/*monitoring_state.json` files

Important branch hygiene:
- This worktree already has broad untracked IFBench/project files. Do not assume every dirty file is from this final handoff.
- Do not revert user/other-agent changes.

## What The Next Agent Should Do

1. Check the active v6e job:
   - `uv run iris --config lib/iris/examples/marin.yaml job summary /ahmed/ifbench-vllm-70b-20k-v6e8-r1 --json`
   - `uv run iris --config lib/iris/examples/marin.yaml job logs /ahmed/ifbench-vllm-70b-20k-v6e8-r1 --tail --max-lines 300`

2. If the v6e job reaches progress again:
   - let it continue.
   - update this logbook at each material checkpoint.
   - watch for repeated preemption; if preemptions keep wiping progress, patch the runner to write incremental shards/checkpoints before relaunching again.

3. If the v6e job fails or preempts repeatedly:
   - Use the proven v5p config as fallback:
     - `--tpu v5p-8`
     - TP=4
     - memory >= 400GB
     - disk 100GB is okay if cache stays under `/app/.hf_cache`
     - `max_model_len=4096`, `max_num_seqs=16` or 64 depending on risk tolerance.
   - Do not conclude "70B cannot run on v5p"; the 400GB probe proved it can start and serve.

4. After a full 70B rollout exists:
   - copy/download it if needed.
   - count rows and empties.
   - identify `finish_reason=http_400`.
   - rerun only HTTP-400 prompt IDs with long context.
   - merge repaired rows.

5. Build final 20k extraction:
   - Use Iris 8B 20k, Gemini Flash 20k, and the full 70B Iris output if available.
   - Optionally include Together 70B's 2,299 successes as an auxiliary source, but do not treat its missing 17,701 provider-error rows as model failures.
   - Re-run strict and loose scoring.
   - Extract DPO pairs and SFT examples.
   - Compare final pair yield against the partial 19.895% and smoke 34%.

6. Provider follow-up:
   - Re-check Together auth/key before trusting provider polling.
   - If auth is fixed, poll/download Together 8B batch `9b109f48-a95a-4fb6-a658-7d16c52053d3`.
   - Together 8B is redundant with Iris 8B but can be useful for provider-vs-local comparison.

7. Training plan after data is finalized:
   - Start with fixed-size ablations, not one giant run:
     - SFT-only on chosen/pass examples.
     - DPO-only on chosen/rejected pairs.
     - SFT warmup then DPO.
     - mixture ratios such as 0/100, 25/75, 50/50, 75/25, 100/0 SFT/DPO by token or example budget.
   - DPO rejected-margin ablation:
     - fixed number of pairs per bucket,
     - compare close-margin rejects vs medium-margin rejects vs far-failing rejects,
     - keep chosen examples constant where possible.
   - Evaluation arms should include strict and loose IFBench/IFEval-style metrics, plus held-out constraints by `num_constraints`.

## Immediate Live Status Summary

As of `2026-04-27 17:25 UTC`, the only live compute we are actively waiting on is:
- `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`

Everything else is either complete, failed with an explained cause, or blocked by current-session provider auth:
- 8B Iris: complete.
- Gemini 20k batch: complete.
- Together 70B batch: complete but under-delivered due to provider internal errors.
- Together 8B batch: last trustworthy state was still running; current shell cannot poll because Together returns 401.
- v5p 70B probe: complete and successful.

### 2026-04-27 18:17 UTC — v6e 70B restarted run is making progress

**Active job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- State: `running`.
- Preemption count remains `1`.
- Current worker: `marin-tpu-v6e-preemptible-8-europe-west4-20260427-1716-2c14b3a6-worker-0`.
- Latest observed progress after restart: `13600/20000` at ~4.5 req/s.
- Current memory: ~149GB; peak memory: ~281GB.
- No output object yet; the runner writes at completion.
- Context-window 400s are still present for long-tail prompts, e.g. prompt IDs `c9877d5cc60a9b5a` and `21664a7e5a34ba03`. This matches the expected repair-pass plan.

**Next checkpoint:** if throughput holds, the remaining ~6.4k requests should finish in roughly 25 minutes, barring another preemption. On completion, verify GCS object, count rows, count `http_400`, then launch the long-context repair pass only for those failed prompt IDs.

### 2026-04-27 18:18 UTC — current source matrix clarification

**Usable rollout files on disk right now:**
- Gemini Flash file-batch: `/tmp/ifbench_port/overnight_20k/rollouts/gemini_gemini-3-flash-preview.jsonl` — 20,000 rows.
- Iris 8B local vLLM: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_8b.jsonl` — 20,000 rows. GCS copy exists at `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_8b.jsonl`.
- Together 70B batch successes: `/tmp/ifbench_port/overnight_20k/rollouts/together_meta-llama_Llama-3.3-70B-Instruct-Turbo.jsonl` — 2,299 rows.
- Together 70B error file: `/tmp/ifbench_port/overnight_20k/rollouts/together_70b_error_file.jsonl` — 17,701 provider-error rows.

**Submitted but not usable as a complete rollout yet:**
- Iris 70B full local vLLM: `/ahmed/ifbench-vllm-70b-20k-v6e8-r1` — running at last check; no GCS output object yet.
- Together 8B batch `9b109f48-a95a-4fb6-a658-7d16c52053d3` — submitted, but no downloaded output file; last trustworthy provider poll had it still in progress. Current shell gets Together `401`, so do not trust fresh provider polling until auth is fixed.

**Bottom line:** Gemini Flash finished fully; Iris 8B finished fully; Together 70B technically finished but under-delivered; Together 8B has no usable output yet; Iris 70B is currently the active full-70B source.

### 2026-04-27 18:20 UTC — Together 8B-Lite live status cannot be re-polled from current shell

**Batch:** `9b109f48-a95a-4fb6-a658-7d16c52053d3`
**Model:** `meta-llama/Meta-Llama-3-8B-Instruct-Lite`

Current local facts:
- Handle exists in `/tmp/ifbench_port/overnight_20k/handles.json`.
- No Together 8B rollout file exists under `/tmp/ifbench_port/overnight_20k/rollouts/`.
- `TOGETHER_API_KEY` is set in this shell, but direct GET `https://api.together.xyz/v1/batches/9b109f48-a95a-4fb6-a658-7d16c52053d3` returns HTTP `401` with an empty body.
- Last trustworthy successful poll remains the earlier one: `IN_PROGRESS`, progress `55.89%`.

Interpretation:
- I cannot currently say whether the Together 8B-Lite batch is still in progress, completed, failed, or expired; the current shell cannot authenticate to Together.
- Since Iris 8B already finished all 20k, Together 8B is not blocking the main dataset path. It is only useful as provider-vs-local comparison or an extra rollout source if auth is restored and the batch has completed.

### 2026-04-27 18:21 UTC — decision: treat Together batch path as not working for this run

Per user instruction, mark the Together path as not working for the active plan:
- Together 70B batch `9afbd565-a6be-48c8-beaf-29be61d2b81d` technically completed, but only returned 2,299/20,000 rows and put 17,701 rows in a provider `Internal Server Error` file.
- Together 8B-Lite batch `9b109f48-a95a-4fb6-a658-7d16c52053d3` was last validly seen as `IN_PROGRESS`, but current shell polling returns HTTP `401 Unauthorized`; no 8B-Lite output file exists locally.
- Therefore, do not wait on Together for the main 20k dataset. Use Together 70B's 2,299 successes only as optional auxiliary signal, never as the required full 70B source.

Main path from here:
- Finish Iris 70B v6e rollout.
- Repair its context-window failures.
- Build the final extraction from Iris 8B + Gemini Flash + Iris 70B.

### 2026-04-27 18:23 UTC — repair-run support added while 70B runs

**Active 70B state:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- Latest observed progress: `15300/20000` at ~4.5 req/s.

**Code change:** `experiments/ifbench/rollout/vllm_iris_infer.py`
- Added `--prompt-ids-file` so a follow-up Iris job can rerun only selected prompt IDs from the first `n` prepared rows.
- Added `--max-new-tokens` so repair jobs can reduce or adjust generation length without editing the module constant.
- Threaded a per-run `SamplingConfig` through rollout creation so the output `sampling_config_hash` reflects the repair job's max-new-token setting.

**Validation:**
- `uv run python -m py_compile experiments/ifbench/rollout/vllm_iris_infer.py` passed.
- Local prompt-id filter smoke test on `/tmp/ifbench_port/overnight_20k/iris/prompts_20k.jsonl` with known failed IDs `c9877d5cc60a9b5a` and `21664a7e5a34ba03` returned exactly those two rows.

**Use after completion:**
1. Read the completed Iris 70B rollout jsonl.
2. Write prompt IDs where `finish_reason == "http_400"` to a local/GCS text file.
3. Launch a long-context repair job using the same runner with `--prompt-ids-file`, lower `--max-num-seqs`, and larger `--max-model-len`.

### 2026-04-27 18:24 UTC — repair JSONL helper added

**Active 70B state:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- Latest observed progress: `15600/20000` at ~4.5 req/s.

**Added:** `experiments/ifbench/rollout/repair_rollouts.py`
- `failed-ids`: read a rollout jsonl and write unique prompt IDs matching a finish reason, default `http_400`.
- `merge`: replace rows in a base rollout jsonl with rows from a repair rollout jsonl, keyed by `prompt_id`; extra repair rows append at the end.

**Validation:**
- `uv run python -m py_compile experiments/ifbench/rollout/repair_rollouts.py` passed.
- Temp-file smoke test verified `failed-ids` emits the expected prompt id and `merge` replaces the failed row.

### 2026-04-27 18:28 UTC — Iris 70B progress checkpoint

**Active job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- State: `running`.
- Preemption count remains `1`.
- Latest progress visible in Iris logs: `16100/20000` at ~4.5 req/s.
- Resource summary: current memory ~150GB; peak memory ~281GB; worker unchanged.
- No output object yet; continue waiting for completion.

### 2026-04-27 18:33 UTC — second v6e preemption; must make rollout durable

**Active job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- State: `running`, but `preemption_count` increased from 1 to 2.
- Previous worker `marin-tpu-v6e-preemptible-8-europe-west4-20260427-1716-2c14b3a6-worker-0` failed with `worker ping threshold exceeded`.
- The second attempt had reached at least `16100/20000` at ~4.5 req/s before the worker died.
- Iris restarted the job on new worker `marin-tpu-v6e-preemptible-8-europe-west4-20260427-1821-d034c72f-worker-0`.

**Problem:** the current runner writes only at the end, so both partial attempts were lost. Waiting is no longer the right strategy.

**Next action:** patch `experiments/ifbench/rollout/vllm_iris_infer.py` to write durable GCS shards incrementally and skip already-written prompt IDs on resume, then stop/relaunch the 70B job with shard output enabled.

### 2026-04-27 18:36 UTC — old 70B job stopped; sharded 70B job submitted

**Stopped:** `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`
- Reason: two preemptions after large partial progress (`~2000/20000`, then `~16100/20000`) and no durable partial output.

**Submitted:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r1`
- TPU: `v6e-8`
- Priority: `interactive`
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- TP=8
- Client concurrency=256
- Server `max_num_seqs=64`
- `max_model_len=4096`
- Memory/disk: 500GB RAM, 100GB disk
- Final output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`
- Durable shard dir: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_shards`
- Resume mode: `--resume-from-shards`
- Shard size: 100 rollouts per `part-*.jsonl`

**Exact submit command:**
```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name ifbench-vllm-70b-20k-v6e8-sharded-r1 \
  --priority interactive \
  --enable-extra-resources \
  --tpu v6e-8 \
  --cpu 32 \
  --memory 500GB \
  --disk 100GB \
  -- \
  python -m experiments.ifbench.rollout.vllm_iris_infer \
    --model-id meta-llama/Llama-3.3-70B-Instruct \
    --prepared gs://marin-us-central2/scratch/ifbench/overnight_20k/prompts_20k.jsonl \
    --n 20000 \
    --output gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl \
    --concurrency 256 \
    --tensor-parallel-size 8 \
    --max-num-seqs 64 \
    --max-model-len 4096 \
    --mode native \
    --seed 0 \
    --cache-dir /app/.hf_cache \
    --shard-output-dir gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_shards \
    --resume-from-shards \
    --shard-size 100
```

**Next:** monitor startup. Once shards appear, future preemptions should lose at most the in-flight shard rather than all progress.

### 2026-04-27 18:37 UTC — sharded job pending; old monitor stopped

**Sharded job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r1`
- State: `pending`.
- Task state: `pending`.
- No worker allocated yet.
- No shard objects exist yet.

**Cleanup:** stopped old tmux session `ifbench_v6e_monitor` because it was hardcoded to the stopped non-sharded job `/ahmed/ifbench-vllm-70b-20k-v6e8-r1`.

### 2026-04-27 18:40 UTC — Together batch path marked broken; sharded Iris relaunch needs explicit vLLM install

**Together status:** not a viable blocking path for tonight's 20k run.
- 70B Together batch already under-delivered: 2,299 successes and 17,701 provider-side `batch_client_error` / `Internal Server Error` rows.
- 8B Lite Together batch handle `9b109f48-a95a-4fb6-a658-7d16c52053d3` was last known `IN_PROGRESS` at 55.89%, but direct API polling now returns HTTP 401 with an empty body using the current shell's `TOGETHER_API_KEY`.
- Decision: do not wait on Together for the dataset. The usable 8B source is the completed Iris vLLM 8B rollout, and the remaining blocker is a complete 70B rollout on Iris.

**Iris sharded 70B status:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r1` failed immediately.
- Terminal state: `JOB_STATE_FAILED`.
- Error: `FileNotFoundError: [Errno 2] No such file or directory: 'vllm'`.
- Root cause: the job command invoked `python -m experiments.ifbench.rollout.vllm_iris_infer` directly, but this container did not have the `vllm` console script available. Previous successful probes/jobs had `vllm-tpu` available before the runner started.

**Next action:** relaunch as `ifbench-vllm-70b-20k-v6e8-sharded-r2` with the same durable shard settings, but add an explicit preflight install/check for `vllm-tpu` before starting the runner. Continue treating Together as best-effort only.

### 2026-04-27 18:41 UTC — sharded Iris 70B relaunched with vLLM extra

**Submitted:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- Fix vs r1: added Iris environment extras `--extra tpu --extra vllm`; Marin's `vllm` extra pins `vllm-tpu==0.13.2.post6`, which should provide the `vllm` console script before runner startup.
- Same durable shard settings as r1:
  - final output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`
  - shard dir: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_shards`
  - `--resume-from-shards`
  - `--shard-size 100`
- Same runtime shape: v6e-8, interactive priority, TP=8, client concurrency 256, server `max_num_seqs=64`, `max_model_len=4096`, 500GB RAM, 100GB disk.

**Next:** check after startup. Need see either setup installing extras, vLLM readiness logs, or first `part-*.jsonl` shards in GCS. If r2 still cannot find `vllm`, fall back to a direct CLI command prefix that runs `uv pip install vllm-tpu==0.13.2.post6` before `python -m ...`.

### 2026-04-27 18:42 UTC — r2 passed the r1 failure point

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: `JOB_STATE_RUNNING`.
- Setup logs show `syncing deps`, `installing pip deps`, activating `/app/.venv/bin/python`, then `running user command`.
- Runner logs reached `Configured model cache under /app/.hf_cache` and printed storage state.
- This clears the r1 issue (`vllm` executable missing before runner startup).

**Current wait:** vLLM server startup/readiness, then first durable GCS shards under `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_shards/`.

### 2026-04-27 18:47 UTC — r2 still in vLLM startup, resources healthy

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Logs show minute heartbeats: `Still waiting for vLLM startup for meta-llama/Llama-3.3-70B-Instruct` through 18:46 UTC.
- Task summary:
  - worker: `marin-tpu-v6e-preemptible-8-europe-west4-20260427-1821-d034c72f-worker-0`
  - current memory: ~149GB
  - peak memory: ~280GB
  - disk: ~141GB
- No shard files yet; still waiting for server readiness.

**Interpretation:** no r1-style missing-binary failure and no memory pressure. Startup is longer than the earlier tiny probe but still plausibly model-load/compile. Continue monitoring.

### 2026-04-27 18:49 UTC — r2 serving and writing durable shards

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- vLLM is serving requests.
- Progress observed: `300/20000` at ~4.2 req/s.
- Durable shards present:
  - `part-1777315642-00000.jsonl` — 100 rows
  - `part-1777315642-00001.jsonl` — 100 rows
  - `part-1777315642-00002.jsonl` — 100 rows
- First shard timestamp: 18:47:50 UTC; startup-to-first-shard was ~6m34s after server start.

**Known issue during rollout:** some prompts return vLLM HTTP 400 because `max_tokens=1024` plus long input exceeds the 4096 context cap. These rows should be marked as `finish_reason=http_400` by the runner and repaired later using `experiments/ifbench/rollout/repair_rollouts.py` plus a focused long-context repair job.

**Next:** keep monitoring for progress, preemptions, and final consolidated output. Because shards are durable, a preemption should now lose at most the current in-flight shard instead of the whole run.

### 2026-04-27 18:52 UTC — r2 at ~1k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `1000/20000` at ~4.5 req/s.
- GCS shard count observed: 11 `part-*.jsonl` files (first 10 full progress shards plus current listing including `00010`).
- Resource summary remains healthy: current memory ~152GB, peak ~280GB, disk ~141GB.

**ETA:** at ~4.5 req/s, remaining wall time is roughly 70 minutes if the worker is not preempted.

### 2026-04-27 18:57 UTC — r2 at ~2.5k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `2500/20000` at ~4.5 req/s.
- GCS shard count observed: 26 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00025.jsonl` at 18:57:06 UTC.
- Resource summary remains healthy: current memory ~153GB, peak ~280GB, disk ~141GB.

**Notes:** 4096-context HTTP 400 rows continue for long prompts, as expected. Repair after base rollout completes.

### 2026-04-27 19:03 UTC — r2 at ~4k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `4000/20000` at ~4.5 req/s.
- GCS shard count observed: 41 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00040.jsonl` at 19:02:30 UTC.
- Resource summary remains healthy: current memory ~152GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 60 minutes remaining at current rate.

### 2026-04-27 19:08 UTC — r2 at ~5.5k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `5500/20000` at ~4.5 req/s.
- GCS shard count observed: 56 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00055.jsonl` at 19:08:09 UTC.
- Resource summary remains healthy: current memory ~152GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 54 minutes remaining at current rate.

### 2026-04-27 19:14 UTC — r2 at ~7.1k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `7100/20000` at ~4.5 req/s.
- GCS shard count observed: 71 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00070.jsonl` at 19:13:34 UTC.
- Resource summary remains healthy: current memory ~152GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 48 minutes remaining at current rate.

### 2026-04-27 19:19 UTC — r2 at ~8.5k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `8500/20000` at ~4.5 req/s.
- GCS shard count observed: 85 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00084.jsonl` at 19:18:51 UTC.
- Resource summary remains healthy: current memory ~152GB, peak ~280GB, disk ~141GB.

**Repair note:** context-cap failures include some very long inputs (seen examples at ~20k+ input tokens). The repair pass may need a larger context setting than 8k for a subset; after base rollout, inspect failed prompt token lengths before picking repair `max_model_len`.

### 2026-04-27 19:25 UTC — r2 reached halfway

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `10000/20000` at ~4.5 req/s.
- GCS shard count observed: 100 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00099.jsonl` at 19:24:24 UTC.
- Resource summary remains healthy: current memory ~153GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 37 minutes remaining at current rate.

### 2026-04-27 19:31 UTC — r2 at ~11.6k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `11600/20000` at ~4.5 req/s.
- GCS shard count observed: 116 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00115.jsonl` at 19:30:24 UTC.
- Resource summary remains healthy: current memory ~153GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 31 minutes remaining at current rate.

### 2026-04-27 19:37 UTC — r2 at ~13.2k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `13200/20000` at ~4.5 req/s.
- GCS shard count observed: 132 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00131.jsonl` at 19:36:12 UTC.
- Resource summary remains healthy: current memory ~153GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 25 minutes remaining at current rate.

### 2026-04-27 19:46 UTC — r2 at ~15.8k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `15800/20000` at ~4.5 req/s.
- GCS shard count observed: 158 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00157.jsonl` at 19:45:36 UTC.
- Resource summary remains healthy: current memory ~153GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 16 minutes remaining at current rate.

### 2026-04-27 19:52 UTC — r2 at ~17.3k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `17300/20000` at ~4.5 req/s.
- GCS shard count observed: 173 `part-*.jsonl` files.
- Latest visible shard: `part-1777315642-00172.jsonl` at 19:51:06 UTC.
- Resource summary remains healthy: current memory ~153GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 10 minutes remaining at current rate.

### 2026-04-27 19:56 UTC — r2 at ~18.6k/20k

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `18600/20000` at ~4.5 req/s.
- GCS shard count observed: 186 `part-*.jsonl` files.
- No final consolidated output object yet.
- Resource summary remains healthy: current memory ~153GB, peak ~280GB, disk ~141GB.

**ETA:** roughly 5 minutes remaining at current rate, plus whatever time final consolidation/upload takes.

### 2026-04-27 20:03 UTC — r2 reached 20k/20k; waiting on final consolidation

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- State: still running.
- Failure/preemption counts: 0 / 0.
- Latest progress line: `20000/20000` at ~4.5 req/s.
- GCS shard count observed: 200 `part-*.jsonl` files.
- Latest shard: `part-1777315642-00199.jsonl` at 20:01:13 UTC.
- Final consolidated output object not visible yet at `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`.

**Next:** wait for terminal success/final object, then validate row count and finish-reason distribution locally. If final consolidation hangs, reconstruct final jsonl directly from the 200 durable shard files.

### 2026-04-27 20:06 UTC — r2 succeeded; final 70B rollout object exists

**Job:** `/ahmed/ifbench-vllm-70b-20k-v6e8-sharded-r2`
- Terminal state: succeeded.
- Failure/preemption counts: 0 / 0.
- Duration: ~4,999s task duration.
- Final progress: `20000/20000` at ~4.5 req/s.
- GCS shard count: 200 `part-*.jsonl` files.
- Final consolidated output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`
- Final object size: ~39.95 MiB.
- Logs confirm: `Wrote 20000 consolidated sharded rollouts to gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`.

**Next:** copy final output local, validate 20k row count, count `finish_reason` values, then identify/repair HTTP 400 context failures before final DPO/SFT extraction.

### 2026-04-27 20:14 UTC — 70B output validated; repair IDs prepared

**Local file:** `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`
- Rows: 20,000
- Unique prompt IDs: 20,000
- Duplicate prompt IDs: 0
- Finish reasons:
  - `stop`: 17,311
  - `length`: 2,579
  - `http_400`: 110
- Empty `response_text`: 110, exactly the `http_400` set.

**Repair IDs:**
- Local: `/tmp/ifbench_port/overnight_20k/iris/llama_3_3_70b_http400_ids.txt`
- GCS: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/llama_3_3_70b_http400_ids.txt`
- Count: 110

**Failed prompt token lengths (Llama chat-template tokenization):**
- min: 3,090
- p50: 4,682
- max: 91,055
- Cumulative buckets: 46 <=4k, 86 <=8k, 97 <=16k, 107 <=32k, 109 <=64k, 110 <=98k.

**Runner patch for repair:** updated `experiments/ifbench/rollout/vllm_iris_infer.py` to expose:
- `--max-num-batched-tokens` passed through to `vllm serve`
- `--request-timeout` for long prompt prefill/generation
- `uv run python -m py_compile experiments/ifbench/rollout/vllm_iris_infer.py` passed.

**Repair plan:** launch one v6e-8 repair job for the 110 IDs with `max_model_len=98304`, `max_num_batched_tokens=98304`, `max_num_seqs=1`, `concurrency=8`, `request_timeout=1800`, and shard size 10. If this is too large for vLLM/TPU startup, fall back to bucketed repair by context length.

### 2026-04-27 20:26 UTC — 70B HTTP 400 repair job submitted

**Submitted:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-r1`
- TPU: v6e-8
- Priority: interactive
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Repair IDs: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/llama_3_3_70b_http400_ids.txt` (110 IDs)
- Output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400.jsonl`
- Shards: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_shards`
- Runtime shape: TP=8, `max_model_len=98304`, `max_num_batched_tokens=98304`, `max_num_seqs=1`, concurrency=8, request timeout=1800s, shard size=10.

**Next:** monitor startup. If vLLM rejects the 98k context shape or OOMs, relaunch bucketed repairs by prompt length.

### 2026-04-27 20:29 UTC — repair job pending on v6e-8 autoscale

**Job:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-r1`
- State: pending.
- Pending reason: no current worker has sufficient v6e-8 resources; autoscaler is scaling up `tpu_v6e-preemptible_8-europe-west4-a` for 1 slice.
- Failure/preemption counts: 0 / 0.

**Action:** keep waiting. This is scheduler/capacity wait, not a runtime failure.

### 2026-04-27 20:31 UTC — preliminary full 20k extraction without repair

**Inputs:**
- Iris 8B: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_8b.jsonl`
- Gemini Flash: `/tmp/ifbench_port/overnight_20k/rollouts/gemini_gemini-3-flash-preview.jsonl`
- Iris 70B v6e-8 base: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8.jsonl`

**Output dir:** `/tmp/ifbench_port/overnight_20k/outputs_iris8b_70b_v6e8_gemini_pre_repair/`

**Result:**
- Prompts: 20,000
- DPO pairs: 6,581
- Pair yield: 32.905%
- SFT examples: 13,801
- No passers: 11,805
- No failers: 1,614
- Yield by `num_constraints`:
  - 1: 2,419
  - 2: 2,152
  - 3: 1,297
  - 4: 596
  - 5: 117

**Interpretation:** the full 20k three-model pool clears the 30% pair-yield gate even before repairing the 110 70B context failures. Repair should only make a small difference in aggregate yield.

### 2026-04-27 20:32 UTC — v5p-8 repair race submitted

**Reason:** v6e repair was still pending on autoscaler capacity. To avoid blocking on one TPU family, launched a separate v5p-8 repair attempt to different output paths.

**Submitted:** `/ahmed/ifbench-vllm-70b-http400-repair-v5p8-r1`
- TPU: v5p-8
- Priority: interactive
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Repair IDs: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/llama_3_3_70b_http400_ids.txt` (110 IDs)
- Output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v5p8_repair_http400.jsonl`
- Shards: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v5p8_repair_http400_shards`
- Runtime shape: TP=4, `max_model_len=98304`, `max_num_batched_tokens=98304`, `max_num_seqs=1`, concurrency=8, request timeout=1800s, shard size=10.

**Race policy:** use whichever repair job finishes cleanly first. Do not merge both into the same base file.

### 2026-04-27 20:33 UTC — v5p repair relaunched with feasible memory

**Stopped:** `/ahmed/ifbench-vllm-70b-http400-repair-v5p8-r1`
- Reason: pending unsatisfied resources; requested 500GB RAM while v5p workers reported <500GB available.

**Submitted:** `/ahmed/ifbench-vllm-70b-http400-repair-v5p8-r2`
- Same command/output paths as v5p r1.
- Memory lowered to 450GB. Earlier v5p 70B startup probe succeeded at 400GB, so this should be sufficient for host memory while fitting the scheduler's available worker memory.

**Active repair jobs now:**
- `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-r1` — pending on v6e-8 capacity.
- `/ahmed/ifbench-vllm-70b-http400-repair-v5p8-r2` — newly submitted.

### 2026-04-27 20:34 UTC — v6e repair running; v5p backup stopped

**v6e repair:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-r1`
- State: running.
- Failure/preemption counts: 0 / 0.

**v5p backup:** `/ahmed/ifbench-vllm-70b-http400-repair-v5p8-r2`
- Still unschedulable due memory; even 450GB decimal exceeds the v5p available byte limit.
- Stopped once v6e started running, to avoid duplicate work/capacity churn.

**Next:** monitor v6e repair startup. Risk is whether vLLM accepts/compiles `max_model_len=98304`, not scheduling.

### 2026-04-27 20:38 UTC — v6e repair still in startup, no crash

**Job:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-r1`
- State: running.
- Failure/preemption counts: 0 / 0.
- Logs show vLLM startup heartbeats through 20:37 UTC.
- Resource summary:
  - current memory: ~150GB
  - peak memory: ~282GB
  - disk: ~139GB
- No repair shards yet.

**Interpretation:** the 98k context setting has not failed immediately. Continue waiting for vLLM readiness or a startup failure.

### 2026-04-27 20:44 UTC — Together batch marked broken for this run; 98k repair failed

**Together batch status:** treat Together Batch as **not working / non-blocking** for the 20k run.
- 70B batch produced only 2,299 usable rows and 17,701 provider-side `batch_client_error` / `Internal Server Error` rows.
- 8B-Lite batch never became a usable source for the final dataset; latest polling had hit auth/empty-body issues and the complete 8B source is already the Iris vLLM output.
- Decision: do **not** wait on Together for tonight's dataset. The usable three-model pool is Iris 8B + Iris 70B + Gemini Flash.

**98k 70B repair attempt:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-r1` failed after vLLM startup.
- TPU: v6e-8, TP=8.
- Shape: `max_model_len=98304`, `max_num_batched_tokens=98304`, `max_num_seqs=1`.
- vLLM loaded the model and initialized KV cache, but HBM was tight:
  - model init: ~16.46 / 31.25 GiB per chip
  - KV cache init: ~28.12 / 31.25 GiB per chip
  - reported KV capacity: 305,680 tokens, max concurrency 3.11x for 98,304-token requests
- Failure root cause: `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED` while loading `jit_step_fun`; it attempted to reserve 3.21G at the bottom of memory with only 2.12G free/reservable.
- This is TPU HBM/JAX compile pressure, not host RAM and not v6e scheduling.

**Next action:** generate context-length repair buckets for the 110 HTTP-400 IDs and rerun smaller shapes first. The preliminary extraction already clears the pair-yield gate at 32.905%, so this repair is cleanup rather than a blocker.

### 2026-04-27 20:45 UTC — 32k bulk repair submitted

**Token bucket split for 110 HTTP-400 IDs** (`max_new_tokens=1024` included in capacity check):
- `<=8192`: 80 prompts, input token range 3,090-6,948.
- `<=16384`: 16 prompts, input token range 7,369-12,991.
- `<=32768`: 11 prompts, input token range 15,736-31,417.
- `<=65536`: 2 prompts, input token range 36,381-41,570.
- `<=98304`: 1 prompt, input tokens 91,055.

**Uploaded bucket artifacts:**
- `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/llama_3_3_70b_http400_ids_le32768.txt` — 107 IDs.
- `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/llama_3_3_70b_http400_token_buckets.json`.
- Individual `ctx8192`, `ctx16384`, `ctx32768`, `ctx65536`, `ctx98304` ID files are in the same GCS directory.

**Submitted:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-32k-r1`
- TPU: v6e-8, priority interactive.
- Shape: TP=8, `max_model_len=32768`, `max_num_batched_tokens=32768`, `max_num_seqs=4`, client concurrency 16.
- IDs: 107 prompts from `llama_3_3_70b_http400_ids_le32768.txt`.
- Output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_le32768.jsonl`.
- Shards: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_le32768_shards`.
- Monitor state: `scratch/20260427T204535Z_ifbench_vllm_70b_http400_repair_v6e8_32k_r1_monitoring_state.json`.

**Next:** monitor startup and shards. If the 32k repair succeeds, merge those 107 repairs into the 70B base and rerun extraction; then decide whether the 2-prompt 65k bucket and 1-prompt 92k outlier are worth more Iris time.

### 2026-04-27 20:52 UTC — 32k repair passed startup and is writing shards

**Job:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-32k-r1`
- State: running.
- Failure/preemption counts: 0 / 0.
- Worker: v6e-8 in `europe-west4`.
- Memory peak so far: ~277GB host memory.
- vLLM startup survived the compile window that killed the 98k attempt.
- First shard written at 20:52 UTC:
  `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_le32768_shards/part-1777323093-00000.jsonl` (10 rows).

**Interpretation:** smaller `max_model_len=32768` is viable for 70B on v6e-8. Continue monitoring until 107/107 repair rows are consolidated.

### 2026-04-27 20:54 UTC — 65k repair submitted for the two medium-long failures

**32k repair progress:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-32k-r1`
- Shards written: 3 × 10 rows = 30/107.
- No errors observed.

**Submitted parallel 65k repair:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-65k-r1`
- TPU: v6e-8, priority interactive.
- Shape: TP=8, `max_model_len=65536`, `max_num_batched_tokens=32768`, `max_num_seqs=1`, client concurrency 2.
- IDs: 2 prompts from `llama_3_3_70b_http400_ids_ctx65536.txt`.
- Output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_ctx65536.jsonl`.
- Shards: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_ctx65536_shards`.
- Monitor state: `scratch/20260427T205406Z_ifbench_vllm_70b_http400_repair_v6e8_65k_r1_monitoring_state.json`.

**Outlier still unrepaired:** 1 prompt needs ~92k total context. Do not retry the 98k shape until the 32k/65k repairs are safe, because the 98k startup already failed from HBM compile pressure.

### 2026-04-27 20:59 UTC — 32k repair validated

**32k repair job:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-32k-r1`
- State: succeeded.
- Duration: ~12m37s.
- Failure/preemption counts: 0 / 0.
- Final GCS output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_le32768.jsonl`.
- Local copy: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_le32768.jsonl`.
- Validation:
  - rows: 107
  - unique prompt IDs: 107
  - finish reasons: `stop=86`, `length=21`
  - empty responses: 0
  - token usage missing: 0

**65k repair:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-65k-r1`
- State: running after pending capacity.
- Same v6e worker family; no startup verdict yet.

**Next:** wait for the 65k two-prompt repair. If it succeeds, merge 109 repair rows into the base; otherwise merge the 107-row 32k repair only and leave 3 long-context failures in place.

### 2026-04-27 21:08 UTC — repaired 20k extraction complete

**65k repair job:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-65k-r1`
- State: succeeded.
- Duration: ~6m27s.
- Failure/preemption counts: 0 / 0.
- Final GCS output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_ctx65536.jsonl`.
- Local copy: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_ctx65536.jsonl`.
- Validation: 2 rows, 2 unique prompt IDs, `finish_reason=stop` for both, 0 empty responses.

**Merged 70B output:**
- Local: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repaired.jsonl`.
- GCS: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repaired.jsonl`.
- Rows: 20,000.
- Unique prompt IDs: 20,000.
- Finish reasons after 109/110 repairs:
  - `stop`: 17,399
  - `length`: 2,600
  - `http_400`: 1
- Empty responses: 1 (the remaining ~91k-token outlier).

**Final 20k extraction with repaired 70B:**
- Output dir local: `/tmp/ifbench_port/overnight_20k/outputs_iris8b_70b_v6e8_gemini_repaired/`.
- Output dir GCS: `gs://marin-us-central2/scratch/ifbench/overnight_20k/outputs_iris8b_70b_v6e8_gemini_repaired/`.
- DPO pairs: 6,595.
- SFT examples: 13,825.
- Pair yield: 32.975%.
- No passers: 11,790.
- No failers: 1,615.
- Yield by `num_constraints`:
  - 1: 2,427
  - 2: 2,149
  - 3: 1,301
  - 4: 601
  - 5: 117

**Delta vs pre-repair extraction:**
- DPO pairs: +14 (6,581 → 6,595).
- SFT examples: +24 (13,801 → 13,825).
- No-pass prompts: -15.
- No-fail prompts: +1.

**Interpretation:** 20k generation/extraction is complete enough for training ablations. Only one 70B rollout remains empty; it is a 91,055-token prompt that requires ~92k context with the current 1024-token generation budget.

### 2026-04-27 21:08 UTC — one-row 98k chunked outlier repair submitted

**Why:** 109/110 70B HTTP-400 repairs are complete and the final repaired extraction is already usable. There is one remaining 91,055-token prompt. The earlier 98k repair failed with `max_num_batched_tokens=98304`; this retry tests whether chunked prefill capped at 32k avoids the HBM compile failure.

**Submitted:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-98k-chunked-r1`
- TPU: v6e-8, priority interactive.
- Shape: TP=8, `max_model_len=98304`, `max_num_batched_tokens=32768`, `max_num_seqs=1`, client concurrency 1.
- IDs: 1 prompt from `llama_3_3_70b_http400_ids_ctx98304.txt`.
- Output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_ctx98304_chunked.jsonl`.
- Shards: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_ctx98304_chunked_shards`.
- Monitor state: `scratch/20260427T210840Z_ifbench_vllm_70b_http400_repair_v6e8_98k_chunked_r1_monitoring_state.json`.

**Policy:** if this fails with the same HBM/JAX compile error, stop retrying the outlier and keep the 109/110 repaired dataset.

### 2026-04-27 21:16 UTC — final fully repaired 20k bundle complete

**98k chunked outlier repair:** `/ahmed/ifbench-vllm-70b-http400-repair-v6e8-98k-chunked-r1`
- State: succeeded.
- Duration: ~4m55s.
- Failure/preemption counts: 0 / 0.
- Key finding: `max_model_len=98304` works on v6e-8 when `max_num_batched_tokens` is capped at 32768. The previous 98k failure was caused by compiling with `max_num_batched_tokens=98304`, not by the context length alone.
- Final GCS output: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_ctx98304_chunked.jsonl`.
- Local copy: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repair_http400_ctx98304_chunked.jsonl`.
- Validation: 1 row, `finish_reason=stop`, 0 empty responses, 91,055 input tokens, 21 output tokens.

**Fully repaired 70B output:**
- Local: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repaired_full.jsonl`.
- GCS: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repaired_full.jsonl`.
- Rows: 20,000.
- Unique prompt IDs: 20,000.
- Finish reasons:
  - `stop`: 17,400
  - `length`: 2,600
- Empty responses: 0.

**Final 20k extraction with fully repaired 70B:**
- Output dir local: `/tmp/ifbench_port/overnight_20k/outputs_iris8b_70b_v6e8_gemini_repaired_full/`.
- Output dir GCS: `gs://marin-us-central2/scratch/ifbench/overnight_20k/outputs_iris8b_70b_v6e8_gemini_repaired_full/`.
- DPO pairs: 6,599.
- SFT examples: 13,820.
- Pair yield: 32.995%.
- No passers: 11,790.
- No failers: 1,611.
- Yield by `num_constraints`:
  - 1: 2,425
  - 2: 2,154
  - 3: 1,303
  - 4: 600
  - 5: 117

**Delta vs 109/110 repaired extraction:**
- DPO pairs: +4 (6,595 → 6,599).
- SFT examples: -5 (13,825 → 13,820) because the formerly empty 70B answer changed the per-prompt pass/fail mix.
- No-fail prompts: -4.

**Current best dataset:** use `outputs_iris8b_70b_v6e8_gemini_repaired_full/` for training ablations. It is the cleanest 20k bundle: Gemini Flash + Iris Llama-3 8B + fully repaired Iris Llama-3.3 70B, no Together dependency.

### 2026-04-27 21:21 UTC — handoff / what next

**Current state:** 20k IFBench data generation is complete for this phase. No repair jobs remain active. Do not wait on Together for this run; Together Batch was unreliable and is no longer on the critical path.

**Best artifacts to use:**
- DPO/SFT bundle, local: `/tmp/ifbench_port/overnight_20k/outputs_iris8b_70b_v6e8_gemini_repaired_full/`
  - `dpo_pairs.jsonl`: 6,599 rows
  - `sft_examples.jsonl`: 13,820 rows
  - `summary.json`: pair yield 32.995%
- DPO/SFT bundle, GCS: `gs://marin-us-central2/scratch/ifbench/overnight_20k/outputs_iris8b_70b_v6e8_gemini_repaired_full/`
- Fully repaired 70B rollouts, local: `/tmp/ifbench_port/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repaired_full.jsonl`
- Fully repaired 70B rollouts, GCS: `gs://marin-us-central2/scratch/ifbench/overnight_20k/iris/rollouts_llama_3_3_70b_v6e8_repaired_full.jsonl`

**Provider/inference conclusion:**
- Gemini Flash file batch completed and is in the final bundle.
- Iris 8B vLLM completed and is in the final bundle.
- Iris 70B vLLM completed, then all 110 HTTP-400 failures were repaired.
- Together Batch should be treated as a negative result for this run: 70B returned mostly provider `Internal Server Error`s; 8B-Lite never became a reliable complete source. Keep Together code paths for future tiny probes only, not for this 20k dataset.

**Important vLLM lesson for future long-context repair:**
- For 70B on v6e-8, `max_model_len=98304` can work if `max_num_batched_tokens=32768`.
- The failed 98k repair used `max_num_batched_tokens=98304` and died during JAX/XLA compile from HBM pressure. The context length itself was not the only issue.

**Recommended next work:**
1. **Freeze the dataset.** Add a small manifest/README or dataset config that points training code at the final GCS bundle. Avoid regenerating rollouts unless the user explicitly asks for a larger run.
2. **Validate the training loaders on the final files.** Load `dpo_pairs.jsonl` and `sft_examples.jsonl`, check schema fields, tokenize a small batch, and confirm sequence length/truncation behavior for the intended base model.
3. **Run fixed-dataset ablations before scaling beyond 20k.**
   - SFT-only on `sft_examples.jsonl`.
   - DPO-only on `dpo_pairs.jsonl`.
   - DPO+SFT mixture using the same fixed data.
   - Keep evals identical across arms so the training signal comparison is clean.
4. **Mixture default for first pass:** start with one SFT batch and one DPO batch per optimizer step, loss `dpo_loss + alpha_sft * ce_loss`. Try `alpha_sft in {0.05, 0.1, 0.2}`. The likely first run is `alpha_sft=0.1`, then adjust based on IFBench/IFEval and general instruction quality.
5. **DPO pair-margin ablation:** construct fixed-size pair subsets from the same 6,599 pairs:
   - close margin: chosen barely beats rejected
   - medium margin
   - wide margin
   - mixed/all pairs
   Keep the number of pairs constant across arms to isolate margin quality from dataset size.
6. **Scoring analysis before training if time permits:** compute per-model strict/loose pass rates on the final 20k rollouts and summarize why each pair was formed. This will make the margin bins defensible.

**Immediate best next command sequence for the next agent:**
1. Inspect final pair/SFT schema:
   `head -n 2 /tmp/ifbench_port/overnight_20k/outputs_iris8b_70b_v6e8_gemini_repaired_full/dpo_pairs.jsonl`
2. Find existing DPO/SFT training entrypoints/configs:
   `rg -n "SimpleDPOConfig|dpo_pairs|sft_examples|alpha_sft|DPO" experiments lib -g '*.py' -g '*.yaml'`
3. Decide whether to implement the mixture trainer hook now or first run a loader-only smoke test.

**Do not spend time on next:**
- Do not debug Together Batch further unless the user asks. It is no longer blocking.
- Do not launch 95k generation yet. The next empirical question is training signal quality on the clean 20k dataset.
- Do not restart or bounce Iris clusters; all needed jobs completed.
