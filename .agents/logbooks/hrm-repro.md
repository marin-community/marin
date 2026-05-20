# HRM-Text Reproduction with marin MoE: Research Logbook

## Scope
- Goal: reproduce the HRM-Text training data + token budget with the current
  marin MoE recipe (`experiments/grug/moe/`).
- Reference: `sapientinc/HRM-Text` (model + training), `sapientinc/data_io`
  (data pipeline), `sapientinc/HRM-Text-1B` HF model card.
- User intent: start with a d768 test run; full reproduction at d1280 once
  the data + heuristic + launch wiring is validated.

## Reference numbers (HRM-Text XL, 1B params)
- Architecture: hidden 1536, 32 layers (16 H + 16 L, 2×3 recurrence), 12 heads
- Tokenizer: BPE 65k
- Global batch size: 196,608 tokens
- LR: 2.2e-4, warmup 2000, AdamATan2, beta 0.9/0.95, wd 0.1, EMA 0.9999
- **Unique tokens trained on: 40B** (per HF model card)
- Data: stratified-sampled instruction/response mixture (FLAN, SYNTH,
  openthoughts2, openmathinstruct2, acereason, sudoku, math train,
  Platypus, gsm8k_train, no_robots, etc.), 4 epochs, PrefixLM with
  target-only loss.

## Port mapping → marin MoE

| HRM-Text | marin MoE (`hrm_repro.py`) |
|----------|----------------------------|
| BPE 65k tokenizer | `meta-llama/Meta-Llama-3.1-8B` (128_256 vocab, matches `MoeAdamHHeuristic.vocab_size`) |
| Cleaned data + tokenization + stratified sample | `sapientinc/HRM-Text-data-io-cleaned-20260515` via `HfTokenizeConfig` with `SupervisedLmDatasetFormat(input_key="instruction", target_key="response")` |
| Target-only PrefixLM loss | `SupervisedTextProcessor` writes `loss_weight=0` on input tokens and `=1` on response tokens; `GrugLmExample.causal()` honors it |
| HRM XL: 32 layers, hidden 1536 (1B params, recurrent) | marin d1280: 13 layers, hidden 1280 (heuristic-derived; MoE, E=64 K=4 + shared) |
| 40B unique tokens, 4 epochs | `HRM_REPRO_TARGET_TOKENS=4e10` × 1 marin epoch (Levanter handles document shuffling) |
| AdamATan2 + warmup 2000 | `MoeAdamHHeuristic.build_optimizer_config(batch, tokens, dim)` (AdamH with constant-token-half-life β2, linear LR sched, warmup 10%) |

## Sizing & wall-clock (v5p-8)

Token count is pinned to HRM XL's 40B and the heuristic derives the batch:
```
batch_exact = 4e10 / (2**14 * 4096) ≈ 596 → 1024 (next power of two)
num_steps   = round(4e10 / (1024 * 4096))   = 9537
```

| Dim   | Layers | Tokens | Batch | Steps | fpt        | C ≈ 3·fpt·T | v5p-8 tok/s | ETA   |
|-------|--------|--------|-------|-------|------------|-------------|-------------|-------|
| 768   | 8      | 4.0e10 | 1024  | 9537  | 2.09e8     | 2.51e19     | 273k        | 40.7 h |
| 1280  | 13     | 4.0e10 | 1024  | 9537  | 7.62e8     | 9.14e19     | 128k        | 86.8 h |

`v5p-8 tok/s` from `experiments/grug/moe/README.md` compute-optimal table.

Note: at d768 / 40B, we are well past the v16 compute-optimal point
(2.19e17 → 2.7B tokens). The d768 test run is therefore overtrained — it
validates the data pipeline and heuristic end-to-end but isn't on the
compute-optimal frontier.

## Evaluation

- **In-training perplexity**: `add_validation_sets_to_mixture(HRM_TEXT_MIX,
  default_validation_sets(...))` registers Paloma + uncheatable_eval shards.
  Eval runs every 1000 steps via the existing `GrugEvalConfig`.
- **HRM-Text downstream table (MMLU, ARC-C, HellaSwag, Winogrande, BoolQ,
  GSM8k, MATH, DROP)**: declared in `HRM_TEXT_BENCHMARKS` (matching marin task
  aliases). Running them requires either:
  (a) a Grug-MoE → Levanter `LmHeadModel` adapter (would let
      `levanter.eval_harness.lm_eval_harness` run in-training), or
  (b) a Grug-MoE → HF checkpoint exporter (for the vLLM-based
      `evaluate_lm_evaluation_harness` step).
  Neither exists today for the MoE template — left as a follow-up.

## Submission

```bash
.venv/bin/iris --config lib/iris/config/marin.yaml job run \
  --no-wait --preemptible --reserve v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe.hrm_repro
```

Override knobs:
- `HRM_REPRO_HIDDEN_DIM=1280` for the full reproduction
- `HRM_REPRO_TARGET_TOKENS=...` to override the 40B default
- `HRM_REPRO_RUN_SUFFIX=...` to label re-runs

## Initial submission

- Job: `/kaiyue/iris-run-job-20260520-025649` (d768, 40B tokens, v5p-8
  preemptible). Submitted 2026-05-20 02:56 UTC. Reservation pending behind
  the other two in-flight runs (`context_norm_no_xsa_gate1` d512 + d768).

## Update 2026-05-20: full HRM-Text mix ported as 13 per-source cleaners

After the dmmath-only mix turned out to be ~5% of HRM XL's training data,
ported the rest of `sapientinc/data_io`'s cleaners as marin ExecutorSteps in
`experiments/grug/moe/hrm_text_data.py`. Each source streams the upstream HF
dataset, applies the same filter/transform as the data_io cleaner, and writes
`(instruction, response, condition)` parquet to GCS. Streaming + chunked
`pyarrow.ParquetWriter` means even AceReason (93 GB on Hub) runs with only
32 GB RAM / 64 GB disk per worker.

| Source                | HF id                                | Notes                       |
|-----------------------|--------------------------------------|-----------------------------|
| openmathinstruct2     | nvidia/OpenMathInstruct-2            | cot + direct (single pass)  |
| acereason             | nvidia/AceReason-1.1-SFT             | filter category=math, strip `<think>` |
| openthoughts2         | open-thoughts/OpenThoughts2-1M       | drop code-heavy sources     |
| sudoku_extreme        | sapientinc/sudoku-extreme            | "Solve the Sudoku" prompt   |
| textbookreasoning     | MegaScience/TextbookReasoning        | cot + direct                |
| gsm8k_train           | openai/gsm8k                          | split on "#### "            |
| math_train            | EleutherAI/hendrycks_math             | all subsets                 |
| omnimath              | KbsdJames/Omni-MATH                   | cot + direct                |
| numinamath            | AI-MO/NuminaMath-1.5                  | filter synthetic / invalid  |
| natural_reasoning     | facebook/natural_reasoning            | filter proofs               |
| principia             | facebook/principia-collection         | direct                       |
| webinstruct_verified  | TIGER-Lab/WebInstruct-verified        | direct                       |
| no_robots             | HuggingFaceH4/no_robots               | concat system+user          |
| dmmath                | sapientinc/HRM-Text-data-io-cleaned-20260515 | already cleaned on HF |

Mixture weights are taken directly from `data_io/prefix_config.yaml`
(`max_per_file × repeat`), so each source's share of one HRM-Text epoch is
preserved in the marin mixture. After normalization the breakdown is:

- dmmath 67%, openmathinstruct2 / acereason / webinstruct_verified 8% each,
  sudoku 4%, openthoughts2 2%, textbookreasoning / numinamath /
  natural_reasoning / principia / no_robots ~0.4% each, math_train /
  gsm8k_train / omnimath the rest.

**Skipped** (need raw downloads or heavier transforms; logged as follow-ups
in code docstring): amps_khan, ampsmathematica, Platypus/scibench, SYNTH,
tasksource (297-line subset allowlist), Platypus/ARB, openbookqa, reclor.

Resubmitted at d768 with the full mix as
`/kaiyue/iris-run-job-20260520-032117`.

## Caveat: HF "cleaned" dataset is a subset of HRM XL's actual training mix

Inspecting `sapientinc/HRM-Text-data-io-cleaned-20260515` directly:

```
train files: 168    (~4.08 GB total parquet, all under data_clustered/dmmath/)
val files:   8
test files:  30
unique data_clustered/ subdirs: ['dmmath']
```

The HF dataset page advertises "104M rows / 350 GB", but
`datasets.load_dataset_builder` only resolves the DeepMind Mathematics
(`dmmath`) subset — 168 parquet files / ~4 GB. The other cleaned subsets in
the data_io README (FLAN, SYNTH, openthoughts2, openmathinstruct2, acereason,
sudoku, Platypus, gsm8k_train, math_train, omnimath, ampsmathematica,
webinstruct_verified, no_robots, tasksource) are listed in
`data_io/prefix_config.yaml` but their cleaned outputs do not appear under
`data_clustered/` on this HF dataset.

So the current reproduction trains on the math-only slice of HRM XL's data
(short instruction / numeric response pairs from DeepMind Mathematics). Marin
will see ~ a few B tokens of unique content and revisit them many times to
hit the 40B-token budget — comparable to what HRM XL does with `repeat: N`
prefixes in `prefix_config.yaml`, just much narrower in domain.

Faithful full-mix reproduction requires either re-running data_io's pipeline
locally on all 14 source datasets (needs ~512 GiB RAM + Rust tokenizer), or
waiting for sapient to publish the remaining cleaned subsets to HF. Logged as
a follow-up.
