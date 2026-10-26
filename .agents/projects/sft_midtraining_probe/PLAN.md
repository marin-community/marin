# SFT probe: do midtraining-style two-phase schedules help after fine-tuning? (Delphi 3e18)

Status: draft for discussion with Calvin (2026-10-14). Becomes the handoff once the open decisions at the end are settled.
Checkpoint facts come from `checkpoint_inventory.md` / `.csv` beside this file.

## Question

The one-phase MARINER paper reports that, at the same swarm budget, two-phase surrogates and schedules did not find better
mixtures than a single-phase swarm and surrogate (Section 8, Appendix E). Every objective there is a base-model loss
(Uncheatable BPB, OlmoBaseEval Easy BPB). Liu, Neubig and Xiong (2025, arXiv:2510.14865, "Midtraining Bridges Pretraining
and Posttraining Distributions") argue that midtraining helps mainly by preparing the model for post-training, and they
measure it by validation loss after supervised fine-tuning (SFT). This probe asks whether our two-phase schedules, whose
second phase (the learning-rate decay, the last 20% of training) was meant to stand in for midtraining, show an SFT
benefit that base-model objectives missed.

## What Liu et al. did (for matching our protocol)

- Pythia 70M/160M/410M/1B trained on C4 for 128B tokens; a specialized mix (StarCoder code, math, FLAN, KnowledgeQA, DCLM)
  enters at a chosen step and stays to the end, at 5-20% of tokens.
- SFT on four sets: Pycode (CodeSearchNet Python, 5,000 examples, no prompt masking), GSM8K (7,500, prompt masked), LIMA
  (1,000, no masking), SciQ (13,679, prompt masked). 4 epochs, global batch 64, 10% linear warmup then cosine decay, peak LR
  from a grid on the base model. Metric: target-domain validation loss after SFT; forgetting: C4 validation loss after SFT.
  5 seeds.
- Effects at 1B: StarCoder midtraining -> Pycode SFT loss 2.174 -> 1.888; math -> GSM8K 0.942 -> 0.851; most other pairs move
  0.02 or less. Gains track the midtraining data's token-level proximity to the SFT data. Late introduction tolerates only
  small weights, and raising the weight later does not recover a late start.

## Arms

All arms are Delphi 3e18 (Qwen3 360M, 1.6B tokens, simulated epoching toward the 6.33T target), so every comparison shares
model size, token budget and repetition.

### Main comparison (new pretraining where checkpoints are missing)

| Arm | Mixture | Isolates |
| --- | --- | --- |
| A | MARINER's OlmoBaseEval Easy optimum (`lwspu_t9_snc_cap08`, the paper's 3e18 proposal) | baseline |
| B | `dphase_t9b075_can_e0p005`, the Table-9 two-phase frontier policy (code, math and synthetic QA moved into the second phase) | B - A: the practical comparison |
| C | B's phase-weighted average 0.8 w0 + 0.2 w1, trained single-phase | B - C: timing (midtraining) effect at fixed per-bucket tokens; C - A: composition |

- Three trainer seeds per arm at A's data seed(s), so arms pair by (data seed, trainer seed). Reuse existing checkpoints where
  the inventory finds them; train the missing ones with the same launcher and data seeds.
- Optional B': a Dolmino-late schedule whose phase-weighted average is A's mixture (late Dolmino share 0.75, the only share
  that did not hurt base BPB in the July phase-asymmetry panel). Then A is B''s own C, so B' - A isolates timing at MARINER's
  optimum.

### Existing matched pairs (no new pretraining)

1. Swarm pairs: the 280-run two-phase swarm and the 280-run single-phase swarm trained on its phase-weighted averages
   (Fieldbook exp_01kxhhy6k73fwz111ab42q5ssq and exp_01kxm37amarnnd1wey37nj8ed5). 238 designs differ between the two;
   42 phase-constant designs are shared. Pairs share data seeds (verify in the inventory). SFT a stratified subset (e.g. the
   40 designs with the largest phase-1 shift toward code and math plus 40 random designs), or all 238 if cheap.
2. July 2026 aggressive phase-asymmetry panel (exp_01ky48sv686wh45tnyc1pnrgym): fixed-aggregate two-phase contrasts around two
   single-phase frontier anchors (Uncheatable and OlmoBaseEval Easy), each with 16 tied single-phase controls, 9
   Dolmino-late schedules (late Dolmino share 0.75/0.9/1.0, 3 repeats), 24 handcrafted late-quality recipes and 96 balanced
   antithetic partitions. Base-model result: no generic gain; 0.9-1.0 late Dolmino shares hurt base BPB by 0.006-0.009.
   The Dolmino-late family is the closest analogue of conventional midtraining.

## SFT protocol

- Sets and masking as in Liu et al.: Pycode (no masking), GSM8K (prompt masked), LIMA (no masking), SciQ (prompt masked).
  Hold out a fixed validation split per set (GSM8K test, SciQ validation, a fixed 10% of Pycode, LIMA's test set), identical
  for every checkpoint.
- 4 epochs, global batch 64, 10% warmup then cosine. Peak LR per SFT set from a small grid (1e-5, 3e-5, 1e-4, 3e-4) on the
  proportional-mixture checkpoint (not on any arm), then fixed for all arms, so tuning favors no arm.
- Two SFT seeds (data order) per base checkpoint.
- Forgetting: after SFT, Uncheatable BPB and Paloma C4 BPB with the existing evaluation steps.
- Launcher: `experiments/sft/launcher.py` (`sft_step`, `SFTSpec`, `LevanterCheckpointModel` or an HF-checkpoint source;
  Delphi chat template in `experiments/sft/delphi_chat_template.py`). Plain prompt/completion examples with completion-only
  masking cover GSM8K and SciQ; Pycode and LIMA put the whole example in the completion.

## Analysis (fixed before running)

- Primary: paired difference B - C in GSM8K and Pycode SFT validation loss, averaged over (trainer seed, SFT seed) pairs, with
  SE across the three trainer seeds. One-sided: midtraining helps if B - C < 0 on both, with no worse forgetting.
- Secondary: B - A and C - A; LIMA and SciQ; forgetting losses.
- Swarm pairs: distribution of (two-phase - single-phase) SFT loss per design; regression of the Pycode and GSM8K deltas on the
  phase-1 minus phase-0 share of code and of math buckets (Liu et al.'s proximity prediction: larger late code/math shift,
  larger SFT gain).
- Asymmetry panel: Dolmino-late schedules and recipes against their anchor's tied controls (anchor + seed block), as in the July
  report.
- Context for effect sizes: base-model two-phase vs single-phase differences at 3e18 are about 0.002-0.008 BPB; tied-control
  repeat SD is 0.0010 (Uncheatable) and 0.0031 (OlmoBaseEval Easy) BPB.

## Interpretation

- B (or the Dolmino-late family) lower than its single-phase control after SFT, beyond seed noise: the paper's negative
  two-phase result holds only for base-model objectives; Section 8 should say so.
- No difference: the negative result extends to post-SFT loss at this scale. Our second phase starts at 80% of training, which
  Liu et al. call late introduction, so a null does not rule out an earlier switch.

## Constraints for the executing session

- Never stop or restart an Iris cluster. Keep training and SFT in the region of the checkpoints (likely us-east5); no
  cross-region copies of checkpoints.
- Submit through `marin.run.iris_run` and follow the memory notes on bundle size, preemption retries, early release after a
  canary, and progress logging with ETAs.
- Record the experiment in Fieldbook (new experiment id) and results under
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/sft_midtraining_probe_<date>/`.

## Decisions (defaults set 2026-10-14; Calvin may override)

Updated after the checkpoint inventory (`checkpoint_inventory.md`): every needed checkpoint exists in us-east5 (HF export
at `hf/step-3006`), and the lowest single runs are within seed noise (the Table-9 anchor's tied mixture spans 1.0553 to
1.0717 over 25 data seeds, SD 0.0040), so B is chosen for its schedule, not its rank.

1. B = `dphase_t9b075_can_e0p005`, the Table-9 two-phase frontier policy: phase TV 0.115, with code, math and synthetic QA
   moved into the second phase, so it resembles midtraining. C = `dphase_t9b075_tied`, its exact phase average, which
   already exists at the same seeds (data seed 690300, trainer seed 0, v5p-8; gap 0.00084 BPB). Lower single runs
   (`rphase_216` 1.0533, `fiber_1_domain_13_minus` 1.0562 with phase TV 0.011) are not used.
2. To pair with A (data seed 662009, trainer seeds 0-2, v6e-8 in us-east5-b), train B and C at that data seed and those three
   trainer seeds on v6e-8: six new runs. Keep the existing 690300 B/C pair as an extra pair.
3. B' = a Dolmino-late schedule whose phase average is A's mixture (late Dolmino share 0.75, the share that did not hurt
   base BPB in the July panel), three trainer seeds at data seed 662009 on v6e-8: three new runs. A is its own C.
4. Swarm pairs: all 238 two-phase/one-phase design pairs (same data seed 7141000 + run order, trainer seed 0). The one-phase
   weights are the realized phase average with alpha0 = 2400/3007 = 0.798. On base BPB the two-phase run averages 0.0033
   (Table-9) and 0.0018 (Uncheatable) worse, with phase TV 0.38-0.66.
5. July panel (panel 3 in the inventory): all 290 runs, paired by anchor and seed block; its Dolmino-late family and tied
   controls are the priority if SFT throughput is limited. The phase-fiber DOE (panel 4, 200 runs) is optional.
6. Uncheatable: OlmoBaseEval Easy is primary. Add the Uncheatable A/B/C only after the primary results are in.
7. Storage: the bucket uses Autoclass; reading a Nearline or Coldline checkpoint has no retrieval fee and moves it back to
   Standard. Keep reads in us-east5.

## Execution steps

1. Inventory: done (`checkpoint_inventory.md/.csv`, 2026-10-14). Spot-check a few URIs before relying on it.
2. Materialize the arm mixtures: B's two phases and C's average from the decoupled panel's mixture CSVs (launcher
   `experiments/domain_phase_mix/launch_delphi_decoupled_phase_information_panel.py`); B' from A's weights (policy
   `lwspu_t9_snc_cap08`). Store the exact weights and hashes.
3. Pretrain the nine new runs (B, C, B' x trainer seeds 0-2 at data seed 662009) at 3e18 on v6e-8 in us-east5-b with the
   launcher A used (`experiments/domain_phase_mix/launch_delphi_frozen_procedure_validation_3e18.py` /
   `launch_delphi_fairness_repeats_3e18.py`); release one canary first and the rest at its first committed checkpoint.
4. Build the SFT pipeline: the four datasets with fixed validation splits, masking as above, `experiments/sft/launcher.py`.
   Run the LR grid on the proportional checkpoint; freeze the LR per SFT set.
5. Run SFT for every base checkpoint (main arms, swarm pairs, July panel) at two SFT seeds; then the forgetting evals.
6. Analyze as above; write the report with per-arm tables and the swarm-pair regression; record in Fieldbook.
