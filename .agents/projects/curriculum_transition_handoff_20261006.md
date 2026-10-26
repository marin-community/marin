# Handoff: do abrupt mixture switches cost loss in the StarCoder two-phase setting?

## Goal

Test a claim in the one-phase paper's Appendix E ("Two-phase curricula") in the one setting where two-phase
schedules beat single-phase mixtures under fresh seeds. The paper source is
`/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/sections/appendix_analysis.tex`
(read it for context, do not edit it). The paragraph under test:

> Our schedules switch mixtures abruptly, which may raise the loss temporarily after the switch. Replaying earlier
> data in a later stage helps most when the stages differ most, possibly by reducing the loss spike at the transition
> (Kotha and Liang 2026, "Replaying Pre-Training Data Improves Fine-Tuning", arXiv:2603.04964). Interpolating smoothly
> between the phase mixtures would reduce this change.

Answer two questions with measurements:

1. Does the abrupt switch raise loss temporarily, beyond the change the new mixture implies at steady state? Measure
   it on programming-languages BPB (`eval/paloma/dolma_100_programing_languages/bpb`, the objective) and on a web
   eval. How large is it, how long does it last, and does it grow with the phase contrast p1 - p0?
2. Does a smooth interpolation between the two phase mixtures, at the same per-bucket token totals and the same LR
   schedule, change final programming-languages BPB and the two-phase gain over the best single-phase mixture?

Do not edit the paper. Report back as described under Deliverable; Calvin brings the results into the paper in the
paper session.

## Where the two-phase advantage is (paper Figure 24, `fig:a-curricula-sweeps`)

The setting mixes Nemotron-CC with StarCoder on a 210M Qwen3 proxy trained with MuonH and a warmup-stable-decay
schedule whose decay starts at 80% of training, which is also the phase boundary. Each step is 128 sequences of 2,048
tokens. The first sweep crossed four token budgets (1.0B to 7.4B) with seven StarCoder supports (the full pool, or a
subset giving 1/8 to 4 times the StarCoder epochs of a 5.7T-token target run): 28 blocks of 125 schedules at one seed.
Each block's best single-phase mixture and best two-phase schedule (phase weights differing by at least 0.04) were
retrained with five fresh paired seeds.

All Holm-positive gains are at D = 7.41B tokens (28,260 steps, boundary at step 22,608). The winning schedules hold
StarCoder near zero for the first 80% of training and switch it on at the decay, so the largest mixture change the
design allows coincides with the decay onset:

| StarCoder subset (x target repetition) | Best single-phase p | Best two-phase p0 -> p1 | Gain, 5 paired seeds |
|---|---|---|---|
| 1x (m100) | 0.70 | 0.01 -> 0.86 | +0.0076 [0.0061, 0.0090] BPB, 5/5 |
| 2x (m200) | 0.40 | 0.00 -> 0.50 | +0.0105 [0.0083, 0.0126] BPB, 5/5 |
| 4x (m400) | 0.20 | 0.00 -> 0.28 | +0.0138 [0.0101, 0.0175] BPB, 5/5 |

Across these three blocks the gain is largest where the phase contrast p1 - p0 is smallest (0.28 at 4x) and smallest
where it is largest (0.85 at 1x). Repetition also differs between the blocks, so this pattern is descriptive.

The matched-compute sweep's confirmation at the same 7.41B cell (StarCoder at 1x) found that 0.02 -> 0.82 beat the
single-phase 0.7 by +0.0061 [0.0053, 0.0069] BPB in 8/8 fresh paired seeds. None of the four full-pool blocks has a
positive mean gain.

Sources, under `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/`:

- `starcoder_wsd80_dense_horizon_replay_confirmation_scaling_20260811/confirmed_selected_policy_gain_by_horizon_and_replay.csv`
  (gains, intervals, schedules, phase epochs);
- `starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811/` (run names, seeds and phase
  weights of the confirmation runs, whose W&B histories are the first place to look).

## Code

- Launchers in `experiments/domain_phase_mix/`: `launch_starcoder_wsd80_dense_support_surfaces.py` (the sweep),
  `launch_starcoder_wsd80_dense_support_empirical_optimum_confirmation.py` (the fresh-seed confirmation), and the
  base configuration `launch_starcoder_wsd_80_20_surface.py` (BATCH_SIZE 128, SEQ_LEN 2048, MIXTURE_BLOCK_SIZE 2048,
  `steps_per_eval=1_000`, WSD aligned to the phase boundary). Cells and StarCoder subset sizes are in
  `experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json`.
- A phase schedule is a Levanter stage list in steps, `train_weights=[(0, w0), (boundary_step, w1)]`.
  `rescale_mixture_schedule_for_batch_schedule` in `lib/levanter/src/levanter/data/mixture.py` converts steps to
  sequence indices, and `MixtureDataset` requires every stage start to be a multiple of the 2,048-sequence block,
  which is 16 steps here. `MixtureDataset` accepts any number of stages and fixes integer per-block counts within each
  stage, so a ramp can probably be written as a staircase of 16-step stages without changing the loader. Check the
  stage count and startup cost, the offsets into the finite StarCoder subsets across many stages (the sweep's
  `audit_runtime_sequence_counts` uses `_counts_per_block_per_stage`), and the launcher audits that assume exactly two
  stages and an 80/20 boundary (`audit_materialized_runtime_configs`). Only a schedule finer than 16 steps would need
  loader work.
- A ramp moves tokens between phases. To isolate the shape of the transition, hold each bucket's total tokens (and so
  StarCoder's epochs) fixed, for example with a ramp centred on the boundary whose area equals the step's, and keep the
  LR schedule unchanged. State the choice: a ramp that starts at the boundary lowers StarCoder exposure in the decay
  unless it is compensated.

## Measurement problem

The existing runs evaluate every 1,000 steps (confirm this for the confirmation launcher), so a transient shorter than
about 1,000 steps after step 22,608 does not appear in the eval curves. `train/loss` is logged far more often, but it
averages over the current mixture, so it jumps when the composition changes. I did not find per-bucket training-loss
logging in Levanter's `train_lm`. Options: rerun selected pairs with dense evaluation in a window around the boundary
(a windowed eval callback, or a small `steps_per_eval` with only the programming-languages set and one web set), save
dense checkpoints and evaluate them offline, or add per-bucket training-loss logging.

## Prior work to read before designing

- Gradient-conflict experiment: `reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811_v9/design.md`,
  launchers `experiments/domain_phase_mix/launch_starcoder_wsd80_gradient_conflict*.py`. It froze trajectories that
  include the confirmed winner (0.02 -> 0.82 at step 22,608) with full-state checkpoints at -256, -64, 0, +64 and
  +256 steps around the switch, and a switch-time intervention that moves the data switch to 0.6-0.9 of training with
  the decay fixed at 0.8. Its 2026-08-13 output inventory
  (`reference_outputs/starcoder_wsd80_gradient_conflict_output_inventory_20260811/`) shows all 256 trajectory roots
  empty, and later gate runs exist. Check what actually ran under
  `gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_gradient_conflict_review_v9_20260811/`
  (run `gcloud auth login` first). If those checkpoints exist, evaluating them may answer question 1 without new
  training.
- Coupled phase/LR-onset sweep:
  `reference_outputs/starcoder_wsd80_coupled_onset_refinement_confirmation_results_20260902/results.md`. Moving the
  coupled switch and decay onset to 0.6, 0.8 and 0.9 of training gave fresh eight-seed gains of +0.0050, +0.0071 and
  +0.0047 BPB, and one fixed schedule pair changed sign between onsets, so the value of a switch depends on the
  training state at the switch.
- LR-onset gradient probe: `reference_outputs/starcoder_wsd80_lr_onset_gradient_probe_results_20260823/report.md`.
- Logbook: `.agents/logbooks/wsd80-mechanistic-surrogate.md`.

## Suggested order

1. Without training: read the prior work, pull the W&B histories of the 7.41B confirmation pairs, and establish
   whether the gradient-conflict checkpoints exist.
2. Trace: rerun the 4x and 1x two-phase winners and their single-phase partners with dense evaluation from about step
   22,000 to 24,000, at two or three seeds. This is a trace within each run, not an endpoint test.
3. Intervention: compare the abrupt switch with linear ramps of a few widths (for example 512 and 2,048 steps, and the
   whole 5,652-step decay) at matched totals, with at least five fresh paired seeds as in the confirmation. The
   endpoint is programming-languages BPB at the final step; report paired-t 95% intervals and the change in the
   two-phase gain.

Get Calvin's approval for the run matrix and its compute before launching steps 2 and 3.

## Repo rules that matter here

Read `AGENTS.md` and the `use-iris` skill. Submit with `marin.run.iris_run`. Never stop or restart an Iris cluster.
Keep compute and data in us-central1, where the StarCoder subsets and checkpoints live. Detach long launches (nohup,
own log, Monitor) and log done/total with an ETA. Use paired seeds as the inferential unit. Report times in Pacific.
Commit only when asked, with no attribution lines.

## Deliverable

A short report with one figure in a new directory under `reference_outputs/`, covering:

- whether loss rises temporarily after the switch, by how much and for how long on each eval, and whether the rise
  grows with p1 - p0;
- the endpoint effect of smoothing on programming-languages BPB and on the two-phase gain, with intervals;
- replacement wording for the paper paragraph above, written as final sentences.
