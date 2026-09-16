# A finite-pool test of epoch matching

This experiment tests whether an epoch-matched small proxy selects a better mixture for a larger training budget than an unmatched proxy. It supplies the three observed curves proposed for Figure 2A: an unmatched proxy, a target that naturally repeats a finite StarCoder corpus, and a proxy whose smaller corpus matches the target's repetition. The experiment is prepared for review; no new training has been submitted.

## Background

Existing target curve C40 already supplies 25 usable measurements at 7.408B training tokens. Its interior optimum is near StarCoder fraction 0.70. The original p=1 endpoint used a different StarCoder pool because zero-weight web components were removed before assigning sequential shuffle keys. We replace that endpoint and freeze the interior pool for every new run. The [source audit](target_reuse_audit.md) records the measurements, cache identities, shuffle keys, and remaining runtime verification; [research.md](research.md) relates the design to earlier experiments and repetition-mismatch work.

## Design

All runs use the same Qwen3 architecture: 210,052,480 total parameters, including 45,884,800 nonembedding parameters. Only training horizon and available StarCoder support change. The initial StarCoder subset is the finite benchmark corpus, shared across conditions; it is distinct from the subsequent reduction for simulated epoching. The six Nemotron web components retain their full caches and remain nonrepeating.

| Condition | Training tokens | Available StarCoder tokens | Maximum nominal StarCoder epochs |
|---|---:|---:|---:|
| Small, unmatched | 277,872,640 | 279,969,792 | 0.9925 |
| Target | 7,408,189,440 | 279,969,792 | 26.4607 |
| Small, epoch-matched | 277,872,640 | 10,485,760 | 26.5000 |

For StarCoder fraction \(p\), the nominal epoch counts are the last column multiplied by \(p\). The unmatched proxy cannot exhaust its parent even at p=1. Exact epoch matching would use \(P D_s/D_t\) tokens; whole-batch slicing gives a 0.149% nominal discrepancy. The local allocator replay gives at most 0.174% discrepancy across the full grid after mixture-block rounding; [the audit](sequence_allocation_audit.json) records its runtime and every coordinate. The matched subset is the first 40 batches of the same ordered parent whose full size is 1,068 batches. It is fixed across all mixture weights and trainer seeds.

Both proxies use 1,060 steps with batch size 128 and sequence length 2,048. The target uses 28,260 steps. The inherited MuonH optimizer uses Muon/Adam learning rates 0.02/0.008, one-percent warmup, and WSD with the last fifth decaying; phase boundaries are aligned to 2,048-sequence mixture blocks. The two phase weight vectors are identical, so each run uses a constant mixture. The smaller run has about 6.06 tokens per nonembedding parameter, against 161.45 for the target; residual horizon effects are expected even after epoch matching.

The frozen manifest contains all 26 observed target mixture fractions. The pilot uses p=0, 0.10, 0.30, 0.70, and 1.00. The reference trainer seed is 20260711, matching the historical target; replication adds trainer seeds 20260908 and 20260909. Data seed and all seven component shuffle keys remain fixed. At p=0 the matched and unmatched proxies are identical, so a single run supplies both labels.

| Cumulative stage | New proxy runs | Corrected target endpoints | Total estimated new training FLOPs |
|---|---:|---:|---:|
| Pilot | 9 | 1 | 8.705e18 |
| Primary, full grid | 51 | 1 | 1.896e19 |
| Replicated, full grid | 153 | 1 | 4.386e19 |

These costs include the corrected target endpoint once and exclude the 25 reused measurements, evaluation, startup and failures. They use three times the architecture's forward FLOPs, 878,499,840 FLOPs per token. Stage totals include preceding stages; successful matching artifacts are reused. Pilot-to-primary and primary-to-replicated promotion require the earlier stage to be complete. Promotion remains an explicit command after reviewing the outcomes.

The primary metric is held-out Paloma programming-languages BPB. Select each proxy's minimum on the common measured grid, then evaluate that selected fraction on the target curve. Report target loss and regret relative to the target's minimum on that same grid, alongside the selected fractions. Use observed losses, not surrogate fits, for selection. Exact ties select the smaller fraction and retain the full tie set in the report. The replicated analysis averages the three proxy trainer seeds before selection. Target regret remains conditional on the archived target measurements and the corrected endpoint; this is not a replicated target-performance claim.

The figure plots measured BPB against StarCoder fraction, marks the three minima, and reports target loss at the two proxy-selected fractions. Different absolute loss levels across scales are expected. An epoch-matched proxy need not reproduce the target curve vertically, and a negative result is a valid outcome.

## Costs and limitations

The short proxy or its roughly 10M-token matched subset may give an inaccurate optimum. The pilot checks this regime before releasing the larger sweep. Its outcome is recorded regardless of whether matching helps; an unfavorable ordering is not a reason to alter the protocol silently. Trainer repeats measure training randomness conditional on one parent and one nested subset. They do not measure uncertainty from choosing different parent corpora or proxy subsets. Repeating target measurements at the selected mixtures would be a separate confirmation experiment.

After GCS access was restored, the historical configuration, final metric, child package versions and actual cache length were verified. The archived and current shufflers agree on every parent position under the historical JAX version, and the matched subset is the exact parent prefix. The [reviewed receipt](reuse_audit.json) records those checks and is required for submission. The exact original launch Git SHA remains unavailable; this is configuration and mapping equivalence, not a complete original-binary replay. The current runtime/cache audit also passed. No training has been submitted.

## Testing and execution

Sequence-level tests verify that the fixed StarCoder shuffle reproduces the historical interior ordering at p=1, that the matched subset is nested in the unmatched parent, and that unaffected components and validation data retain their behavior. Launcher tests cover materialized caps and seeds, cumulative stage selection, and durable fingerprint checks. Analysis tests cover stale or incomplete measurements, the replaced endpoint, p=0 aliasing, and target regret when proxy loss levels differ.

Use [spec.md](spec.md) for interfaces and [review_packet.md](review_packet.md) for commands and verification results. All proposed jobs use us-central1/us-central1-a, keeping the existing caches and outputs in their original region. Nothing is submitted by the default launcher command.

## Questions for CC's review

- Is the 278M-token horizon sufficiently informative for this fixed model, or should the pilot be regarded only as a feasibility check?
- Is 0.149% nominal epoch mismatch acceptable, or is a sequence-level cap worth adding to remove most rounding error?
- For the motivating figure, are conditional target regret and proxy trainer repeats sufficient, or should a later target-seed confirmation be budgeted?
