# Complete proxy and target math curves

All fifteen target checkpoints and fifteen unique proxy checkpoints have verified MATH-500 and GSM8K reference-solution likelihoods. The final release reused the eleven first-stage target measurements by their exact payload hashes and evaluated only the four newly completed FineMath targets. The CPU parent and TPU child both succeeded, without retry. The full plot contains eight points per domain and model scale; each scale's web-only point is shared across domains.

| Training domain | Scale | MATH-500 minimum fraction | Epochs | Perplexity | GSM8K minimum fraction | Epochs | Perplexity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wikipedia | Matched proxy | 30% | 4.76836 | 21.17463 | 20% | 3.16191 | 18.62546 |
| Wikipedia | Target | 50% | 7.93719 | 7.83202 | 20% | 3.16097 | 7.61101 |
| FineMath-3+ | Matched proxy | 70% | 11.09805 | 7.53399 | 70% | 11.09805 | 11.43987 |
| FineMath-3+ | Target | 100% | 15.82782 | 3.55485 | 70% | 11.09793 | 4.03641 |

FineMath's target MATH-500 likelihood continues improving through the 100% endpoint. This establishes a boundary minimum on the observed grid, not an interior optimum or a turnover point. FineMath's GSM8K target minimum is at 70%, matching its proxy's selected fraction; perplexity rises from 4.03641 at 70% to 4.22917 at 100%. Wikipedia favors lower fractions on both math evaluations, although its MATH-500 target valley is shallow and its minimum shifts from 30% at proxy scale to 50% at target scale.

These are single-seed, token-weighted reference-solution likelihood measurements. They do not measure generated-answer accuracy. The training subsets have not undergone a benchmark decontamination audit, so these remain exploratory likelihood diagnostics. Epoch sweeps also change how much web data the varied domain replaces. The results should not be interpreted as pure causal effects of repeating a fixed mixture.

The common seven-component Uncheatable objective gives a different result: both domains, at both scales, have their observed minimum at 30% (4.768 epochs). Its completed, corrected BPB publication and plots are recorded separately in `../native_closeout_20260912/`.

## Verification and provenance

Frozen completed spec: `c788d33ddd3b357a7f80de6ed1d5ecb632104e4beabd5cc0934d580ecac77d39`. The scorer, tokenizer, source generations, response mask, windowing, batch size, and metric aggregation match the previous proxy and target evaluations. All thirty live permanent checkpoint metadata snapshots match their frozen identities. All result receipts use the same scored-population hash, `0b8b22970794a9bedb4f0cb55bb5643c38f9d9bf43e7c19c0bb6aec1f59376eb`.

Every restored PALOMA token-loss control passes the original 0.00005 tolerance. The maximum absolute discrepancy across all thirty checkpoints is 0.000003815; across the four newly evaluated targets it is 0.000000358. No tolerance was changed. `final_validation.json` records every checkpoint's discrepancy and the four new result pairs.

Iris parent `/calvinxu/tpp10-target-math-completion` and child `/calvinxu/tpp10-target-math-completion/target-math-completion` succeeded. The child took 2 minutes 5 seconds. Fieldbook parent `job_01m2bq8ta6wg2w5d3bgjhkkxxs` and child `job_01m2bqq7b7dxhpvdgm15aevfny` are linked to the four existing training runs. Only those four runs receive new math result artifacts and metric records; the eleven reused target measurements are not duplicated.

The complete plot, CSV, per-run receipts and summaries are under `results/c788d33ddd3b357a7f80de6ed1d5ecb632104e4beabd5cc0934d580ecac77d39/`. The PNG/PDF was visually checked for label overlap and clipping. It uses measured points and straight joins, with no fitted curves. No paper or frozen evaluator source was changed during collection.

Reproduce from the repository root:

```bash
uv run python -m experiments.domain_phase_mix.analyze_tpp10_target_math \
  --complete-spec .agents/projects/starcoder_tpp10/domain_sweeps/target_math_complete_20260912/spec.json \
  --archive .agents/projects/starcoder_tpp10/domain_sweeps/native_closeout_20260912/plots \
  --require-complete
```
