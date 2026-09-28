# Wikipedia and FineMath under the same math evaluation

Wikipedia's observed minimum occurs at 4.768 epochs on MATH-500 and 3.162 on GSM8K; FineMath-3+ minimizes both at 11.098 epochs. All eight points per curve are verified, including the shared web-only checkpoint. This comparison shows different preferred repetition levels under a common evaluation. The common Uncheatable aggregate still selects 4.768 epochs for both domains.

| Evaluation | Wikipedia fraction / epochs | Wikipedia minimum PPL | FineMath fraction / epochs | FineMath minimum PPL |
|---|---:|---:|---:|---:|
| MATH-500, primary | 30% / 4.768 | 21.1746 | 70% / 11.098 | 7.5340 |
| GSM8K, secondary | 20% / 3.162 | 18.6255 | 70% / 11.098 | 11.4399 |

The separation has a measurable cost on this grid. On MATH-500, choosing 11.098 epochs for Wikipedia raises perplexity by 21.2% from its own minimum; choosing 4.768 for FineMath raises it by 17.6%. On GSM8K, using the other domain's preferred epoch count raises perplexity by 29.2% for Wikipedia and 27.3% for FineMath. These comparisons apply an exact epoch choice, not an upper bound: a sufficiently loose common epoch cap could permit both optima. They motivate adaptive selection without establishing that every fixed cap must fail.

| Domain fraction | Realized epochs | Wikipedia MATH-500 | FineMath MATH-500 | Wikipedia GSM8K | FineMath GSM8K |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.000 | 25.0401 | 25.0401 | 19.5368 | 19.5368 |
| 5% | 0.788 | 24.1551 | 13.7057 | 19.8673 | 16.6431 |
| 10% | 1.576 | 23.0834 | 11.4709 | 19.8598 | 14.8448 |
| 20% | 3.162 | 21.8695 | 9.6672 | 18.6255 | 14.5677 |
| 30% | 4.768 | 21.1746 | 8.8616 | 19.8314 | 12.7086 |
| 50% | 7.938 | 22.3303 | 7.8880 | 19.2944 | 12.0467 |
| 70% | 11.098 | 25.6703 | 7.5340 | 24.0726 | 11.4399 |
| 100% | 15.825 | 121.7698 | 8.1328 | 260.0919 | 12.9085 |

Both domains improve MATH-500 over the shared zero-fraction control before worsening. Wikipedia's GSM8K curve is less smooth and its best improvement over the control is only 4.7%; MATH-500 gives the cleaner primary illustration. All values are reference-solution perplexities conditioned on the problem, not generated-answer accuracy. The two domains use the same evaluation examples, prompt format, solution-token mask, tokenizer and frozen scorer. MATH-500 includes 500 problems and 112,697 scored solution tokens; GSM8K includes 1,319 problems and 173,832 tokens. No examples were truncated, and original GSM8K solution markup was retained.

Seven existing Wikipedia matched proxies were newly evaluated. The zero-fraction receipt was reused from the completed FineMath evaluation with its source URI, specification hash and receipt hash recorded. The job performed no training or target evaluation. Both central1 parent and TPU worker succeeded; all checkpoint identities and PALOMA controls passed. An independent live audit reproduced all 16 raw curve receipts, every reported minimum and penalty, all seven new checkpoint metadata hashes, and shared-p0 lineage. Maximum Wikipedia PALOMA token-loss discrepancy is 3.81470e-6 against the unchanged 5e-5 tolerance.

The minima are observed grid minima from one trainer seed and one matched subset per domain. Varying the fraction changes both repetition and the web/domain allocation, so these curves do not isolate an intrinsic repetition tolerance of the corpus. MATH-500 and GSM8K were chosen after inspecting existing broad evaluations; MATH-500 was designated primary before either math evaluation ran. No benchmark decontamination audit or seed uncertainty is claimed. Keep the earlier common-Uncheatable result alongside this exploratory comparison.

## Reproduction and provenance

Run `uv run python -m experiments.domain_phase_mix.analyze_tpp10_wikipedia_math --require-complete` from the repository root. It verifies all raw receipts and shared scoring fields, then writes the complete point table, minima, neighboring gaps, opposite-choice penalties and plots under `results/12ea6861093bc224babb05709a17b4ce78082ffaf8e8035e9f7d154cdfc4ea51/`. Curves connect measured points; no surrogate fits are overlaid. The linear full-range plot preserves the large Wikipedia p100 failures; the logarithmic full-grid companion makes all points and both FineMath upturns visible. A separately labeled valley-detail view ends at p70 and refers to the full-range endpoints. All three rendered views were visually checked.

- Wikipedia specification: `12ea6861093bc224babb05709a17b4ce78082ffaf8e8035e9f7d154cdfc4ea51`.
- FineMath reference specification: `8d4f18d0a9dfbf4a065a09e621bf35f67e9547874f8b3392af0d3aaf57f938e8`.
- Iris parent: `/calvinxu/tpp10-wikipedia-math-eval`; child: `/calvinxu/tpp10-wikipedia-math-eval/wikipedia-math-proxies`.
- Raw Wikipedia receipts: `gs://marin-us-central1/experiments/tpp10_finemath_math/12ea6861093bc224babb05709a17b4ce78082ffaf8e8035e9f7d154cdfc4ea51/`.
- `spec.json`, `build_receipt.json`, `region_check.txt`, `bundle_preflight.json` and `submit.sh` preserve the reviewed scoring identity, checkpoint preflight and submission.

No further training or target evaluation is released. The four FineMath target cancellations remain in force. Manuscript and outline are unchanged; CC can propagate these measured results when the illustrative figure is selected.
