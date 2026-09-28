# Wikipedia benchmark candidates

AO3 English gives the clearest early improvement among the saved English-prose evaluations. Its observed minimum is at 3.16 epochs for the proxy and 1.58 for the target. It evaluates fiction, so a figure using it should describe transfer to English prose. Wikipedia English is the only directly on-domain evaluation; its minima are later, at 7.94 and 11.10 epochs.

BBC News has earlier minima (0.79 and 1.58 epochs), but its initial BPB improvement is only 0.43% for the proxy and 0.20% for the target. Its near-zero slope makes it a weaker visual example. AO3 improves by 11.28% and 11.67%, although neighboring grid points near its minimum are close (within 0.23%).

| Saved benchmark view | Proxy minimum (epochs) | Target minimum (epochs) |
| --- | ---: | ---: |
| Wikipedia English | 7.94 | 11.10 |
| BBC News | 0.79 | 1.58 |
| AO3 English | 3.16 | 1.58 |
| Uncheatable aggregate | 4.77 | 4.77 |
| arXiv Physics | 4.77 | 0.79 |
| arXiv Computer Science | 0.79 | 0.79 |
| GitHub Python | 7.94 | 7.94 |
| GitHub C++ | 4.77 | 11.10 |
| PALOMA Programming Languages | 4.77 | 4.77 |
| MATH-500 | 4.77 | 7.94 |
| GSM8K | 3.16 | 3.16 |

The PDF shows every saved benchmark: seven Uncheatable components, their established equal mean, PALOMA Programming Languages, MATH-500 and GSM8K. All 22 curves contain all eight measured fractions, totaling 176 plotted values. Logging aliases and token-loss transformations of an existing score are not counted as additional benchmarks. No new composite was constructed.

The first two columns give absolute values at each scale over their full linear range. The third column divides each curve by its own measured minimum and subtracts one, expressed as a percentage; its upper limit is 5%, with larger values clipped. This preserves the measured minimum and makes local shape visible. It does not normalize statistical uncertainty or make percentage BPB and percentage perplexity equivalent measures. No fit or smoother is used.

The previously described held-out Wikipedia evaluation is the same Uncheatable Wikipedia-English component. Its corpus provenance differs from the training source, but there is no article-level decontamination audit. No separate Wikitext, Dolma Wikipedia holdout or factual-QA accuracy is saved for these checkpoints.

For the proposed common-MATH-500 panel, all four tracks are complete. FineMath target loss still decreases to the highest sampled point (15.83 epochs), so that endpoint is the lowest observed value rather than an established interior optimum. These exploratory benchmark comparisons use one trainer seed and one matched subset. The mixture sweep changes both repetition and the amount of web data.

Sources and SHA-256 hashes are in provenance.json; the exact values and minima are in points.csv and minima.csv. Reproduce locally with `uv run plot_candidates.py` from this directory. No training or evaluation was launched, and the manuscript is unchanged.
