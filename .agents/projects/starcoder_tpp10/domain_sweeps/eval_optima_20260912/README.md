# Proxy minima across recorded evaluations

Wikipedia and FineMath-3+ select different mixtures on six of the eight recorded component evaluations. PALOMA Programming Languages provides a common objective with two positive, interior minima: Wikipedia at 30% (4.7684 epochs), FineMath at 50% (7.9379 epochs). Both proxies still minimize the Uncheatable mean at 30%.

The table uses both complete proxy grids from the audited 06:02 UTC snapshot on 12 September 2026. Each proxy is a 16.6M-parameter model trained at ten tokens per total parameter, with a 10.49M-token pool of the varied domain and the fixed web background specified in the [survey setup](../plots_20260912/README.md). Epochs count passes through the varied-domain pool. Grid minima are chosen from 0, 5, 10, 20, 30, 50, 70 and 100% domain weight. BPB means bits per byte; lower is better. The Uncheatable mean equally weights the seven component evaluations listed below, excluding PALOMA.

| Evaluation | Wikipedia proxy minimum | FineMath proxy minimum |
| --- | --- | --- |
| Uncheatable mean | 30%, 4.77 epochs | 30%, 4.77 epochs |
| PALOMA Programming Languages | 30%, 4.77 epochs | 50%, 7.94 epochs |
| Wikipedia English | 50%, 7.94 epochs | 0%, 0 epochs |
| GitHub Python | 50%, 7.94 epochs | 50%, 7.94 epochs |
| GitHub C++ | 30%, 4.77 epochs | 50%, 7.94 epochs |
| BBC News | 5%, 0.79 epochs | 0%, 0 epochs |
| arXiv physics | 30%, 4.77 epochs | 30%, 4.77 epochs |
| arXiv computer science | 5%, 0.79 epochs | 20%, 3.16 epochs |
| AO3 English | 20%, 3.16 epochs | 5%, 0.79 epochs |

On PALOMA, applying FineMath's selected fraction to the Wikipedia sweep (50% Wikipedia instead of 30%) costs 0.025175 BPB. Applying Wikipedia's selected fraction to the FineMath sweep (30% FineMath instead of 50%) costs 0.007624 BPB. The second-best Wikipedia point is 20%, worse by 0.016533 BPB. At the target scale, with 301.2M parameters and 3.012B training tokens, Wikipedia also selects 30%; FineMath's target curve is incomplete in this snapshot.

The other separations require caution. FineMath's C++ minimum at 50% improves on 30% by only 0.001349 BPB. Its arXiv-CS minimum at 20% improves on 10% by 0.000781 BPB. AO3's Wikipedia minimum at 20% improves on 30% by 0.001411 BPB. These single-seed differences do not establish distinct continuous optima or statistical significance. Wikipedia-English and BBC minima at zero FineMath weight show that adding FineMath does not improve those metrics on this grid; they are not positive-repetition turnover estimates.

For domain-specific evaluations, Wikipedia has a held-out Wikipedia-English metric, selecting 50% or 7.94 epochs on the proxy. No math-specific evaluation was recorded for these checkpoints. arXiv physics is an adjacent domain, not a direct FineMath evaluation. A comparison using each domain's own metric needs evaluation of the saved checkpoints on an independent math set; it does not require retraining.

This is an exploratory comparison across recorded metrics. Retain the unchanged Uncheatable result alongside it. The plotted minima include the 100% endpoints, while the displayed range stops at 70% to reveal the valleys. [Figure](proxy_eval_differences.png), [complete table with runner-up gaps and penalties](proxy_eval_optima.csv), [source snapshot](source.json).

Reproduce from the repository root:

```bash
uv run python -m experiments.domain_phase_mix.analyze_tpp10_domain_eval_optima \
  --snapshot .agents/projects/starcoder_tpp10/domain_sweeps/plots_20260912 \
  --output .agents/projects/starcoder_tpp10/domain_sweeps/eval_optima_20260912
```
