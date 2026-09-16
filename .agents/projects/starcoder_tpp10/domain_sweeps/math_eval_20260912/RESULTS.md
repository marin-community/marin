# FineMath proxy math likelihood results

Both math evaluations select the 70% FineMath mixture, at 11.098 realized epochs. The 50% and 100% neighbors are worse. The same checkpoints minimize Uncheatable at 4.768 epochs, so the high-epoch preference is evaluation-dependent.

| FineMath fraction | Realized epochs | MATH-500 perplexity | GSM8K perplexity | Uncheatable BPB |
|---:|---:|---:|---:|---:|
| 0% | 0.000 | 25.0401 | 19.5368 | 1.62048 |
| 5% | 0.788 | 13.7057 | 16.6431 | 1.54333 |
| 10% | 1.576 | 11.4709 | 14.8448 | 1.52324 |
| 20% | 3.162 | 9.6672 | 14.5677 | 1.49881 |
| 30% | 4.768 | 8.8616 | 12.7086 | 1.49387 |
| 50% | 7.938 | 7.8880 | 12.0467 | 1.51184 |
| 70% | 11.098 | 7.5340 | 11.4399 | 1.55748 |
| 100% | 15.825 | 8.1328 | 12.9085 | 2.00461 |

MATH-500 is primary; GSM8K is secondary. Perplexity is the exponential of reference-solution token NLL conditioned on the problem. These scores measure solution likelihood, not generated-answer accuracy. Original GSM8K solution markup is retained. The full populations contain 500 and 1,319 problems, with 112,697 and 173,832 scored tokens respectively; none was truncated.

These are observed coarse-grid minima from one trained checkpoint at each mixture. Both neighboring gaps are substantive relative to numerical restore variation, but no trainer-seed uncertainty has been measured. No benchmark decontamination audit of the FineMath subset is claimed.

All eight checkpoint/control checks passed: maximum PALOMA token-loss discrepancy 1.19209e-6, below the unchanged 5e-5 tolerance. Retry parent and worker both succeeded. An independent raw-receipt check reproduced every local CSV value. The rendered three-panel plot has been visually checked. The first failed attempt produced no accepted results and is retained under attempt1/.

## Next decision

The user subsequently approved evaluating the existing Wikipedia proxies under this same math protocol. That evaluation is complete: Wikipedia minimizes MATH-500 at 4.768 epochs and GSM8K at 3.162, while FineMath minimizes both at 11.098. See [the common-evaluation comparison](../wikipedia_math_eval_20260912/RESULTS.md). Wikipedia still prefers 7.938 epochs on its own Wikipedia-English BPB, and both domains' common Uncheatable minima coincide at 4.77.

If more proxy-only screening is agreed, discuss Stack-Edu Python first, arXiv second, and Nemotron Math Textbooks as a distinct math-content reserve. CC reviewed the implementation, corrected evidence and final math results; see CC_DISPOSITION.md and CANDIDATES.md. No new training or target evaluation has been submitted.

Results: results/8d4f18d0a9dfbf4a065a09e621bf35f67e9547874f8b3392af0d3aaf57f938e8/{receipt.json,points.csv,summary.json,math_likelihood.png,math_likelihood.pdf}.
