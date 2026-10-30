# SFT reproduction policy scores

The first plot compares the qk175-skew8 base with reproduction Step 222. A second set follows the reproduction through Steps 471, 1024, and 2048. The final set adds the original October 5 Datakit SFT model. The October 5 checkpoint is a separately trained reference; dashed segments distinguish it from the reproduction trajectory.

Scores come from the [completed evaluation tracker](/Users/benfeuer/Documents/experiments/complete/eval-policy-09-25/TRACKER.md), SHA-256 `fe26792f72cbd9092f0f33fa416213a80b4b4303f89472d743aa65e71f93d3b1`. [scores.csv](scores.csv) freezes all 156 displayed-model/benchmark cells, including each tracker result URI. The plots use the tracker's three-decimal scores. Step2048 SWE-bench Verified was early sealed at 453/500 policy-valid trials; its [frozen receipt](s3://marin-us-east-02a/marin/evals/20261029-100020-open-athena-Grug-67B-A2B-Datakit-SFT-Repro-262K-Step2048-2026.10.29-50e07a18efd14e93910d24b82877c66f-swebench-verified-e544f9a3bae75619948a68992bca6d26/results/frozen-seals/swebench-verified-result-20261030-093828.json) records the full-precision score.

Each stage has its own filter: a benchmark appears when its absolute score change between that stage's endpoints is **greater than 5.0 percentage points**. Seventeen of 26 benchmarks qualify for [base→Step 222](base_to_step222.csv). Eight qualify for [Step 222→Step 2048](step222_to_step2048.csv), with four gains and four declines. Fifteen qualify for [Step 222→full SFT](step222_to_full.csv), including two declines. The CSVs list each stage's qualifying benchmarks and endpoint changes.

- [Base to Step 222](base_to_step222.png) ([SVG](base_to_step222.svg), [PDF](base_to_step222.pdf))
- [Step 222 to Step 2048: all qualifying benchmarks](step222_to_step2048_all.png) ([SVG](step222_to_step2048_all.svg), [PDF](step222_to_step2048_all.pdf))
- [Step 222 to Step 2048: gains](step222_to_step2048_gains.png) ([SVG](step222_to_step2048_gains.svg), [PDF](step222_to_step2048_gains.pdf))
- [Step 222 to Step 2048: declines](step222_to_step2048_declines.png) ([SVG](step222_to_step2048_declines.svg), [PDF](step222_to_step2048_declines.pdf))
- [Step 222 to full SFT: all qualifying benchmarks](step222_to_full_all.png) ([SVG](step222_to_full_all.svg), [PDF](step222_to_full_all.pdf))
- [Step 222 to full SFT: eight largest changes](step222_to_full_largest.png) ([SVG](step222_to_full_largest.svg), [PDF](step222_to_full_largest.pdf))
- [Step 222 to full SFT: seven other qualifying changes](step222_to_full_other.png) ([SVG](step222_to_full_other.svg), [PDF](step222_to_full_other.pdf))

Regenerate the plots from the frozen CSV with Python 3.13 and Matplotlib 3.11.2:

```bash
cd /Users/benfeuer/Documents/experiments/active/sft-pipe-repro/artifacts/figures
uv run --no-project --with matplotlib==3.11.2 python plot_sft_repro.py
```

To re-extract scores from the tracker before plotting, add `--tracker /Users/benfeuer/Documents/experiments/complete/eval-policy-09-25/TRACKER.md`. That rewrites `scores.csv` and `source.json`; it may produce different figures if the tracker has changed.
