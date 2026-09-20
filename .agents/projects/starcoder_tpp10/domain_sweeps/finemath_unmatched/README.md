# FineMath-3+ unmatched proxy (no simulated epoching), 19-20 September 2026

Seven 16.6M proxy runs (p = 5, 10, 20, 30, 50, 70, 100 percent; p = 0 is the shared web-only proxy) trained
on the full 190,316,544-token FineMath parent pool with no downsampling, so FineMath exposure stays below
0.87 epochs. Plan `plan.json` (sha256 7827602b…), release `release.json`, script `submit.sh`, Iris parent
`/calvinxu/tpp10-finemath-unmatched-sweep` (submitted 23:03 PDT, succeeded 23:50 PDT; the first launch of the
`<domain>_unmatched` stage of `launch_tpp10_domain_sweeps.py`, released at the canary's first checkpoint).
`results/measurements.csv` carries the Uncheatable components and Paloma code; the Uncheatable macro
minimizes at p = 50 (1.4839; 1.4858 at p = 30, 1.4942 at p = 70).

GSM8K and MATH-500 are scored separately by `evaluate_tpp10_finemath_math.py --plan plan.json --arm unmatched`
(spec `math_spec.json`, sha256 65b19cf3…; the spec builder gained `--plan`/`--arm` for this). Coordinator
`/calvinxu/tpp10-finemath-unmatched-math-eval` (submitted 23:55 PDT via `submit_math_eval.sh`) launches one
v5p-8 worker over the eight checkpoints; results land under
`gs://marin-us-central1/experiments/tpp10_finemath_math/<spec_sha256>/<run_name>.json`.

Math results (scored 00:15 PDT 20 September, coordinator succeeded): the unmatched GSM8K curve keeps falling to
p = 100 (1.525 BPB at 0.87 epochs; 1.542 at p = 70, 1.613 at p = 50), so the no-simulated-epoching pick is the
boundary, as in Figure 3; the matched proxy minimizes at p = 70 (11.1 epochs). MATH-500 behaves the same way.
Token losses were converted to BPB with the ratio fixed by the shared p = 0 checkpoint (identical in both
evaluations); `results/measurements.csv` carries `gsm8k_bpb` and `math500_bpb`. Read on the target GSM8K curve,
the unmatched pick (p = 100, 15.8 epochs) costs +3.3% against 0 for the simulated-epoching pick; FLAN's unmatched
pick (p = 20) costs +4.0% on SciQ.
