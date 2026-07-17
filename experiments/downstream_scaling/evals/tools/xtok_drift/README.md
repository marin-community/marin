# xtok drift tool — end-to-end workflow

Scores the GSM8K joint-decode-avg-xtok advisor (Qwen3-4B-Base) on recorded
rollouts under its exact forced token history vs the canonical tokenization of
the same bytes, then browses the divergence. Design and metrics:
`.agents/projects/20260715_xtok_advisor_drift_viz_plan.md`.

The tool is `rollouts.py` (replay + step-path resolution), `scoring.py` (GPU
scoring CLI), and `app.py` (Gradio browser) in this directory. All commands
run from the repo root. `XTOK_DRIFT_CACHE` names the cache directory —
scored rollouts plus downloaded inputs, a few hundred MB per slug;
`--cache-dir` overrides it per invocation.

## Setup (once)

```bash
uv venv .venv-xtok
uv pip install -p .venv-xtok/bin/python -e .   # orchestration stack for --slug resolution
uv pip install -p .venv-xtok/bin/python torch transformers gradio  # GPU torch; not in base deps
gcloud auth application-default login          # GCS read access
export XTOK_DRIFT_CACHE=<scratch or shared-FS path>   # put in your shell profile
.venv-xtok/bin/python -c "import experiments.downstream_scaling.evals.run_delphi_gsm8k_joint_decode_avg_xtok"  # env gate
```

If the GPU nodes cannot reach Hugging Face, download `Qwen/Qwen3-4B-Base`
(revision `906bfd4`) once and pass `--model /path/to/it` below.

## Score

```bash
sbatch --gres=gpu:1 --time=0:30:00 --wrap "\
  .venv-xtok/bin/python -m experiments.downstream_scaling.evals.tools.xtok_drift.scoring \
    --slug 1e22 --weights 0.4 --num-rollouts 5"
```

The job log ends with the KL summary — the drift measurement. Inputs are
cached under `$XTOK_DRIFT_CACHE/inputs/`, so reruns against the same slug pay
no egress. Each scoring run replaces the previous run's results; the app
always shows the latest run. `--problems` / `--samples` select specific
problems or sample ranks; `--only-misaligned` screens without the GPU and
scores only rollouts whose segmentations diverge, printing the misaligned
fraction; `--step-output` + `--prompts` bypass slug resolution.

## Browse

```bash
srun --gres=gpu:1 --pty .venv-xtok/bin/python \
    -m experiments.downstream_scaling.evals.tools.xtok_drift.app --port 7860
# from your laptop, then open http://localhost:7860:
ssh -J <login-host> <node> -L 7860:localhost:7860
```

Only the probe tab uses the GPU; an rsynced cache dir browses on a laptop
with the same commands minus the allocation.
