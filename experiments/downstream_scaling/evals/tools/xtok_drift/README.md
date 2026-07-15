# xtok drift tool — end-to-end workflow

Scores the GSM8K joint-decode-avg-xtok advisor (Qwen3-4B-Base) on recorded
rollouts under its exact forced token history vs the canonical tokenization of
the same bytes, then browses the divergence. Design and metrics:
`.agents/projects/20260715_xtok_advisor_drift_viz_plan.md`.

The tool is `rollouts.py` (replay + step-path resolution), `scoring.py` (GPU
scoring CLI), and `app.py` (Gradio browser) in this directory; it runs from a
marin checkout containing them (`~/marin` below).

## Setup (once)

```bash
uv venv ~/.venvs/xtok-drift
uv pip install -p ~/.venvs/xtok-drift/bin/python -e ~/marin   # orchestration stack for --slug resolution
uv pip install -p ~/.venvs/xtok-drift/bin/python torch transformers gradio  # GPU torch; not in base deps
gcloud auth application-default login                         # GCS read access
cd ~/marin && ~/.venvs/xtok-drift/bin/python -c \
    "import experiments.downstream_scaling.evals.run_delphi_gsm8k_joint_decode_avg_xtok"  # env gate
```

If the GPU nodes cannot reach Hugging Face, download `Qwen/Qwen3-4B-Base`
(revision `906bfd4`) once and pass `--model /path/to/it` below.

## Score

```bash
sbatch --gres=gpu:1 --time=0:30:00 --chdir ~/marin --wrap "\
  ~/.venvs/xtok-drift/bin/python -m experiments.downstream_scaling.evals.tools.xtok_drift.scoring \
    --slug 1e22 --weights 0.4 --num-rollouts 5 --cache-dir ~/xtok_drift_cache"
```

The job log ends with the KL summary — the drift measurement. Inputs are
cached under `~/xtok_drift_cache/inputs/`, so reruns against the same slug pay
no egress. `--problems` / `--samples` select specific problems or sample
ranks; `--step-output` + `--prompts` bypass slug resolution.

## Browse

```bash
srun --gres=gpu:1 --chdir ~/marin --pty \
    ~/.venvs/xtok-drift/bin/python -m experiments.downstream_scaling.evals.tools.xtok_drift.app \
    --cache-dir ~/xtok_drift_cache --port 7860
# from your laptop, then open http://localhost:7860:
ssh -J <login-host> <node> -L 7860:localhost:7860
```

Only the probe tab uses the GPU; an rsynced cache dir browses on a laptop with
the same commands minus the allocation.
