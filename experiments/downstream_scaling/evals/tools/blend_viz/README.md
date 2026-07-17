# A/B blend visualizer

`blend_viz` inspects the same-tokenizer HumanEval rollouts produced by
`run_delphi_humaneval_joint_decode_avg_xtok_llama.py`. Scoring teacher-forces
the recorded token IDs through the Delphi decoder and Llama advisor, then
writes one JSON/NPZ pair per rollout. The app browses those artifacts and can
query the same two models on an editable prompt and generated prefix.

The scoring process and the live probe require enough memory to keep both
models resident. CUDA forwards use bfloat16; probability calculations use
float32. Rollout browsing alone does not run either model.

## Setup (once)

Run all commands from the repo root.

```bash
uv venv .venv-blend-viz
uv pip install -p .venv-blend-viz/bin/python -e .
uv pip install -p .venv-blend-viz/bin/python torch transformers gradio
gcloud auth application-default login
export BLEND_VIZ_CACHE=<scratch or shared-FS path>
.venv-blend-viz/bin/python -c \
  "import experiments.downstream_scaling.evals.run_delphi_humaneval_joint_decode_avg_xtok_llama"
```

Put `BLEND_VIZ_CACHE` in your shell profile. If the GPU nodes cannot reach
Hugging Face, download the Delphi checkpoint and
`meta-llama/Llama-3.1-8B` revision `d04e592` once, then pass their local paths
with `--decoder-model` and `--advisor-model`.

## Score rollouts

Resolve the completed HumanEval sweep by Delphi slug:

```bash
sbatch --gres=gpu:1 --time=0:30:00 --wrap "\
  .venv-blend-viz/bin/python -m experiments.downstream_scaling.evals.tools.blend_viz.scoring \
    --slug 3e18 \
    --weights 0.0 \
    --num-rollouts 2"
```

To use local or already-resolved artifacts, provide all three step outputs and
the decoder checkpoint:

```bash
sbatch --gres=gpu:1 --time=0:30:00 --wrap "\
  .venv-blend-viz/bin/python -m experiments.downstream_scaling.evals.tools.blend_viz.scoring \
    --step-output /path/to/completions-step \
    --prompts /path/to/prompts-step \
    --grades /path/to/grade-step \
    --decoder-model /path/to/delphi-checkpoint \
    --weights 0.0 0.5 1.0 \
    --samples 0 1 \
    --grade-filter pass"
```

Pass `--advisor-model` or `--decoder-model` to replace the Hugging Face model
with a local checkpoint. Each scoring run removes old top-level JSON/NPZ
results from the selected cache but preserves downloaded inputs under
`inputs/`.

Each artifact has one row for every recorded token decision plus one terminal
EOS or cut row. It stores exact full-vocabulary `KL(A || B)`, entropy and
committed-token statistics, along with each side's top 256 IDs and normalized
log probabilities. Cross-gathered log probabilities make the displayed
per-token A/B probability differences exact.

## Run the app

```bash
GRADIO_SHARE=True srun --gres=gpu:1 --pty .venv-blend-viz/bin/python \
  -m experiments.downstream_scaling.evals.tools.blend_viz.app
```

Gradio prints the public URL after startup. The app has no authentication;
anyone with the URL can browse the cache and submit live probes.

The rollout tab is cache-backed. The probe tab loads both models on first use
and keeps them resident. Loading a selected rollout row uses the recorded token
IDs before that decision, so the initial probe matches the cached row exactly.
Editing the generated prefix switches to canonical text encoding. Prefixes that
end inside a UTF-8 codepoint are displayed with byte escapes while their
recorded IDs remain active.

For cache-only browsing, run
`.venv-blend-viz/bin/python -m experiments.downstream_scaling.evals.tools.blend_viz.app`
without a GPU allocation. Opening the probe still requires a GPU that can hold
both models.
