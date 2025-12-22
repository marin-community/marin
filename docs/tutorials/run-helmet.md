# Running HELMET Evaluations with Marin

Marin can run the Princeton NLP **HELMET** long-context benchmark (https://github.com/princeton-nlp/HELMET) using **vLLM on TPU**.

HELMET requires downloading its data separately; Marin models this as a dedicated `ExecutorStep` so it can be cached and shared across runs.

## Quickstart

See `experiments/evals/run_helmet.py` for an end-to-end example.

```bash
uv run python experiments/evals/run_helmet.py --prefix <YOUR_PREFIX>
```

Key knobs:

- `HelmetConfig.use_chat_template` (**required**): set `True` for instruct/chat models and `False` for base models (matches HELMET’s own slurm scripts).
- `HELMET_PIPELINE_AUTOMATIC` / `HELMET_PIPELINE_SHORT` / `HELMET_PIPELINE_ALL`:
  - `AUTOMATIC`: full HELMET suite, no OpenAI judging
  - `SHORT`: `*_short.yaml` configs, no OpenAI judging
  - `ALL`: full suite + OpenAI judging (requires `OPENAI_API_KEY`)
- `HelmetConfig.evals_per_instance`: `1` (default) runs each HELMET config in its own TPU job; `"all"` runs everything in one job.
- `HelmetConfig.vllm_serve_args`: forwarded to `vllm serve` (useful for long-context models, e.g. `("--max-model-len", "131072")`).
- `helmet_steps(..., model_path=...)` can take a `str`, `InputName`, or `ExecutorStep`. If the resolved path is not already under `gs://$PREFIX/gcsfuse_mount/...`, Marin will add a staging step that copies it to `gcsfuse_mount/models/helmet-staged/<hash>/` for TPU-friendly access.

## About “missing” HELMET metrics

Some HELMET headline metrics (e.g. `gpt-4-score` / `gpt-4-f1` for parts of LongQA and Summarization) require separate model-based judging scripts in the upstream HELMET repo.

Marin runs those judging passes when you use `HELMET_PIPELINE_ALL`; otherwise, the report will include only automatically-computable metrics.
