This reproduces the lm-eval multi-host TPU bug using the TraitGym eval smoke test forced onto a v4-32 slice (multi-host) instead of the default v5p-8 (single-host).

Clone the `dna` branch and apply the resource override:
```bash
git clone --branch dna git@github.com:marin-community/marin.git && cd marin
```

The change in `experiments/dna/smoke_tests/train_traitgym_eval.py` overrides `SHORT_RUN_CONFIG_V1` to use `ResourceConfig.with_tpu("v4-32")` via `dataclasses.replace`. This forces the training + eval job onto a multi-host TPU slice where the lm-eval harness fails.

Submit to `us-central2` (adjust `.env` path as needed):
```bash
source /path/to/marin/.env && uv run lib/marin/src/marin/run/ray_run.py \
  --cluster us-central2 \
  --no_wait \
  --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
  --env_vars HUGGING_FACE_HUB_TOKEN ${HUGGING_FACE_HUB_TOKEN} \
  -- python experiments/dna/smoke_tests/train_traitgym_eval.py --prefix gs://marin-dna-us-central2
```

Monitor logs with `uv run scripts/ray/cluster.py --cluster us-central2 job-logs <JOB_ID> --tail 50`.
