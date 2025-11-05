# Launches the "gpt_small_fast" model on a TPU node

uv run python infra/launch.py -e JAX_TRACEBACK_FILTERING off \
   --foreground --tpu_name $(whoami)-levanter-test-8 --zone us-central2-b --tpu_type v4-8 --preemptible -- \
    uv run pytest tests/ -m '"not ray and not slow"' $*
