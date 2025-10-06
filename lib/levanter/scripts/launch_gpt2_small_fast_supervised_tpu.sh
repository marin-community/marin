# Launches the "gpt_small_fast" model on a TPU node

python infra/launch.py --foreground --tpu_name $(whoami)-levanter-itest-32 --zone us-central2-b --tpu_type v4-32 --preemptible -- \
    python -m levanter.main.train_lm \
    --config_path config/llama_small_fast.yaml \
    --trainer.checkpointer.base_path gs://levanter-checkpoints/gpt-itest/ --trainer.checkpointer.save_interval 30m $*
