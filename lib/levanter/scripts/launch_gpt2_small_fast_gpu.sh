# 4 gpus, 3090s
# TODO: maybe move to the a100s or a6000s?
srun --account=nlp --cpus-per-task=2 --gres=gpu:3090:4 --job-name=dlwh-job-1681253 --mem=16G --open-mode=append --partition=jag-standard --time=14-0 \
    bash infra/run-slurm.sh python src/levanter/main/train_lm.py \
    --config_path config/llama_small_fast.yaml \
    --trainer.checkpointer.save_interval 30m \
    --trainer.per_device_parallelism -1 $*
