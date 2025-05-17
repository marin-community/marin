import wandb

api = wandb.Api()
# Define your details
username = "stanford-mercury"
project = "optimizer-scaling"


# tag = "sweep-130m-10B-muonzf7964alr0.008-wd0.1-minlr0-warmup0-b10.8-b20-a9a044"
# tag = "sweep-130m-2B-nadamw96aba0lr0.008-wd0.1-minlr0-warmup2000-b10.95-2ac247"
tag = "sweep-1.2b-48B-nadamwcb7c09lr0.004-wd0.1-minlr0.0-warmup4000-b10-fdd79f"

run = api.run(f"{username}/{project}/{tag}")

print(run.summary)
benchmarks = ['lambada_openai', 'openbookqa', 'winogrande', 'piqa', 'boolq', 'wsc273', 'hellaswag_0shot', 'arc_challenge', 'arc_easy', 'copa']

def get_benchmark_acc(run):
    acc_dict = {}
    for benchmark in benchmarks:
        acc = run.summary.get(f'lm_eval/{benchmark}/acc', 0.0)
        acc_norm = run.summary.get(f'lm_eval/{benchmark}/acc_norm', 0.0)
        acc_dict[benchmark] = max(acc, acc_norm)
    print(acc_dict)
    return acc_dict

import pandas as pd

# df = pd.read_csv('optimizer_loss_eval.csv')

acc_dict = get_benchmark_acc(run)
for benchmark in benchmarks:
    print(f"{acc_dict[benchmark]},", end="")