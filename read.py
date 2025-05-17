import pickle
import copy
import pandas as pd
import wandb
with open('losses_all.pkl', 'rb') as f:
    losses_all = pickle.load(f)



# ensure every (model_size, data_size) for 'soap' has a best_config
for model_size, data_size in losses_all['soap']:
    if 'best_config' not in losses_all['soap'][(model_size, data_size)]:
        losses_all['soap'][(model_size, data_size)] = copy.deepcopy(
            losses_all['soape'][(model_size, data_size)]
        )

# patch a bit
api = wandb.Api()
# Define your details
username = "stanford-mercury"
project = "optimizer-scaling"

tag = "sweep-300m-24B-soapei11857elr0.008-wd0.1-minlr0-warmup2000-b10.9-29e88d"

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
losses_all['soap'][('300m', '24B')]['lm_eval_acc'] = get_benchmark_acc(run)
losses_all['soap'][('300m', '24B')]['paloma_loss'] = run.history(keys=['eval/paloma/c4_en/loss'])
correct_name = {
    'adamw': 'AdamW', 'lion': 'Lion', 'mini': 'Adam-Mini',
    'scion': 'Scion', 'cautious': 'Cautious', 'mars': 'Mars',
    'nadamw': 'NAdamW', 'muon': 'Muon', 'soap': 'Soap', 'kron': 'Kron',
}

with open('best_hyper_params.tex', 'w') as f:
    for optimizer, grid in losses_all.items():
        if optimizer in ('soape', 'soapb'):
            continue

        # 1) Gather all best_configs into a flat list
        records = []
        for (model_size, data_size), info in grid.items():
            bc = info['best_config'].copy()
            bc['model_size'] = model_size
            bc['data_size']  = data_size
            records.append(bc)

        df = pd.DataFrame(records)

        # 2) Loop over each model_size and pivot
        for model_size in sorted(df['model_size'].unique()):
            df_ms = df[df['model_size'] == model_size].copy()
            # pivot so rows are hyper-parameters, columns are data_size
            df_pivot = (
                df_ms
                .set_index('data_size')            # index on data_size
                .drop(columns=['model_size'])      # no longer needed
                .T                                  # transpose
            )

            # format row index (hyperparam names) nicely
            df_pivot.index = [
                ' '.join(tok.upper() for tok in name.split('_'))
                for name in df_pivot.index
            ]
            df_pivot.index.name = 'HYPER-PARAMETER'
            df_pivot.columns.name = 'DATA SIZE'

            caption = (
                f"Best Hyper-parameters for {correct_name[optimizer]}, "
                f"Model Size = {model_size}"
            )
            label = (
                f"tab:best-hyper-params-"
                f"{correct_name[optimizer].lower()}-{model_size.upper()}"
            )

            f.write(
                df_pivot.to_latex(
                    index=True,
                    caption=caption,
                    label=label,
                    float_format="%.2e"
                )
            )
            f.write('\n\n')


with open('eval_performance.tex', 'w') as f:
    for optimizer, grid in losses_all.items():
        if optimizer in ('soape', 'soapb'):
            continue

        # 1) Gather all best_configs into a flat list
        records = []
        for (model_size, data_size), info in grid.items():
            if 'lm_eval_acc' not in info:
                print(optimizer, model_size, data_size)
                continue
            bc = info['lm_eval_acc'].copy()
            bc['final C4 loss'] = list(info["paloma_loss"]["eval/paloma/c4_en/loss"])[-1]
            bc['model_size'] = model_size
            bc['data_size']  = data_size
            records.append(bc)

        df = pd.DataFrame(records)

        # 2) Loop over each model_size and pivot
        for model_size in sorted(df['model_size'].unique()):
            df_ms = df[df['model_size'] == model_size].copy()
            # pivot so rows are hyper-parameters, columns are data_size
            df_pivot = (
                df_ms
                .set_index('data_size')            # index on data_size
                .drop(columns=['model_size'])      # no longer needed
                .T                                  # transpose
            )

            # format row index (hyperparam names) nicely
            df_pivot.index = [
                ' '.join(tok.upper() for tok in name.split('_'))
                for name in df_pivot.index
            ]
            df_pivot.index.name = 'PERFORMANCE METRIC'
            df_pivot.columns.name = 'DATA SIZE'

            caption = (
                f"Evaluation Performance for {correct_name[optimizer]}, "
                f"Model Size = {model_size}"
            )
            label = (
                f"tab:eval-performance-"
                f"{correct_name[optimizer].lower()}-{model_size.upper()}"
            )

            f.write(
                df_pivot.to_latex(
                    index=True,
                    caption=caption,
                    label=label,
                    float_format="%.3f"
                )
            )
            f.write('\n\n')


with open('eval_performance.tex', 'w') as f:
    for optimizer, grid in losses_all.items():
        if optimizer in ('soape', 'soapb'):
            continue

        # 1) Gather all best_configs into a flat list
        records = []
        for (model_size, data_size), info in grid.items():
            if 'lm_eval_acc' not in info:
                print(optimizer, model_size, data_size)
                continue
            bc = info['lm_eval_acc'].copy()
            bc['final C4 loss'] = list(info["paloma_loss"]["eval/paloma/c4_en/loss"])[-1]
            bc['model_size'] = model_size
            bc['data_size']  = data_size
            records.append(bc)

        df = pd.DataFrame(records)

        # 2) Loop over each model_size and pivot
        for model_size in sorted(df['model_size'].unique()):
            df_ms = df[df['model_size'] == model_size].copy()
            # pivot so rows are hyper-parameters, columns are data_size
            df_pivot = (
                df_ms
                .set_index('data_size')            # index on data_size
                .drop(columns=['model_size'])      # no longer needed
                .T                                  # transpose
            )

            # format row index (hyperparam names) nicely
            df_pivot.index = [
                ' '.join(tok.upper() for tok in name.split('_'))
                for name in df_pivot.index
            ]
            df_pivot.index.name = 'PERFORMANCE METRIC'
            df_pivot.columns.name = 'DATA SIZE'

            caption = (
                f"Evaluation Performance for {correct_name[optimizer]}, "
                f"Model Size = {model_size}"
            )
            label = (
                f"tab:eval-performance-"
                f"{correct_name[optimizer].lower()}-{model_size.upper()}"
            )

            f.write(
                df_pivot.to_latex(
                    index=True,
                    caption=caption,
                    label=label,
                    float_format="%.3f"
                )
            )
            f.write('\n\n')

cnt = 0
with open('loss_ablation.tex', 'w') as f:
    from matplotlib import pyplot as plt
    for optimizer, grid in sorted(losses_all.items()):
        if optimizer in ('soape', 'soapb'):
            continue
        f.write("\\subsubsection{Optimizer: " + correct_name[optimizer] + "}\n")
        for (model_size, data_size), info in grid.items():
            baseline_loss = list(info['paloma_loss']['eval/paloma/c4_en/loss'])[-1]
            best_config = info['best_config']
            losses = info['losses']
            hyperparameters = set([x[0] for x in losses])
            # print(f"Processing {optimizer} {model_size} {data_size}")
            if len(hyperparameters) == 0:
                print("Missing hyperparameters for ", optimizer, model_size, data_size)
                # print(losses_all['soap'][('300m', '6B')])
            else:
                f.write("\\paragraph{Model Size: " + model_size + " Data Size: " + data_size + "}\n")
            for hyper in hyperparameters:
                grid = sorted([x[1] for x in losses if x[0] == hyper] + [best_config[hyper]])
                corresponding_losses = []
                for value in grid:
                    if (hyper, value) in losses:
                        corresponding_losses.append(losses[(hyper, value)])
                    else:
                        corresponding_losses.append(baseline_loss)
                for loss in corresponding_losses:
                    if type(loss) is not float or loss > 4 :
                        corresponding_losses[corresponding_losses.index(loss)] = 'Diverged'

                hyper = ' '.join(tok.upper() for tok in hyper.split('_'))
                df2 = pd.DataFrame(
                    [corresponding_losses],
                    index=['loss'],
                    columns=[f"{v}" for v in grid]   # optional: use the grid values as column names
                )

                f.write(df2.to_latex(
                    index=False,
                    caption=f"Loss Ablation for {optimizer} with respect to {hyper}",
                    label=f"tab:loss-ablation-{optimizer.lower()}-{hyper}-{model_size}-{data_size}",
                    float_format="%.3f",
                ).replace("\\begin{tabular}", "\\centering \\begin{tabular}".replace("{table}", "{table}[H]")))
                cnt += 1

                    