"""Fetch LR sweep data from wandb for v10/v11/v12 groups."""
import json
import wandb

api = wandb.Api()
results = []

for group in ['isoflop-moe-v10', 'isoflop-moe-v11', 'isoflop-moe-v12', 'isoflop-moe-v12-retry']:
    runs = api.runs('marin-community/dial_moe', filters={'group': group})
    for r in runs:
        # Include finished AND crashed runs that have bpb (some crashed at final step)
        bpb = r.summary.get('eval/paloma/c4_en/bpb')
        if bpb is None:
            continue
        dim = None
        tok_ratio = None
        adam_lr = None
        for p in r.name.split('-'):
            if p.startswith('d') and p[1:].isdigit():
                dim = int(p[1:])
            if p.endswith('x') and p[0] == 't':
                try:
                    tok_ratio = float(p[1:-1])
                except ValueError:
                    pass
            if p.startswith('adam'):
                try:
                    adam_lr = float(p[4:])
                except ValueError:
                    pass
        trainer = r.config.get('trainer', {}).get('trainer', {})
        bs = trainer.get('train_batch_size')
        steps = trainer.get('num_train_steps')
        results.append({
            'name': r.name, 'group': group, 'state': r.state, 'dim': dim,
            'tok_ratio': tok_ratio, 'adam_lr': adam_lr, 'bpb': bpb,
            'batch_size': bs, 'steps': steps,
            'last_step': r.summary.get('_step'),
        })

print(f"Fetched {len(results)} runs with bpb")
for g in ['isoflop-moe-v10', 'isoflop-moe-v11', 'isoflop-moe-v12']:
    finished = sum(1 for r in results if r['group'] == g and r['state'] == 'finished')
    crashed = sum(1 for r in results if r['group'] == g and r['state'] == 'crashed')
    print(f"  {g}: {finished} finished, {crashed} crashed-with-bpb")

with open('experiments/grug/moe_apr2/new_plots/all_lr_sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to new_plots/all_lr_sweep_results.json")
