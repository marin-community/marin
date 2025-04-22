import json
from optimizer_sweep.utils_simp import create_configs, check_baseline_run, calculate_data_tag
import hashlib
key_name = "learning_rate"
def num_left_eq_0(baseline_config, sweep_grids, target_data, tags):
    """Modified num_left function that takes required parameters"""
    target_steps, config_in_dict = create_configs(baseline_config, sweep_grids, target_data=target_data)
    num_left = 0
    losses = {}
    for config in config_in_dict:
        exist, loss = check_baseline_run(config, tags, return_loss = True)
        if(not exist):
            return False, losses
        losses[config[key_name]] = loss
    return True, losses
def get_cache_key(optimizer, model_size, data_size, target_chinchilla):
    """Generate a unique cache key for the query"""
    key_str = f"{optimizer}_{model_size}_{data_size}_{target_chinchilla}"
    return hashlib.md5(key_str.encode()).hexdigest()

cache = json.load(open("wandb_cache.json", "r"))

model_and_data_size = [('130m', '2B', 1), ('130m', '5B', 2), ('130m', '10B', 4), 
                        ('130m', '21B', 8), ('300m', '6B', 1), ('520m', '10B', 1)]
# model_and_data_size = [('300m', '6B', 1)]
optimizers = ["mini", "lion", "sophia", "adamw", "nadamw", "mars", "cautious",  "soap","muon", "scion", "soape", "soapb", "kron"]


actual_list = {}

for optimizer in optimizers:
    for model_size, data_size, target_chinchilla in model_and_data_size:
        cache_key = get_cache_key(optimizer, model_size, data_size, target_chinchilla)
        if cache_key in cache:
            actual_list[(optimizer, model_size, target_chinchilla)] = cache[cache_key]


optimizer = "adamw"
with open('sweep_configurations.json', 'r') as f:
    sweep_configs = json.load(f)
optimizer_configs = sweep_configs.get(optimizer.lower(), {})

# plot the loss vs lr with x axis as lr and y axis as loss for all model_size, target_chinchilla
import matplotlib.pyplot as plt


fig, ax = plt.subplots()


for model_size, data_size, target_chinchilla in model_and_data_size:
    model_configs = optimizer_configs.get(model_size, {})
    sweep_config = model_configs.get(str(target_chinchilla), [{}])[0]
    sweep_grids = sweep_config.get('sweep_grids', None)
    approximate_best_config_list = actual_list[(optimizer, model_size, target_chinchilla)]["approximate_best_config_list"]
    for approximate_best_config in approximate_best_config_list:
        target_data, data_size = calculate_data_tag(model_size, target_chinchilla)
        tags = (model_size, data_size, optimizer)
        sweep_grids = {key_name: sweep_grids[key_name]}
        is_baseline_run, losses = num_left_eq_0(approximate_best_config, sweep_grids, target_data, tags)
        if is_baseline_run:
            break
    losses_keys = sorted(losses.keys())
    losses_values = [losses[key] for key in losses_keys]
    ax.plot(losses_keys, losses_values, label=f'{model_size} {target_chinchilla}')
    

ax.legend()
plt.ylim(3, 4)
plt.savefig(f'figs/{optimizer}_loss_vs_{key_name}.png')
    







