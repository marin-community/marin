import pickle
from collections import defaultdict
directory = "PhaseIII_1B"

def clean_loss_value(value):
    """Clean and format loss values for LaTeX table"""
    if value is None:
        return None
    elif value == 'NaN':
        return 'NaN'
    elif isinstance(value, (int, float)):
        if value > 10:  # Very large values
            return f'>10'
        else:
            return f'{value:.3f}'
    else:
        return str(value)


correct_name = {
    "adamw": "AdamW",
    "lion": "Lion",
    "mini": "Adam-Mini",
    "scion": "Scion",
    "cautious": "Cautious",
    "mars": "Mars",
    "nadamw": "NAdamW",
    "muon": "Muon",
    "soape": "Soap",
    "kron": "Kron",
}
def escape_latex(text: str) -> str:
    """Escape underscores for LaTeX"""
    return text.replace('_', '\\_')

def hyperparameter_to_latex(param_name: str, unmapped_list=None) -> str:
    """Convert hyperparameter names to LaTeX symbols, using Greek/math symbols where possible."""
    latex_symbols = {
        'beta1': r'$\beta_1$',
        'beta2': r'$\beta_2$',
        'learning_rate': r'$\eta$',
        'weight_decay': r'$\lambda$',
        'epsilon': r'$\epsilon$',
        'gamma': r'$\gamma$',
        'momentum': r'$\mathrm{\beta_{muon}}$',
        'decay': r'$\mathrm{Decay (WSD)}$',
        'warmup': r'$\mathrm{warmup}$',  # Not standard, but for completeness
        'train_batch_size': r'$\mathrm{BSZ}$',
        'partition_grads_into_blocks': r'$\mathrm{Blocking}$',
        'min_lr_ratio': r'$\eta_{min}$',
        'max_grad_norm': r'$\gradnorm$',
        'block_size': r'$\mathrm{block size}$',
        'preconditioner_lr': r'$\eta_{pc}$',
        'preconditioner_update_probability': r'$p_{pc}$',
        'update_prob_flat_start': r'$Step_{pc}$',
        'muon_epsilon': r'$\epsilon_{muon}$',
        'adam_lr': r'$\eta_{adam}$',
        'scion_epsilon': r'$\epsilon_{scion}$',
        'precondition_frequency': r'$f_{pc}$',
        'shampoo_beta': r'$\beta_{shampoo}$',
        'lr_schedule': r'$\mathrm{Schedule}$',
        'preconditioner_init_scale': r'$Init_{pc}$',
        'normalize_grads': r'$\mathrm{NormGrad}$',
    }
    if param_name in latex_symbols:
        return latex_symbols[param_name]
    else:
        if unmapped_list is not None:
            unmapped_list.add(param_name)
        return r'\texttt{' + escape_latex(param_name) + '}'

def format_config_name(config_key):
    """Format config key as a readable name"""
    if isinstance(config_key, tuple) and len(config_key) >= 2:
        model_size = config_key[0]
        data_size = config_key[1]
        return f"{model_size} on {data_size}x Chinchilla Data"
    else:
        return str(config_key)

def find_baseline_config(result_data, hyperparams):
    """Find the baseline configuration (typically the one with most common/default values)"""
    # Group hyperparameters by their values
    hyperparam_values = defaultdict(list)
    
    for key, loss_value in result_data.items():
        if isinstance(key, tuple) and len(key) == 2:
            param_name, param_value = key
            if param_name in hyperparams:
                hyperparam_values[param_name].append(param_value)
    
    # Find the most common value for each hyperparameter (baseline)
    baseline_config = {}
    for param_name in hyperparams:
        if param_name in hyperparam_values:
            # Use the first value as baseline (you might want to modify this logic)
            baseline_config[param_name] = min(hyperparam_values[param_name])
    
    return baseline_config

def get_baseline_loss(result_data, baseline_config):
    """Get the loss for the baseline configuration"""
    # Look for a configuration that matches all baseline values
    return 0

def process_adam_lr_data(result_data, baseline_config):
    """Process data to calculate adam_lr from learning_rate * muon_to_adam_lr or scion_to_signum_lr"""
    processed_data = {}
    
    # Find baseline learning_rate (the one used in the baseline configuration)
    baseline_lr = baseline_config['learning_rate']
    
    # Process each parameter
    for key, loss_value in result_data.items():
        if isinstance(key, tuple) and len(key) == 2:
            param_name, param_value = key
            
            if param_name in ['muon_to_adam_lr', 'scion_to_signum_lr']:
                # Calculate adam_lr using baseline learning_rate
                if baseline_lr is not None:
                    adam_lr_value = baseline_lr * param_value
                    processed_data[('adam_lr', adam_lr_value)] = loss_value
                else:
                    # If no baseline learning_rate found, keep original
                    processed_data[key] = loss_value
            else:
                # Keep other parameters as is
                processed_data[key] = loss_value
        else:
            processed_data[key] = loss_value

    
    return processed_data

def generate_config_table(optimizer_name, config_key, config_data, unmapped_list=None):
    """Generate LaTeX table for a specific config showing baseline and ablations"""
    
    if 'result' not in config_data or not config_data['result']:
        return None
    
    result_data = config_data['result']
    result_name = config_data['name']
    baseline_config = config_data['best_config']
    
    # Process data to calculate adam_lr
    result_data = process_adam_lr_data(result_data, baseline_config)
    if 'muon_to_adam_lr' in baseline_config:
        baseline_config['adam_lr'] = baseline_config['learning_rate'] * baseline_config['muon_to_adam_lr']
        baseline_config.pop('muon_to_adam_lr')
        for key in list(result_name.keys()):
            if type(key) == tuple and key[0] == 'muon_to_adam_lr':
                result_name[('adam_lr', key[1] * baseline_config['learning_rate'])] = result_name[key]
    elif 'scion_to_signum_lr' in baseline_config:
        baseline_config['adam_lr'] = baseline_config['learning_rate'] * baseline_config['scion_to_signum_lr']
        baseline_config.pop('scion_to_signum_lr')
        for key in list(result_name.keys()):
            if type(key) == tuple and key[0] == 'scion_to_signum_lr':
                result_name[('adam_lr', key[1] * baseline_config['learning_rate'])] = result_name[key]
    if 'nesterov' in baseline_config:
        baseline_config.pop('nesterov')
    
    
    
    # Extract unique hyperparameters
    hyperparams = set()
    hyperparam_data = defaultdict(dict)
    
    for key, loss_value in result_data.items():
        if isinstance(key, tuple) and len(key) == 2:
            param_name, param_value = key
            hyperparams.add(param_name)
            hyperparam_data[param_name][param_value] = loss_value
    

    
    hyperparams = sorted(set(baseline_config.keys()))
    
    # Find baseline configuration
    baseline_loss = result_data['Baseline']
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(f"\\begin{{table}}[H]")
    latex_lines.append(f"\\centering")
    latex_lines.append(f"\\caption{{Hyperparameter ablation for {correct_name[optimizer_name]} on {format_config_name(config_key)}}}")
    latex_lines.append(f"\\label{{tab:ablation_{correct_name[optimizer_name].lower()}_{format_config_name(config_key).replace(', ', '_').replace(' ', '_').lower()}}}")
    
    # Create table with hyperparameter columns + loss column + link column
    num_cols = len(hyperparams) + 2  # +1 for loss column, +1 for link column
    repeated_c = "c" * num_cols
    latex_lines.append(f"\\begin{{tabular}}{{{repeated_c}}}")
    latex_lines.append(f"\\toprule")
    
    # Header row with LaTeX symbols
    header = " & ".join([hyperparameter_to_latex(hp, unmapped_list) for hp in hyperparams]) + " & Loss & Link \\\\"
    latex_lines.append(header)
    latex_lines.append(f"\\midrule")
    
    # Baseline row
    baseline_row = []
    for param_name in hyperparams:
        if param_name in baseline_config:
            baseline_row.append(str(baseline_config[param_name]))
        else:
            baseline_row.append("N/A")
    baseline_row.append(clean_loss_value(baseline_loss))
    baseline_row.append(f"\\href{{https://wandb.ai/stanford-mercury/optimizer-scaling/runs/{result_name['Baseline']}}}{{0}}")
    
    latex_lines.append(" & ".join(baseline_row) + " \\\\")
    latex_lines.append(f"\\midrule")
    
    # Ablation rows - one for each hyperparameter variation
    idx = 0
    for param_name in hyperparams:
        if param_name in hyperparam_data:
            param_values = sorted(hyperparam_data[param_name].keys())
            baseline_value = baseline_config.get(param_name)
            
            # Create rows for non-baseline values of this hyperparameter
            for param_value in param_values:
                if param_value != baseline_value:
                    
                    row = []
                    for hp in hyperparams:
                        if hp == param_name:
                            row.append(str(param_value))
                        else:
                            row.append("--")  # Dash to indicate baseline
                    
                    # Add loss value
                    loss_value = hyperparam_data[param_name][param_value]
                    row.append(clean_loss_value(loss_value))
                    if row[-1] is None:
                        continue
                    idx += 1
                    # Add W&B link
                    key = (param_name, param_value)
                    if key in result_name:
                        row.append(f"\\href{{https://wandb.ai/stanford-mercury/optimizer-scaling/runs/{result_name[key]}}}{{{idx}}}")
                    else:
                        raise ValueError(f"Key {key} not found in result_name")
                    latex_lines.append(" & ".join(row) + " \\\\")
    
    latex_lines.append(f"\\bottomrule")
    latex_lines.append(f"\\end{{tabular}}")
    latex_lines.append(f"\\end{{table}}")
    latex_lines.append("")
    
    return "\n".join(latex_lines).replace("9.999999999999999e-26", "1e-25")

def main():
    # Load the data
    path = f"experiments/optimizer_sweep/Analysis/{directory}/1d_losses.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # Generate tables for each optimizer and config
    unmapped_hyperparams = set()
    
    for optimizer_name, optimizer_data in data.items():
        all_tables = []
        if optimizer_data:  # Skip empty optimizers
            # Generate table for each config
            for config_key, config_info in optimizer_data.items():
                table = generate_config_table(optimizer_name, config_key, config_info, unmapped_list=unmapped_hyperparams)
                if table:
                    all_tables.append((optimizer_name, config_key, table))
                    
                # Debug: print number of experiments left for this config
                if 'num_left' in config_info:
                    print(f"{optimizer_name} - {format_config_name(config_key)}: {config_info['num_left']} experiments left")
                print(config_info)
    
        # Write all tables to a file
        import os
        os.makedirs(f"experiments/optimizer_sweep/Analysis/{directory}/tex", exist_ok=True)
        with open(f"experiments/optimizer_sweep/Analysis/{directory}/tex/hyperparam_tables_{optimizer_name}.tex", "w") as f:        
            f.write("\subsection{Sweeping Results for " + correct_name[optimizer_name] + "}")
            for optimizer_name, config_key, table in all_tables:
                f.write(f"% {optimizer_name} - {format_config_name(config_key)}\n")
                f.write(table)
                f.write("\n")
            
            # f.write("\\end{document}\n")
        
        print("LaTeX tables generated in hyperparam_tables.tex")
        print(f"Generated {len(all_tables)} tables total")
        if unmapped_hyperparams:
            print("Unmappable hyperparameters (no Greek/math symbol):", sorted(unmapped_hyperparams))
    
    # Also print individual tables
    # for i, (optimizer_name, config_key, table) in enumerate(all_tables):
    #     print(f"\n{'='*80}")
    #     print(f"Table {i+1}: {optimizer_name} - {format_config_name(config_key)}")
    #     print(f"{'='*80}")
    #     print(table)

if __name__ == "__main__":
    main() 