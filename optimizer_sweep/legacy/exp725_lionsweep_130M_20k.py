from optimizer_sweep.template import template



if __name__ == "__main__":
    # round 1
    sweep_grids = {
        'learning_rate': [1e-3, 2e-3, 4e-3],
        'weight_decay': [0, 0.1, 0.3, 0.5, 0.7, 0.9],
        'warmup': [500, 1000, 2000, 4000],
        'beta1': [0.9, 0.95, 0.98],
        'beta2': [0.9, 0.95, 0.98],
        'max_grad_norm': [0, 1.0, 2.0]}

    baseline_config = {
        'learning_rate': 2e-3, 
        'weight_decay': 0.7,
        'min_lr_ratio': 0,
        'warmup': 2000,
        'beta1': 0.95,
        'beta2': 0.95,
        'max_grad_norm': 1.0
    }
    model_size = "130m"
    target_step = 20000
    template(model_size, target_step, 'lion', baseline_config, sweep_grids) 

