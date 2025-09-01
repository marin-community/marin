valid_lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
valid_weight_decays = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
valid_epochs = [1, 2, 4, 8, 16, 32, 64]


def extract_neighbors(value, valid_values):
    index = valid_values.index(value)
    assert 0 <= index < len(valid_values), f"{value} is not in the valid range"
    lower_value = valid_values[index - 1] if index != 0 else value
    upper_value = valid_values[index + 1] if index != len(valid_values) - 1 else value
    return lower_value, upper_value


def get_bounding_box(base_train_steps, epochs, lr, weight_decay, model_name, use_wd=True):
    lower_epochs, upper_epochs = extract_neighbors(epochs, valid_epochs)
    lower_lr, upper_lr = extract_neighbors(lr, valid_lrs)
    if use_wd:
        lower_weight_decay, upper_weight_decay = extract_neighbors(weight_decay, valid_weight_decays)
    else:
        lower_weight_decay, upper_weight_decay = weight_decay, weight_decay

    return [
        (base_train_steps, epochs, lr, weight_decay, model_name),
        (base_train_steps, lower_epochs, lr, weight_decay, model_name),
        (base_train_steps, upper_epochs, lr, weight_decay, model_name),
        (base_train_steps, epochs, lower_lr, weight_decay, model_name),
        (base_train_steps, epochs, upper_lr, weight_decay, model_name),
        (base_train_steps, epochs, lr, lower_weight_decay, model_name),
        (base_train_steps, epochs, lr, upper_weight_decay, model_name),
    ]
