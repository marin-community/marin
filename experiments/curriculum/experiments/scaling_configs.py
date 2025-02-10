learning_rate_dict = {
    "150m": 3e-3,
    "300m": 3e-3,
    "600m_0.001": 1e-3,
    "600m_0.003": 3e-3,
    "1_9b": 3e-4,
    "1_9b_1024_0.001": 1e-3,
    "1_9b_1024_0.0003": 3e-4,
    "8b_1024": 3e-4,
}

chinchilla_steps = {
    "150m": 3000,
    "300m": 6000,
    "600m_0.001": 12000,
    "600m_0.003": 12000,
    "1_9b": 38000,
    "1_9b_1024_0.001": 38000,
    "1_9b_1024_0.0003": 38000,
    "8b_1024": 160000,
}

def version_tag(lr):
    return f"-lr{lr}" if lr != 3e-3 else ""

def correct_model_size(model_size):
    if model_size == "600m_0.003" or model_size == "600m_0.001":
        return "600m"
    if model_size == "1_9b_1024_0.001" or model_size == "1_9b_1024_0.0003":
        return "1_9b_1024"
    return model_size
