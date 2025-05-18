import hashlib

from marin.execution.executor import unwrap_versioned_value


def cautious_train_config(prefix: str, config):
    name = (
        f"lr{unwrap_versioned_value(config.learning_rate)}-"
        f"wd{unwrap_versioned_value(config.weight_decay)}-"
        f"minlr{unwrap_versioned_value(config.min_lr_ratio)}-"
        f"warmup{unwrap_versioned_value(config.warmup)}-"
        f"b1{unwrap_versioned_value(config.beta1)}-"
        f"b2{unwrap_versioned_value(config.beta2)}-"
        f"gn{unwrap_versioned_value(config.max_grad_norm)}-"
        f"steps{unwrap_versioned_value(config.num_train_steps)}"
        f"eps{unwrap_versioned_value(config.epsilon)}-"
        f"cautious"
    )
    first_hash = hashlib.md5(name.encode()).hexdigest()[:6]
    return (prefix + first_hash + name)[:64]
