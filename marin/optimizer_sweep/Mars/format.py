import hashlib

from marin.execution.executor import unwrap_versioned_value


def mars_train_config(prefix: str, config):
    name = (
        f"lr{unwrap_versioned_value(config.optimizer_config.learning_rate)}-"
        f"wd{unwrap_versioned_value(config.optimizer_config.weight_decay)}-"
        f"minlr{unwrap_versioned_value(config.optimizer_config.min_lr_ratio)}-"
        f"warmup{unwrap_versioned_value(config.optimizer_config.warmup)}-"
        f"b1{unwrap_versioned_value(config.optimizer_config.beta1)}-"
        f"b2{unwrap_versioned_value(config.optimizer_config.beta2)}-"
        f"gn{unwrap_versioned_value(config.optimizer_config.max_grad_norm)}-"
        f"steps{unwrap_versioned_value(config.num_train_steps)}"
        f"eps{unwrap_versioned_value(config.optimizer_config.epsilon)}-"
        f"gamma{unwrap_versioned_value(config.optimizer_config.gamma)}"
    )
    first_hash = hashlib.md5(name.encode()).hexdigest()[:6]
    return (prefix + first_hash + name)[:64]
