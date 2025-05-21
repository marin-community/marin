import hashlib

from marin.execution.executor import unwrap_versioned_value


def kron_train_config(prefix: str, config):
    """Format a config into a string that includes all relevant hyperparameter keys."""
    name = (
        f"lr{unwrap_versioned_value(config.optimizer_config.learning_rate)}-"
        f"wd{unwrap_versioned_value(config.optimizer_config.weight_decay)}-"
        f"b1{unwrap_versioned_value(config.optimizer_config.beta1)}-"
        f"plr{unwrap_versioned_value(config.optimizer_config.preconditioner_lr)}-"
        f"pis{unwrap_versioned_value(config.optimizer_config.preconditioner_init_scale)}-"
        f"gn{unwrap_versioned_value(config.optimizer_config.max_grad_norm)}-"
        f"norm{unwrap_versioned_value(config.optimizer_config.normalize_grads)}-"
        f"pgb{unwrap_versioned_value(config.optimizer_config.partition_grads_into_blocks)}-"
        f"bs{unwrap_versioned_value(config.optimizer_config.block_size)}-"
        f"pup{unwrap_versioned_value(config.optimizer_config.preconditioner_update_probability)}-"
        f"upfs{unwrap_versioned_value(config.optimizer_config.update_prob_flat_start)}-"
        f"wu{unwrap_versioned_value(config.optimizer_config.warmup)}-"
        f"steps{unwrap_versioned_value(config.num_train_steps)}"
        f"minlr{unwrap_versioned_value(config.optimizer_config.min_lr_ratio)}"
    )
    # Shorten final name with a hash prefix to avoid overly long identifiers
    first_hash = hashlib.md5(name.encode()).hexdigest()[:6]
    # Return a string of max length 64
    return (prefix + first_hash + name)[:64]
