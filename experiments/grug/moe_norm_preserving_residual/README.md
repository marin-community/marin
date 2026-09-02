# Norm-preserving residual MoE

This variant copies the canonical July Baseline at commit `52d8a9eb8` and
changes only the two residual merges in each transformer layer. For layer
`l`, both the attention and MoE merge use

```text
sqrt(1 - beta_l / L) * residual + sqrt(beta_l / L) * hidden
beta_l = softplus(theta_l)
theta_l = 0 at initialization
```

`L` is the total transformer layer count. One scalar `theta_l` is shared by
the two merges in layer `l`. `beta_l` is capped just below `L` so the residual
coefficient stays real if training drives `theta_l` outside its expected
range. W&B logs the effective values under `train/residual/layer_<l>/beta`.

Gate 1 uses the exact July d512 and d768 recipes:

```bash
.venv/bin/iris --cluster=marin job run --no-wait --reserve v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe_norm_preserving_residual.launch_norm_preserving_residual
```

The experiment is tracked in [#8860](https://github.com/marin-community/marin/issues/8860).
