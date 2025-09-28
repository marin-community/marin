# RL Wishlist

# Refactor and make testing easier

There's no reason we should be able to run things like environment sampling on their own without involving the rollout worker.
We should lift out:

* loss functions -> rl_loss.py
* inference serving -> inference_server.py
* environment sampling -> already somewhat factored out but we can do better

I'd like to be able to write a simple synchronous training loop. That means we should be able to:

ref_model = load_reference_model()
pol_model = load_policy_model()
env = load_env("math")

```python
for i in range(1000):
    batch = []
    for _ in num_groups:
        group = env.sample()
        group.compute_reward()
        batch.append(group)

```
