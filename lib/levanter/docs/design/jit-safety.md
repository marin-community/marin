# JIT Safety Rules

Any method inside an `equinox.Module`, any function decorated with `jax.jit` or one of its variants (e.g. `eqx.filter_jit`, `haliax.named_jit`), and any helpers they call must follow these rules.

## Rules

1. **No data-dependent Python control flow** inside jitted code.
2. **No dynamic shapes or dynamic lengths** when indexing.
3. Use `debug.print` to inspect values (not Python `print`).
4. Use jit-safe variants like `jnp.where` or `hax.where` when branching on data.
5. Do not call `jax.default_backend()` or `jax.devices()` at module import time; resolve backend/device lazily inside runtime functions.

## Examples

```python
# BAD — data-dependent control flow
@jax.jit
def f(x):
    if x > 0:  # Python bool from traced value
        return x
    return -x

# GOOD — jit-safe branching
@jax.jit
def f(x):
    return jnp.where(x > 0, x, -x)
```
