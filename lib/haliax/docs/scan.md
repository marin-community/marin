# Scan and Fold

A common pattern in deep learning is to apply a sequence of layers to an input, feeding the output from one
layer to the next. In JAX, this is often done with [jax.lax.scan][].

As the docs say, scan does an operation sort of like this in Python:

```python
def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)
```

Haliax provides two versions of this pattern: [haliax.fold][] and [haliax.scan][]. [haliax.scan][] works much like JAX's scan,
except it is curried and it works with NamedArrays. [haliax.fold][] is a more restricted version of scan that is easier to
use if you don't need the full generality of scan. (It works with functions that only return `carry`, not `carry, output`.)

## [haliax.scan][haliax.scan]

Unlike JAX's scan, Haliax's scan is curried - it takes the function and configuration first, then the initial carry and scan arguments as a separate call: `scan(f, axis)(init, xs)`.

### Key Features
* Works with named axes using [haliax.NamedArray][]
* Supports gradient checkpointing for memory efficiency, including several advanced checkpointing policies
* Integrates with [equinox.Module][] for building neural networks

### Basic Example

Here's a practical example of using [haliax.scan][] to sum values along an axis while keeping track of intermediates:

```python
Time = Axis("Time", 100)
Features = Axis("Features", 16)

# Create time series data
data = hax.random.normal(PRNGKey(0), (Time, Features))

def running_stats(state, x):
    count, mean, min_val, max_val = state
    count += 1
    # this is a common pattern to improve the robustness of the mean calculation
    delta = x - mean
    mean = mean + delta / count
    min_val = hax.minimum(min_val, x)
    max_val = hax.maximum(max_val, x)

    return (count, mean, min_val, max_val), mean


# Initialize state: (count, mean, min, max)
init_state = (
    0.0,
    hax.zeros((Features,)),
    hax.full((Features,), float('inf')),
    hax.full((Features,), float('-inf'))
)

final_state, running_means = hax.scan(running_stats, Time)(init_state, data)
```

Note that:

* `scan` is curried: `scan(f, axis)(init, xs)`
* `running_stats` returns a tuple of `(carry, output)`, which is why we have two return values from `scan`
* the running_means will have shape `(Time, Features)`, with the mean at each time step
* the final_state will have the same shape as the initial state


### Using `scan` with no inputs
You can also use scan without any inputs if you want:

```python
Time = Axis("Time", 100)
Features = Axis("Features", 16)

def simulate_brownian_motion(state, _):
    return state + hax.random.normal(PRNGKey(0), Features), state

init_state = hax.zeros((Features,))

final_state, path = hax.scan(simulate_brownian_motion, Time)(init_state, None)
```

More commonly, you might use this for an RNN or Transformer model. (See [haliax.nn.Stacked][].)

## [haliax.fold][haliax.fold]

[haliax.fold][] is a simpler version of [haliax.scan][] that is easier to use when you don't need the full generality of `scan`.
Specifically, `fold` is for functions that only return a `carry`, not a `carry, output`.

Morally, `fold` is like this Python code:

```python
def fold(f, init, xs):
  carry = init
  for x in xs:
    carry = f(carry, x)
  return carry
```

### Basic Example

Same example, but we only care about the final state:

```python
Time = Axis("Time", 100)
Features = Axis("Features", 16)

# Create time series data
data = hax.random.normal(PRNGKey(0), (Time, Features))

def running_stats(state, x):
    count, mean, min_val, max_val = state
    count += 1
    # this is a common pattern to improve the robustness of the mean calculation
    delta = x - mean
    mean = mean + delta / count
    min_val = hax.minimum(min_val, x)
    max_val = hax.maximum(max_val, x)

    return (count, mean, min_val, max_val)


# Initialize state: (count, mean, min, max)
init_state = (
    0.0,
    hax.zeros((Features,)),
    hax.full((Features,), float('inf')),
    hax.full((Features,), float('-inf'))
)

final_state = hax.fold(running_stats, Time)(init_state, data)
```

## [haliax.map][haliax.map]

[haliax.map][] is a convenience function that applies a function to each element of an axis. It is similar
to [jax.lax.map][] but works with NamedArrays, providing a similar interface to [haliax.scan][] and [haliax.fold][].

```python

Time = Axis("Time", 100)

data = hax.random.normal(PRNGKey(0), (Time,))

def my_fn(x):
    return x + 1

result = hax.map(my_fn, Time)(data)
```

You should generally prefer to use [haliax.vmap][] instead of [haliax.map][], but it's there if you need it.
(It uses less memory than [haliax.vmap][] but is slower.)


## Gradient Checkpointing / Rematerialization

Both [haliax.scan][] and [haliax.fold][] support gradient checkpointing, which can be useful for deep models.
Typically, you'd use this as part of [haliax.nn.Stacked][] or [haliax.nn.BlockSeq][] but you can also use it directly.

Gradient checkpointing is a technique for reducing memory usage during backpropagation by recomputing some
intermediate values during the backward pass. This can be useful when you have a deep model with many layers.

###  TL;DR Guidance

Here is some guidance on when to use gradient checkpointing:

* Use `remat=False` if you need to reduce computation and have lots of memory. This is the default in [haliax.scan][].
* Use `remat=True` for most models. It's usually good enough. This is the default in [haliax.nn.Stacked][].
* Use `remat="nested"` if you need to reduce memory usage.
* Use `save_block_internals` sparingly, but it is your best tool for trading increased memory usage for reduced computation
if you need something between `remat=True` and `remat=False`.
* Use `save_carries="offload"` if you need to reduce memory usage at the cost of recomputation. This is a new feature
in JAX and doesn't seem to reliably work yet.


### Simple Checkpointing

In the simplest case, you can enable a usually-good-enough checkpointing policy by passing `remat=True`:

```python
final_state = hax.fold(running_stats, Time, remat=True)(init_state, data)
```

("remat" is short for "rematerialization", which is another term for gradient checkpointing.)

This will preserve the intermediate "carries" and other inputs the fold function needs, while rematerializing
(i.e. recomputing) the internal state of each block (i.e. call to the running_stats function) as needed during
backpropagation.


### Nested Scan

Simple checkpointing requires `O(N)` memory where $N$ is the number of blocks. A nested scan lets you reduce
this to `O(sqrt(N))` memory, at the cost of a bit more computation. You can enable this by passing `remat="nested"`:

```python
final_state = hax.fold(running_stats, Time, remat="nested")(init_state, data)
```

This will break the scan into a double loop, where the outer loop has `sqrt(N)` blocks and the inner loop has
`sqrt(N)` blocks (with appropriate rounding).

Functionally, it does something like:

```
outer_size = int(sqrt(N))  # ensuring outer_size divides N
blocks = haliax.rearrange("block -> (outer inner)", blocks, outer=outer_size)

state = init_state
for o in range(outer_size):
    inner_blocks = blocks["outer", o]

    for i in range(inner_size):
        state = f(state, inner_blocks["inner", i])

    # not real jax
    state = save_for_backward(state)
```

where we save only the carries from the outer loop, and fully rematerialize the inner loop.

If `C` is the amount of memory needed for the carry, and `N` is the number of blocks, then the memory usage
of the nested scan is `2 * C * sqrt(N)`. In addition, you need enough memory to do backward in one block.

In practice, nested scan is about 20% slower than simple checkpointing (for Transformers), but uses much less memory.

#### Advanced: customizing the number of blocks

You can also customize the number of blocks in the outer loop by using a policy:

```python
policy = ScanCheckpointPolicy(nested=4)  # 4 outer blocks
```

Note that by itself this doesn't help you at all except potentially requiring more memory. You can potentially
combine it with other policy options to make things faster though.


### Custom Checkpointing Policies

If you need more control over the checkpointing policy, you can pass a [haliax.nn.ScanCheckpointPolicy][] object to
the `scan` or `fold` call:

```python
policy = ScanCheckpointPolicy(
   save_carries=True,  # default
   save_inputs=True,  # default
   save_block_internals=False,  # default
)
```

### Saving Block-Internal Values

"`internals`" refers to the internal computation of the block. If you set `save_block_internals=True`, then
all internals of every block will be saved. This can be expensive and mostly negates the benefits of checkpointing.

Instead you can choose which internals to save by passing a list of strings to `save_block_internals`:

```python
def my_complex_fn(state, x):
    y = x + state
    y = hax.sin(y) + x
    y = hax.tree_checkpoint_name(y, "y")
    y = hax.cos(y) + x
    y = hax.tree_checkpoint_name(y, "z")
    return y

policy = ScanCheckpointPolicy(save_carries=True, save_block_internals=["y"])

final_state = hax.fold(my_complex_fn, Time, remat=policy)(init_state, data)

```

With this policy, the output of the `sin` function will be saved during the forward pass.

This will save an extra `sin` computation in the backward pass, adding $`O(N * Pos * Hidden)`$ memory usage,
which is double that required by the default policy, but it reduces the amount of recomputation needed.
(It's probably not worth it in this case.)

### Offloading Checkpointed Values

Both `save_carries` and `save_inputs` can either be a boolean or the string "offload". If "offload", then the
checkpointed values will be offloaded to the host during the forward pass, and reloaded during the backward pass.

In addition, you can offload block internals by passing a list of strings to `offload_block_internals`:

```
policy = ScanCheckpointPolicy(save_carries=True, save_block_internals=["y"], offload_block_internals=["z"])
```


### Summary of String and Boolean Aliases

* `remat=True` is the same as `remat=ScanCheckpointPolicy(save_carries=True, save_inputs=True)`
* `remat="full"` is the same as `remat=True`
* `remat=False` is the same as `remat=ScanCheckpointPolicy(disable=True)`
* `remat="nested"` is the same as `remat=ScanCheckpointPolicy(nested=True)`
* `remat="offload"` is the same as `remat=ScanCheckpointPolicy(save_carries="offload", save_inputs="offload")`
* `remat="save_all"` is the same as `remat=ScanCheckpointPolicy(save_carries=True, save_inputs=True, save_block_internals=True)`,
which should be the same as not using remat at all...


### Memory and Computation Tradeoffs

Let `N` be the number of blocks, `C` be the memory needed for the carry, and `I` be the internal memory needed
for each block. Let F be the amount of computation needed for each block. Constants are added for a bit more precision
but are not exact. This is assuming that backward requires ~twice the flops as forward, which is roughly right for
Transformers.

| Policy           | Memory Usage             | Computation    |
|------------------|--------------------------|----------------|
| `remat=False`    | `O(N * C + N * I)`       | `O(3 * N * F)` |
| `remat=True`     | `O(N * C + I)`           | `O(4 * N * F)` |
| `remat="nested"` | `O(2 * sqrt(N) * C + I)` | `O(5 * N * F)` |


(Which shows why nested scan is about 20% slower than simple checkpointing. The math says 25% but it's more like 20% in
practice.) Any nested remat will require `5 * N * F` computation, which is about 25% more than simple remat.


## Module Stacks

A core pattern for larger models in JAX is the "scan-over-layers" pattern, where you have a sequence of layers
that get stacked together, and you use [jax.lax.scan][] or [haliax.fold][] or [haliax.scan][] to apply them to a
sequence of inputs. In Haliax, layers are represented as [equinox.nn.Module][]s, and the [haliax.nn.Stacked][] module
provides a way to create a sequence of layers that can be applied to a sequence of inputs that implements the
scan-over-layers pattern.

### Stacked

[haliax.nn.Stacked][] lets you apply a layer sequentially to an input, scanning over a "Layers" axis. For instance,
a Transformer might use a Stacked for its Transformer blocks:


```python
class TransformerBlock(eqx.Module):

    def __init__(self, config: TransformerConfig, layer_index, *, key):
        attn_key, mlp_key = jax.random.split(key)
        self.attention = Attention.init(config, key=attn_key)
        self.mlp = MLP.init(config, key=mlp_key)
        self.ln1 = LayerNorm.init(config.Hidden)
        self.ln2 = LayerNorm.init(config.Hidden)
        self.layer_index = layer_index

    def __call__(self, x):
        y = self.attention(self.ln1(x))
        x = x + y
        y = self.mlp(self.ln2(x))
        return x + y

class Transformer(eqx.Module):
    def __init__(self, config: TransformerConfig):
        self.blocks = Stacked.init(Layers, TransformerBlock)(
            config,  # static configuration
            scale=hax.arange(Layers),  # dynamic configuration. Each layer gets a scalar scale value [0, 1, 2, ...]
            key=jax.random.split(key, Layers),  # dynamic configuration. Each layer gets a different key
        )

    def __call__(self, x: NamedArray) -> NamedArray:
        # morally the equivalent of:
        # for block in self.blocks:
        #     x = block(x)
        # Except that it works better with JAX compile times.

        return self.blocks.fold(x)
```

You can think of [haliax.nn.Stacked][] as an analog to PyTorch's
[torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html), except that
every layer in the sequence must have exactly the same shape and configuration.

Internally, a Stacked is a single copy of the module, except that every NamedArray inside that module
has a Block axis prepended (as though they were stacked with [haliax.stack][]). Similarly, every JAX array
inside the module has its first axis prepended with an axis of the same size as the Block axis, as though
they were stacked with [jax.numpy.stack][].

When you call the Stacked, it scans over the Block axis, applying the module to each element of the Block.

#### Creating a Stacked

To create a Stacked, we provide `Stacked.init`, which takes a "Layers" [haliax.Axis][] and another Module as
well as args and kwargs for that module. The Layer is the axis that the Stacked will scan over, and the `args`
and `kwargs` are implicitly vmapped over the Layers.

For instance, to create a stack of GPT2 blocks, you might do:

```python
import jax.random

blocks = Stacked.init(Layers, Gpt2Block)(
    config,  # static configuration
    scale=hax.arange(Layers),  # dynamic configuration. Each layer gets a scalar scale value [0, 1, 2, ...]
    key=jax.random.split(key, Layers.size),  # dynamic configuration. Each layer gets a different key
)
```

Any NamedArray passed to the Stacked init will have its Layers axis (if present) vmapped over. Any
JAX array will have its first axis vmapped over.

#### Apply Blocks in Parallel with `vmap`

Sometimes you may want to apply each block independently, without feeding the
output of one block into the next.  `Stacked.vmap` does exactly that: it uses
[haliax.vmap][] to broadcast the initial value to every block and evaluates
them in parallel, returning the stack of outputs.

```python
y = stacked.vmap(x)
```


#### Fold Blocks vs Scan Blocks

The Stacked module provides two ways to apply the layers: `fold` and `scan`.  A fold is the moral equivalent of this for loop:

```python
for block in self.blocks:
    x = block(x)
```

while a scan is the moral equivalent of this for loop:

```python
out = []
for block in self.blocks:
    x, y = block(x)
    out.append(y)

return x, stack(out)
```

Blocks can be coded to either support fold or scan, but not both.
A "fold Block" should have the signature `def __call__(self, x: Carry) -> Carry`,
while a "scan Block" should have the signature `def __call__(self, x: Carry) -> Tuple[Carry, Output]`.

(See also [jax.lax.scan][], [haliax.fold][], and [haliax.scan][].)

#### Requirements for Stacked Blocks

As we said above, the Stacked module requires that all the layers have the same shape and configuration.

A further constraint is that the elements of the stack must have the same Python control flow. This is the usual
constraint imposed on jit-compiled functions in JAX. All control flow must use `jax.lax` primitives like
[jax.lax.cond][], [jax.lax.while_loop][], and [jax.lax.scan][]. You can't use Python control flow like `if` or `for`
except for static control flow that is the same for all elements of the stack.

### BlockSeq and BlockFoldable

We also provide a way to create a sequence of layers that can be applied to a sequence of inputs that implements the
same interface as [haliax.nn.Stacked][], but with a different implementation. This is the [haliax.nn.BlockSeq][] module.
BlockSeq implements those for loops directly, rather than using [haliax.fold][] or [haliax.scan][].

[haliax.nn.scan.BlockFoldable][] is an interface that both [haliax.nn.Stacked][] and [haliax.nn.BlockSeq][] implement. It
exposes the usual ``fold`` and ``scan`` methods as well as helpers ``fold_via`` and ``scan_via`` which return
callables that perform the respective operations using a custom block function.

## API

::: haliax.fold
::: haliax.scan
::: haliax.map
::: haliax.ScanCheckpointPolicy

### Modules
::: haliax.nn.Stacked
::: haliax.nn.BlockSeq
::: haliax.nn.scan.BlockFoldable
