# Quantized Training

!!! warning

    FP8 and Int8 training in Haliax is currently experimental and may change in the future.

Haliax supports training with FP8 and int8. This is useful for training on hardware that is optimized for FP8 or Int8,
such as the H100 (fp8) or A100s (int8) and TPU v5 and newer (int8).

## TL;DR

Using FP8 with Haliax is actually pretty straightforward. To enable FP8, do this:

```python
import haliax.quantization as haxq
# setup
module = haxq.quantize_linear_layers(module, haxq.QuantizationConfig(fp8=True))

# if using optax. This saves a tiny amount of memory so you can skip it if you want
_, trainable_module = haxq.partition_for_grad_overwrite(module)
opt_state = opt.initial_state(trainable_module)

# train step
grads = eqx.filter_grad(loss_fn)(module, data)
overwrite, grads = haxq.partition_for_grad_overwrite(grads)
updates, opt_state = opt.update(grads, opt_state, params=module)  # or however you update your optimizer
module = haxq.apply_updates(module, updates, overwrite)
```

And train your model like normal.

Similarly, you can use `Int8` by setting `Int8=True` in the `QuantizationConfig` object.



## What is FP8?

FP8 refers to 8-bit floating point numbers. FP8 is a massively reduced precision compared to the 32-bit floating point numbers
or 16-bit floating point numbers that are typically used in deep learning: there are only 256 possible values in FP8, compared to
the (almost) 2^32 in 32-bit and 2^16 in 16-bit. However, FP8 is still useful for training deep learning models, especially on
hardware that is optimized for FP8. In particular, it can massively accelerate training on hardware that is optimized for FP8:
H100 has 2x FP8 FLOPS compared to FP16 FLOPS and almost 60x(!) compared to F32 FLOPS.

The FP8 in Haliax is currently designed to optimize throughput on FP8-enabled devices (currently H100) rather
than to save memory. In particular, Haliax's FP8 support is not designed to quantize a model to FP8 for deployment,
though this shouldn't be that hard to add for models that were trained using this functionality.
We would be happy to accept contributions to add this functionality,
and are happy to work with you to do so. In particular, adding this for models trained using Haliax's FP8 should be easy.

See this [FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) for more information on FP8.

## What is Int8?

Int8 refers to 8-bit integers. Int8 has the same number of bits as FP8, but the interpretation is different: instead of
exponentially spaced numbers, Int8 has linearly spaced numbers.

In Haliax, we support Int8 training through Google's [AQT](https://github.com/google/aqt) library. AQT (for
"Accurate Quantization Training") is a library that allows you to train models with quantization-aware training (QAT).

## How to use FP8 or Int8 in Haliax

To use quantized training in Haliax, you need to do three things:

* Enable FP8 (or int8) for the layers you want
* Modify your training step to be compatible

Each of these is just a couple of lines of code.

```python
import haliax as hax
import equinox as eqx
import jax

In = hax.Axis("In", 32)
Mid = hax.Axis("Mid", 128)
Out = hax.Axis("Out", 16)
Hidden = hax.Axis("Hidden", 64)


class MyModule(eqx.Module):
    up_proj: hax.nn.Linear
    down_proj: hax.nn.Linear

    @staticmethod
    def init(*, key):
        super().__init__()
        k_up, k_down = jax.random.split(key)
        return MyModule(
            up_proj=hax.nn.Linear.init(In, Mid, key=k_up),
            down_proj=hax.nn.Linear.init(Mid, Out, key=k_down),
        )

    def __call__(self, x):
        x = self.up_proj(x)
        x = hax.nn.relu(x)
        x = self.down_proj(x)
        return x

module = MyModule.init(key=jax.random.PRNGKey(0))

# Enable FP8
module = hax.quantization.quantize_linear_layers(module, QuantizationConfig(fp8=True))

# Enable FP8 for a specific layer
from haliax.quantization import QuantizationConfig

config = QuantizationConfig(targets=["up_proj"], fp8=True)
module = hax.quantization.quantize_linear_layers(module, config)

# Train step
grads = eqx.filter_grad(loss_fn)(module, data)
overwrite, grads = haxq.partition_for_grad_overwrite(grads)
updates, opt_state = opt.update(grads, opt_state, params=module)  # or however you update your optimizer
module = hax.quantization.apply_updates(module, updates, grads)
```

That's it! Just a few lines of code to enable FP8. The `quantize_linear_layers` function will transform your module to use
quantization-aware training for linear layers (or a subset if you want), and the combo of [haliax.quantization.partition_for_grad_overwrite][] and [haliax.quantization.apply_updates][] function will apply the updates to the module
in a way that is compatible with FP8.

## How FP8 works

For an overview of the FP8, see the [FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).
You don't need to understand it though. Haliax's FP8 integration is more or less plug and play, as shown above.
The implementation of FP8 in Haliax is more or less a straightforward port (including some copy and paste) of the
[FP8 implementation in Flax](https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py).

FP8 in JAX (as well as INT8) is typically implemented using "`dot_general` injection", where you pass
a custom implementation of `dot_general` to functions and modules like [haliax.dot][] and [haliax.nn.Linear][].
The `dot_general` for FP8 is implemented by scaling
the inputs, projecting the inputs to FP8, performing the computation in FP8, and then
scaling the result back to the original precision.
The subtle part of FP8 is that the scaling is a parameter that is trained based on a history of the inputs to the layer
(as well as gradients coming in from backward). This means that the FP8 `dot_general` needs to maintain state.
In Equinox, this means that the `dot_general` is actually a `Module` that packages together the state and the
computation. (Unlike [equinox.nn.StatefulLayer][] which returns a state object you pass back into the module, the FP8 `dot_general`
module hijacks the gradient computation to update its state. This is necessary because the FP8 scaling factors
depend on the gradients.)

The way this happens is by "hijacking" the gradient computation. When you call `eqx.filter_grad(loss_fn)(module, data)`,
you will get the gradient computation as normal, but you'll also get the updated state of the FP8 `dot_general` module.
This updated state needs to directly replace the state in the module (rather than be used for a gradient step), which is
why you need to use the [haliax.quantization.partition_for_grad_overwrite][]

The FP8 `dot_general` module is implemented in [haliax.quantization.Fp8DotGeneralOp][]. It's actually not that complicated:

1) It holds a scaling factor and history of maximum values for each of (lhs, rhs, output) and updates them based on the
gradients.
2) When invoked, it scales the inputs, projects them to FP8, performs the computation, and scales the result back to the
original precision.  It remembers the maximum absolute value for each of the inputs.
3) For the gradients, it scales the gradients, projects them to FP8, does the backward computation,
and scales the gradients back to the original precision. It remembers the maximum absolute value for the incoming
gradient and stores it in the gradient.

## How Int8 works

Int8 is in principle the same, though the details differ. AQT is a much more flexible library than the FP8 implementation,
because it can be a bit more finicky. We use AQT directly, and we recommend you look at the
[AQT documentation](https://github.com/google/aqt?tab=readme-ov-file#how-aqt-works-internally) for more
information on how it works.

# API Reference

## Functions

::: haliax.quantization.quantize_linear_layers
::: haliax.quantization.partition_for_grad_overwrite
::: haliax.quantization.apply_updates


## Interfaces
::: haliax.quantization.DotGeneralOp
::: haliax.quantization.OverwriteWithGradient

## Modules


::: haliax.quantization.DefaultDotGeneralOp
::: haliax.quantization.Fp8DotGeneralOp
::: haliax.quantization.Int8DotGeneralOp

## Configuration

::: haliax.quantization.QuantizationConfig
