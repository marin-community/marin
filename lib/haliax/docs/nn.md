# Neural Networks


## Modules

Haliax provides a small number of neural network modules that are compatible with Equinox, though
they naturally all use [haliax.NamedArray][]. (We welcome PRs for more modules! Nothing too exotic though.)

The most interesting of these modules is [haliax.nn.Stacked][], which allows you to create homogeneous "stacks"
of the same module (e.g. transformer blocks), which is a common pattern in deep learning.

### Linear

::: haliax.nn.Embedding
::: haliax.nn.Linear

### Dropout
::: haliax.nn.Dropout

### Normalization

::: haliax.nn.normalization.LayerNormBase
::: haliax.nn.LayerNorm
::: haliax.nn.RmsNorm

### Meta

::: haliax.nn.MLP

### Stacked

See the full documentation of [Stacked](scan.md#stacked).

### Convolution

Unlike other frameworks, Haliax doesn't distinguish between 1D, 2D, and 3D, and general convolutions. Instead, we have
a single [haliax.nn.Conv][] module that can be used for all of these, depending on the number of axes
provided. Similarly, for transposed convolutions, we have [haliax.nn.ConvTranspose][].

::: haliax.nn.Conv
::: haliax.nn.ConvTranspose

### Pooling

As with convolutions, we don't distinguish between 1D, 2D, and 3D pooling, and instead have a single
pooling operation for each of the kinds of reductions:

::: haliax.nn.max_pool
::: haliax.nn.mean_pool
::: haliax.nn.min_pool

## Attention

We don't provide an explicit attention module, but we do provide an attention function and several related functions:

:::haliax.nn.attention.dot_product_attention
:::haliax.nn.attention.dot_product_attention_weights

### Masks
::: haliax.nn.attention.causal_mask
::: haliax.nn.attention.prefix_lm_mask
::: haliax.nn.attention.combine_masks_and
::: haliax.nn.attention.combine_masks_or
::: haliax.nn.attention.forgetful_causal_mask

### Biases

::: haliax.nn.attention.mask_to_bias
::: haliax.nn.attention.alibi_attention_bias

## Functions

These functions wrap the equivalent in [jax.nn][]:

::: haliax.nn.relu
::: haliax.nn.relu6
::: haliax.nn.sigmoid
::: haliax.nn.softplus
::: haliax.nn.soft_sign
::: haliax.nn.silu
::: haliax.nn.swish
::: haliax.nn.log_sigmoid
::: haliax.nn.leaky_relu
::: haliax.nn.hard_sigmoid
::: haliax.nn.hard_silu
::: haliax.nn.hard_swish
::: haliax.nn.hard_tanh
::: haliax.nn.elu
::: haliax.nn.celu
::: haliax.nn.selu
::: haliax.nn.gelu
::: haliax.nn.quick_gelu
::: haliax.nn.glu
::: haliax.nn.logsumexp
::: haliax.nn.log_softmax
::: haliax.nn.softmax
::: haliax.nn.standardize
::: haliax.nn.one_hot

### Loss Functions

::: haliax.nn.cross_entropy_loss
::: haliax.nn.cross_entropy_loss_and_log_normalizers
