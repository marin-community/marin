"""An implementation of the Hyena operator.

Paper: [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)
Official implementation in PyTorch:
https://github.com/HazyResearch/safari/blob/541902aca88cb11af4d816ac762f3303e4ff8eea/src/models/sequence/hyena.py

Current diffences from the official impl:
- We don't support inner_factor.
- We don't support post_order_ffn.
- We more efficiently support multiple blocks. I believe the PyTorch version is inefficient: It uses
  the full sequence length in the PositionalEmbedding, where it should only be operating over a
  single block.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax
import haliax as hax
import haliax.nn as hnn
from haliax import Axis
from haliax.axis import AxisSpec
from haliax.jax_utils import maybe_rng_split, named_call
from haliax.nn.mlp import DEFAULT_WIDTH_NAME
from haliax.quantization import DotGeneralOp

from levanter.utils.activation import ActivationFunction, ActivationFunctionEnum


@dataclass(frozen=True)
class HyenaConfig:
    seq_len: int = 1024  # l_max from PyTorch impl
    hidden_dim: int = 768  # d_model from PyTorch impl
    num_heads: int = 1

    order: int = 2  # depth of the Hyena recurrence
    filter_order: int = 16  # width of the FFN parametrizing the implicit filter
    short_filter_order: int = 3  # length of the explicit input convolutional filter
    outer_mixing: bool = False  # whether to use outer mixing
    activation: ActivationFunctionEnum = ActivationFunctionEnum.gelu_new
    num_blocks: int = 1  # number of blocks to split the sequence into
    num_hidden_layers_filter_mlp: int = 2  # number of inner linear layers inside filter MLP

    # Filter parameters
    filter_emb_dim: int = 3  # dim of input to MLP, augments with positional encoding
    filter_dropout: float = 0.0  # dropout probability for the filter

    # Modulation parameters
    fast_decay_pct: float = 0.3
    slow_decay_pct: float = 1.5
    target: float = 1e-2
    modulate: bool = True
    shift: float = 0.0

    # General parameters
    resid_pdrop: float = 0.0  # Dropout for residual connections
    use_bias: bool = True  # Whether to use bias in linear layers

    # Axes
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Block = property(lambda self: Axis(name="blocks", size=self.num_blocks))
    PosPerBlock = property(lambda self: Axis(name="pos_per_block", size=self.seq_len // self.num_blocks))
    FilterOrder = property(lambda self: Axis(name="hyena_filter_order", size=self.filter_order))
    FilterEmbed = property(lambda self: Axis(name="hyena_filter_embed", size=self.filter_emb_dim))
    HeadSizeOrderMinus1 = property(
        lambda self: Axis(name="head_size_order_minus_1", size=(self.hidden_dim // self.num_heads) * (self.order - 1))
    )
    EmbedOrderPlus1 = property(lambda self: Axis(name="embed_order_plus_1", size=self.hidden_dim * (self.order + 1)))
    OrderMinus1 = property(lambda self: Axis(name="order_minus_1", size=self.order - 1))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))
    HeadSizeOrderPlus1 = property(
        lambda self: Axis(name="head_size_order_plus_1", size=(self.hidden_dim * (self.order + 1)) // self.num_heads)
    )

    def __post_init__(self):
        if self.hidden_dim % self.num_heads:
            raise ValueError(f"hidden_dim {self.hidden_dim} must be divisible by num_heads {self.num_heads}")
        if self.seq_len % self.num_blocks:
            raise ValueError(f"seq_len {self.seq_len} must be divisible by num_blocks {self.num_blocks}")


class PositionalEmbedding(eqx.Module):
    """Complex exponential positional embeddings for Hyena filters."""

    z: hax.NamedArray  # [Pos, Embed]
    t: hax.NamedArray  # [Pos]
    PosPerBlock: Axis = eqx.field(static=True)

    @staticmethod
    def init(PosPerBlock: Axis, Embed: Axis, *, key=None):
        """Initialize positional embeddings for Hyena filters.

        Args:
            PosPerBlock: Position axis, will be in the outputs of __call__.
            Embed: Axis of positional embedding
            key: Optional random key (not used)
        """
        seq_len = PosPerBlock.size

        # Ensure emb_dim is valid
        if Embed.size <= 1:
            raise ValueError("emb_dim must be greater than 1")

        # The time embedding fed to the filters is normalized so that t_f = 1
        t = hax.linspace(PosPerBlock, start=0, stop=1)

        # Calculate number of frequency bands
        bands = (Embed.size - 1) // 2
        Band = Axis("band", bands)

        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = hax.linspace(PosPerBlock, start=0, stop=seq_len - 1)
        w = 2 * jnp.pi * t_rescaled / seq_len

        f = hax.linspace(Band, start=1e-4, stop=bands - 1)
        z_complex = hax.exp(-1j * f.broadcast_axis(PosPerBlock) * w.broadcast_axis(Band))

        # Concatenate time and complex components
        z = hax.concatenate(
            Embed,
            [
                hax.rename(x, {Band: Embed.name})
                for x in (t.broadcast_axis(Band), hax.real(z_complex), hax.imag(z_complex))
            ],
        )

        return PositionalEmbedding(z, t, PosPerBlock)

    def __call__(self, L):
        """Get positional embeddings for the first L positions.

        Args:
            L: Length to get embeddings for

        Returns:
            Tuple of (z, t) embeddings limited to length L
        """
        if L > self.PosPerBlock.size:
            raise ValueError(f"Requested length {L} > max size {self.PosPerBlock.size}")

        return self.z.slice(self.PosPerBlock, length=L), self.t.slice(self.PosPerBlock, length=L)


class ExponentialModulation(eqx.Module):
    """Exponential modulation for the Hyena filter."""

    deltas_abs: hax.NamedArray
    modulate: bool
    shift: float
    PosPerBlock: Axis = eqx.field(static=True)

    @staticmethod
    def init(
        HeadSizeOrderMinus1: Axis,
        PosPerBlock: Axis,
        fast_decay_pct: float,
        slow_decay_pct: float,
        target: float,
        modulate: bool,
        shift: float,
        *,
        key=None,
    ):
        """Initialize exponential modulation for Hyena filter.

        Args:
            HeadSizeOrderMinus1: Embedding dimension axis
            PosPerBlock: Position axis
            fast_decay_pct: Fast decay percentage
            slow_decay_pct: Slow decay percentage
            target: Target value for decay
            modulate: Whether to apply modulation
            shift: Shift value for modulation
            key: Random key (not used)
        """
        max_decay = jnp.log(target) / fast_decay_pct
        min_decay = jnp.log(target) / slow_decay_pct

        decays = jnp.linspace(min_decay, max_decay, HeadSizeOrderMinus1.size)
        deltas = hax.named(decays, (HeadSizeOrderMinus1,))

        return ExponentialModulation(hax.abs(deltas), modulate, shift, PosPerBlock)

    def __call__(self, t, x):
        """Apply exponential modulation to input.

        Args:
            t: Time values
            x: Input tensor to modulate

        Returns:
            Modulated tensor
        """
        if self.modulate:
            decay = hax.exp(-t * self.deltas_abs.broadcast_axis(self.PosPerBlock))
            x = x * (decay + self.shift)

        return x


class MLPTrainableActivation(eqx.Module):
    """
    hax.nn.MLP with a trainable activation function.
    Activation parameters are shared across all layers.
    """

    activation: Callable = eqx.field(static=False)  # this is the main difference from hax.nn.MLP
    layers: Sequence[hax.nn.Linear]
    Width: Axis = eqx.field(static=True)
    Width2: Axis = eqx.field(static=True)

    @staticmethod
    def init(
        Input: AxisSpec,
        Output: AxisSpec,
        width: int | Axis,
        depth: int,
        activation: Callable = hnn.relu,
        *,
        out_first: bool = True,
        use_bias: bool = True,
        use_final_bias: bool = True,
        key: PRNGKeyArray,
        dot_general: Optional[DotGeneralOp] = None,
        init_scale: float = 1.0,
    ):
        Width = _get_width(width)
        Width2 = Width.alias(Width.name + "2")

        keys = jax.random.split(key, depth + 1)

        layers = []

        kwargs: dict = {
            "use_bias": use_bias,
            "dot_general": dot_general,
            "init_scale": init_scale,
            "out_first": out_first,
        }

        last_kwargs: dict = {
            "use_bias": use_final_bias,
            "dot_general": dot_general,
            "init_scale": init_scale,
            "out_first": out_first,
        }

        if depth == 0:
            # special case: no hidden layers
            layers.append(hnn.Linear.init(Input, Output, key=keys[0], **last_kwargs))
        else:
            # first hidden layer
            layers.append(hnn.Linear.init(Input, Width, key=keys[0], **kwargs))
            # middle hidden layers
            cur = Width
            next = Width2
            for i in range(1, depth):
                layers.append(hnn.Linear.init(cur, next, key=keys[i], **kwargs))
                cur, next = next, cur
            # final layer
            layers.append(hnn.Linear.init(cur, Output, key=keys[-1], **last_kwargs))

        return MLPTrainableActivation(
            layers=tuple(layers),
            activation=activation,
            Width=Width,
            Width2=Width2,
        )

    @property
    def In(self) -> AxisSpec:
        return self.layers[0].In

    @property
    def Out(self) -> AxisSpec:
        return self.layers[-1].Out

    def __call__(self, x: hax.NamedArray, *, key=None) -> hax.NamedArray:
        keys = maybe_rng_split(key, len(self.layers))
        for layer, k in zip(self.layers[:-1], keys):
            layer_out = layer(x, key=k)
            # the trainable activation function does a matmul with the width axis,
            # so we need to rename it to the default width axis name.
            renamed = False
            if layer_out.has_axis(self.Width2.name):
                renamed = True
                layer_out = layer_out.rename({self.Width2.name: self.Width.name})
            x = self.activation(layer_out)
            if renamed:
                x = x.rename({self.Width.name: self.Width2.name})
        return self.layers[-1](x, key=keys[-1])


def _get_width(Width: int | Axis) -> Axis:
    if isinstance(Width, int):
        return Axis(DEFAULT_WIDTH_NAME, Width)
    else:
        return Width


class Sin(eqx.Module):
    """Sinusoidal activation function with trainable frequency."""

    freq: hax.NamedArray

    @staticmethod
    def init(Order: Axis, w: float = 10, *, key=None):
        return Sin(w * hax.ones((Order,)))

    def __call__(self, x: hax.NamedArray) -> hax.NamedArray:
        return hax.sin(self.freq * x)


def fft_conv(u: jax.Array, k: jax.Array) -> jax.Array:
    """JAX implementation of FFT convolution."""
    seqlen = u.shape[-2]
    fft_size = 2 * seqlen

    # FFT supports only float32 or float64.
    u_f = jnp.fft.rfft(jnp.astype(u, jnp.float32), n=fft_size, axis=-2)
    k_f = jnp.fft.rfft(jnp.astype(k, jnp.float32), n=fft_size, axis=-2) / fft_size

    # Perform convolution in frequency domain
    y_f = u_f * k_f
    y = jnp.fft.irfft(y_f, n=fft_size, axis=-2)[..., :seqlen, :]

    return jnp.astype(y, u.dtype)


class HyenaFilter(eqx.Module):
    """Implicit long filter with modulation for Hyena."""

    implicit_filter: MLPTrainableActivation
    modulation: ExponentialModulation
    pos_emb: PositionalEmbedding
    bias: hax.NamedArray
    normalized: bool
    use_bias: bool
    dropout: hnn.Dropout

    @staticmethod
    def init(config: HyenaConfig, *, key):
        keys = jrandom.split(key, 4)

        pos_emb = PositionalEmbedding.init(config.PosPerBlock, config.FilterEmbed, key=keys[0])

        implicit_filter = MLPTrainableActivation.init(
            Input=config.FilterEmbed,
            width=config.FilterOrder,
            Output=config.HeadSizeOrderMinus1,
            depth=config.num_hidden_layers_filter_mlp,
            activation=Sin.init(config.FilterOrder, w=1),
            use_bias=config.use_bias,
            key=keys[1],
        )

        modulation = ExponentialModulation.init(
            config.HeadSizeOrderMinus1,
            config.PosPerBlock,
            config.fast_decay_pct,
            config.slow_decay_pct,
            config.target,
            config.modulate,
            config.shift,
            key=keys[2],
        )

        bias = hax.random.normal(keys[3], (config.HeadSize,))

        dropout = hnn.Dropout(pdrop=config.filter_dropout)

        return HyenaFilter(
            implicit_filter, modulation, pos_emb, bias, normalized=False, use_bias=config.use_bias, dropout=dropout
        )

    def generate_filters(self, input_length: int, *, key=None) -> hax.NamedArray:
        """Generate filter kernels for Hyena operation.

        Args:
            input_length: Length of input sequence
            key: Optional PRNG key for dropout

        Returns:
            NamedArray containing filter
        """
        z, t = self.pos_emb(input_length)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)

        if self.normalized:
            # Implement L1 norm manually since there's no haliax.norm
            h_abs = hax.abs(h)
            norm_values = hax.sum(h_abs, axis=h.axes[1], where=None)
            h = h / norm_values.broadcast_axis(h.axes[1])

        return h

    @named_call
    def __call__(self, x: hax.NamedArray, k: hax.NamedArray, bias=None, *, key=None):
        """Apply the hyena filter.

        Args:
            x: Input tensor with shape (batch, seq_len, channels)
            k: Filter to use
            bias: Optional bias to use (if None, uses self.bias)
            key: Optional PRNG key for dropout

        Returns:
            Filtered tensor with same shape as input
        """
        if bias is None:
            bias = self.bias

        bias = bias if self.use_bias else hax.zeros_like(bias)

        # fft_conv is not haliax aware so we have to rearrange and pass in raw arrays.
        x = hax.rearrange(x, (..., self.pos_emb.PosPerBlock))
        k = hax.rearrange(k, (..., self.pos_emb.PosPerBlock))
        x_unnamed = x.array
        k_unnamed = k.array

        y_unnamed = fft_conv(x_unnamed, k_unnamed)
        y = hax.named(y_unnamed, x.axes)
        y += bias

        if key is not None and self.dropout.pdrop > 0:
            dropout_key = haliax.jax_utils.maybe_rng_split(key, 1)[0]
            y = self.dropout(y, key=dropout_key)

        return y


class HyenaOperator(eqx.Module):
    """Hyena operator - the core building block of the Hyena architecture."""

    config: HyenaConfig = eqx.field(static=True)
    in_proj: hnn.Linear
    out_proj: hnn.Linear
    short_filter: hnn.Conv
    filter_fn: HyenaFilter
    dropout: hnn.Dropout
    activation: ActivationFunction = eqx.field(static=True)

    @staticmethod
    def init(config: HyenaConfig, *, key):
        keys = jrandom.split(key, 5)

        in_proj = hnn.Linear.init(
            In=config.Embed,
            Out=config.EmbedOrderPlus1,
            key=keys[0],
            use_bias=config.use_bias,
        )

        # Output projection: hidden_size -> hidden_size
        # Create a new axis with the same dimensions to avoid naming collision
        # We do not support inner_factor from the PyTorch impl.
        out_proj = hnn.Linear.init(
            In=config.Embed, Out=config.Embed.alias("output_embed"), key=keys[1], use_bias=config.use_bias
        )

        short_filter = hnn.Conv.init(
            Spatial=config.Pos,
            In=config.EmbedOrderPlus1,
            Out=config.EmbedOrderPlus1.alias("out_channels"),
            kernel_size=config.short_filter_order,
            groups=config.EmbedOrderPlus1.size,
            padding=config.short_filter_order - 1,
            key=keys[2],
        )

        filter_fn = HyenaFilter.init(config, key=keys[3])
        dropout = hnn.Dropout(pdrop=config.resid_pdrop)

        return HyenaOperator(
            config=config,
            in_proj=in_proj,
            out_proj=out_proj,
            short_filter=short_filter,
            filter_fn=filter_fn,
            dropout=dropout,
            activation=config.activation.to_fn(),
        )

    @named_call
    def __call__(self, u: hax.NamedArray, *, key: PRNGKeyArray | None = None) -> hax.NamedArray:
        key_in_proj, key_dropout = haliax.jax_utils.maybe_rng_split(key, 2)
        Pos = self.config.Pos
        Block = self.config.Block
        PosPerBlock = self.config.PosPerBlock
        EmbedOrderPlus1 = self.config.EmbedOrderPlus1
        Embed = self.config.Embed
        # input has the same axis name as the Pos axis, but possibly different size
        input_length = u.axis_size(Pos.name)
        l_filter = min(input_length, Pos.size)

        # Input projection from [Embed] to [(order+1) * Embed]
        u = self.in_proj(u, key=key_in_proj)

        # trying to keep the variable names from the official impl.
        # I think uc stands for "u convolved".
        uc = self.short_filter(u).rename({"out_channels": EmbedOrderPlus1})
        # Note: after the short filter, uc."position" axis has slightly larger size due to padding.
        # Hence we use Pos.name rather than Pos to slice, otherwise haliax complains the axis sizes
        # are different.
        uc = uc.slice(Pos.name, length=l_filter)

        # Now we need to reshape to match the PyTorch implementation's:
        # 'b (ho v) (z l) -> b ho v z l'
        uc = hax.unflatten_axis(uc, Pos, (Block, PosPerBlock))
        uc = hax.unflatten_axis(uc, EmbedOrderPlus1, (self.config.Heads, self.config.HeadSizeOrderPlus1))

        components = hax.split(uc, self.config.HeadSizeOrderPlus1, [self.config.HeadSize] * (self.config.order + 1))
        v = components[-1]
        x = components[:-1]
        assert len(x) == self.config.order
        filters = self.filter_fn.generate_filters(l_filter, key=key_dropout)
        filters = hax.unflatten_axis(
            filters, self.config.HeadSizeOrderMinus1, (self.config.OrderMinus1, self.config.HeadSize)
        )
        filters_list = filters.unbind(self.config.OrderMinus1)

        # Long-range filtering with recurrence
        for filter_order, x_i in enumerate(reversed(x[1:])):
            # Outer product of Embed with Embed
            if self.config.outer_mixing:
                EmbedPrime = self.config.Embed.alias("embed_prime")
                v_for_outer = hax.rename(v, {Embed: EmbedPrime})
                outer_product = v_for_outer.broadcast_axis(EmbedPrime) * x_i
                v = self.dropout(outer_product, key=key_dropout)
                v = hax.sum(v, EmbedPrime)
            else:
                v = self.dropout(v * x_i, key=key_dropout)

            v = self.filter_fn(v, filters_list[filter_order], key=key_dropout)

            # Not currently supporting the post_order_ffn from the PyTorch impl.

        v = v * x[0]
        v = hax.flatten_axes(v, (Block, PosPerBlock), Pos)
        v = hax.flatten_axes(v, (self.config.Heads, self.config.HeadSize), Embed)
        y = self.activation(v)
        y = self.out_proj(y, key=key_dropout).rename({"output_embed": Embed})
        return y
