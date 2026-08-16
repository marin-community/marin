# **************************************************
# Copyright (c) 2026, Mayank Mishra
# copied from https://github.com/open-lm-engine/accelerated-model-architectures
# **************************************************

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import Axis, AxisSelector, NamedArray

from levanter.kernels.pallas.depthwise_causal_convolution import depthwise_causal_convolution


class DepthwiseCausalConvolution(eqx.Module):
    Embed: Axis = eqx.field(static=True)
    weight: NamedArray  # [Embed, Kernel]
    bias: NamedArray | None  # [Embed]

    kernel_size: int = eqx.field(static=True)
    activation_function: str | None = eqx.field(static=True)

    @property
    def State(self) -> Axis:
        return Axis("state", self.kernel_size - 1)

    @staticmethod
    def init(
        Embed: Axis,
        kernel_size: int,
        activation_function: str | None,
        use_bias: bool,
        *,
        key: PRNGKeyArray,
    ) -> "DepthwiseCausalConvolution":
        assert kernel_size > 1

        weight_key, bias_key = jax.random.split(key, 2)
        bound = kernel_size**-0.5
        Kernel = Axis("kernel", kernel_size)

        weight = hax.random.uniform(weight_key, (Embed, Kernel), minval=-bound, maxval=bound)
        bias = hax.random.uniform(bias_key, (Embed,), minval=-bound, maxval=bound) if use_bias else None

        return DepthwiseCausalConvolution(
            Embed=Embed,
            weight=weight,
            bias=bias,
            kernel_size=kernel_size,
            activation_function=activation_function,
        )

    def __call__(
        self,
        input: NamedArray,
        input_state: NamedArray | None = None,
        attention_mask: NamedArray | None = None,
        output_state: bool = False,
        *,
        Pos: AxisSelector = "position",
    ) -> tuple[NamedArray, NamedArray | None]:
        """Apply the depthwise causal convolution.

        Args:
            input: [Batch, Pos, Embed], Batch being whatever single axis remains once Pos and
                Embed are accounted for.
            input_state: [Batch, Embed, State], the `kernel_size - 1` raw positions preceding
                `input`. None is equivalent to a 0 tensor.
            attention_mask: [Batch, Pos], zeroing out padding positions before and after the
                convolution.
            output_state: whether to also return the trailing `kernel_size - 1` raw input
                positions for use as `input_state` in a subsequent call.
            Pos: the axis (or its name) identifying the sequence dimension in `input`.

        Returns:
            output: [Batch, Pos, Embed].
            output_state: [Batch, Embed, State] if `output_state` is True else None.
        """
        input = hax.rearrange(input, (..., Pos, self.Embed))
        if len(input.axes) != 3:
            raise ValueError(f"expected exactly one batch axis alongside {Pos} and {self.Embed}, got {input.axes}")
        Batch = input.axes[0]

        input_state_array = None
        if input_state is not None:
            input_state_array = hax.rearrange(input_state, (Batch, self.Embed, self.State)).array

        attention_mask_array = None
        if attention_mask is not None:
            attention_mask_array = hax.rearrange(attention_mask, (Batch, Pos)).array

        output_array, output_state_array = depthwise_causal_convolution(
            input=input.array,
            weight=self.weight.array,
            bias=None if self.bias is None else self.bias.array,
            input_state=input_state_array,
            attention_mask=attention_mask_array,
            output_state=output_state,
            activation_function=self.activation_function,
        )

        output = hax.named(output_array, input.axes)
        output_state_named = (
            None if output_state_array is None else hax.named(output_state_array, (Batch, self.Embed, self.State))
        )

        return output, output_state_named
