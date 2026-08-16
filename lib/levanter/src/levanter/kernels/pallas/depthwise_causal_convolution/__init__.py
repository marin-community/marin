# **************************************************
# Copyright (c) 2026, Mayank Mishra
# copied from https://github.com/open-lm-engine/accelerated-model-architectures
# **************************************************

from .op import IMPLEMENTATIONS, Implementation, depthwise_causal_convolution

__all__ = [
    "IMPLEMENTATIONS",
    "Implementation",
    "depthwise_causal_convolution",
]
