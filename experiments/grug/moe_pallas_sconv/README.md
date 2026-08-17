# Grug MoE Pallas SConv smoke

This variant copies `experiments/grug/moe` and adds kernel-size-4 depthwise causal convolutions after the K and V projections and on the attention and MoE branch outputs. It uses the Pallas TPU kernel from #8331, identity-initializes every convolution, and removes cross-document tap contributions for packed examples.

`launch.py` runs `MOE-PSC-001`, a 25-step d512 smoke test on one v5p-8. It verifies TPU compilation and training before the full two-scale Gate 1 comparison in `experiments/grug/moe/agent.md`.

Tracking issue: https://github.com/marin-community/marin/issues/8377
