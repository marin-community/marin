# GPU Tips

Read this file for GPU Pallas/Mosaic work.

Prefer Mosaic GPU for new Pallas GPU kernels. Existing Triton-backend Pallas
kernels may still exist in the repo, such as Haliax's `ragged_dot`; treat them
as legacy references unless the user explicitly asks to work on the Triton
backend.

Start from the official
[JAX GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
and
[JAX Toolbox tips](https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/GPU_performance.md).
Keep benchmark setup aligned with [Performance workflow](performance-workflow.md).
