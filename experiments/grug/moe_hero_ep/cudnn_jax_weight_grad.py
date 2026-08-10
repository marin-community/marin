# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import-stable CuTe launcher for the experimental cuDNN grouped Wgrad bridge."""

from levanter.cutlass_kernel_cache import cute_launcher_factory


@cute_launcher_factory
def build_cudnn_grouped_wgrad_launcher(
    modules,
    *,
    expert_count: int,
    max_active_clusters: int,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
):
    cutlass, cute, _cjax, kernel_type, weight_mode, input_order = modules

    @cute.jit
    def launcher(stream, mat_a, mat_b, offsets, output, workspace):
        kernel = kernel_type(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == 256,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            accumulate_on_output=False,
            expert_cnt=expert_count,
            weight_mode=weight_mode.DENSE,
            input_order=input_order.Tensor2D,
        )
        kernel(mat_a, mat_b, output, offsets, workspace, max_active_clusters, stream, None)

    return launcher
