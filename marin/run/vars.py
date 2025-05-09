# Environment variables that we want to set in Ray
ENV_VARS = {
    "PYTHONPATH": "./submodules/levanter/src:${PYTHONPATH}",
    "LIBTPU_INIT_ARGS": "--xla_tpu_scoped_vmem_limit_kib=81920 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true",
}

PIP_DEPS = []

REMOTE_DASHBOARD_URL = "http://127.0.0.1:8265"
