# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

<<<<<<< HEAD
# Environment variables that we want to set in Ray
ENV_VARS = {
    "TPU_MIN_LOG_LEVEL": "3",
    "TPU_STDERR_LOG_LEVEL": "3",
    "PYTHONPATH": "./submodules/levanter/src:${PYTHONPATH}",
    "LIBTPU_INIT_ARGS": (
        "--xla_tpu_scoped_vmem_limit_kib=81920 "
        "--xla_enable_async_all_gather=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
        "--xla_tpu_data_parallel_opt_different_sized_ops=true "
        "--xla_tpu_megacore_fusion_allow_ags=false "
        "--xla_enable_async_collective_permute=true "
        "--xla_tpu_enable_ag_backward_pipelining=true"
    ),
}

PIP_DEPS = []

=======
>>>>>>> origin/main
REMOTE_DASHBOARD_URL = "http://127.0.0.1:8265"
