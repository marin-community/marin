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

from fray.cluster import ResourceConfig

from marin.evaluation.helmet import HELMET_PIPELINE_AUTOMATIC, HelmetConfig, helmet_steps
from marin.execution.executor import executor_main


def main() -> None:
    # For gcsfuse-backed checkpoints, prefer a path under `gs://$PREFIX/gcsfuse_mount/...`
    model_path = "meta-llama/Llama-3.2-1B"
    model_name = "llama-3.2-1b"

    steps = helmet_steps(
        model_name=model_name,
        model_path=model_path,
        helmet=HelmetConfig(
            # Required: match HELMET's slurm behavior (True for instruct/chat models, False for base models).
            use_chat_template=False,
            resource_config=ResourceConfig.with_tpu("v4-8"),
            # Optional: for long-context models, you may need something like:
            # vllm_serve_args=("--max-model-len", "131072"),
        ),
        pipeline=HELMET_PIPELINE_AUTOMATIC,
        wandb_tags=["helmet", "test"],
    )

    executor_main(steps=steps)


if __name__ == "__main__":
    main()
