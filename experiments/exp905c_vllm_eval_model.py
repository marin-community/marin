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

import logging
import re
import sys

from experiments.evals.evals import evaluate_lm_evaluation_harness
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main
from fray.cluster import ResourceConfig


def extract_checkpoint_number(path: str) -> str | None:
    """Extract checkpoint/step number from a model path.

    Examples:
        gs://bucket/checkpoints/.../hf/step-3000/ -> "3000"
        gs://bucket/checkpoints/.../hf/step-11718/ -> "11718"
        gs://bucket/models/some-model/ -> None (no checkpoint number)
    """
    # Match "step-" followed by digits
    match = re.search(r'step-(\d+)', path)
    if match:
        return match.group(1)
    return None

resource_config = ResourceConfig.with_tpu("v5p-8")

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for even more detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# Enable specific loggers
logging.getLogger("marin.evaluation").setLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.INFO)

"""
Note for people trying to do evals:
- This script uses evaluate_lm_evaluation_harness which uses vLLM and lm-evaluation-harness
- The lm-evaluation-harness library automatically loads evaluation datasets
- Similar to exp905c_levanter_eval_model.py but uses vLLM instead of Levanter engine
- The structure follows exp905c_levanter_eval_model.py with EVAL_TASKS, MODELS, and compile_results

"""
EVAL_TASKS = [
    EvalTaskConfig("aime24", num_fewshot=0, task_alias="aime24_0shot"),
    EvalTaskConfig("aime25", num_fewshot=0, task_alias="aime25_0shot"),
    # EvalTaskConfig("math_500", num_fewshot=0, task_alias="math_500"),
    # EvalTaskConfig("gpqa_diamond_zeroshot", num_fewshot=0, task_alias="gpqa_diamond_zeroshot"),
]

# Seeds for multiple evaluation runs to compute averaged results
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
# SEEDS = [42, 43, 44]

MODELS = [
    # {
    #     "name": "qwen2.5-7b-instruct",
    #     "path": "gs://marin-us-central2/models/qwen2.5-7b-instruct",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "marin-8b-instruct",
    #     "path": "gs://marin-us-central2/models/Marin-8B-Instruct",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-redo_fixed_template",
    #     "path": "gs://marin-us-central2/checkpoints/exp2199b_redo_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-2d659d/hf/step-11718",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-redo_fixed_template_pt2",
    #     "path": "gs://marin-us-central2/checkpoints/exp2199b_redo_pt2_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-1a1aff/hf/step-10500",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen3-8b-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2199c_sft_qwen3_8b_openthoughts3_bsz512_lr8e_5-accb91/hf/step-11718/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "marin-8b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2199a_sft_marin_8b_instruct_openthoughts3_bsz512_lr8e_5-3a3fd2/hf/step-11718/",  # NOTE: No need for padded vocab
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-1.5b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209a1_sft_qwen2pt5_1pt5b_instruct_ot3_bsz512_lr8e_5-eb7076/hf/step-11718/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-3b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209b1_sft_qwen2pt5_3b_instruct_openthoughts3_bsz512_lr8e_5-c7d431/hf/step-11718/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-1.5b-instruct-finetuned-ot4-qwen3-32b-7.5K",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209a2_sft_qwen2pt5_1pt5b_instruct_openthoughts4_1pt2m_qwen3_-0f8594/hf/step-11718/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-3b-instruct-finetuned-ot4-qwen3-32b-7.5K",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209b2_sft_qwen2pt5_3b_instruct_openthoughts4_1pt2m_qwen3_3b_-88f693/hf/step-11718/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-qwen3-32b-7.5K",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209c2_sft_qwen2pt5_7b_instruct_openthoughts4_1pt2m_qwen3_3b_-740b7d/hf/step-11718/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-32b-instruct-finetuned-ot4-qwen3-32b-7.5K",  # TODO
    #     "path": "gs://marin-us-central1/checkpoints/exp2209d2_sft_qwen2pt5_32b_instruct_ot4_1pt2m_qwen3_3b_bsz512_lr-86c9db/hf/step-11718/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "llama-3.1-8b-instruct",
    #     "path": "gs://marin-us-central2/models/llama-3.1-8b-instruct/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-32b-exp2262a",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262a_ot4_math30k_qwen3_32b_bsz128_lr4e_5-ad01cb/hf/step-1170",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262b",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262b_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-41ff16/hf/step-1170",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "openthinker3-7b",
    #     "path": "gs://marin-us-central2/models/OpenThinker3-7B/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262c",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262c_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-b39be3/hf/step-3000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },


    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-32b-exp2262e-bsz30",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262e_ot4_math30k_qwen3_32b_bsz128_lr4e_5-51aefe/hf/step-2340/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262f-bsz30",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262f_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-cfac80/hf/step-2340/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },

    # START exp2199
    {
        "name": "marin-8b-longcontext-finetuned-ot3-exp2199a2_redo2",
        "path": "gs://marin-us-central2/checkpoints/exp2199a2_redo2_sft_longcontext_marin_8b_ot3_bsz512_lr8e_5-fe3bab/hf/step-11718/",
        "apply_chat_template": True,
        "tensor_parallel_size": 4,
    },
    {
        "name": "qwen2.5-7b-instruct-finetuned-ot3-exp2199b_redo3",
        "path": "gs://marin-us-central2/checkpoints/exp2199b_redo3_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-c05011/hf/step-11718/",
        "apply_chat_template": True,
        "tensor_parallel_size": 1,
    },

    # START exp2262
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-32b-exp2262g",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262g_ot4_math30k_qwen3_32b_bsz128_lr4e_5-42ab13/hf/step-1000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-32b-exp2262g",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262g_ot4_math30k_qwen3_32b_bsz128_lr4e_5-42ab13/hf/step-2000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-32b-exp2262g",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262g_ot4_math30k_qwen3_32b_bsz128_lr4e_5-42ab13/hf/step-3000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-32b-exp2262g",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262g_ot4_math30k_qwen3_32b_bsz128_lr4e_5-42ab13/hf/step-4000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-32b-exp2262g",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262g_ot4_math30k_qwen3_32b_bsz128_lr4e_5-42ab13/hf/step-4681/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262h",
    #     "path": "gs://marin-eu-west4/checkpoints/exp2262h_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-4b4170/hf/step-1000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262h",
    #     "path": "gs://marin-eu-west4/checkpoints/exp2262h_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-4b4170/hf/step-2000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262h",
    #     "path": "gs://marin-eu-west4/checkpoints/exp2262h_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-4b4170/hf/step-3000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262h",
    #     "path": "gs://marin-eu-west4/checkpoints/exp2262h_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-4b4170/hf/step-4000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262h",
    #     "path": "gs://marin-eu-west4/checkpoints/exp2262h_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-4b4170/hf/step-4681/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-32b-exp2262i",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262i_longcontext_marin_ot4_math30k_qwen3_32b_bsz128_lr1e_5-e4da9a/hf/step-500/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262j",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262j_longcont_marin_ot4_math30k_qwen3_235b_bsz128_lr1e_5-385ee7/hf/step-500/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-32b-exp2262i",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262i_longcontext_marin_ot4_math30k_qwen3_32b_bsz128_lr1e_5-e4da9a/hf/step-1000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262j",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262j_longcont_marin_ot4_math30k_qwen3_235b_bsz128_lr1e_5-385ee7/hf/step-1000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-32b-exp2262i",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262i_longcontext_marin_ot4_math30k_qwen3_32b_bsz128_lr1e_5-e4da9a/hf/step-2000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262j",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262j_longcont_marin_ot4_math30k_qwen3_235b_bsz128_lr1e_5-385ee7/hf/step-2000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-32b-exp2262i",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262i_longcontext_marin_ot4_math30k_qwen3_32b_bsz128_lr1e_5-e4da9a/hf/step-3000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262j",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262j_longcont_marin_ot4_math30k_qwen3_235b_bsz128_lr1e_5-385ee7/hf/step-3000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-32b-exp2262i",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262i_longcontext_marin_ot4_math30k_qwen3_32b_bsz128_lr1e_5-e4da9a/hf/step-4000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262j",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262j_longcont_marin_ot4_math30k_qwen3_235b_bsz128_lr1e_5-385ee7/hf/step-4000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-32b-exp2262i",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262i_longcontext_marin_ot4_math30k_qwen3_32b_bsz128_lr1e_5-e4da9a/hf/step-4681/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math30k-qwen3-235b-a22b-exp2262j",
    #     "path": "gs://marin-us-central1/checkpoints/exp2262j_longcont_marin_ot4_math30k_qwen3_235b_bsz128_lr1e_5-385ee7/hf/step-4681/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },

    # START exp2262 - part 2: agreed/disagreed subsets
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math11k-qwen3-32b-agreed-exp2262k",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262k_ot4_agreed_qwen3_32b_bsz128_lr4e_5-afb092/hf/step-1000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math11k-qwen3-32b-agreed-exp2262k",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262k_ot4_agreed_qwen3_32b_bsz128_lr4e_5-afb092/hf/step-1711/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math11k-qwen3-235b-a22b-agreed-exp2262l",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262l_ot4_agreed_qwen3_235b_a22b_bsz128_lr4e_5-72433f/hf/step-1000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math11k-qwen3-235b-a22b-agreed-exp2262l",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262l_ot4_agreed_qwen3_235b_a22b_bsz128_lr4e_5-72433f/hf/step-1711/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math11k-qwen3-32b-agreed-exp2262o",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262o_longcontext_marin_ot4_agreed_qwen3_32b_bsz128_lr1e_5-abb099/hf/step-1000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math11k-qwen3-32b-agreed-exp2262o",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262o_longcontext_marin_ot4_agreed_qwen3_32b_bsz128_lr1e_5-abb099/hf/step-1711/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math11k-qwen3-235b-a22b-agreed-exp2262p",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262p_longcont_marin_ot4_agreed_qwen3_235b_bsz128_lr1e_5-472040/hf/step-1000/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math11k-qwen3-235b-a22b-agreed-exp2262p",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262p_longcont_marin_ot4_agreed_qwen3_235b_bsz128_lr1e_5-472040/hf/step-1711/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math6k-qwen3-32b-disagreed-exp2262m",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262m_ot4_disagreed_qwen3_32b_bsz128_lr4e_5-9bf2dd/hf/step-500/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math6k-qwen3-32b-disagreed-exp2262m",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262m_ot4_disagreed_qwen3_32b_bsz128_lr4e_5-9bf2dd/hf/step-934/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math6k-qwen3-235b-a22b-disagreed-exp2262n",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262n_ot4_disagreed_qwen3_235b_a22b_bsz128_lr4e_5-6dea42/hf/step-500/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned-ot4-math6k-qwen3-235b-a22b-disagreed-exp2262n",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262n_ot4_disagreed_qwen3_235b_a22b_bsz128_lr4e_5-6dea42/hf/step-934/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 1,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math6k-qwen3-32b-disagreed-exp2262q",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262q_longcontext_marin_ot4_disagreed_qwen3_32b_bsz128_lr1e_5-ccc1db/hf/step-500/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math6k-qwen3-32b-disagreed-exp2262q",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262q_longcontext_marin_ot4_disagreed_qwen3_32b_bsz128_lr1e_5-ccc1db/hf/step-934/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math6k-qwen3-235b-a22b-disagreed-exp2262r",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262r_longcont_marin_ot4_disagreed_qwen3_235b_bsz128_lr1e_5-9aa333/hf/step-500/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
    # {
    #     "name": "marin-8b-longcont-finetuned-ot4-math6k-qwen3-235b-a22b-disagreed-exp2262r",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262r_longcont_marin_ot4_disagreed_qwen3_235b_bsz128_lr1e_5-9aa333/hf/step-934/",
    #     "apply_chat_template": True,
    #     "tensor_parallel_size": 4,
    # },
]


def compile_results(steps: list[ExecutorStep]) -> ExecutorStep:
    """
    Takes in a list of ExecutorSteps for lm-eval tasks and compiles the results into a single DataFrame.

    Newer lm-eval runs (with EvaluationTracker + log_samples=True) write per-task outputs under
    paths like:

        gs://.../evaluation/lm_evaluation_harness/{model_name}-{hash}/{dataset}_0shot/__tmp__{model}/samples_{dataset}_TIMESTAMP.jsonl

    This helper scans each step's eval root for those `samples_*.jsonl` files, and aggregates the
    JSONL records into a flat DataFrame, annotating each row with `dataset_name` and `model_name`.
    """
    import json
    import logging
    import pandas as pd
    import fsspec
    from marin.execution.executor import InputName, OutputName

    logger = logging.getLogger(__name__)

    def _compile_results_fn(config) -> None:
        """Function that will be executed by the ExecutorStep to compile results."""
        all_results = []

        # Extract parameters from config
        input_paths = config["input_paths"]
        output_path = config["output_path"]

        logger.info(f"Input paths: {input_paths}")

        if not input_paths:
            raise Exception("No input paths found!")

        # Read results from each step's output path
        for i, input_path in enumerate(input_paths):
            # Normalise input_path to the evaluation root directory.
            # Older code passed .../{hash}/results.json; newer runs conceptually
            # treat .../{hash}/ as the root. We strip any trailing "results.json"
            # and then look for per-task sample files under that root.
            base_dir = input_path
            if base_dir.endswith("results.json"):
                base_dir = base_dir.rsplit("/", 1)[0]

            logger.info(f"Loading lm-eval samples from root {base_dir}")

            # Normalize to a GCS URL if the scheme was stripped by the executor packaging.
            # We assume eval outputs live in the "marin-us-east1" bucket when no scheme is present.
            if base_dir.startswith("gs://"):
                gcs_root = base_dir
            else:
                # Avoid accidental relative local paths like "marin-us-east1/..."
                gcs_root = "gs://" + base_dir.lstrip("/")

            # Pattern for per-task sample files:
            # {root}/{dataset}_0shot/{model_name_escaped}/samples_{dataset}_TIMESTAMP.jsonl
            # Note: model_name_escaped replaces "/" with "__" (e.g. gs__bucket__path__model)
            fs = fsspec.filesystem("gcs")
            pattern = gcs_root.rstrip("/") + "/*/*/samples_*.jsonl"
            sample_files: list[str] = fs.glob(pattern)

            if not sample_files:
                logger.warning(f"No samples_*.jsonl files found for input root {base_dir}")
                continue

            for sample_file in sample_files:
                logger.info(f"Reading samples from {sample_file}")
                path_parts = sample_file.split("/")

                # Infer dataset_name from the task directory: {dataset}_0shot
                # e.g. .../model-task-seed42-hash/aime24_0shot/gs__bucket__model/samples_aime24_*.jsonl
                if len(path_parts) >= 3:
                    task_dir = path_parts[-3]
                    # Strip trailing "_{shot}" suffix (e.g. "_0shot", "_5shot")
                    if "_" in task_dir:
                        dataset_name = task_dir.rsplit("_", 1)[0]
                    else:
                        dataset_name = task_dir
                else:
                    dataset_name = "unknown_dataset"

                # Infer model_name from the model/hash directory (4 levels up from samples file)
                # path: .../{model-task-seed42-hash}/{task}_0shot/{model_escaped}/samples_*.jsonl
                if len(path_parts) >= 4:
                    model_dir = path_parts[-4]
                elif len(path_parts) >= 2:
                    model_dir = path_parts[-2]
                else:
                    model_dir = "unknown_model"

                # Strip hash suffix from model_dir (e.g. "model-name-seed42-47227c" -> "model-name-seed42")
                if "-" in model_dir:
                    model_name = model_dir.rsplit("-", 1)[0]
                else:
                    model_name = model_dir

                # Read JSONL samples from GCS using the same filesystem
                with fs.open(sample_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            logger.warning(f"Failed to parse JSON line in {sample_file}: {line[:200]}")
                            continue

                        # Annotate with dataset and model metadata
                        record["dataset_name"] = dataset_name
                        record["model_name"] = model_name
                        all_results.append(record)

        if not all_results:
            raise Exception("No results found in any of the provided steps")

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Extract base model name (without seed suffix) and seed for averaging
        def extract_base_model_and_seed(model_name):
            """Extract base model name and seed from model_name like 'openthinker3-7b-seed42-aime25'

            The model_name format is: {original_model_name}-seed{N}-{task_suffix}
            We want to extract the original_model_name and N.
            """
            import re
            # Match -seed followed by digits, followed by - (task suffix) or end of string
            match = re.search(r'-seed(\d+)(?=-|$)', model_name)
            if match:
                seed = int(match.group(1))
                # Base model is everything before -seed
                base_model = model_name[:match.start()]
                return base_model, seed
            return model_name, None

        df[['base_model_name', 'seed']] = df['model_name'].apply(
            lambda x: pd.Series(extract_base_model_and_seed(x))
        )

        # Save compiled results (all individual runs)
        results_file = f"{output_path}/compiled_results.json"
        with fsspec.open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        csv_file = f"{output_path}/compiled_results.csv"
        with fsspec.open(csv_file, "w") as f:
            df.to_csv(f, index=False)

        logger.info(f"Compiled results saved to: {results_file}")

        # Compute averaged results across seeds
        # Check for common accuracy-related columns
        accuracy_cols = [col for col in df.columns if col in ['exact_match', 'acc', 'accuracy', 'correct']]

        if accuracy_cols and 'base_model_name' in df.columns and 'dataset_name' in df.columns:
            # First compute per-seed accuracy, then average across seeds
            # This gives meaningful std (variance across seeds, not variance of 0/1 values)
            avg_results = []
            for (base_model, dataset), group in df.groupby(['base_model_name', 'dataset_name']):
                # Compute accuracy for each seed separately
                per_seed_accuracies = {}
                for col in accuracy_cols:
                    if col in group.columns:
                        # Group by seed and compute mean accuracy for each seed
                        seed_accs = group.groupby('seed')[col].mean()
                        per_seed_accuracies[col] = seed_accs

                result = {
                    'base_model_name': base_model,
                    'dataset_name': dataset,
                    'num_seeds': group['seed'].nunique(),
                    'seeds': sorted(group['seed'].dropna().unique().tolist()),
                }

                # Compute mean and std across seeds (not across individual samples)
                for col in accuracy_cols:
                    if col in per_seed_accuracies:
                        seed_accs = per_seed_accuracies[col]
                        result[f'{col}_mean'] = seed_accs.mean()
                        result[f'{col}_std'] = seed_accs.std()
                        # Also store per-seed values for reference
                        result[f'{col}_per_seed'] = seed_accs.to_dict()

                avg_results.append(result)

            avg_df = pd.DataFrame(avg_results)

            # Save averaged results
            avg_results_file = f"{output_path}/averaged_results.json"
            with fsspec.open(avg_results_file, "w") as f:
                json.dump(avg_results, f, indent=2)

            avg_csv_file = f"{output_path}/averaged_results.csv"
            with fsspec.open(avg_csv_file, "w") as f:
                avg_df.to_csv(f, index=False)

            logger.info(f"Averaged results saved to: {avg_results_file}")
            logger.info(f"Averaged results:\n{avg_df.to_string()}")

            # Log averaged results to W&B - one run per model
            try:
                import wandb

                # Create a separate summary run for each base model
                num_seeds = len(config.get("seeds", []))
                for base_model in avg_df['base_model_name'].unique():
                    model_df = avg_df[avg_df['base_model_name'] == base_model]

                    wandb.init(
                        project="marin",
                        name=f"{base_model}-averaged-{num_seeds}seeds",
                        tags=["averaged-results", base_model[:64]],  # Truncate to 64 chars
                        config={
                            "base_model_name": base_model,
                            "num_seeds": len(config.get("seeds", [])),
                            "seeds": config.get("seeds", []),
                        },
                        reinit=True,
                    )

                    # Log averaged metrics for each dataset
                    for _, row in model_df.iterrows():
                        dataset = row['dataset_name']

                        # Log as summary metrics
                        for col in accuracy_cols:
                            mean_col = f'{col}_mean'
                            std_col = f'{col}_std'
                            if mean_col in row and std_col in row:
                                wandb.log({
                                    f"{dataset}/{col}_mean": row[mean_col],
                                    f"{dataset}/{col}_std": row[std_col],
                                })

                    # Log this model's averaged results table
                    wandb.log({"averaged_results": wandb.Table(dataframe=model_df)})

                    wandb.finish()
                    logger.info(f"Averaged results for {base_model} logged to W&B")

            except Exception as e:
                logger.warning(f"Failed to log averaged results to W&B: {e}")
        else:
            logger.warning("Could not compute averaged results: missing accuracy columns or grouping columns")

    # Create input paths and output path
    input_paths = [step.cd("results.json") for step in steps]
    output_path = OutputName("compiled_results")

    return ExecutorStep(
        name="evaluation/lm_evaluation_harness/compile_results",
        fn=_compile_results_fn,
        config={"input_paths": input_paths, "output_path": output_path, "seeds": SEEDS},
        description="Compile results from multiple lm-eval steps into a single DataFrame",
    )


if __name__ == "__main__":
    # Quiet ray logs for this experiment
    logging.getLogger("ray").setLevel(logging.WARNING)

    all_steps = []

    for model_config in MODELS:
        # Build full model name: base name + checkpoint suffix (if path contains step-N)
        # e.g., "qwen2.5-7b-exp2262g" + path ".../step-3000/" -> "qwen2.5-7b-exp2262g-chkpt3000"
        base_name = model_config["name"]
        chkpt_num = extract_checkpoint_number(model_config["path"])
        full_model_name = f"{base_name}-chkpt{chkpt_num}" if chkpt_num else base_name

        for task in EVAL_TASKS:
            engine_kwargs = {
                # Default to TP=1 (no tensor parallelism) to avoid vocab padding requirements.
                # 7B models fit on single v5p chip (95GB HBM). Set to 4 for larger models.
                "tensor_parallel_size": model_config.get("tensor_parallel_size", 1),
                "max_num_seqs": 30,  # Batch size for parallel generation
                "enforce_eager": False,  # Allow graph optimization for better throughput
            }
            # Ensure that max_model_len > max_gen_toks + prompt len.
            # Note that max_gen_toks is controlled by lm-eval
            engine_kwargs["max_model_len"] = int(32768+2048)
            engine_kwargs["max_gen_toks"] = int(32768)  # Overwritten by lm-eval task's yaml config

            # Create individual jobs for each seed
            for seed in SEEDS:
                lm_eval_task_step = evaluate_lm_evaluation_harness(
                    full_model_name,
                    model_config["path"],
                    evals=[task],
                    max_eval_instances=None,
                    engine_kwargs=engine_kwargs,
                    resource_config=resource_config,
                    apply_chat_template=model_config.get("apply_chat_template", False),
                    wandb_tags=[
                        # Truncate to 64 chars (wandb tag limit)
                        base_name[:64],  # Base model name for grouping
                        full_model_name[:64],  # Full name with checkpoint for filtering
                    ],
                    generation_params={
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "do_sample": True,
                        "n": 1,  # Generate 1 sample per prompt
                        "seed": seed,
                    },
                    seed=seed,  # Append seed to step name
                )

                all_steps.append(lm_eval_task_step)

    # Add compile results step
    compile_step = compile_results(all_steps)
    all_steps.append(compile_step)

    executor_main(steps=all_steps)
