// One real retention-archive record: a SkyRL schema_v7 turn record (token-id lists trimmed).
export const SCHEMA_V7_STOP = {
  "disposition": {
    "error_treatment": null,
    "exception_type": null,
    "server_error": null
  },
  "global_step": null,
  "metrics": {
    "episode/assistant_turns": 1,
    "episode/conversations": 1,
    "episode/generated_tokens": 83,
    "episode/tool_calls": 0,
    "generate/avg_num_tokens": 83.0,
    "generate/avg_tokens_non_zero_rewards": 83.0,
    "generate/avg_tokens_zero_rewards": 0.0,
    "generate/failed_trajectory_fraction": 0.0,
    "generate/max_num_tokens": 83,
    "generate/min_num_tokens": 83,
    "generate/num_failed_instances": 0,
    "generate/num_failed_trajectories": 0,
    "generate/num_masked_trajectories": 0,
    "generate/num_trials": 1,
    "generate/optimized_avg_num_tokens": 83.0,
    "generate/outcome/graded": 1.0,
    "generate/outcome/infra_error": 0.0,
    "generate/outcome/invalid_task": 0.0,
    "generate/outcome/rollout_failure": 0.0,
    "generate/outcome/submission_failure": 0.0,
    "generate/outcome/unavailable": 0.0,
    "generate/std_num_tokens": 0.0,
    "generate/task_rollout/generated_tokens": 83.0,
    "generate/task_rollout/missing_logprob_tokens": 83.0,
    "generate/task_rollout/multi_turn_tasks": 0.0,
    "generate/task_rollout/tasks": 1.0,
    "generate/task_rollout/tool_tasks": 0.0,
    "generate/task_rollout/turns": 1.0,
    "timing/grading/max": 2.719920917006675,
    "timing/grading/mean": 2.719920917006675,
    "timing/grading/p50": 2.719920917006675,
    "timing/grading/p95": 2.719920917006675,
    "timing/setup/max": 0.4099281739909202,
    "timing/setup/mean": 0.4099281739909202,
    "timing/setup/p50": 0.4099281739909202,
    "timing/setup/p95": 0.4099281739909202
  },
  "phase": "eval",
  "prompt": {
    "messages": [
      {
        "content": "Question: There are 40 pupils in a class. Half of the class likes blue. One-fourth of the remaining likes green and the rest likes yellow. H",
        "role": "user"
      },
      {
        "content": "Give your answer as plain text.",
        "role": "user"
      }
    ],
    "token_ids": [
      128000,
      128006,
      9125,
      128007
    ]
  },
  "provenance": {
    "inference_backend": "vllm",
    "model_path": "/tmp/marinskyrl/models/model",
    "model_source_identity": "micro210-reval-exp1-rl16/model@2026.11.24.28:c557835e",
    "model_version_step": null,
    "resume_path": null,
    "reward_shaping_schema_version": null,
    "runner": "TaskRolloutWorker",
    "sampling": {
      "ignore_eos": false,
      "include_stop_str_in_output": true,
      "logprobs": null,
      "max_tokens": 16384,
      "min_p": 0.0,
      "min_tokens": 1,
      "repetition_penalty": 1.0,
      "seed": null,
      "skip_special_tokens": false,
      "stop": null,
      "stop_token_ids": [
        128001,
        128009
      ],
      "temperature": 0.7,
      "top_k": -1,
      "top_p": 0.95
    }
  },
  "record_id": "ee260d9873060396c4bd87dfdfd2f653e793c6eb3b9fed1bc8e9b5effedaa3be",
  "response": {
    "generation_limit": 16384,
    "loop_spans": [],
    "loss_mask": [
      1,
      1,
      1,
      1
    ],
    "messages": [
      {
        "content": "Question: There are 40 pupils in a class. Half of the class likes blue. One-fourth of the remaining likes green and the rest likes yellow. H",
        "role": "user"
      },
      {
        "content": "Give your answer as plain text.",
        "role": "user"
      },
      {
        "annotations": null,
        "audio": null,
        "content": "15",
        "function_call": null,
        "reasoning": null,
        "refusal": null,
        "role": "assistant"
      }
    ],
    "step_boundaries": [
      {
        "is_last_step": null,
        "prompt_token_ids": [
          128000,
          128006,
          9125,
          128007
        ],
        "row_index": 0,
        "stop_reason": "stop",
        "token_end": 83,
        "token_start": 0
      }
    ],
    "stop_reason": "stop",
    "text": "<|start_think|>We need to solve the problem. There are 40 pupils. Half like blue: 40/2 = 20 like blue. Remaining = 40 - 20 = 20. One-fourth ",
    "token_ids": [
      128002,
      1687,
      1205,
      311
    ],
    "trainable_spans": [
      {
        "end": 83,
        "start": 0
      }
    ]
  },
  "reward": {
    "components": null,
    "outcome": 1.0,
    "shaped": 1.0
  },
  "run_id": "micro210-reval-exp1-rl16@2026.11.24.28",
  "schema_version": 7,
  "trajectory": {
    "environment_class": "taskcompendium",
    "environment_extras": {
      "data_source": "openai/gsm8k",
      "teacher_route": null
    },
    "instance_id": "eval-None-0--399",
    "repetition_id": 0,
    "row_indices": [
      0
    ]
  },
  "verification_result": {
    "diagnostics": {
      "exit_code": 0,
      "stderr": "/tests/grader_lib/lm_eval/tasks/hendrycks_math/utils.py:169: SyntaxWarning: invalid escape sequence '\\%'\n  string = string.replace(\"\\%\", \"\")",
      "stderr_truncated": false,
      "stdout": "1.0\n",
      "stdout_truncated": false
    },
    "passed": null,
    "reason": null,
    "score": 1.0,
    "score_max": 1.0,
    "score_min": 0.0,
    "status": "verified"
  },
  "verifier": null
};

// A real length-stopped schema_v7 record: the model ran out of tokens mid-thinking.
export const SCHEMA_V7_LENGTH = {
  "disposition": {
    "error_treatment": null,
    "exception_type": null,
    "server_error": null
  },
  "global_step": null,
  "metrics": {
    "episode/assistant_turns": 1,
    "episode/conversations": 1,
    "episode/generated_tokens": 16384,
    "episode/tool_calls": 0,
    "generate/avg_num_tokens": 16384.0,
    "generate/avg_tokens_non_zero_rewards": 0.0,
    "generate/avg_tokens_zero_rewards": 16384.0,
    "generate/failed_trajectory_fraction": 1.0,
    "generate/max_num_tokens": 16384,
    "generate/min_num_tokens": 16384,
    "generate/num_failed_instances": 1,
    "generate/num_failed_trajectories": 1,
    "generate/num_masked_trajectories": 0,
    "generate/num_trials": 1,
    "generate/optimized_avg_num_tokens": 16384.0,
    "generate/outcome/graded": 0.0,
    "generate/outcome/infra_error": 0.0,
    "generate/outcome/invalid_task": 0.0,
    "generate/outcome/rollout_failure": 0.0,
    "generate/outcome/submission_failure": 1.0,
    "generate/outcome/unavailable": 0.0,
    "generate/std_num_tokens": 0.0,
    "generate/task_rollout/generated_tokens": 16384.0,
    "generate/task_rollout/missing_logprob_tokens": 16384.0,
    "generate/task_rollout/multi_turn_tasks": 0.0,
    "generate/task_rollout/tasks": 1.0,
    "generate/task_rollout/tool_tasks": 0.0,
    "generate/task_rollout/turns": 1.0,
    "timing/grading/max": 0.0006597019964829087,
    "timing/grading/mean": 0.0006597019964829087,
    "timing/grading/p50": 0.0006597019964829087,
    "timing/grading/p95": 0.0006597019964829087,
    "timing/setup/max": 0.0,
    "timing/setup/mean": 0.0,
    "timing/setup/p50": 0.0,
    "timing/setup/p95": 0.0
  },
  "phase": "eval",
  "prompt": {
    "messages": [
      {
        "content": "Problem: $g=9.8 \\mathrm{~N} / \\mathrm{kg}$\n\n\n\n$k=1 / 4 \\pi \\epsilon_{0}=8.99 \\times 10^{9} \\mathrm{~N} \\cdot \\mathrm{m}^{2} / \\mathrm{C}^{2}",
        "role": "user"
      },
      {
        "content": "Give your answer as plain text.",
        "role": "user"
      }
    ],
    "token_ids": [
      128000,
      128006,
      9125,
      128007
    ]
  },
  "provenance": {
    "inference_backend": "vllm",
    "model_path": "/tmp/marinskyrl/models/model",
    "model_source_identity": "micro210-reval-exp1-rl16/model@2026.11.24.28:c557835e",
    "model_version_step": null,
    "resume_path": null,
    "reward_shaping_schema_version": null,
    "runner": "TaskRolloutWorker",
    "sampling": {
      "ignore_eos": false,
      "include_stop_str_in_output": true,
      "logprobs": null,
      "max_tokens": 16384,
      "min_p": 0.0,
      "min_tokens": 1,
      "repetition_penalty": 1.0,
      "seed": null,
      "skip_special_tokens": false,
      "stop": null,
      "stop_token_ids": [
        128001,
        128009
      ],
      "temperature": 0.7,
      "top_k": -1,
      "top_p": 0.95
    }
  },
  "record_id": "14ac91de7cfa801e20c9f0306ccdafa5b8996c4f3880e8445c1efeba068b508f",
  "response": {
    "generation_limit": 16384,
    "loop_spans": [],
    "loss_mask": [
      1,
      1,
      1,
      1
    ],
    "messages": [
      {
        "content": "Problem: $g=9.8 \\mathrm{~N} / \\mathrm{kg}$\n\n\n\n$k=1 / 4 \\pi \\epsilon_{0}=8.99 \\times 10^{9} \\mathrm{~N} \\cdot \\mathrm{m}^{2} / \\mathrm{C}^{2}",
        "role": "user"
      },
      {
        "content": "Give your answer as plain text.",
        "role": "user"
      },
      {
        "annotations": null,
        "audio": null,
        "content": "",
        "function_call": null,
        "reasoning": null,
        "refusal": null,
        "role": "assistant"
      }
    ],
    "step_boundaries": [
      {
        "is_last_step": null,
        "prompt_token_ids": [
          128000,
          128006,
          9125,
          128007
        ],
        "row_index": 0,
        "stop_reason": "length",
        "token_end": 16384,
        "token_start": 0
      }
    ],
    "stop_reason": "length",
    "text": "<|start_think|>The user wants me to solve a physics problem based on the provided context.\nThe context includes a problem about a ship's com",
    "token_ids": [
      128002,
      791,
      1217,
      6944
    ],
    "trainable_spans": [
      {
        "end": 16384,
        "start": 0
      }
    ]
  },
  "reward": {
    "components": null,
    "outcome": 0.0,
    "shaped": 0.0
  },
  "run_id": "micro210-reval-exp1-rl16@2026.11.24.28",
  "schema_version": 7,
  "trajectory": {
    "environment_class": "taskcompendium",
    "environment_extras": {
      "data_source": "lmms-lab/olympiadbench",
      "teacher_route": null
    },
    "instance_id": "eval-None-0--407",
    "repetition_id": 0,
    "row_indices": [
      0
    ]
  },
  "verification_result": {
    "diagnostics": {},
    "passed": null,
    "reason": null,
    "score": 0.0,
    "score_max": 1.0,
    "score_min": 0.0,
    "status": "verified"
  },
  "verifier": null
};

