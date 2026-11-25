import test from 'node:test'
import assert from 'node:assert/strict'

import {
  isSkyrlEvalDumpRow,
  isSkyrlTurnRecord,
  skyrlTurnsToTrajectory,
} from '../src/utils/skyrlTrajectory.ts'

// One real published blob: the eval-dump row for bfclparity-pi doc 12 (token-id lists trimmed).
// Its assistant turn carries a tool call with null content, followed by the tool's reply.
const EVAL_DUMP_ROW = {
  "uid": "12",
  "attempt_id": "a73813cd0729",
  "step": -1,
  "ordinal": 87,
  "is_last_step": true,
  "identity_disposition": "task_spec",
  "task_id": "bfclparity-pi/bfcl-multiple-16",
  "prompt_token_ids": [
    128000,
    128006,
    9125,
    128007
  ],
  "response_token_ids": [
    128005,
    198,
    5018,
    609
  ],
  "input_prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /think\n### Tools\n\nYou may call one or more functions to assist with th",
  "output_response": "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo '[{\\\"population_genetics.calculate_ne\\\": {\\\"species\\\": \\\"tiger\\\", \\\"generations\\",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "# Task\n\nWhat's the required minimum population size (Ne) for maintaining the genetic diversity of a wild tiger population for the next 100 g"
        }
      ]
    },
    {
      "role": "assistant",
      "content": null,
      "refusal": null,
      "annotations": null,
      "audio": null,
      "function_call": null,
      "tool_calls": [
        {
          "id": "chatcmpl-tool-b14c4605f90e32a0",
          "type": "function",
          "function": {
            "name": "bash",
            "arguments": "{\"command\": \"echo '[{\\\"population_genetics.calculate_ne\\\": {\\\"species\\\": \\\"tiger\\\"}}]' > /app/result.json\"}"
          }
        }
      ],
      "reasoning": null
    },
    {
      "role": "tool",
      "content": "(no output)",
      "tool_call_id": "chatcmpl-tool-b14c4605f90e32a0"
    },
    {
      "role": "assistant",
      "content": "The function call has been written to `/app/result.json`. The required minimum effective population size (Ne) for maintaining the genetic di",
      "refusal": null,
      "annotations": null,
      "audio": null,
      "function_call": null,
      "reasoning": null
    }
  ],
  "score": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "stop_reason": "stop",
  "exception_type": null,
  "error_treatment": null,
  "server_error": null,
  "env_class": "harbor_trial",
  "data_source": "bfclparity-pi",
  "env_metrics": {
    "rollout_retries": 0.0,
    "truncation_penalized": 0.0
  },
  "verification": {
    "status": "verified",
    "reason": null,
    "score": 1.0,
    "passed": true,
    "score_min": 0.0,
    "score_max": 1.0,
    "diagnostics": {}
  }
};

// The same shape with the writer's null `messages` (evidence messages absent).
const EVAL_DUMP_ROW_NULL_MESSAGES = {
  "uid": "13",
  "attempt_id": "a73813cd0729",
  "step": -1,
  "ordinal": 87,
  "is_last_step": true,
  "identity_disposition": "task_spec",
  "task_id": "bfclparity-pi/bfcl-multiple-16",
  "prompt_token_ids": [
    128000,
    128006,
    9125,
    128007
  ],
  "response_token_ids": [
    128005,
    198,
    5018,
    609
  ],
  "input_prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /think\n### Tools\n\nYou may call one or more functions to assist with th",
  "output_response": "<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"echo '[{\\\"population_genetics.calculate_ne\\\": {\\\"species\\\": \\\"tiger\\\", \\\"generations\\",
  "messages": null,
  "score": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "stop_reason": "stop",
  "exception_type": null,
  "error_treatment": null,
  "server_error": null,
  "env_class": "harbor_trial",
  "data_source": "bfclparity-pi",
  "env_metrics": {
    "rollout_retries": 0.0,
    "truncation_penalized": 0.0
  },
  "verification": {
    "status": "verified",
    "reason": null,
    "score": 1.0,
    "passed": true,
    "score_min": 0.0,
    "score_max": 1.0,
    "diagnostics": {}
  }
};

// One real retention-archive record: a SkyRL schema_v7 turn record (token-id lists trimmed).
const SCHEMA_V7_STOP = {
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
const SCHEMA_V7_LENGTH = {
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

test('the two real record shapes are recognized; other artifact shapes are not', () => {
  assert.equal(isSkyrlEvalDumpRow(EVAL_DUMP_ROW), true)
  assert.equal(isSkyrlEvalDumpRow(EVAL_DUMP_ROW_NULL_MESSAGES), true)
  assert.equal(isSkyrlTurnRecord(SCHEMA_V7_STOP), true)
  assert.equal(isSkyrlTurnRecord(EVAL_DUMP_ROW), false)
  assert.equal(isSkyrlEvalDumpRow(SCHEMA_V7_STOP), false)
  assert.notEqual(skyrlTurnsToTrajectory(EVAL_DUMP_ROW), null)
  assert.notEqual(skyrlTurnsToTrajectory(SCHEMA_V7_STOP), null)
  assert.equal(skyrlTurnsToTrajectory({ steps: [] }), null)
  assert.equal(skyrlTurnsToTrajectory('nonsense'), null)
  assert.equal(skyrlTurnsToTrajectory([]), null)
})

test('an eval-dump row renders tool-carrying assistant turns, the tool reply, and the raw response', () => {
  const trajectory = skyrlTurnsToTrajectory(EVAL_DUMP_ROW)
  const sources = trajectory.steps.map((step) => step.source)
  assert.deepEqual(sources, ['user', 'system', 'user', 'agent', 'tool', 'agent', 'agent'])
  const callStep = trajectory.steps[3]
  assert.equal(callStep.tool_calls.length, 1)
  assert.equal(callStep.tool_calls[0].tool_call_id, 'chatcmpl-tool-b14c4605f90e32a0')
  assert.equal(callStep.tool_calls[0].function_name, 'bash')
  assert.ok(String(callStep.tool_calls[0].arguments.command).includes('result.json'))
  assert.equal(trajectory.steps[4].message, '(no output)')
  const raw = trajectory.steps.at(-1)
  assert.ok(raw.message.includes('<tool_call>'))
  assert.equal(raw.annotations['verification'], 'verified')
  assert.equal(raw.annotations['verifier score'], '1')
  assert.equal(trajectory.schema_version, 'skyrl-eval-dump')
  assert.equal(trajectory.session_id, 'a73813cd0729')
})

test('a null-messages eval-dump row still renders its prompt and outcome', () => {
  const trajectory = skyrlTurnsToTrajectory(EVAL_DUMP_ROW_NULL_MESSAGES)
  assert.equal(trajectory.steps.length, 2)
  assert.equal(trajectory.steps[0].annotations['content'], 'raw input prompt')
  assert.equal(trajectory.steps[1].annotations['verification'], 'verified')
})

test('a schema_v7 turn record renders its conversation, raw response, and outcome', () => {
  const trajectory = skyrlTurnsToTrajectory(SCHEMA_V7_STOP)
  const sources = trajectory.steps.map((step) => step.source)
  assert.deepEqual(sources, ['user', 'user', 'agent', 'agent'])
  assert.equal(trajectory.steps[2].message, '15')
  const raw = trajectory.steps.at(-1)
  assert.ok(raw.message.includes('<|start_think|>'))
  assert.equal(raw.annotations['stop reason'], 'stop')
  assert.equal(raw.annotations['verifier score'], '1')
  assert.equal(trajectory.schema_version, 'skyrl-turn-v7')
  assert.equal(trajectory.agent.name, 'taskcompendium')
  assert.equal(trajectory.session_id, 'micro210-reval-exp1-rl16@2026.11.24.28')
})

test('a length-stopped schema_v7 record shows its stop_reason', () => {
  const trajectory = skyrlTurnsToTrajectory(SCHEMA_V7_LENGTH)
  const raw = trajectory.steps.at(-1)
  assert.equal(raw.annotations['stop reason'], 'length')
  assert.equal(raw.annotations['verifier score'], '0')
  assert.ok(raw.message.includes('<|start_think|>'))
})

test('a list spanning uids groups by uid and renders the viewed sample\'s group', () => {
  const stopUid = SCHEMA_V7_STOP.trajectory.instance_id
  const lengthUid = SCHEMA_V7_LENGTH.trajectory.instance_id
  assert.notEqual(stopUid, lengthUid)
  // No preferred uid: the first record's group renders, and the other uid's content stays out.
  const first = skyrlTurnsToTrajectory([SCHEMA_V7_STOP, SCHEMA_V7_LENGTH])
  assert.ok(first.steps.every((step) => !(step.message ?? '').includes('physics problem')))
  assert.equal(first.steps.at(-1).annotations['stop reason'], 'stop')
  // The viewed sample's uid selects its own group.
  const viewed = skyrlTurnsToTrajectory([SCHEMA_V7_STOP, SCHEMA_V7_LENGTH], lengthUid)
  assert.equal(viewed.steps.at(-1).annotations['stop reason'], 'length')
  assert.equal(viewed.agent.name, 'taskcompendium')
})

test('a list of cumulative per-step records does not duplicate earlier turns', () => {
  const first = {
    ...EVAL_DUMP_ROW,
    uid: 'cumulative',
    ordinal: 0,
    input_prompt: '',
    output_response: '<|start_think|>reading the task<|end_think|>',
    messages: [
      { role: 'user', content: 'Compare two numbers digit by digit.' },
      { role: 'assistant', content: 'Reading the task.<|end_think|>' },
    ],
  }
  const second = {
    ...first,
    ordinal: 1,
    messages: [...first.messages, { role: 'assistant', content: '7808931378968545' }],
  }
  const trajectory = skyrlTurnsToTrajectory([first, second])
  const texts = trajectory.steps.map((step) => step.message)
  assert.equal(texts.filter((text) => text === 'Reading the task.<|end_think|>').length, 1)
  assert.equal(texts.filter((text) => text === '7808931378968545').length, 1)
  // each record still reports its own outcome
  const rawSteps = trajectory.steps.filter((step) => step.annotations?.['stop reason'])
  assert.equal(rawSteps.length, 2)
})
