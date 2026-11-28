// One real published blob: the eval-dump row for bfclparity-pi doc 12 (token-id lists trimmed).
// Its assistant turn carries a tool call with null content, followed by the tool's reply.
export const EVAL_DUMP_ROW = {
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
export const EVAL_DUMP_ROW_NULL_MESSAGES = {
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

