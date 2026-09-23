export interface ChatExample {
  label: string
  prompt: string
  pythonTools?: string
  workspaceFiles?: Record<string, string>
}

/** Starter prompts shown on an empty conversation. */
export const CHAT_EXAMPLES: ChatExample[] = [
  {
    label: 'Explain how a KV cache speeds up LLM inference.',
    prompt: 'Explain how a KV cache speeds up LLM inference.',
  },
  {
    label: 'Write a limerick about tensor parallelism.',
    prompt: 'Write a limerick about tensor parallelism.',
  },
  {
    label: 'Summarize the plot of Hamlet in three sentences.',
    prompt: 'Summarize the plot of Hamlet in three sentences.',
  },
  {
    label: 'What is 17 × 24? Think step by step.',
    prompt: 'What is 17 × 24? Think step by step.',
  },
  {
    label: 'Python tool · Compound interest',
    prompt: 'Use the Python tool to calculate the balance on $2,500 at 4.5% annual interest for 8 years.',
    pythonTools: `def compound_interest(principal: float, annual_rate_percent: float, years: int) -> float:
    """Return the balance after annual compounding."""
    return principal * (1 + annual_rate_percent / 100) ** years`,
  },
  {
    label: 'Python tool · Word statistics',
    prompt: 'Use the Python tool to count the words and unique words in: the quick brown fox jumps over the lazy dog.',
    pythonTools: `def word_statistics(text: str) -> dict[str, int]:
    """Count all words and case-insensitive unique words."""
    words = text.split()
    return {"words": len(words), "unique_words": len({word.lower() for word in words})}`,
  },
  {
    label: 'Agent · Fix a failing test',
    prompt:
      'Use the bash tool to inspect this repository, run python3.14 test_calculator.py, fix the bug, rerun the test, and show the Git diff before summarizing the change.',
    workspaceFiles: {
      'calculator.py': `def add(left: int, right: int) -> int:
    return left - right
`,
      'test_calculator.py': `from calculator import add

assert add(17, 24) == 41
print("test passed")
`,
    },
  },
  {
    label: 'Agent · Investigate an application log',
    prompt:
      'Use the bash tool to inspect the application log. Identify the failure, write a concise incident_summary.md with evidence and a recommended next step, then display the new file and Git diff.',
    workspaceFiles: {
      'README.md': '# Checkout service\n\nThis workspace contains a short production log for investigation.\n',
      'logs/application.log': `2026-09-17T10:14:20Z INFO checkout request_id=ab12 started
2026-09-17T10:14:21Z WARN payments request_id=ab12 retry=1 status=503
2026-09-17T10:14:22Z WARN payments request_id=ab12 retry=2 status=503
2026-09-17T10:14:24Z ERROR checkout request_id=ab12 failed reason="payment service unavailable"
`,
    },
  },
  {
    label: 'Agent · Repair a configuration',
    prompt:
      'Use the bash tool to run python3.14 check_config.py, update config.json so the check passes, rerun it, and show the Git diff before explaining what changed.',
    workspaceFiles: {
      'config.json': `{
  "workers": 2,
  "request_timeout_seconds": 5
}
`,
      'check_config.py': `import json

with open("config.json") as stream:
    config = json.load(stream)

assert config["workers"] == 4, "workers must be 4"
assert config["request_timeout_seconds"] == 30, "request timeout must be 30 seconds"
print("configuration is valid")
`,
    },
  },
]

/** Starter prefixes for completion mode (base models continue the text). */
export const COMPLETION_EXAMPLES = [
  'The capital of France is',
  'def fibonacci(n):',
  'Once upon a time,',
]
