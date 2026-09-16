export interface ChatExample {
  label: string
  prompt: string
  pythonTools?: string
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
    return round(principal * (1 + annual_rate_percent / 100) ** years, 2)`,
  },
  {
    label: 'Python tool · Word statistics',
    prompt: 'Use the Python tool to count the words and unique words in: the quick brown fox jumps over the lazy dog.',
    pythonTools: `def word_statistics(text: str) -> dict[str, int]:
    """Count all words and case-insensitive unique words."""
    words = text.split()
    return {"words": len(words), "unique_words": len({word.lower() for word in words})}`,
  },
]

/** Starter prefixes for completion mode (base models continue the text). */
export const COMPLETION_EXAMPLES = [
  'The capital of France is',
  'def fibonacci(n):',
  'Once upon a time,',
]
