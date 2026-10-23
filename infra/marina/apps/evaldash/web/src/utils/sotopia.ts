export interface SotopiaSummary {
  transcript: { sender: string; receiver: string; message: string }[][]
  reasoning: string | null
  scores: Record<string, number | boolean>[]
  overall_scores: number[]
  evaluated_agent_index: number
  agent_ids: string[]
  models: string[]
  environment_id: string
  combo_id: string
  context_budget: {
    max_input_tokens: number
    max_prompt_tokens: number
    rejected_requests: number
  } | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isSotopiaSummary(value: unknown): value is SotopiaSummary {
  if (!isRecord(value) || !Array.isArray(value.transcript) || !Array.isArray(value.scores)) return false
  if (!Array.isArray(value.overall_scores) || !Array.isArray(value.agent_ids) || !Array.isArray(value.models)) {
    return false
  }
  if (typeof value.evaluated_agent_index !== 'number' ||
      typeof value.environment_id !== 'string' || typeof value.combo_id !== 'string' ||
      (value.reasoning !== null && typeof value.reasoning !== 'string')) return false
  if (!value.agent_ids.every((id: unknown) => typeof id === 'string') ||
      !value.models.every((model: unknown) => typeof model === 'string') ||
      !value.overall_scores.every((score: unknown) => typeof score === 'number') ||
      !value.scores.every((score: unknown) => isRecord(score) &&
        Object.values(score).every((field) => typeof field === 'number' || typeof field === 'boolean'))) return false
  const budget = value.context_budget
  if (budget !== null && (!isRecord(budget) || typeof budget.max_input_tokens !== 'number' ||
      typeof budget.max_prompt_tokens !== 'number' || typeof budget.rejected_requests !== 'number')) return false
  return value.transcript.every((turn: unknown) => Array.isArray(turn) && turn.every((message: unknown) =>
    isRecord(message) && typeof message.sender === 'string' &&
    typeof message.receiver === 'string' && typeof message.message === 'string',
  ))
}
