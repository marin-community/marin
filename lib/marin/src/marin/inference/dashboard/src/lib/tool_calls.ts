import type { ToolCall } from './types'

const INLINE_TOOL_CALL = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g

interface ToolCallDelta {
  index?: number
  id?: string
  type?: string
  function?: {
    name?: string
    arguments?: string | Record<string, unknown>
  }
}

function callId(): string {
  return crypto.randomUUID ? `call_${crypto.randomUUID()}` : `call_${Date.now()}_${Math.random().toString(36).slice(2)}`
}

/** Merge OpenAI streaming tool-call deltas by their stable choice index. */
export function mergeToolCallDeltas(current: ToolCall[], deltas: ToolCallDelta[]): ToolCall[] {
  const merged = current.map((call) => ({ ...call, function: { ...call.function } }))
  for (const [position, delta] of deltas.entries()) {
    const index = delta.index ?? position
    const existing = merged[index] ?? {
      id: '',
      type: 'function' as const,
      function: { name: '', arguments: '' },
    }
    if (delta.id) existing.id = delta.id
    if (delta.function?.name) existing.function.name += delta.function.name
    if (delta.function?.arguments !== undefined) {
      existing.function.arguments +=
        typeof delta.function.arguments === 'string'
          ? delta.function.arguments
          : JSON.stringify(delta.function.arguments)
    }
    merged[index] = existing
  }
  return merged
}

/** Parse raw chat-template tool tags used when a backend has no structured parser. */
export function inlineToolCalls(content: string): { visible: string; calls: ToolCall[] } {
  const calls: ToolCall[] = []
  const visible = content.replace(INLINE_TOOL_CALL, (_match, payload: string) => {
    try {
      const parsed = JSON.parse(payload)
      const functionPayload = parsed.function ?? parsed
      const name = functionPayload.name
      if (typeof name !== 'string' || !name) return _match
      const rawArguments = functionPayload.arguments ?? functionPayload.parameters ?? {}
      calls.push({
        id: callId(),
        type: 'function',
        function: {
          name,
          arguments: typeof rawArguments === 'string' ? rawArguments : JSON.stringify(rawArguments),
        },
      })
      return ''
    } catch {
      return _match
    }
  })
  return { visible: visible.trim(), calls }
}

/** Fill optional IDs and discard malformed calls that cannot be executed. */
export function executableToolCalls(calls: ToolCall[]): ToolCall[] {
  return calls
    .filter((call) => call.function.name)
    .map((call) => ({
      ...call,
      id: call.id || callId(),
      type: 'function',
      function: {
        name: call.function.name,
        arguments: call.function.arguments || '{}',
      },
    }))
}
