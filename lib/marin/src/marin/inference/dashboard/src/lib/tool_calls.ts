import type { ChatTemplateProtocol, ToolCall } from './types'

const TOOL_CALL_ID_PREFIX = 'call_'

interface PendingToolCall {
  id: string | null
  name: string
  argumentsText: string
  argumentsObject: Record<string, unknown> | null
}

export interface ToolCallAccumulator {
  calls: Map<number, PendingToolCall>
}

function objectValue(value: unknown, message: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(message)
  return value as Record<string, unknown>
}

function toolArguments(value: unknown): Record<string, unknown> {
  const parsed = typeof value === 'string' ? JSON.parse(value) : value
  return objectValue(parsed, 'Tool call arguments must be a JSON object')
}

function parseToolCall(payload: string, newId: () => string): ToolCall {
  const fields = objectValue(JSON.parse(payload), 'Tool call must contain a JSON object')
  const functionFields = fields.function === undefined ? fields : objectValue(fields.function, 'Invalid function call')
  const name = functionFields.name
  if (typeof name !== 'string' || !name) throw new Error('Tool call must contain a function name')
  const arguments_ = functionFields.arguments ?? functionFields.parameters ?? functionFields.args
  if (arguments_ === undefined) throw new Error('Tool call must contain function arguments')
  return {
    id: typeof fields.id === 'string' ? fields.id : `${TOOL_CALL_ID_PREFIX}${newId()}`,
    name,
    arguments: toolArguments(arguments_),
  }
}

/** Parse inline tool calls using the active template's output format. */
export function inlineToolCalls(
  content: string,
  protocol: ChatTemplateProtocol | null,
  newId: () => string,
): { visible: string; calls: ToolCall[] } {
  if (protocol?.tool_call_format === 'json') {
    const payload = content.trim()
    if (!payload.startsWith('{') || !payload.endsWith('}')) return { visible: content, calls: [] }
    try {
      return { visible: '', calls: [parseToolCall(payload, newId)] }
    } catch {
      return { visible: content, calls: [] }
    }
  }

  const start = protocol?.tool_call_start
  const end = protocol?.tool_call_end
  if (!start || !end) return { visible: content, calls: [] }

  const escapedStart = start.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const escapedEnd = end.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const pattern = new RegExp(`${escapedStart}\\s*([\\s\\S]*?)\\s*${escapedEnd}`, 'g')
  const calls: ToolCall[] = []
  const visible = content.replace(pattern, (_match, payload: string) => {
    const call = parseToolCall(payload, newId)
    calls.push(call)
    return ''
  })
  return { visible: visible.trim(), calls }
}

export function createToolCallAccumulator(): ToolCallAccumulator {
  return { calls: new Map() }
}

/** Accumulate OpenAI tool-call deltas, whose JSON arguments may span SSE events. */
export function appendToolCallDelta(accumulator: ToolCallAccumulator, value: unknown): void {
  if (!Array.isArray(value)) throw new Error('Model tool_calls must be an array')
  for (const [position, item] of value.entries()) {
    const fields = objectValue(item, 'Model tool call must be an object')
    const index = typeof fields.index === 'number' ? fields.index : position
    const pending = accumulator.calls.get(index) ?? {
      id: null,
      name: '',
      argumentsText: '',
      argumentsObject: null,
    }
    if (typeof fields.id === 'string') pending.id = fields.id
    if (fields.function !== undefined) {
      const functionFields = objectValue(fields.function, 'Model tool call function must be an object')
      if (typeof functionFields.name === 'string') pending.name += functionFields.name
      if (typeof functionFields.arguments === 'string') {
        pending.argumentsText += functionFields.arguments
      } else if (functionFields.arguments !== undefined) {
        pending.argumentsObject = toolArguments(functionFields.arguments)
      }
    }
    accumulator.calls.set(index, pending)
  }
}

/** Validate accumulated OpenAI calls once the response stream has finished. */
export function finalizeToolCalls(accumulator: ToolCallAccumulator, newId: () => string): ToolCall[] {
  return [...accumulator.calls.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, pending]) => {
      if (!pending.name) throw new Error('Tool call must contain a function name')
      const arguments_ = pending.argumentsObject ?? toolArguments(pending.argumentsText)
      return {
        id: pending.id ?? `${TOOL_CALL_ID_PREFIX}${newId()}`,
        name: pending.name,
        arguments: arguments_,
      }
    })
}
