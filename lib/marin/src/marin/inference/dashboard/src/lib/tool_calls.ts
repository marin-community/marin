import type { ToolCall } from './types'
import { TOOL_CALL_TAG } from './python_tools'
import { newId } from './storage'

const INLINE_TOOL_CALL = new RegExp(`<${TOOL_CALL_TAG}>\\s*([\\s\\S]*?)\\s*</${TOOL_CALL_TAG}>`, 'g')

function parseToolCall(payload: string): ToolCall {
  const parsed: unknown = JSON.parse(payload)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Tool call must contain a JSON object')
  }
  const fields = parsed as Record<string, unknown>
  const name = fields.name
  if (typeof name !== 'string' || !name) throw new Error('Tool call must contain a function name')
  const arguments_ = fields.arguments ?? {}
  if (!arguments_ || typeof arguments_ !== 'object' || Array.isArray(arguments_)) {
    throw new Error('Tool call arguments must be a JSON object')
  }
  return { id: `call_${newId()}`, name, arguments: arguments_ as Record<string, unknown> }
}

/** Parse XML tool calls, raising when a tagged payload does not match the protocol. */
export function inlineToolCalls(content: string): { visible: string; calls: ToolCall[] } {
  const calls: ToolCall[] = []
  const visible = content.replace(INLINE_TOOL_CALL, (_match, payload: string) => {
    const call = parseToolCall(payload)
    calls.push(call)
    return ''
  })
  return { visible: visible.trim(), calls }
}
