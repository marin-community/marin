import type { ToolCall } from './types'
import { TOOL_CALL_TAG } from './python_tools'
import { newId } from './storage'

const INLINE_TOOL_CALL = new RegExp(`<${TOOL_CALL_TAG}>\\s*([\\s\\S]*?)\\s*</${TOOL_CALL_TAG}>`, 'g')

function callId(): string {
  return `call_${newId()}`
}

function parseToolCall(payload: string): ToolCall | null {
  try {
    const parsed = JSON.parse(payload)
    const name = parsed.name
    if (typeof name !== 'string' || !name) return null
    const rawArguments = parsed.arguments ?? {}
    return {
      id: callId(),
      type: 'function',
      function: {
        name,
        arguments: typeof rawArguments === 'string' ? rawArguments : JSON.stringify(rawArguments),
      },
    }
  } catch (error) {
    console.warn('failed to parse inline tool call', error)
    return null
  }
}

/** Parse XML tool calls and the compatible bare-JSON form emitted by some chat templates. */
export function inlineToolCalls(content: string): { visible: string; calls: ToolCall[] } {
  const calls: ToolCall[] = []
  const visible = content.replace(INLINE_TOOL_CALL, (_match, payload: string) => {
    const call = parseToolCall(payload)
    if (!call) return _match
    calls.push(call)
    return ''
  })
  return { visible: visible.trim(), calls }
}
