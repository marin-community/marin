import type { ToolCall } from './types'
import { newId } from './storage'

const INLINE_TOOL_CALL = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g

function callId(): string {
  return `call_${newId()}`
}

function parseToolCall(payload: string): ToolCall | null {
  try {
    const parsed = JSON.parse(payload)
    const functionPayload = parsed.function ?? parsed
    const name = functionPayload.name
    if (typeof name !== 'string' || !name) return null
    const rawArguments = functionPayload.arguments ?? functionPayload.parameters ?? {}
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
  if (calls.length) return { visible: visible.trim(), calls }

  const barePayload = visible.trim()
  if (barePayload.startsWith('{') && barePayload.endsWith('}')) {
    const call = parseToolCall(barePayload)
    if (call) return { visible: '', calls: [call] }
  }
  return { visible: visible.trim(), calls }
}
