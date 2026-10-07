import type { Conversation, ToolCall, ToolMessage } from './types'

export interface ToolDefinition {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown>
  }
}

interface ModelToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

export type ModelMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string; tool_calls?: ModelToolCall[] }
  | { role: 'tool'; name: string; tool_call_id: string; content: string }

/** Build structured history for the served model's active chat template. */
export function modelMessages(conversation: Conversation): ModelMessage[] {
  const request: ModelMessage[] = []
  const system = conversation.system.trim()
  if (system) request.push({ role: 'system', content: system })
  for (const message of conversation.messages) {
    if (message.role === 'assistant') {
      const assistant: Extract<ModelMessage, { role: 'assistant' }> = {
        role: 'assistant',
        content: message.content,
      }
      if (message.toolCalls?.length) assistant.tool_calls = message.toolCalls.map(modelToolCall)
      request.push(assistant)
    } else if (message.role === 'tool') {
      request.push(modelToolResult(message))
    } else {
      request.push({ role: 'user', content: message.content })
    }
  }
  return request
}

function modelToolCall(call: ToolCall): ModelToolCall {
  return {
    id: call.id,
    type: 'function',
    function: { name: call.name, arguments: JSON.stringify(call.arguments) },
  }
}

function modelToolResult(message: ToolMessage): Extract<ModelMessage, { role: 'tool' }> {
  return {
    role: 'tool',
    name: message.name,
    tool_call_id: message.toolCallId,
    content: JSON.stringify(message.result) ?? 'null',
  }
}
