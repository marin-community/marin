import type { AssistantMessage, Conversation, ToolCall } from './types'

/** Render the complete conversation as one plain-text debugging transcript. */
export function plainTextChat(conversation: Conversation): string {
  const sections: string[] = []
  if (conversation.system.trim()) sections.push(section('system', conversation.system))

  for (const message of conversation.messages) {
    if (message.role === 'user') {
      sections.push(section('user', message.content))
      continue
    }
    if (message.role === 'tool') {
      sections.push(section(`tool ${message.name} (${message.toolCallId})`, stringify(message.result)))
      continue
    }
    appendAssistantSections(sections, message)
  }
  return sections.join('\n\n')
}

function appendAssistantSections(sections: string[], message: AssistantMessage) {
  const hasRawResponse = message.rawContent !== undefined || message.rawReasoning !== undefined
  const reasoning = hasRawResponse ? message.rawReasoning : message.thinking
  const content = hasRawResponse ? message.rawContent : message.content

  if (reasoning) sections.push(section('assistant reasoning', reasoning))
  if (content) sections.push(section('assistant', content))
  if (message.toolCalls?.length) sections.push(section('assistant tool_calls', stringifyToolCalls(message.toolCalls)))
  if (message.error) sections.push(section('assistant error', message.error))
  if (!reasoning && !content && !message.toolCalls?.length && !message.error) sections.push(section('assistant', ''))
}

function stringifyToolCalls(calls: ToolCall[]): string {
  return stringify(
    calls.map((call) => ({
      id: call.id,
      name: call.name,
      arguments: call.arguments,
    })),
  )
}

function stringify(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? 'null'
}

function section(role: string, content: string): string {
  return `[${role}]\n${content}`
}
