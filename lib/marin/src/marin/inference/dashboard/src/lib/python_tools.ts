import type { ChatMessage, Conversation, ToolCall } from './types'

const CDATA_END = ']]>'
const CDATA_CONTINUATION = ']]]]><![CDATA[>'
export const TOOL_CALL_TAG = 'tool_call'

export interface ModelMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

function cdata(value: string): string {
  return value.split(CDATA_END).join(CDATA_CONTINUATION)
}

/** Build model history using the XML protocol understood by user-authored Python tools. */
export function modelMessages(conversation: Conversation, pythonTools: string): ModelMessage[] {
  const request: ModelMessage[] = []
  const system = [conversation.system.trim(), pythonTools ? pythonToolInstructions(pythonTools) : '']
    .filter(Boolean)
    .join('\n\n')
  if (system) request.push({ role: 'system', content: system })
  for (let index = 0; index < conversation.messages.length; index += 1) {
    const message = conversation.messages[index]
    if (message.role === 'assistant') {
      request.push({
        role: 'assistant',
        content: [message.content, ...(message.toolCalls ?? []).map(pythonToolCallMessage)].filter(Boolean).join('\n'),
      })
    } else if (message.role === 'tool') {
      const results = [pythonToolResultMessage(message)]
      while (conversation.messages[index + 1]?.role === 'tool') {
        index += 1
        results.push(pythonToolResultMessage(conversation.messages[index]))
      }
      request.push({ role: 'user', content: results.join('\n') })
    } else {
      request.push({ role: 'user', content: message.content })
    }
  }
  return request
}

function pythonToolInstructions(source: string): string {
  return `The user provided executable Python functions inside this XML block:
<python_tools><![CDATA[
${cdata(source)}
]]></python_tools>
Call a function only by emitting this exact XML form with JSON arguments:
<${TOOL_CALL_TAG}>{"name":"function_name","arguments":{"parameter":"value"}}</${TOOL_CALL_TAG}>
The application will execute the function and return a <tool_result> XML element in the next user message. Use that
result to answer the user or make another call. Do not invent functions outside the block.`
}

function pythonToolCallMessage(call: ToolCall): string {
  let arguments_: unknown = {}
  try {
    const parsed = JSON.parse(call.function.arguments)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) arguments_ = parsed
  } catch {
    // The matching tool result tells the model that its arguments were invalid.
  }
  return `<${TOOL_CALL_TAG}>${JSON.stringify({ name: call.function.name, arguments: arguments_ })}</${TOOL_CALL_TAG}>`
}

function pythonToolResultMessage(message: ChatMessage): string {
  let result: unknown = message.content
  try {
    result = JSON.parse(message.content)
  } catch {
    // Tool endpoints normally return JSON; preserve unexpected output as a string.
  }
  return `<tool_result><![CDATA[${cdata(JSON.stringify({ name: message.name, result }))}]]></tool_result>`
}
