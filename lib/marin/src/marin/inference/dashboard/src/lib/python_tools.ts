import type { Conversation, ToolCall, ToolMessage } from './types'

const CDATA_END = ']]>'
const CDATA_CONTINUATION = ']]]]><![CDATA[>'
export const TOOL_CALL_TAG = 'tool_call'
export const TOOL_RESULT_TAG = 'tool_result'

export interface ModelMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

function escapeCdataContent(value: string): string {
  return value.split(CDATA_END).join(CDATA_CONTINUATION)
}

/** Build model history using the dashboard's XML protocol for user-authored Python tools. */
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
        content: [message.content, ...(message.toolCalls ?? []).map(pythonToolCallXml)].filter(Boolean).join('\n'),
      })
    } else if (message.role === 'tool') {
      const results = [pythonToolResultXml(message)]
      while (true) {
        const next = conversation.messages[index + 1]
        if (!next || next.role !== 'tool') break
        index += 1
        results.push(pythonToolResultXml(next))
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
${escapeCdataContent(source)}
]]></python_tools>
Call a function only by emitting this exact XML form with JSON arguments:
<${TOOL_CALL_TAG}>{"name":"function_name","arguments":{"parameter":"value"}}</${TOOL_CALL_TAG}>
The application will execute the function and return a <${TOOL_RESULT_TAG}> XML element in the next user message. Use that
result to answer the user or make another call. Do not invent functions outside the block.`
}

function pythonToolCallXml(call: ToolCall): string {
  return `<${TOOL_CALL_TAG}>${JSON.stringify({ name: call.name, arguments: call.arguments })}</${TOOL_CALL_TAG}>`
}

function pythonToolResultXml(message: ToolMessage): string {
  const result: unknown = JSON.parse(message.content)
  return `<${TOOL_RESULT_TAG}><![CDATA[${escapeCdataContent(JSON.stringify({ name: message.name, result }))}]]></${TOOL_RESULT_TAG}>`
}
