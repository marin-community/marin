/** Static serving metadata returned by the dashboard server's /info route. */
export interface ServingInfo {
  model: string
  backend: string
  tensor_parallel_size: number
  max_model_len: number | null
  dtype: string
  has_chat_template: boolean
  endpoint: string
  streaming: boolean
  tools: ToolDefinition[]
}

export type ServerStatus = 'connecting' | 'ok' | 'loading' | 'bad'

export interface ToolDefinition {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown>
  }
}

export interface ToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'tool'
  /** Visible text with any thinking segment stripped. */
  content: string
  thinking: string
  thinkingSeconds: number | null
  error: string | null
  name?: string
  toolCallId?: string
  toolCalls?: ToolCall[]
}

export interface Conversation {
  id: string
  title: string
  model: string
  system: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
}

export interface SamplingParams {
  temperature: number
  maxTokens: number
  topP: number
}
