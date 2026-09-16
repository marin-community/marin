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
}

export type ServerStatus = 'connecting' | 'ok' | 'loading' | 'bad'

export interface ToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
}

export interface UserMessage {
  role: 'user'
  content: string
}

export interface AssistantMessage {
  role: 'assistant'
  /** Visible text with any thinking segment stripped. */
  content: string
  thinking: string
  thinkingSeconds: number | null
  error: string | null
  toolCalls?: ToolCall[]
}

export interface ToolMessage {
  role: 'tool'
  content: string
  name: string
  toolCallId: string
}

export type ChatMessage = UserMessage | AssistantMessage | ToolMessage

export interface Conversation {
  id: string
  title: string
  model: string
  system: string
  pythonTools: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
}

export interface SamplingParams {
  temperature: number
  maxTokens: number
  topP: number
}
