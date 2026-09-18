import type { ThinkingMode } from './chat_template'

/** Output conventions discovered from the served model's active chat template. */
export interface ChatTemplateProtocol {
  thinking_start: string | null
  thinking_end: string | null
  tool_call_start: string | null
  tool_call_end: string | null
  tool_call_format: 'delimited' | 'json' | null
}

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
  chat_template_protocol: ChatTemplateProtocol
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
  /** Unparsed decoded text received in the response content field. */
  rawContent?: string
  /** Unparsed decoded text received in reasoning_content or reasoning. */
  rawReasoning?: string
  thinkingSeconds: number | null
  error: string | null
  toolCalls?: ToolCall[]
}

export interface ToolMessage {
  role: 'tool'
  result: unknown
  name: string
  toolCallId: string
}

export type ChatMessage = UserMessage | AssistantMessage | ToolMessage

export interface ShellWorkspace {
  filesJson: string
  history: string[]
  repositoryUrl: string
}

export interface Conversation {
  id: string
  title: string
  model: string
  system: string
  pythonTools: string
  shellWorkspace: ShellWorkspace | null
  thinkingMode: ThinkingMode
  customInstructions: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
}

export interface SamplingParams {
  temperature: number
  maxTokens: number
  topP: number
}
