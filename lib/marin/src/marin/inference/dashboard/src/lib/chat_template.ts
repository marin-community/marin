import type { ToolDefinition } from './python_tools'

export const ThinkingMode = {
  TemplateDefault: 'default',
  Enabled: 'enabled',
  Disabled: 'disabled',
} as const

export type ThinkingMode = (typeof ThinkingMode)[keyof typeof ThinkingMode]

export interface ChatRequestFields {
  skip_special_tokens: false
  tools?: ToolDefinition[]
  chat_template_kwargs?: Record<string, unknown>
}

/** Build the model-facing fields required by the chat UI. */
export function chatRequestFields(
  thinkingMode: ThinkingMode,
  customInstructions: string,
  tools: ToolDefinition[],
): ChatRequestFields {
  const chatTemplateArgs: Record<string, unknown> = {}
  if (thinkingMode === ThinkingMode.Enabled) chatTemplateArgs.enable_thinking = true
  if (thinkingMode === ThinkingMode.Disabled) chatTemplateArgs.enable_thinking = false
  if (customInstructions.trim()) chatTemplateArgs.custom_instructions = customInstructions.trim()

  // The response parser needs model-native thinking and tool-call delimiters.
  const fields: ChatRequestFields = { skip_special_tokens: false }
  if (tools.length) fields.tools = tools
  if (Object.keys(chatTemplateArgs).length) fields.chat_template_kwargs = chatTemplateArgs
  return fields
}
