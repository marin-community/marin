import type { ToolDefinition } from './python_tools'

export const ThinkingMode = {
  TemplateDefault: 'default',
  Enabled: 'enabled',
  Disabled: 'disabled',
} as const

export type ThinkingMode = (typeof ThinkingMode)[keyof typeof ThinkingMode]

export interface ChatTemplateRequestFields {
  tools?: ToolDefinition[]
  chat_template_kwargs?: Record<string, unknown>
}

/** Build the template arguments exposed by the chat UI. */
export function chatTemplateRequestFields(
  thinkingMode: ThinkingMode,
  customInstructions: string,
  tools: ToolDefinition[],
): ChatTemplateRequestFields {
  const chatTemplateArgs: Record<string, unknown> = {}
  if (thinkingMode === ThinkingMode.Enabled) chatTemplateArgs.enable_thinking = true
  if (thinkingMode === ThinkingMode.Disabled) chatTemplateArgs.enable_thinking = false
  if (customInstructions.trim()) chatTemplateArgs.custom_instructions = customInstructions.trim()

  const fields: ChatTemplateRequestFields = {}
  if (tools.length) fields.tools = tools
  if (Object.keys(chatTemplateArgs).length) fields.chat_template_kwargs = chatTemplateArgs
  return fields
}
