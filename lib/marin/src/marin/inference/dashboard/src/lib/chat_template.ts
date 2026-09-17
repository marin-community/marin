import type { ToolDefinition } from './python_tools'

export interface ChatTemplateRequestFields {
  tools?: ToolDefinition[]
  chat_template_kwargs?: Record<string, unknown>
}

/** Build the template arguments exposed by the chat UI. */
export function chatTemplateRequestFields(
  enableThinking: boolean | null,
  customInstructions: string,
  tools: ToolDefinition[],
): ChatTemplateRequestFields {
  const chatTemplateArgs: Record<string, unknown> = {}
  if (enableThinking !== null) chatTemplateArgs.enable_thinking = enableThinking
  if (customInstructions.trim()) chatTemplateArgs.custom_instructions = customInstructions.trim()

  const fields: ChatTemplateRequestFields = {}
  if (tools.length) fields.tools = tools
  if (Object.keys(chatTemplateArgs).length) fields.chat_template_kwargs = chatTemplateArgs
  return fields
}
