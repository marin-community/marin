import type { ToolDefinition } from './python_tools'
import type { WebSearchAvailability } from './types'

export const WEB_SEARCH_TOOL_NAME = 'web_search'

export const WEB_SEARCH_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: WEB_SEARCH_TOOL_NAME,
    description: 'Search the web. Returns ranked URLs with titles and excerpts relevant to the objective.',
    parameters: {
      type: 'object',
      properties: {
        objective: {
          type: 'string',
          description: 'What you are trying to find, in one sentence.',
        },
        search_queries: {
          type: 'array',
          items: { type: 'string' },
          description: 'Two or three keyword search queries.',
        },
      },
      required: ['objective', 'search_queries'],
      additionalProperties: false,
    },
  },
}

/** Why the Web search option is disabled, or null when Parallel accepts searches (or search is not configured). */
export function webSearchUnavailableReason(availability: WebSearchAvailability | null): string | null {
  switch (availability) {
    case 'no_credit':
      return 'Web search is unavailable: the Parallel account has no credit.'
    case 'key_rejected':
      return "Web search is unavailable: Parallel rejected this server's API key."
    default:
      return null
  }
}
