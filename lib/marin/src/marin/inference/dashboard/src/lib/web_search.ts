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

export const WEB_EXTRACT_TOOL_NAME = 'web_extract'

export const WEB_EXTRACT_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: WEB_EXTRACT_TOOL_NAME,
    description:
      'Read web pages. Returns the parts of each page relevant to the objective, and the page text when full_content is true.',
    parameters: {
      type: 'object',
      properties: {
        urls: {
          type: 'array',
          items: { type: 'string' },
          minItems: 1,
          maxItems: 5,
          description: 'One to five URLs to read, for example from web_search results.',
        },
        objective: {
          type: 'string',
          description: 'What you need from the pages, in one sentence.',
        },
        full_content: {
          type: 'boolean',
          description: 'Also return the page text from the top of each page (truncated). Use only when excerpts are not enough.',
        },
      },
      required: ['urls', 'objective'],
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
