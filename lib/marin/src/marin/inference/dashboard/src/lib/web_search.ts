import type { ToolDefinition } from './python_tools'

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
