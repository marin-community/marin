import type { ToolDefinition } from './python_tools'

export const BASH_TOOL_NAME = 'bash'

export const BASH_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: BASH_TOOL_NAME,
    description:
      'Run a Bash command in the persistent workspace rooted at /work. Use it to inspect and edit files, run programs and tests, and inspect Git state. The workspace has no network access.',
    parameters: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description: 'The Bash command to run from /work.',
        },
      },
      required: ['command'],
      additionalProperties: false,
    },
  },
}

export function parseWorkspaceFiles(source: string): Record<string, string> {
  const value: unknown = JSON.parse(source)
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Shell workspace files must be a JSON object mapping paths to text')
  }
  const files = value as Record<string, unknown>
  if (!Object.values(files).every((content) => typeof content === 'string')) {
    throw new Error('Shell workspace file contents must be strings')
  }
  return files as Record<string, string>
}

export function bashCommand(arguments_: Record<string, unknown>): string {
  const command = arguments_.command
  if (typeof command !== 'string' || !command.trim()) {
    throw new Error('Bash tool requires a non-empty command string')
  }
  return command
}
