import type { ImportedGitCommit, ServingInfo } from './types'
import type { ToolDefinition } from './python_tools'
import type { SharedChatSnapshot } from './shared_chat'

/** Resolve a path relative to a dashboard URL. The dashboard is served under the
 * Iris controller proxy at /proxy/<name>/, and the proxy does not rewrite
 * bodies — an absolute path like /v1/chat/completions would escape the prefix. */
export const api = (path: string, baseUrl = location.href) => new URL(path, baseUrl).toString()

export async function fetchInfo(baseUrl?: string): Promise<ServingInfo> {
  const response = await fetch(api('info', baseUrl))
  if (!response.ok) throw new Error(`info returned ${response.status}`)
  return response.json()
}

export interface HealthResult {
  ok: boolean
  model: string | null
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export async function fetchHealth(baseUrl?: string): Promise<HealthResult> {
  const response = await fetch(api('health', baseUrl))
  const body = await response.json().catch(() => ({}))
  return { ok: response.ok, model: body.model ?? null }
}

export async function fetchToolDefinitions(source: string, signal: AbortSignal, baseUrl?: string): Promise<ToolDefinition[]> {
  return postJsonResult('tools', { source }, signal, 'tool definitions', baseUrl)
}

export async function invokeTool(
  name: string,
  source: string,
  arguments_: Record<string, unknown>,
  signal: AbortSignal,
  baseUrl?: string,
): Promise<unknown> {
  return postJsonResult(`tools/${encodeURIComponent(name)}`, { source, arguments: arguments_ }, signal, 'tool', baseUrl)
}

export interface ShellCommandResult {
  exit_code: number
  stdout: string
  stderr: string
  stop_reason: string | null
  unsupported: string[]
  partial_commands: string[]
}

export async function invokeShell(
  files: Record<string, string>,
  commits: ImportedGitCommit[],
  history: string[],
  command: string,
  signal: AbortSignal,
  baseUrl?: string,
): Promise<ShellCommandResult> {
  return postJsonResult('shell', { files, commits, history, command }, signal, 'shell command', baseUrl)
}

export interface RepositorySnapshot {
  files: Record<string, string>
  commits: ImportedGitCommit[]
  skipped_files: number
  truncated_history: boolean
}

export async function importRepository(url: string, signal: AbortSignal): Promise<RepositorySnapshot> {
  return postJsonResult('shell/repository', { url }, signal, 'repository import')
}

export async function createChatShare(snapshot: SharedChatSnapshot): Promise<string> {
  const result = await postJsonResult<{ id: string }>('chat-shares', snapshot, undefined, 'chat share')
  if (typeof result.id !== 'string') throw new Error('chat share returned an invalid ID')
  return result.id
}

export async function fetchChatShare(shareId: string): Promise<unknown> {
  const response = await fetch(api(`chat-shares/${encodeURIComponent(shareId)}`))
  if (response.ok) return response.json()
  throw new Error(`chat share returned ${response.status}: ${await response.text()}`)
}

/** POST an OpenAI request and invoke onData for either buffered JSON or SSE events. */
export async function requestCompletion(
  path: string,
  body: Record<string, unknown>,
  streaming: boolean,
  signal: AbortSignal,
  onData: (data: any) => void,
  baseUrl?: string,
  contextLength?: number | null,
): Promise<string | null> {
  if (contextLength != null) {
    const tokenization: Record<string, unknown> = { model: body.model }
    if (body.messages) {
      tokenization.messages = body.messages
      tokenization.add_generation_prompt = true
      tokenization.chat_template_kwargs = body.chat_template_kwargs
      tokenization.tools = body.tools
    } else {
      tokenization.prompt = body.prompt
      tokenization.add_special_tokens = body.add_special_tokens ?? true
    }
    const { count } = await postJsonResult<{ count: number }>(
      'tokenize', tokenization, signal, 'tokenization', baseUrl,
    )
    const available = contextLength - count
    if (available <= 0) throw new Error('Prompt fills the model context. Shorten it before generating.')
    body = { ...body, max_tokens: Math.min(Number(body.max_tokens), available) }
  }
  const response = await postJson(path, body, signal, baseUrl)
  if (!response.ok || !response.body) {
    throw new Error(`${response.status} — ${await response.text()}`)
  }
  if (!streaming) {
    const data = await response.json()
    onData(data)
    return data.choices?.[0]?.finish_reason ?? null
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finishReason: string | null = null
  while (true) {
    const { done, value } = await reader.read()
    buffer += done ? decoder.decode() + '\n' : decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue
      const payload = trimmed.slice(5).trim()
      if (payload === '[DONE]') return finishReason
      const data = JSON.parse(payload)
      finishReason = data.choices?.[0]?.finish_reason ?? finishReason
      onData(data)
    }
    if (done) break
  }
  return finishReason
}

function postJson(path: string, body: object, signal: AbortSignal | undefined, baseUrl?: string): Promise<Response> {
  return fetch(api(path, baseUrl), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

async function postJsonResult<T>(
  path: string,
  body: object,
  signal: AbortSignal | undefined,
  label: string,
  baseUrl?: string,
): Promise<T> {
  const response = await postJson(path, body, signal, baseUrl)
  if (response.ok) return response.json()
  throw new Error(`${label} returned ${response.status}: ${await response.text()}`)
}
