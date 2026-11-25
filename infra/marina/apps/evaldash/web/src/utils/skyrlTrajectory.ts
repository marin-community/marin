// Adapter: SkyRL evaluation turn records onto the ATIF Trajectory type, so TrajectoryViewer
// renders them. Two real shapes exist in published blobs:
// - the eval-dump row: one trial's flat record (`uid`, `output_response`, `messages`,
//   `verification`, `input_prompt`, ...) — what the long-loop archive writer stores per trial;
// - the SkyRL schema_v7 turn record (`schema_version: 7`): `response.text`, `response.messages`,
//   `verification_result`, `trajectory.instance_id` — what the retention archives store.
// One artifact blob holds one record or a list of them. A list can span several uids, so records
// group by uid and the group matching the viewed sample renders. Messages map to steps by role —
// including assistant turns that carry tool_calls — and each record's stop_reason and verification
// ride that record's final step as string annotations.
import type { Trajectory, TrajectoryStep, TrajectoryToolCall } from '@/types/api'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** The eval-dump row shape: one trial's flat record. `messages` may be null. */
export function isSkyrlEvalDumpRow(value: unknown): boolean {
  return isRecord(value) && typeof value.uid === 'string' && typeof value.output_response === 'string'
}

/** The SkyRL schema_v7 turn record shape (`schema_version: 7`, nested response/verification). */
export function isSkyrlTurnRecord(value: unknown): boolean {
  return isRecord(value) && value.schema_version === 7 && isRecord(value.response)
}

/** Flatten one message's content: a string, or a list of `{type, text}` parts. */
function messageText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === 'string' ? part : isRecord(part) && typeof part.text === 'string' ? part.text : ''))
      .filter(Boolean)
      .join('\n')
  }
  return ''
}

/** The step source a message role maps to; ATIF's agent step carries the assistant turns. */
function messageSource(role: unknown): string {
  if (role === 'assistant') return 'agent'
  return typeof role === 'string' && role ? role : 'user'
}

/** OpenAI-style tool calls (`{id, function: {name, arguments}}`) onto ATIF's typed calls. */
function toolCallsOf(value: unknown): TrajectoryToolCall[] | undefined {
  if (!Array.isArray(value)) return undefined
  const calls: TrajectoryToolCall[] = []
  for (const call of value) {
    if (!isRecord(call)) continue
    const fn = isRecord(call.function) ? call.function : null
    const name = fn && typeof fn.name === 'string' ? fn.name : ''
    if (!name) continue
    const raw = fn && typeof fn.arguments !== 'undefined' ? fn.arguments : call.arguments
    let args: Record<string, unknown> = {}
    if (typeof raw === 'string') {
      try {
        args = JSON.parse(raw) as Record<string, unknown>
      } catch {
        args = { raw }
      }
    } else if (isRecord(raw)) {
      args = raw
    }
    calls.push({
      tool_call_id: typeof call.id === 'string' ? call.id : undefined,
      function_name: name,
      arguments: args,
    })
  }
  return calls.length ? calls : undefined
}

/** A record's outcome facts (stop reason, verification) as the string annotations a step carries. */
function recordAnnotations(verification: Record<string, unknown> | null, stopReason: string | null, exception: unknown): Record<string, string> | undefined {
  const annotations: Record<string, string> = {}
  if (stopReason) annotations['stop reason'] = stopReason
  if (typeof exception === 'string' && exception) annotations['exception'] = exception
  if (isRecord(verification)) {
    if (typeof verification.status === 'string' && verification.status) annotations['verification'] = verification.status
    if (typeof verification.score === 'number') annotations['verifier score'] = String(verification.score)
    else if (typeof verification.score === 'string' && verification.score) annotations['verifier score'] = verification.score
    if (typeof verification.reason === 'string' && verification.reason) annotations['verifier note'] = verification.reason
  }
  return Object.keys(annotations).length ? annotations : undefined
}

function numericMetrics(value: unknown): Record<string, number> | undefined {
  if (!isRecord(value)) return undefined
  const metrics: Record<string, number> = {}
  for (const [name, metric] of Object.entries(value)) {
    if (typeof metric === 'number') metrics[name] = metric
  }
  return Object.keys(metrics).length ? metrics : undefined
}

function sameMessage(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

/** Drop the head of ``next`` that repeats ``previous`` verbatim: cumulative per-step transcripts. */
function dropRepeatedPrefix(previous: unknown[], next: unknown[]): unknown[] {
  if (previous.length === 0 || next.length < previous.length) return next
  for (let index = 0; index < previous.length; index++) {
    if (!sameMessage(previous[index], next[index])) return next
  }
  return next.slice(previous.length)
}

// One record normalized to the fields the adapter renders, whatever shape it arrived in.
interface NormalizedTurn {
  uid: string
  step: number
  kind: 'skyrl-eval-dump' | 'skyrl-turn-v7'
  sessionId: string
  envClass: string
  promptText: string
  responseText: string
  messages: unknown[] | null
  verification: Record<string, unknown> | null
  stopReason: string | null
  exception: unknown
  metrics: Record<string, number> | undefined
}

function normalizeEvalDumpRow(record: Record<string, unknown>): NormalizedTurn {
  return {
    uid: record.uid,
    step: typeof record.ordinal === 'number' ? record.ordinal : typeof record.step === 'number' ? record.step : 0,
    kind: 'skyrl-eval-dump',
    sessionId: typeof record.attempt_id === 'string' ? record.attempt_id : '',
    envClass: typeof record.env_class === 'string' ? record.env_class : '',
    promptText: typeof record.input_prompt === 'string' ? record.input_prompt : '',
    responseText: record.output_response,
    messages: Array.isArray(record.messages) ? record.messages : null,
    verification: isRecord(record.verification) ? record.verification : null,
    stopReason: typeof record.stop_reason === 'string' ? record.stop_reason : null,
    exception: record.exception_type,
    metrics: numericMetrics(record.env_metrics),
  }
}

function normalizeTurnRecord(record: Record<string, unknown>): NormalizedTurn {
  const response = isRecord(record.response) ? record.response : {}
  const trajectory = isRecord(record.trajectory) ? record.trajectory : {}
  const disposition = isRecord(record.disposition) ? record.disposition : {}
  const instanceId = typeof trajectory.instance_id === 'string' ? trajectory.instance_id : ''
  const rows = Array.isArray(trajectory.row_indices) ? trajectory.row_indices : []
  const boundaries = Array.isArray(response.step_boundaries) ? response.step_boundaries : []
  const boundary = boundaries.find(isRecord)
  return {
    uid: instanceId || (typeof record.record_id === 'string' ? record.record_id : ''),
    step: boundary && typeof boundary.row_index === 'number' ? boundary.row_index : rows.length === 1 && typeof rows[0] === 'number' ? rows[0] : 0,
    kind: 'skyrl-turn-v7',
    sessionId: typeof record.run_id === 'string' ? record.run_id : '',
    envClass: typeof trajectory.environment_class === 'string' ? trajectory.environment_class : '',
    promptText: '',
    responseText: typeof response.text === 'string' ? response.text : '',
    messages: Array.isArray(response.messages) ? response.messages : null,
    verification: isRecord(record.verification_result) ? record.verification_result : null,
    stopReason: typeof response.stop_reason === 'string' ? response.stop_reason : null,
    exception: disposition.exception_type,
    metrics: numericMetrics(record.metrics),
  }
}

function normalize(value: unknown): NormalizedTurn | null {
  if (isSkyrlEvalDumpRow(value)) return normalizeEvalDumpRow(value)
  if (isSkyrlTurnRecord(value)) return normalizeTurnRecord(value)
  return null
}

function turnSteps(turn: NormalizedTurn, firstStepId: number, includePrompt: boolean, messages: unknown[]): TrajectoryStep[] {
  const steps: TrajectoryStep[] = []
  let stepId = firstStepId
  if (includePrompt && turn.promptText) {
    steps.push({ step_id: stepId++, source: 'user', message: turn.promptText, annotations: { 'content': 'raw input prompt' } })
  }
  for (const message of messages ?? []) {
    if (!isRecord(message)) continue
    const text = messageText(message.content)
    const calls = toolCallsOf(message.tool_calls)
    if (!text && !calls) continue
    steps.push({
      step_id: stepId++,
      source: messageSource(message.role),
      message: text || undefined,
      tool_calls: calls,
    })
  }
  // The record's terminal step: the raw response with its think block, plus its outcome facts.
  const annotations = recordAnnotations(turn.verification, turn.stopReason, turn.exception)
  if (turn.responseText || annotations) {
    steps.push({ step_id: stepId++, source: 'agent', message: turn.responseText || undefined, annotations })
  }
  return steps
}

/**
 * Map one SkyRL record, or a list of them, onto the Trajectory type; null when not SkyRL. A list
 * groups by uid — one trial's records can span several uids — and `preferredUid` picks the group
 * matching the viewed sample (the first group renders when absent or unmatched).
 */
export function skyrlTurnsToTrajectory(parsed: unknown, preferredUid?: string): Trajectory | null {
  const records = (Array.isArray(parsed) ? parsed : [parsed]) as unknown[]
  if (records.length === 0) return null
  const turns: NormalizedTurn[] = []
  for (const record of records) {
    const turn = normalize(record)
    if (turn === null) return null
    turns.push(turn)
  }
  const groups = new Map<string, NormalizedTurn[]>()
  for (const turn of turns) {
    const group = groups.get(turn.uid) ?? []
    group.push(turn)
    groups.set(turn.uid, group)
  }
  const selected = (preferredUid !== undefined && groups.get(preferredUid)) || groups.values().next().value
  if (!selected) return null
  selected.sort((left, right) => left.step - right.step)
  const steps: TrajectoryStep[] = []
  let previous: unknown[] = []
  for (const [index, turn] of selected.entries()) {
    const fresh = dropRepeatedPrefix(previous, turn.messages ?? [])
    previous = turn.messages ?? previous
    steps.push(...turnSteps(turn, steps.length + 1, index === 0, fresh))
  }
  return {
    schema_version: selected[0].kind,
    session_id: selected[0].sessionId || undefined,
    agent: { name: selected[0].envClass || 'skyrl' },
    steps,
    final_metrics: selected[selected.length - 1].metrics,
  }
}
