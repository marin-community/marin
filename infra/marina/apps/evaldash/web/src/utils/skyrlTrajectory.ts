// Render flat SkyRL eval dumps and schema-v7 retention records as ATIF trajectories.
import type { Trajectory, TrajectoryStep, TrajectoryToolCall } from '@/types/trajectory'

const EVAL_DUMP_KIND = 'skyrl-eval-dump'
const RETENTION_KIND = 'skyrl-turn-v7'

interface VerificationOutcome {
  status?: string
  score?: number | string
  reason?: string
}

function verificationOutcome(value: unknown): VerificationOutcome | null {
  if (!isRecord(value)) return null
  return {
    status: typeof value.status === 'string' ? value.status : undefined,
    score: typeof value.score === 'number' || typeof value.score === 'string' ? value.score : undefined,
    reason: typeof value.reason === 'string' ? value.reason : undefined,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** The eval-dump row shape: one trial's flat record. `messages` may be null. */
function isSkyrlEvalDumpRow(value: unknown): value is Record<string, unknown> {
  return isRecord(value) && typeof value.uid === 'string' && typeof value.output_response === 'string'
}

/** The SkyRL schema_v7 turn record shape (`schema_version: 7`, nested response/verification). */
function isSkyrlTurnRecord(value: unknown): value is Record<string, unknown> {
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
        // A length-stopped dump can contain incomplete JSON; keep its arguments visible as unparsed text.
        args = { unparsed_arguments: raw }
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
function recordAnnotations(verification: VerificationOutcome | null, stopReason: string | null, exception: unknown): Record<string, string> | undefined {
  const annotations: Record<string, string> = {}
  if (stopReason) annotations['stop reason'] = stopReason
  if (typeof exception === 'string' && exception) annotations['exception'] = exception
  if (verification !== null) {
    if (verification.status) annotations['verification'] = verification.status
    if (typeof verification.score === 'number') annotations['verifier score'] = String(verification.score)
    else if (typeof verification.score === 'string' && verification.score) annotations['verifier score'] = verification.score
    if (verification.reason) annotations['verifier note'] = verification.reason
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
  return isRecord(left) && isRecord(right)
    && left.role === right.role
    && messageText(left.content) === messageText(right.content)
    && JSON.stringify(toolCallsOf(left.tool_calls)) === JSON.stringify(toolCallsOf(right.tool_calls))
}

/** Drop messages already rendered at the start of a cumulative transcript. */
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
  trialKey: string | null
  step: number
  kind: typeof EVAL_DUMP_KIND | typeof RETENTION_KIND
  sessionId: string
  envClass: string
  promptText: string
  responseText: string
  messages: unknown[] | null
  verification: VerificationOutcome | null
  stopReason: string | null
  exception: unknown
  metrics: Record<string, number> | undefined
}

function normalizeEvalDumpRow(record: Record<string, unknown>): NormalizedTurn {
  return {
    uid: typeof record.uid === 'string' ? record.uid : '',
    trialKey: null,
    step: typeof record.ordinal === 'number' ? record.ordinal : typeof record.step === 'number' ? record.step : 0,
    kind: EVAL_DUMP_KIND,
    sessionId: typeof record.attempt_id === 'string' ? record.attempt_id : '',
    envClass: typeof record.env_class === 'string' ? record.env_class : '',
    promptText: typeof record.input_prompt === 'string' ? record.input_prompt : '',
    responseText: typeof record.output_response === 'string' ? record.output_response : '',
    messages: Array.isArray(record.messages) ? record.messages : null,
    verification: verificationOutcome(record.verification),
    stopReason: typeof record.stop_reason === 'string' ? record.stop_reason : null,
    exception: record.exception_type,
    metrics: numericMetrics(record.env_metrics),
  }
}

function normalizeTurnRecord(record: Record<string, unknown>): NormalizedTurn {
  const response = isRecord(record.response) ? record.response : {}
  const trajectory = isRecord(record.trajectory) ? record.trajectory : {}
  const disposition = isRecord(record.disposition) ? record.disposition : {}
  const prompt = isRecord(record.prompt) ? record.prompt : {}
  const promptMessages = Array.isArray(prompt.messages) ? prompt.messages : []
  const responseMessages = Array.isArray(response.messages) ? response.messages : []
  const includesPrompt = promptMessages.every((message, index) => sameMessage(message, responseMessages[index]))
  const instanceId = typeof trajectory.instance_id === 'string' ? trajectory.instance_id : ''
  const rows = Array.isArray(trajectory.row_indices) ? trajectory.row_indices : []
  const boundaries = Array.isArray(response.step_boundaries) ? response.step_boundaries : []
  const boundary = boundaries.find(isRecord)
  return {
    uid: instanceId || (typeof record.record_id === 'string' ? record.record_id : ''),
    trialKey: instanceId && Number.isInteger(trajectory.repetition_id)
      ? JSON.stringify([RETENTION_KIND, record.run_id, instanceId, trajectory.repetition_id])
      : null,
    step: boundary && typeof boundary.row_index === 'number' ? boundary.row_index : rows.length === 1 && typeof rows[0] === 'number' ? rows[0] : 0,
    kind: RETENTION_KIND,
    sessionId: typeof record.run_id === 'string' ? record.run_id : '',
    envClass: typeof trajectory.environment_class === 'string' ? trajectory.environment_class : '',
    promptText: '',
    responseText: typeof response.text === 'string' ? response.text : '',
    messages: includesPrompt ? responseMessages : [...promptMessages, ...responseMessages],
    verification: verificationOutcome(record.verification_result),
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
  // Attach outcomes to an identical rendered answer; otherwise retain a separate raw response step.
  const annotations = recordAnnotations(turn.verification, turn.stopReason, turn.exception)
  const last = steps[steps.length - 1]
  if (turn.responseText && last?.source === 'agent' && last.message === turn.responseText) {
    last.annotations = annotations
  } else if (turn.responseText || annotations) {
    steps.push({ step_id: stepId++, source: 'agent', message: turn.responseText || undefined, annotations })
  }
  return steps
}

function normalizedTurns(parsed: unknown): NormalizedTurn[] | null {
  const records = (Array.isArray(parsed) ? parsed : [parsed]) as unknown[]
  if (records.length === 0) return null
  const turns: NormalizedTurn[] = []
  for (const record of records) {
    const turn = normalize(record)
    if (turn === null) return null
    turns.push(turn)
  }
  return turns
}

function selectedTrial(turns: NormalizedTurn[], preferredUid?: string): NormalizedTurn[] {
  const selected = preferredUid !== undefined
    ? turns.filter((turn) => turn.uid === preferredUid)
    : turns
  const documentId = preferredUid === undefined ? '(unspecified)' : JSON.stringify(preferredUid)
  if (selected.length === 0) throw new Error(`No SkyRL record matches doc_id ${documentId}.`)
  if (selected.length > 1) {
    const key = selected[0].trialKey
    // Flat dumps lack a repetition identity; matching uids alone cannot establish one trial.
    if (key === null || selected.some((turn) => turn.trialKey !== key)) {
      throw new Error(`Ambiguous SkyRL records for doc_id ${documentId}: multiple trials or repetitions cannot be distinguished.`)
    }
  }
  selected.sort((left, right) => left.step - right.step)
  return selected
}

function trajectoryFromTurns(selected: NormalizedTurn[]): Trajectory {
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

/** Return null for unknown formats; throw when a recognized list cannot identify one trial. */
export function skyrlTurnsToTrajectory(parsed: unknown, preferredUid?: string): Trajectory | null {
  const turns = normalizedTurns(parsed)
  if (turns === null) return null
  const selected = selectedTrial(turns, Array.isArray(parsed) ? preferredUid : undefined)
  return trajectoryFromTurns(selected)
}
