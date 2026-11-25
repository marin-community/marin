import type { Trajectory, TrajectoryStep } from '@/types/api'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isSkyrlTurnRecord(value: unknown): boolean {
  return (
    isRecord(value) &&
    typeof value.uid === 'string' &&
    typeof value.output_response === 'string' &&
    Array.isArray(value.messages)
  )
}

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

function messageSource(role: unknown): string {
  if (role === 'assistant') return 'agent'
  return typeof role === 'string' && role ? role : 'user'
}

function recordAnnotations(record: Record<string, unknown>): Record<string, string> | undefined {
  const annotations: Record<string, string> = {}
  if (typeof record.stop_reason === 'string' && record.stop_reason) annotations['stop reason'] = record.stop_reason
  if (typeof record.exception_type === 'string' && record.exception_type) annotations['exception'] = record.exception_type
  const verification = record.verification
  if (isRecord(verification)) {
    if (typeof verification.status === 'string' && verification.status) annotations['verification'] = verification.status
    if (typeof verification.score === 'number') annotations['verifier score'] = String(verification.score)
    else if (typeof verification.score === 'string' && verification.score) {
      annotations['verifier score'] = verification.score
    }
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

function dropRepeatedPrefix(previous: unknown[], next: unknown[]): unknown[] {
  if (previous.length === 0 || next.length < previous.length) return next
  for (let index = 0; index < previous.length; index++) {
    if (!sameMessage(previous[index], next[index])) return next
  }
  return next.slice(previous.length)
}

function recordSteps(record: Record<string, unknown>, firstStepId: number, includePrompt: boolean, messages: unknown[]): TrajectoryStep[] {
  const steps: TrajectoryStep[] = []
  let stepId = firstStepId
  if (includePrompt && typeof record.input_prompt === 'string' && record.input_prompt) {
    steps.push({ step_id: stepId++, source: 'user', message: record.input_prompt, annotations: { 'content': 'raw input prompt' } })
  }
  for (const message of messages) {
    if (!isRecord(message)) continue
    const text = messageText(message.content)
    if (!text) continue
    steps.push({ step_id: stepId++, source: messageSource(message.role), message: text })
  }
  const annotations = recordAnnotations(record)
  const response = typeof record.output_response === 'string' ? record.output_response : ''
  if (response || annotations) {
    steps.push({ step_id: stepId++, source: 'agent', message: response, annotations })
  }
  return steps
}

/** Map SkyRL eval-dump rows onto Trajectory; return null for other formats. */
export function skyrlTurnsToTrajectory(parsed: unknown): Trajectory | null {
  const records = (Array.isArray(parsed) ? parsed : [parsed]) as unknown[]
  if (records.length === 0 || !records.every(isSkyrlTurnRecord)) return null
  const steps: TrajectoryStep[] = []
  let previous: unknown[] = []
  let finalMetrics: Record<string, number> | undefined
  let agentName = ''
  let sessionId = ''
  records.forEach((value, index) => {
    const record = value as Record<string, unknown>
    const fresh = dropRepeatedPrefix(previous, record.messages as unknown[])
    previous = record.messages as unknown[]
    steps.push(...recordSteps(record, steps.length + 1, index === 0, fresh))
    finalMetrics = numericMetrics(record.env_metrics) ?? finalMetrics
    agentName ||= typeof record.env_class === 'string' ? record.env_class : ''
    sessionId ||= typeof record.attempt_id === 'string' ? record.attempt_id : ''
  })
  return {
    schema_version: 'skyrl-turn-v7',
    session_id: sessionId || undefined,
    agent: { name: agentName || 'skyrl' },
    steps,
    final_metrics: finalMetrics,
  }
}
