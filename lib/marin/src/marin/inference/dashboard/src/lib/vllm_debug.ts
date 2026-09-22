export interface VllmRequestMetrics {
  time_to_first_token_ms: number | null
  queue_time_ms: number | null
  generation_time_ms: number | null
  mean_itl_ms: number | null
  tokens_per_second: number | null
}

export interface VllmTokenUsage {
  prompt_tokens: number
  completion_tokens: number
}

export interface VllmRequestDebug {
  metrics: VllmRequestMetrics | null
  usage: VllmTokenUsage | null
}

export function vllmDebugStreamOptions(enabled: boolean, streaming: boolean): Record<string, unknown> {
  return enabled && streaming ? { stream_options: { include_usage: true } } : {}
}

/** Content chunks have neither field; the final streaming usage chunk can have no choices. */
export function requestDebugData(data: unknown): VllmRequestDebug | null {
  if (typeof data !== 'object' || data === null || (!('metrics' in data) && !('usage' in data))) return null
  const metrics = ('metrics' in data ? data.metrics : null) as VllmRequestMetrics | null | undefined
  const hasTiming = metrics && Object.values(metrics).some((value) => typeof value === 'number')
  return {
    metrics: hasTiming ? metrics : null,
    usage: (('usage' in data ? data.usage : null) as VllmTokenUsage | null | undefined) ?? null,
  }
}
