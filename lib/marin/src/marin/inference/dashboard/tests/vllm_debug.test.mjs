import assert from 'node:assert/strict'
import test from 'node:test'

import { useVllmDebug } from '../src/composables/useVllmDebug.ts'
import { requestCompletion } from '../src/lib/api.ts'
import { requestDebugData, vllmDebugStreamOptions } from '../src/lib/vllm_debug.ts'

globalThis.location = new URL('https://example.test/proxy/serve/')

const metrics = {
  time_to_first_token_ms: 85.2,
  queue_time_ms: 12.3,
  generation_time_ms: 1240.5,
  mean_itl_ms: 9.1,
  tokens_per_second: 103.2,
}
const usage = { prompt_tokens: 42, completion_tokens: 128, total_tokens: 170 }
const content = { choices: [{ delta: { content: 'Hello' } }] }
const buffered = { choices: [{ message: { content: 'Hello' } }], usage, metrics }
const sse = (...events) => [...events.map((event) => `data: ${JSON.stringify(event)}`), 'data: [DONE]'].join('\n\n') + '\n\n'
const withUsage = { stream: true, stream_options: { include_usage: true } }
const debug = { metrics, usage }

for (const { name, streaming, enabled, response, expectedBody, expectedDebug } of [
  { name: 'streaming final event without choices', streaming: true, enabled: true,
    response: sse(content, { choices: [], usage, metrics }), expectedBody: withUsage, expectedDebug: debug },
  { name: 'buffered response', streaming: false, enabled: true,
    response: buffered, expectedBody: { stream: false }, expectedDebug: debug },
  { name: 'missing metrics', streaming: true, enabled: true,
    response: sse(content, { choices: [], usage }), expectedBody: withUsage, expectedDebug: { metrics: null, usage } },
  { name: 'all-null metrics', streaming: true, enabled: true,
    response: sse(content, { choices: [], usage, metrics: Object.fromEntries(Object.keys(metrics).map((key) => [key, null])) }),
    expectedBody: withUsage, expectedDebug: { metrics: null, usage } },
  { name: 'debug disabled', streaming: true, enabled: false,
    response: sse(content), expectedBody: { stream: true }, expectedDebug: null },
]) {
  test(name, async (t) => {
    let posted
    t.mock.method(globalThis, 'fetch', async (url, options) => {
      posted = { url, body: JSON.parse(options.body) }
      return new Response(typeof response === 'string' ? response : JSON.stringify(response))
    })
    const events = []
    let result = null
    await requestCompletion(
      'v1/chat/completions', { stream: streaming, ...vllmDebugStreamOptions(enabled, streaming) },
      streaming, new AbortController().signal,
      (event) => {
        events.push(event)
        if (enabled) result = requestDebugData(event) ?? result
      },
    )
    assert.equal(posted.url, 'https://example.test/proxy/serve/v1/chat/completions')
    assert.deepEqual(posted.body, expectedBody)
    assert.equal(events[0].choices[0][streaming ? 'delta' : 'message'].content, 'Hello')
    assert.deepEqual(result, expectedDebug)
  })
}

test('changing debug mode clears the prior result', () => {
  const { enabled, lastRequest } = useVllmDebug()
  enabled.value = true
  lastRequest.value = debug
  enabled.value = false
  assert.equal(lastRequest.value, null)
  enabled.value = true
  assert.equal(lastRequest.value, null)
})
