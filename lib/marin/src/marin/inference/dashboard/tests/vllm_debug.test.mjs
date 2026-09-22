import assert from 'node:assert/strict'
import test from 'node:test'

import { useVllmDebug } from '../src/composables/useVllmDebug.ts'
import { requestCompletion } from '../src/lib/api.ts'
import { requestDebugData, vllmDebugStreamOptions } from '../src/lib/vllm_debug.ts'

const timings = {
  time_to_first_token_ms: 85.2,
  queue_time_ms: 12.3,
  generation_time_ms: 1240.5,
  mean_itl_ms: 9.1,
  tokens_per_second: 103.2,
}
const usage = { prompt_tokens: 42, completion_tokens: 128, total_tokens: 170 }

function fakeResponse(body, streaming) {
  return new Response(streaming ? body : JSON.stringify(body), {
    headers: { 'content-type': streaming ? 'text/event-stream' : 'application/json' },
  })
}

test('debug-enabled streaming request captures the final usage event without choices', async () => {
  globalThis.location = new URL('https://example.test/proxy/serve/')
  const originalFetch = globalThis.fetch
  let postedBody
  globalThis.fetch = async (_url, options) => {
    postedBody = JSON.parse(options.body)
    return fakeResponse(
      'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n' +
        `data: ${JSON.stringify({ choices: [], usage, metrics: timings })}\n\n` +
        'data: [DONE]\n\n',
      true,
    )
  }
  try {
    const events = []
    let debug = null
    await requestCompletion(
      'v1/chat/completions',
      { stream: true, ...vllmDebugStreamOptions(true, true) },
      true,
      new AbortController().signal,
      (data) => {
        events.push(data)
        debug = requestDebugData(data) ?? debug
      },
    )
    assert.deepEqual(postedBody.stream_options, { include_usage: true })
    assert.equal(events[0].choices[0].delta.content, 'Hello')
    assert.deepEqual(debug, { metrics: timings, usage })
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('debug-enabled buffered response captures timings and token counts', async () => {
  globalThis.location = new URL('https://example.test/proxy/serve/')
  const originalFetch = globalThis.fetch
  let postedBody
  globalThis.fetch = async (_url, options) => {
    postedBody = JSON.parse(options.body)
    return fakeResponse({ choices: [{ message: { content: 'Hello' } }], usage, metrics: timings }, false)
  }
  try {
    let debug = null
    await requestCompletion('v1/chat/completions', { stream: false, ...vllmDebugStreamOptions(true, false) }, false, new AbortController().signal, (data) => {
      debug = requestDebugData(data)
    })
    assert.equal('stream_options' in postedBody, false)
    assert.deepEqual(debug, { metrics: timings, usage })
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('a response without server timings can still show usage', () => {
  assert.deepEqual(requestDebugData({ choices: [], usage }), { metrics: null, usage })
  assert.deepEqual(requestDebugData({ usage, metrics: Object.fromEntries(Object.keys(timings).map((key) => [key, null])) }), {
    metrics: null,
    usage,
  })
  assert.equal(requestDebugData({ choices: [{ delta: { content: 'Hello' } }] }), null)
})

test('debug-disabled streaming request keeps its prior request shape and content events', async () => {
  globalThis.location = new URL('https://example.test/proxy/serve/')
  const originalFetch = globalThis.fetch
  let postedBody
  globalThis.fetch = async (_url, options) => {
    postedBody = JSON.parse(options.body)
    return fakeResponse('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n', true)
  }
  try {
    const events = []
    await requestCompletion('v1/chat/completions', { stream: true, ...vllmDebugStreamOptions(false, true) }, true, new AbortController().signal, (data) => {
      events.push(data)
    })
    assert.equal('stream_options' in postedBody, false)
    assert.equal(events[0].choices[0].delta.content, 'Hello')
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('re-enabling debug after an unmeasured request does not show the older request', async () => {
  const { enabled, lastRequest } = useVllmDebug()
  enabled.value = true
  lastRequest.value = { metrics: timings, usage }

  enabled.value = false
  assert.equal(lastRequest.value, null)
  globalThis.location = new URL('https://example.test/proxy/serve/')
  const originalFetch = globalThis.fetch
  let postedBody
  globalThis.fetch = async (_url, options) => {
    postedBody = JSON.parse(options.body)
    return fakeResponse('data: {"choices":[{"delta":{"content":"Request B"}}]}\n\ndata: [DONE]\n\n', true)
  }
  try {
    const events = []
    await requestCompletion(
      'v1/chat/completions',
      { stream: true, ...vllmDebugStreamOptions(enabled.value, true) },
      true,
      new AbortController().signal,
      (data) => events.push(data),
    )
    assert.equal('stream_options' in postedBody, false)
    assert.equal(events[0].choices[0].delta.content, 'Request B')
    enabled.value = true
    assert.equal(lastRequest.value, null)
  } finally {
    globalThis.fetch = originalFetch
  }
})
