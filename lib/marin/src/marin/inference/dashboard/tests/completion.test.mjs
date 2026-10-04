import assert from 'node:assert/strict'
import test from 'node:test'

import { requestCompletion } from '../src/lib/api.ts'

for (const streaming of [false, true]) {
  test(`reports token exhaustion with ${streaming ? 'streamed' : 'buffered'} output`, async (t) => {
    const content = { choices: [{ [streaming ? 'delta' : 'message']: { content: '**Hipoteza' } }] }
    const finished = { choices: [{ finish_reason: 'length' }] }
    t.mock.method(globalThis, 'fetch', async () => new Response(streaming
      ? `data: ${JSON.stringify(content)}\n\ndata: ${JSON.stringify(finished)}`
      : JSON.stringify({ choices: [{ ...content.choices[0], finish_reason: 'length' }] })))
    const events = []
    const finishReason = await requestCompletion(
      'v1/chat/completions', {}, streaming, new AbortController().signal,
      (event) => events.push(event), 'https://example.test/serve/',
    )
    assert.equal(events[0].choices[0][streaming ? 'delta' : 'message'].content, '**Hipoteza')
    assert.equal(finishReason, 'length')
  })
}

test('does not hide response-handler errors behind a completed stream', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'))
  await assert.rejects(requestCompletion(
    'v1/chat/completions', {}, true, new AbortController().signal,
    () => { throw new Error('render failed') }, 'https://example.test/serve/',
  ), /render failed/)
})

for (const path of ['v1/chat/completions', 'v1/completions']) {
  test(`fits output to the active context for ${path}`, async (t) => {
    const input = path.includes('/chat/')
      ? { messages: [{ role: 'user', content: 'Hello' }], chat_template_kwargs: { enable_thinking: true } }
      : { prompt: 'Hello' }
    const requests = []
    t.mock.method(globalThis, 'fetch', async (url, options) => {
      requests.push({ url, body: JSON.parse(options.body) })
      return new Response(JSON.stringify(url.endsWith('/tokenize')
        ? { count: 32 }
        : { choices: [{ finish_reason: 'length' }] }))
    })
    const body = { model: 'small-context', max_tokens: 16384, ...input }
    await requestCompletion(path, body, false, new AbortController().signal,
      () => {}, 'https://example.test/second/', 4096)
    assert.equal(requests[0].url, 'https://example.test/second/tokenize')
    for (const [key, value] of Object.entries(input)) assert.deepEqual(requests[0].body[key], value)
    assert.equal(requests[1].body.max_tokens, 4064)
    assert.equal(body.max_tokens, 16384)
  })
}
