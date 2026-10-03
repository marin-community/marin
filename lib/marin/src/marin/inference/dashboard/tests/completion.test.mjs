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
