import assert from 'node:assert/strict'
import test from 'node:test'

import { fetchInfo, requestCompletion } from '../src/lib/api.ts'
import { comparisonBaseUrl, comparisonTurns, comparisonUrl } from '../src/lib/comparison.ts'

test('routes each model request to its own dashboard', async () => {
  const first = 'https://iris.example/proxy/t/first/serve/a/'
  const second = comparisonBaseUrl('https://iris.example/proxy/t/second/serve/b/dashboard#chat=unused', first)
  const requests = []
  const originalFetch = globalThis.fetch
  globalThis.fetch = async (url, init) => {
    requests.push({ url, body: init?.body ? JSON.parse(init.body) : null })
    if (url.endsWith('/info')) return Response.json({ model: url.includes('/first/') ? 'model-a' : 'model-b' })
    return Response.json({ choices: [{ message: { content: url.includes('/first/') ? 'A' : 'B' } }] })
  }
  try {
    const firstInfo = await fetchInfo(first)
    const secondInfo = await fetchInfo(second)
    const replies = []
    await Promise.all([
      requestCompletion('v1/chat/completions', { model: firstInfo.model, messages: [{ role: 'user', content: 'Hi' }] }, false, new AbortController().signal, (data) => replies.push(data.choices[0].message.content), first),
      requestCompletion('v1/chat/completions', { model: secondInfo.model, messages: [{ role: 'user', content: 'Hi' }] }, false, new AbortController().signal, (data) => replies.push(data.choices[0].message.content), second),
    ])
    assert.deepEqual(replies.sort(), ['A', 'B'])
    assert.deepEqual(requests.map(({ url }) => url), [
      `${first}info`, `${second}info`, `${first}v1/chat/completions`, `${second}v1/chat/completions`,
    ])
    assert.deepEqual(requests.slice(2, 4).map(({ body }) => body.model), ['model-a', 'model-b'])
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('offers a turn for selection after both models answer the same prompt', () => {
  const comparison = { left: { messages: [] }, right: { messages: [] }, votes: {} }
  comparison.left.messages = [
    { role: 'user', content: 'First' },
    { role: 'assistant', content: 'A1', thinking: 'private', error: null, completed: true, thinkingSeconds: 1 },
    { role: 'user', content: 'Second' },
    { role: 'assistant', content: 'A2', thinking: 'private', error: null, completed: true, thinkingSeconds: 1 },
  ]
  comparison.right.messages = [
    { role: 'user', content: 'First' },
    { role: 'assistant', content: 'B1', thinking: 'private', error: null, completed: true, thinkingSeconds: 1 },
    { role: 'user', content: 'Second' },
    { role: 'assistant', content: 'B2', thinking: 'private', error: null, completed: true, thinkingSeconds: 1 },
  ]

  assert.deepEqual(comparisonTurns(comparison.left, comparison.right), [
    { index: 0, prompt: 'First' }, { index: 1, prompt: 'Second' },
  ])
  comparison.right.messages[3].completed = false
  assert.deepEqual(comparisonTurns(comparison.left, comparison.right), [{ index: 0, prompt: 'First' }])
  comparison.right.messages[3].completed = true
  comparison.right.messages[3].error = 'request failed'
  assert.deepEqual(comparisonTurns(comparison.left, comparison.right), [{ index: 0, prompt: 'First' }])
})

test('rejects a second dashboard on another origin', () => {
  assert.throws(
    () => comparisonBaseUrl('https://other.example/proxy/t/second/serve/b/', 'https://iris.example/proxy/t/first/serve/a/'),
    /this Iris origin/,
  )
})

test('comparison links restore both capability endpoints without a saved URL', () => {
  const first = 'https://iris.example/proxy/t/first-token/serve/a/#chat=old'
  const second = 'https://iris.example/proxy/t/second-token/serve/b/dashboard'
  const shared = new URL(comparisonUrl(first, second))
  const restored = new URLSearchParams(shared.hash.slice(1)).get('compare')
  assert.equal(shared.origin + shared.pathname, 'https://iris.example/proxy/t/first-token/serve/a/')
  assert.equal(comparisonBaseUrl(restored, shared.href), 'https://iris.example/proxy/t/second-token/serve/b/')
})
