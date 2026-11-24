import assert from 'node:assert/strict'
import test from 'node:test'

import { ThinkingMode } from '../src/lib/chat_template.ts'
import {
  conversationFromSharedChat,
  copySharedChatUrl,
  sharedChatIdFromHash,
  sharedChatSnapshot,
  sharedChatUrl,
} from '../src/lib/shared_chat.ts'

test('copies a shared link while the click is active even if creating it takes time', async () => {
  const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator')
  const originalClipboardItem = Object.getOwnPropertyDescriptor(globalThis, 'ClipboardItem')
  let resolveUrl
  const url = new Promise((resolve) => { resolveUrl = resolve })
  let clickActive = true
  let clipboardText

  Object.defineProperty(globalThis, 'ClipboardItem', {
    configurable: true,
    value: class {
      constructor(data) { this.data = data }
    },
  })
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: {
      clipboard: {
        write(items) {
          assert.equal(clickActive, true)
          return items[0].data['text/plain'].then((blob) => blob.text()).then((text) => {
            clipboardText = text
          })
        },
      },
    },
  })

  try {
    const copied = copySharedChatUrl(url)
    clickActive = false
    resolveUrl('https://serve.example/proxy/chat/#chat=abcdefghijklmnop')
    assert.equal(await copied, true)
    assert.equal(clipboardText, 'https://serve.example/proxy/chat/#chat=abcdefghijklmnop')

    navigator.clipboard.write = () => Promise.reject(new DOMException('Not allowed', 'NotAllowedError'))
    assert.equal(await copySharedChatUrl(Promise.resolve(clipboardText)), false)
  } finally {
    if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator)
    else delete globalThis.navigator
    if (originalClipboardItem) Object.defineProperty(globalThis, 'ClipboardItem', originalClipboardItem)
    else delete globalThis.ClipboardItem
  }
})

test('builds a short link and imports a non-executable chat snapshot', () => {
  const conversation = {
    id: 'original',
    title: 'Billing analysis',
    model: 'example/model',
    system: 'Hidden system prompt',
    pythonTools: 'def refund(): ...',
    shellWorkspace: {
      filesJson: '{"secret.txt":"hidden"}',
      commits: [],
      history: [],
      repositoryUrl: '',
    },
    thinkingMode: ThinkingMode.Enabled,
    customInstructions: 'Hidden template instruction',
    createdAt: 10,
    updatedAt: 20,
    messages: [
      { role: 'user', content: 'Analyze this duplicate charge. 😞' },
      {
        role: 'assistant',
        content: '<priority>medium</priority>',
        thinking: 'Check the ledger.',
        rawContent: '<hidden>raw protocol text</hidden>',
        rawReasoning: 'Hidden raw reasoning',
        thinkingSeconds: 1.25,
        error: null,
        toolCalls: [{ id: 'call_refund', name: 'refund', arguments: { amount: 20 } }],
        requestDebug: { metrics: null, usage: { prompt_tokens: 42, completion_tokens: 128 } },
      },
      { role: 'tool', name: 'refund', toolCallId: 'call_refund', result: { approved: true } },
    ],
  }

  const snapshot = sharedChatSnapshot(conversation)
  const url = new URL(sharedChatUrl('https://serve.example/chat?token=capability', 'abcdefghijklmnop'))
  const imported = conversationFromSharedChat(snapshot, 'imported', 30, ThinkingMode.TemplateDefault)
  const serializedSnapshot = JSON.stringify(snapshot)

  assert.equal(url.search, '?token=capability')
  assert.equal(url.hash, '#chat=abcdefghijklmnop')
  assert.doesNotMatch(serializedSnapshot, /Hidden|Check the ledger|call_refund|def refund|secret\.txt|approved/)
  assert.doesNotMatch(serializedSnapshot, /requestDebug|prompt_tokens/)
  assert.deepEqual(imported, {
    id: 'imported',
    title: 'Billing analysis',
    model: 'example/model',
    system: '',
    pythonTools: '',
    shellWorkspace: null,
    webSearch: false,
    thinkingMode: ThinkingMode.TemplateDefault,
    customInstructions: '',
    createdAt: 30,
    updatedAt: 30,
    messages: [
      { role: 'user', content: 'Analyze this duplicate charge. 😞' },
      {
        role: 'assistant',
        content: '<priority>medium</priority>',
        thinking: '',
        thinkingSeconds: null,
        error: null,
      },
    ],
  })
})

test('rejects malformed shared chat IDs and snapshots', () => {
  assert.equal(sharedChatIdFromHash('#chat=short'), null)
  assert.equal(sharedChatIdFromHash('#section=chat'), null)
  assert.equal(
    conversationFromSharedChat(
      { version: 2, title: '', model: '', messages: [] },
      'unused',
      0,
      ThinkingMode.TemplateDefault,
    ),
    null,
  )
})
