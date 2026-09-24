import assert from 'node:assert/strict'
import test from 'node:test'

import { ThinkingMode } from '../src/lib/chat_template.ts'
import {
  conversationFromSharedChat,
  sharedChatIdFromHash,
  sharedChatSnapshot,
  sharedChatUrl,
} from '../src/lib/shared_chat.ts'

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
  assert.deepEqual(imported, {
    id: 'imported',
    title: 'Billing analysis',
    model: 'example/model',
    system: '',
    pythonTools: '',
    shellWorkspace: null,
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
