import assert from 'node:assert/strict'
import test from 'node:test'

import { ThinkingMode } from '../src/lib/chat_template.ts'
import { ISSUE_TRANSCRIPT_FILENAME, issueDraft } from '../src/lib/github_issue.ts'
import { plainTextChat } from '../src/lib/plain_text_chat.ts'

const INFO = {
  model: 'marin/hero-sft',
  backend: 'vllm',
  tensor_parallel_size: 4,
  max_model_len: 32768,
  dtype: 'bfloat16',
  has_chat_template: true,
  endpoint: 'http://localhost:8000',
  streaming: true,
  chat_template_protocol: {
    thinking_start: null,
    thinking_end: null,
    tool_call_start: null,
    tool_call_end: null,
    tool_call_format: null,
  },
}
const PARAMS = { temperature: 0.7, maxTokens: 2048, topP: 0.95, maxToolRounds: 0 }

function conversation(messages) {
  return {
    id: 'c1',
    title: 'Fix   my\nPython script',
    model: 'marin/hero-sft',
    system: 'Be terse.',
    pythonTools: '',
    shellWorkspace: null,
    webSearch: false,
    thinkingMode: ThinkingMode.Enabled,
    customInstructions: '',
    createdAt: 0,
    updatedAt: 0,
    messages,
  }
}

function draftParts(url) {
  const parsed = new URL(url)
  return {
    target: `${parsed.origin}${parsed.pathname}`,
    title: parsed.searchParams.get('title'),
    body: parsed.searchParams.get('body'),
  }
}

test('drafts an issue with setup and a transcript fence longer than any backtick run', () => {
  const chat = conversation([
    { role: 'user', content: 'Why does this fail?\n```py\nprint(x)\n```' },
    { role: 'assistant', content: 'Because x is undefined.', thinking: 'x?', thinkingSeconds: 1, error: null },
  ])

  const draft = issueDraft(chat, plainTextChat(chat), INFO, PARAMS)
  const { target, title, body } = draftParts(draft.url)

  assert.equal(draft.fullTranscript, null)
  assert.equal(target, 'https://github.com/marin-community/marin/issues/new')
  assert.equal(title, '[model feedback] marin/hero-sft: Fix my Python script')
  assert.match(body, /Serving: vllm, bfloat16, tensor parallel 4, context 32768 tokens/)
  assert.match(body, /temperature 0\.7, top_p 0\.95, max_tokens 2048/)
  assert.match(body, /Thinking: enabled/)
  assert.match(body, /^<!-- This repository is public\. .* -->$/m)
  assert.ok(body.includes(`\`\`\`\`text\n${plainTextChat(chat)}\n\`\`\`\``))
})

test('shortens long transcripts to fit the URL limit and returns the full transcript', () => {
  const filler = '😞 lorem ipsum '.repeat(2000)
  const chat = conversation([
    { role: 'user', content: `FIRST-PROMPT ${filler}` },
    { role: 'assistant', content: `${filler} LAST-REPLY`, thinking: '', thinkingSeconds: null, error: null },
  ])

  const draft = issueDraft(chat, plainTextChat(chat), INFO, PARAMS, 6000)
  const { body } = draftParts(draft.url)

  assert.ok(draft.url.length <= 6000)
  assert.ok(draft.url.length > 5000, 'truncation should use most of the available budget')
  assert.equal(draft.fullTranscript, plainTextChat(chat))
  assert.match(body, /FIRST-PROMPT/)
  assert.match(body, /LAST-REPLY/)
  assert.match(body, /\[… \d+ characters omitted …\]/)
  assert.ok(body.includes(ISSUE_TRANSCRIPT_FILENAME))
})

test('shortens long custom template instructions with the transcript', () => {
  const instructions = `START-INSTRUCTIONS ${'x'.repeat(10000)} END-INSTRUCTIONS`
  const chat = {
    ...conversation([{ role: 'user', content: 'Hello' }]),
    customInstructions: instructions,
  }

  const draft = issueDraft(chat, plainTextChat(chat), INFO, PARAMS, 6000)

  assert.ok(draft.url.length <= 6000)
  assert.match(draftParts(draft.url).body, /START-INSTRUCTIONS/)
  assert.ok(draft.fullTranscript.includes(instructions))
  assert.ok(draft.fullTranscript.endsWith(plainTextChat(chat)))
})

test('omits the conversation, its title, and custom instructions when no transcript is given', () => {
  const chat = {
    ...conversation([{ role: 'user', content: 'PRIVATE-PROMPT' }]),
    title: 'PRIVATE-TITLE',
    customInstructions: 'PRIVATE-INSTRUCTIONS',
  }

  const draft = issueDraft(chat, null, INFO, PARAMS)
  const { title, body } = draftParts(draft.url)

  assert.equal(draft.fullTranscript, null)
  assert.equal(title, '[model feedback] marin/hero-sft')
  assert.match(body, /Thinking: enabled/)
  assert.doesNotMatch(draft.url, /PRIVATE/)
})
