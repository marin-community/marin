import assert from 'node:assert/strict'
import test from 'node:test'

import { chatTemplateRequestFields } from '../src/lib/chat_template.ts'
import { modelMessages } from '../src/lib/python_tools.ts'
import { plainTextChat } from '../src/lib/plain_text_chat.ts'
import { splitThinking } from '../src/lib/thinking.ts'
import {
  appendToolCallDelta,
  createToolCallAccumulator,
  finalizeToolCalls,
  inlineToolCalls,
} from '../src/lib/tool_calls.ts'

const DATAKIT_PROTOCOL = {
  thinking_start: '<|start_think|>',
  thinking_end: '<|end_think|>',
  tool_call_start: '<tool_call>',
  tool_call_end: '</tool_call>',
  tool_call_format: 'delimited',
}

const DELPHI_PROTOCOL = {
  thinking_start: '<|start_think|>',
  thinking_end: '<|end_think|>',
  tool_call_start: '<|tool_call|>',
  tool_call_end: '<|tool_call_end|>',
  tool_call_format: 'delimited',
}

const LLAMA_PROTOCOL = {
  thinking_start: null,
  thinking_end: null,
  tool_call_start: null,
  tool_call_end: null,
  tool_call_format: 'json',
}

test('builds the exposed chat template arguments alongside typed tools', () => {
  const tools = [
    {
      type: 'function',
      function: { name: 'lookup_weather', description: '', parameters: { type: 'object' } },
    },
  ]

  assert.deepEqual(chatTemplateRequestFields(false, '  Be concise.  ', tools), {
    tools,
    chat_template_kwargs: {
      enable_thinking: false,
      custom_instructions: 'Be concise.',
    },
  })
  assert.deepEqual(chatTemplateRequestFields(null, '   ', []), {})
})

test('splits Marin special-token reasoning from visible content', () => {
  assert.deepEqual(
    splitThinking(
      '<|start_think|>Use the calculator.<|end_think|><tool_call>{}</tool_call>',
      DATAKIT_PROTOCOL,
    ),
    {
      thinking: 'Use the calculator.',
      visible: '<tool_call>{}</tool_call>',
      inThinking: false,
    },
  )
})

test('splits reasoning when the template emitted its opening token in the prompt', () => {
  assert.deepEqual(splitThinking('Use the calculator.<|end_think|>The result is ready.', DATAKIT_PROTOCOL), {
    thinking: 'Use the calculator.',
    visible: 'The result is ready.',
    inThinking: false,
  })
})

test('recognizes known thinking tokens when template metadata is absent or stale', () => {
  assert.deepEqual(splitThinking('<|start_th', null), {
    thinking: '',
    visible: '',
    inThinking: false,
  })
  assert.deepEqual(splitThinking('<think>Use the calculator.</think>The result is ready.', null), {
    thinking: 'Use the calculator.',
    visible: 'The result is ready.',
    inThinking: false,
  })
  assert.deepEqual(
    splitThinking('<|start_think|>Use the calculator.<|end_think|>The result is ready.', LLAMA_PROTOCOL),
    {
      thinking: 'Use the calculator.',
      visible: 'The result is ready.',
      inThinking: false,
    },
  )
  assert.deepEqual(splitThinking('Use the calculator.<|end_think|>The result is ready.', null), {
    thinking: 'Use the calculator.',
    visible: 'The result is ready.',
    inThinking: false,
  })
})

test('accumulates an OpenAI streamed tool call', () => {
  const accumulator = createToolCallAccumulator()
  appendToolCallDelta(accumulator, [
    {
      index: 0,
      id: 'call_compound',
      type: 'function',
      function: { name: 'compound_interest', arguments: '{"principal":2500,' },
    },
  ])
  appendToolCallDelta(accumulator, [
    {
      index: 0,
      function: { arguments: '"annual_rate_percent":4.5,"years":8}' },
    },
  ])

  assert.deepEqual(finalizeToolCalls(accumulator, () => 'unused'), [
    {
      id: 'call_compound',
      name: 'compound_interest',
      arguments: { principal: 2500, annual_rate_percent: 4.5, years: 8 },
    },
  ])
})

test('sends a matched tool result to the model on the next request', () => {
  const conversation = {
    system: '',
    enableThinking: null,
    customInstructions: '',
    messages: [
      { role: 'user', content: 'Calculate compound interest.' },
      {
        role: 'assistant',
        content: '',
        thinking: 'Use the calculator.',
        thinkingSeconds: 0.2,
        error: null,
        toolCalls: [
          {
            id: 'call_compound',
            name: 'compound_interest',
            arguments: { principal: 2500, annual_rate_percent: 4.5, years: 8 },
          },
        ],
      },
      { role: 'tool', name: 'compound_interest', toolCallId: 'call_compound', result: 3555.25 },
    ],
  }

  assert.deepEqual(modelMessages(conversation), [
    { role: 'user', content: 'Calculate compound interest.' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        {
          id: 'call_compound',
          type: 'function',
          function: {
            name: 'compound_interest',
            arguments: '{"principal":2500,"annual_rate_percent":4.5,"years":8}',
          },
        },
      ],
    },
    {
      role: 'tool',
      name: 'compound_interest',
      tool_call_id: 'call_compound',
      content: '3555.25',
    },
  ])
})

test('renders the whole chat as one plain-text debugging transcript', () => {
  const conversation = {
    system: 'Use tools when helpful.',
    messages: [
      { role: 'user', content: 'Calculate compound interest.' },
      {
        role: 'assistant',
        content: '',
        thinking: 'parsed reasoning',
        rawContent: '<tool_call>{"name":"compound_interest"}</tool_call>',
        rawReasoning: 'Use the calculator.',
        thinkingSeconds: 0.2,
        error: null,
        toolCalls: [
          {
            id: 'call_compound',
            name: 'compound_interest',
            arguments: { principal: 2500 },
          },
        ],
      },
      { role: 'tool', name: 'compound_interest', toolCallId: 'call_compound', result: 3555.25 },
      {
        role: 'assistant',
        content: 'The balance is $3,555.25.',
        thinking: '',
        thinkingSeconds: null,
        error: null,
        toolCalls: [],
      },
    ],
  }

  assert.equal(
    plainTextChat(conversation),
    `[system]
Use tools when helpful.

[user]
Calculate compound interest.

[assistant reasoning]
Use the calculator.

[assistant]
<tool_call>{"name":"compound_interest"}</tool_call>

[assistant tool_calls]
[
  {
    "id": "call_compound",
    "name": "compound_interest",
    "arguments": {
      "principal": 2500
    }
  }
]

[tool compound_interest (call_compound)]
3555.25

[assistant]
The balance is $3,555.25.`,
  )
})

test('parses the Datakit inline tool-call format', () => {
  assert.deepEqual(
    inlineToolCalls(
      '<tool_call>{"name":"compound_interest","arguments":{"principal":2500}}</tool_call>',
      DATAKIT_PROTOCOL,
      () => 'inline',
    ),
    {
      visible: '',
      calls: [
        {
          id: 'call_inline',
          name: 'compound_interest',
          arguments: { principal: 2500 },
        },
      ],
    },
  )
})

test('parses Delphi tool tokens selected from its chat template', () => {
  assert.deepEqual(
    inlineToolCalls(
      '<|tool_call|>{"name":"compound_interest","arguments":{"principal":2500}}<|tool_call_end|>',
      DELPHI_PROTOCOL,
      () => 'delphi',
    ),
    {
      visible: '',
      calls: [
        {
          id: 'call_delphi',
          name: 'compound_interest',
          arguments: { principal: 2500 },
        },
      ],
    },
  )
})

test('parses the bare JSON call selected by a Llama tool template', () => {
  assert.deepEqual(
    inlineToolCalls(
      '{"name":"compound_interest","parameters":{"principal":2500}}',
      LLAMA_PROTOCOL,
      () => 'llama',
    ),
    {
      visible: '',
      calls: [
        {
          id: 'call_llama',
          name: 'compound_interest',
          arguments: { principal: 2500 },
        },
      ],
    },
  )
  assert.deepEqual(inlineToolCalls('{"answer":42}', LLAMA_PROTOCOL, () => 'unused'), {
    visible: '{"answer":42}',
    calls: [],
  })
})
