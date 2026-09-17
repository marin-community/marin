import assert from 'node:assert/strict'
import test from 'node:test'

import { chatTemplateRequestFields, ThinkingMode } from '../src/lib/chat_template.ts'
import { modelMessages } from '../src/lib/python_tools.ts'
import { plainTextChat } from '../src/lib/plain_text_chat.ts'
import { BASH_TOOL_DEFINITION, bashCommand, parseWorkspaceFiles } from '../src/lib/shell_workspace.ts'
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

  assert.deepEqual(chatTemplateRequestFields(ThinkingMode.Disabled, '  Be concise.  ', tools), {
    tools,
    chat_template_kwargs: {
      enable_thinking: false,
      custom_instructions: 'Be concise.',
    },
  })
  assert.deepEqual(chatTemplateRequestFields(ThinkingMode.TemplateDefault, '   ', []), {})
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
    thinkingMode: ThinkingMode.TemplateDefault,
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

  const transcript = plainTextChat(conversation)
  assert.ok(transcript.startsWith('[system]\nUse tools when helpful.\n\n[user]\nCalculate compound interest.'))
  assert.ok(transcript.includes('[assistant reasoning]\nUse the calculator.'))
  assert.ok(transcript.includes('[assistant]\n<tool_call>{"name":"compound_interest"}</tool_call>'))
  assert.ok(transcript.includes('"id": "call_compound"'))
  assert.ok(transcript.includes('[tool compound_interest (call_compound)]\n3555.25'))
  assert.ok(transcript.endsWith('[assistant]\nThe balance is $3,555.25.'))
  assert.ok(!transcript.includes('parsed reasoning'))
})

for (const [format, protocol, payload] of [
  [
    'Datakit delimiters',
    DATAKIT_PROTOCOL,
    '<tool_call>{"name":"compound_interest","arguments":{"principal":2500}}</tool_call>',
  ],
  [
    'Delphi delimiters',
    DELPHI_PROTOCOL,
    '<|tool_call|>{"name":"compound_interest","arguments":{"principal":2500}}<|tool_call_end|>',
  ],
  ['Llama JSON', LLAMA_PROTOCOL, '{"name":"compound_interest","parameters":{"principal":2500}}'],
]) {
  test(`parses the ${format} tool-call format`, () => {
    assert.deepEqual(inlineToolCalls(payload, protocol, () => 'generated'), {
      visible: '',
      calls: [
        {
          id: 'call_generated',
          name: 'compound_interest',
          arguments: { principal: 2500 },
        },
      ],
    })
  })
}

test('leaves a non-call JSON response visible', () => {
  assert.deepEqual(inlineToolCalls('{"answer":42}', LLAMA_PROTOCOL, () => 'unused'), {
    visible: '{"answer":42}',
    calls: [],
  })
})

test('defines and validates the Bash workspace tool', () => {
  assert.equal(BASH_TOOL_DEFINITION.function.name, 'bash')
  assert.doesNotMatch(BASH_TOOL_DEFINITION.function.description, /ShellSim/)
  assert.deepEqual(parseWorkspaceFiles('{"README.md":"# Example\\n","src/main.py":"print(42)\\n"}'), {
    'README.md': '# Example\n',
    'src/main.py': 'print(42)\n',
  })
  assert.equal(bashCommand({ command: 'git status --short' }), 'git status --short')
  assert.throws(() => parseWorkspaceFiles('["not", "a", "file map"]'), /JSON object/)
  assert.throws(() => bashCommand({ command: 42 }), /non-empty command string/)
})
