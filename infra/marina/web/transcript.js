/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
/* The chat-transcript renderer shared by Marina apps, extracted from EvalDash's sample views.
 *
 * One renderer for chat evidence everywhere: role-labeled turn cards, fenced code and prose
 * segments, long messages collapsed to a preview, tool calls and their results as labeled blocks
 * with folded output, and reasoning as a collapsed disclosure. Styling lives in `transcript.css`,
 * which carries neutral defaults; a host app maps its `--tr-*` variables onto its own tokens.
 *
 * The file is consumed two ways, so it must not use `import`/`export` syntax: bundlers load it
 * as CommonJS through the `@marina` alias, and no-build apps load it as a classic script and read
 * the `TranscriptView` global.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  } else {
    root.TranscriptView = api;
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // The message preview EvalDash shows before offering the full text.
  const MESSAGE_PREVIEW = 600;
  // Tool outputs longer than this start fully collapsed instead of scrolling.
  const RESULT_FOLD_CHARS = 2400;

  function el(tag, className, text) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (text !== undefined && text !== null) item.textContent = text;
    return item;
  }

  /** Free-form model output that may contain ``` fenced blocks: prose stays wrapped text,
   * fenced segments render as monospace blocks (their language-tag line is dropped). */
  function fenceSegments(text) {
    const parts = String(text).split('```');
    const segments = [];
    parts.forEach((part, index) => {
      if (index % 2 === 0) {
        if (part) segments.push({ kind: 'prose', text: part });
        return;
      }
      const newline = part.indexOf('\n');
      const body = newline >= 0 ? part.slice(newline + 1) : part;
      segments.push({ kind: 'code', text: body.replace(/\n$/, '') });
    });
    return segments;
  }

  function renderFenced(container, text) {
    for (const segment of fenceSegments(text)) {
      if (segment.kind === 'prose') container.append(el('p', 'tr-prose', segment.text));
      else container.append(el('pre', 'tr-code', segment.text));
    }
  }

  function prettyJson(value) {
    return JSON.stringify(value, null, 2);
  }

  // Terminal panes pad their output with runs of blank lines to a fixed height; collapse those
  // runs and drop trailing whitespace so an observation shows its content instead of a screen
  // of blank lines.
  function normalizeOutput(text) {
    return String(text).replace(/\n{3,}/g, '\n\n').replace(/\s+$/, '');
  }

  /** The message body of one turn: fenced rendering, with long text collapsed to a preview
   * behind a Show-full toggle. */
  function renderMessageBody(container, text, previewChars) {
    const limit = previewChars == null ? MESSAGE_PREVIEW : previewChars;
    const long = text.length > limit;
    const body = el('div', 'tr-message');
    renderFenced(body, long ? text.slice(0, limit) + '…' : text);
    if (long) {
      const toggle = el('button', 'tr-toggle', `Show full message (${text.length} chars)`);
      toggle.type = 'button';
      let expanded = false;
      toggle.addEventListener('click', () => {
        expanded = !expanded;
        body.replaceChildren();
        renderFenced(body, expanded ? text : text.slice(0, limit) + '…');
        toggle.textContent = expanded ? 'Collapse' : `Show full message (${text.length} chars)`;
      });
      container.append(body, toggle);
      return;
    }
    container.append(body);
  }

  /** Reasoning that a trace carries beside the answer, collapsed by default. */
  function renderReasoning(container, text) {
    const details = el('details', 'tr-reasoning');
    details.append(el('summary', 'tr-reasoning-summary', 'Reasoning'));
    const body = el('div', 'tr-reasoning-body');
    renderFenced(body, normalizeOutput(text));
    details.append(body);
    container.append(details);
  }

  /** One tool call: its name and its JSON arguments. */
  function renderToolCall(container, call) {
    const block = el('div', 'tr-tool-call');
    const head = el('div', 'tr-tool-name');
    head.append(el('span', 'tr-tool-label', 'Tool call'));
    head.append(el('span', 'tr-tool-target', call.name));
    block.append(head);
    if (call.callId) block.append(el('div', 'tr-tool-id', call.callId));
    const args = call.arguments;
    if (args !== undefined && args !== null && !(typeof args === 'object' && Object.keys(args).length === 0)) {
      const pre = el('pre', 'tr-tool-args', typeof args === 'string' ? args : prettyJson(args));
      block.append(pre);
    }
    container.append(block);
  }

  /** One tool result: its content (or structured shape), folded for long output. */
  function renderToolResult(container, result, options) {
    const block = el('div', 'tr-tool-result');
    const head = el('div', 'tr-result-label');
    head.append(el('span', 'tr-tool-label', 'Tool result'));
    if (result.callId) head.append(el('span', 'tr-tool-target', result.callId));
    block.append(head);
    const collapsedMode = options.resultMode === 'collapsed';
    const text = typeof result.content === 'string' ? normalizeOutput(result.content) : null;
    const body = text === null ? el('pre', 'tr-tool-args', prettyJson(result.content ?? result)) : el('pre', 'tr-tool-output', text);
    const long = text !== null && text.length > RESULT_FOLD_CHARS;
    if (collapsedMode && long) {
      const toggle = el('button', 'tr-toggle', `Show full output (${text.length} chars)`);
      toggle.type = 'button';
      const preview = el('pre', 'tr-tool-output tr-folded', text.slice(0, 800) + '…');
      let expanded = false;
      toggle.addEventListener('click', () => {
        expanded = !expanded;
        if (expanded) {
          preview.classList.remove('tr-folded');
          preview.textContent = text;
          toggle.textContent = 'Fold output';
        } else {
          preview.classList.add('tr-folded');
          preview.textContent = text.slice(0, 800) + '…';
          toggle.textContent = `Show full output (${text.length} chars)`;
        }
      });
      block.append(preview, toggle);
    } else {
      block.append(body);
    }
    container.append(block);
  }

  /** Normalize any accepted transcript shape into turns.
   * Accepted: `{events: [...]}` (the typed trace), a bare event array, or plain
   * `{role, content}` messages. */
  function normalizeTurns(source) {
    const events = Array.isArray(source) ? source : source && source.events;
    if (!Array.isArray(events)) return [];
    return events.map((event) => {
      if (event && typeof event === 'object') {
        if (event.type === 'assistant_tool_calls') {
          return {
            kind: 'assistant_tool_calls',
            role: 'assistant',
            content: event.content ?? null,
            reasoning: event.reasoning ?? event.reasoning_content ?? null,
            calls: (event.calls ?? []).map((call) => ({
              callId: call.call_id ?? call.callId ?? null,
              name: call.name ?? call.function_name ?? '',
              arguments: call.arguments ?? call.arguments_json ?? {},
            })),
          };
        }
        if (event.type === 'tool_result') {
          return {
            kind: 'tool_result',
            role: 'tool',
            callId: event.call_id ?? event.callId ?? null,
            content: event.content,
          };
        }
        if (typeof event.role === 'string' || typeof event.content === 'string') {
          return {
            kind: 'message',
            role: typeof event.role === 'string' ? event.role : 'assistant',
            content: typeof event.content === 'string' ? event.content : '',
            reasoning: event.reasoning ?? event.reasoning_content ?? null,
          };
        }
      }
      return { kind: 'message', role: 'assistant', content: typeof event === 'string' ? event : prettyJson(event) };
    });
  }

  // Left-border accents per role, matching the chat transcript palette: system and developer
  // read as scaffolding, the user as the request, the assistant as the answer.
  const ROLE_TONES = { system: 'muted', developer: 'muted', user: 'accent', assistant: 'success' };

  /** One labeled turn card: role header, reasoning disclosure, message, tool blocks. */
  function renderTurn(container, turn, options) {
    const card = el('div', 'tr-turn');
    card.classList.add('tr-tone-' + (ROLE_TONES[turn.role] || 'neutral'));
    const head = el('div', 'tr-turn-label');
    head.textContent = (options && options.labelFor ? options.labelFor(turn) : turn.kind === 'tool_result' ? 'tool result' : turn.role);
    card.append(head);
    if (turn.reasoning) renderReasoning(card, turn.reasoning);
    if (turn.kind === 'assistant_tool_calls') {
      if (turn.content) renderMessageBody(card, turn.content, options && options.messagePreviewChars);
      const calls = el('div', 'tr-calls');
      for (const call of turn.calls) renderToolCall(calls, call);
      card.append(calls);
    } else if (turn.kind === 'tool_result') {
      renderToolResult(card, turn, options || {});
    } else {
      renderMessageBody(card, turn.content ?? '', options && options.messagePreviewChars);
    }
    container.append(card);
  }

  /** A sequence of turn cards. `source` accepts any shape `normalizeTurns` understands. */
  function renderTranscript(container, source, options) {
    container.classList.add('tr-transcript');
    for (const turn of normalizeTurns(source)) renderTurn(container, turn, options);
  }

  /** One step card in EvalDash's trajectory shape: a labeled header row (step id, source chip,
   * model, metrics), the model message, tool calls, and the resulting observations. */
  function renderStepCard(container, step, options) {
    container.classList.add('tr-step-card');
    container.classList.add('tr-tone-' + ((step.source === 'agent' ? 'success' : step.source === 'user' ? 'muted' : null) || 'neutral'));
    const head = el('div', 'tr-step-head');
    head.append(el('span', 'tr-step-title', 'Step ' + step.step_id));
    if (step.source) head.append(el('span', 'tr-step-chip', step.source));
    if (step.model_name) head.append(el('span', 'tr-step-model', step.model_name));
    for (const [name, value] of Object.entries(step.metrics ?? {})) {
      head.append(el('span', 'tr-step-metric', `${name} ${value}`));
    }
    container.append(head);
    if (step.message) renderMessageBody(container, step.message, options && options.messagePreviewChars);
    if (step.tool_calls && step.tool_calls.length) {
      const calls = el('div', 'tr-calls');
      for (const call of step.tool_calls) {
        renderToolCall(calls, {
          callId: call.tool_call_id ?? null,
          name: call.function_name,
          arguments: call.arguments,
        });
      }
      container.append(calls);
    }
    const results = (step.observation && step.observation.results) || [];
    for (const result of results) {
      renderToolResult(
        container,
        { callId: result.source_call_id ?? null, content: result.content ?? result },
        Object.assign({ resultMode: 'scroll' }, options),
      );
    }
  }

  return {
    fenceSegments: fenceSegments,
    normalizeTurns: normalizeTurns,
    renderTranscript: renderTranscript,
    renderStepCard: renderStepCard,
    renderFenced: renderFenced,
    renderMessageBody: renderMessageBody,
  };
});
