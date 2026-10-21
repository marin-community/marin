"""In-process Pi HTTP transport for exact served vLLM evidence.

Pi's normal events lack sampled IDs/logprobs. Its custom provider redirects
through this loopback server in the same Pi process, which forwards SSE and
retains completed calls in memory. No model evidence is written to disk.
"""

RECORDER_SCRIPT = r"""
import http from 'node:http';
import fs from 'node:fs';
import { randomUUID } from 'node:crypto';

export function recordDiagnostic(phase, fields) {
  if (process.env.HARBOR_PI_SANDBOX_DIAGNOSTICS !== '1') return;
  fs.appendFileSync(process.env.HARBOR_PI_SANDBOX_DIAGNOSTICS_PATH,
    JSON.stringify({ source: 'recorder', phase, ...fields }) + '\n', { mode: 0o600 });
}

const upstream = process.env.HARBOR_PI_UPSTREAM_BASE;
const records = [];
const daggerTurns = [];
const dagger = process.env.HARBOR_PI_DAGGER_PLAN ? JSON.parse(process.env.HARBOR_PI_DAGGER_PLAN) : null;
const captureActor = process.env.HARBOR_PI_CAPTURE_ACTOR ?? null;
const critiquePlan = process.env.HARBOR_PI_CRITIQUE_PLAN ? JSON.parse(process.env.HARBOR_PI_CRITIQUE_PLAN) : null;
let critique = null;
let critiqueFailure = null;
if (!upstream) throw new Error('HARBOR_PI_UPSTREAM_BASE is required');
if ((dagger && critiquePlan) || (critiquePlan && captureActor)) {
  throw new Error('Pi critique cannot share another actor controller');
}
if (captureActor !== null && (captureActor !== 'teacher' || dagger)) {
  throw new Error('Pi capture actor must be teacher-only and cannot accompany DAgger');
}

function tokenIds(choice) {
  return choice?.delta?.provider_specific_fields?.token_ids ?? choice?.delta?.token_ids ??
    choice?.provider_specific_fields?.token_ids ?? choice?.token_ids;
}

function logprobs(choice) {
  const lp = choice?.logprobs;
  if (Array.isArray(lp?.token_logprobs)) return lp.token_logprobs;
  if (Array.isArray(lp?.content)) return lp.content.map(item => item.logprob);
  return undefined;
}

function accumulator() {
  return {
    prompt_token_ids: null, completion_token_ids: [], logprobs: [], model: null,
    response_id: null, assistant_content: '', assistant_tool_calls: new Map(),
    assistant_complete: false, assistant_invalid: false,
  };
}

function accumulate(state, chunk) {
  if (state.prompt_token_ids === null && Array.isArray(chunk.prompt_token_ids)) {
    state.prompt_token_ids = chunk.prompt_token_ids;
  }
  for (const choice of chunk.choices ?? []) {
    if (state.prompt_token_ids === null && Array.isArray(choice.prompt_token_ids)) {
      state.prompt_token_ids = choice.prompt_token_ids;
    }
    const ids = tokenIds(choice);
    if (Array.isArray(ids)) state.completion_token_ids.push(...ids);
    const probs = logprobs(choice);
    if (Array.isArray(probs)) state.logprobs.push(...probs);
    const delta = choice.delta;
    if (delta && typeof delta === 'object') {
      if (delta.content !== null && delta.content !== undefined) {
        if (typeof delta.content === 'string') state.assistant_content += delta.content;
        else state.assistant_invalid = true;
      }
      for (const piece of delta.tool_calls ?? []) {
        if (!Number.isSafeInteger(piece.index) || piece.index < 0) {
          state.assistant_invalid = true;
          continue;
        }
        const call = state.assistant_tool_calls.get(piece.index) ??
          { id: '', type: 'function', function: { name: '', arguments: '' } };
        if (piece.id !== undefined) {
          if (typeof piece.id !== 'string') state.assistant_invalid = true;
          else call.id += piece.id;
        }
        if (piece.type !== undefined && piece.type !== 'function') state.assistant_invalid = true;
        for (const field of ['name', 'arguments']) {
          const value = piece.function?.[field];
          if (value !== undefined) {
            if (typeof value !== 'string') state.assistant_invalid = true;
            else call.function[field] += value;
          }
        }
        state.assistant_tool_calls.set(piece.index, call);
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) state.assistant_complete = true;
  }
  if (chunk.id) state.response_id = chunk.id;
  if (chunk.model) state.model = chunk.model;
}

function literal(state) {
  const calls = [...state.assistant_tool_calls.entries()].sort(([a], [b]) => a - b).map(([, call]) => call);
  let assistant = null;
  if (state.assistant_complete && !state.assistant_invalid &&
      calls.every(call => call.id && call.function.name)) {
    assistant = { role: 'assistant', content: state.assistant_content || null };
    if (calls.length) assistant.tool_calls = calls;
  }
  return {
    prompt_token_ids: state.prompt_token_ids,
    completion_token_ids: state.completion_token_ids.length ? state.completion_token_ids : null,
    logprobs: state.logprobs.length ? state.logprobs : null,
    model: state.model, response_id: state.response_id, assistant_message: assistant,
  };
}

function enrich(body, teacher = false) {
  // Teacher-trace SFT consumes actions, not teacher probabilities. Never ask
  // either the executed teacher or a private DAgger label for unused scores.
  // Student/revision behavior still needs its sampled per-token logprobs.
  const { logprobs: _logprobs, top_logprobs: _topLogprobs, ...request } = body;
  return { ...request, ...(teacher ? {} : { logprobs: true, top_logprobs: body.top_logprobs ?? 1 }),
    return_token_ids: true };
}

function upstreamUrl(requestPath, baseUrl) {
  // Iris capability URLs carry the credential in the path. An absolute /v1/ URL
  // would silently discard that prefix and never reach the registered endpoint.
  if (!requestPath.startsWith('/v1/')) throw new Error('Unexpected Pi recorder request path');
  const relative = requestPath.slice('/v1/'.length);
  if (!relative || relative.startsWith('/') || relative.startsWith('..')) {
    throw new Error('Invalid Pi recorder request path');
  }
  return new URL(relative, `${baseUrl.replace(/\/$/, '')}/`);
}

async function capture(request, response, payload, baseUrl, headers, bufferResponse) {
  const upstreamResponse = await fetch(upstreamUrl(request.url, baseUrl), {
    method: request.method, headers,
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const contentType = upstreamResponse.headers.get('content-type') ?? 'application/json';
  if (request.url === '/v1/chat/completions') {
    const cap = payload.max_completion_tokens ?? payload.max_tokens;
    recordDiagnostic('upstream_status', { http_status: upstreamResponse.status,
      max_completion_tokens: Number.isSafeInteger(cap) ? cap : null,
      streaming: payload.stream === true });
  }
  if (!bufferResponse) response.writeHead(upstreamResponse.status, { 'content-type': contentType });
  const state = accumulator();
  let pending = '';
  let nonStreaming = '';
  const buffered = [];
  const decoder = new TextDecoder();
  for await (const chunk of upstreamResponse.body) {
    if (bufferResponse) buffered.push(chunk);
    else response.write(chunk);
    const text = decoder.decode(chunk, { stream: true });
    if (request.method === 'POST' && request.url === '/v1/chat/completions' && upstreamResponse.ok) {
      if (payload.stream) {
        pending += text;
        const events = pending.replaceAll('\r\n', '\n').split('\n\n');
        pending = events.pop();
        for (const event of events) {
          const data = event.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
          if (data && data !== '[DONE]') accumulate(state, JSON.parse(data));
        }
      } else nonStreaming += text;
    }
  }
  let captured = null;
  if (request.method === 'POST' && request.url === '/v1/chat/completions' && upstreamResponse.ok) {
    if (payload.stream) {
      if (pending.trim() && pending.trim() !== 'data: [DONE]') throw new Error('incomplete upstream SSE');
      captured = literal(state);
    } else {
      const parsed = JSON.parse(nonStreaming);
      const choice = parsed.choices?.[0];
      captured = {
        prompt_token_ids: parsed.prompt_token_ids ?? choice?.prompt_token_ids ?? null,
        completion_token_ids: tokenIds(choice) ?? null,
        logprobs: logprobs(choice) ?? null,
        model: parsed.model, response_id: parsed.id,
        assistant_message: choice?.message ?? null,
      };
    }
  }
  if (request.url === '/v1/chat/completions') {
    recordDiagnostic('literal_evidence', { http_status: upstreamResponse.status,
      prompt_token_count: Array.isArray(captured?.prompt_token_ids) ? captured.prompt_token_ids.length : null,
      completion_token_count: Array.isArray(captured?.completion_token_ids) ? captured.completion_token_ids.length : null,
      logprob_count: Array.isArray(captured?.logprobs) ? captured.logprobs.length : null,
      has_assistant: captured?.assistant_message?.role === 'assistant',
      tool_call_count: captured?.assistant_message?.tool_calls?.length ?? 0 });
  }
  return {
    record: { request_id: randomUUID(), status_code: upstreamResponse.status, request: payload, literal: captured },
    status: upstreamResponse.status, contentType, buffered,
  };
}

function recordObservation(event, content) {
  const entry = records.at(-1);
  const calls = entry?.literal?.assistant_message?.tool_calls ?? [];
  if (!entry || !calls.some(call => call.id === event.toolCallId)) {
    throw new Error('Pi executed an unrecorded tool call');
  }
  if (!Array.isArray(content) || content.some(block => block.type !== 'text' || typeof block.text !== 'string')) {
    throw new Error('Pi tool observation is not representable as an exact text message');
  }
  entry.observations ??= [];
  if (entry.observations.some(item => item.tool_call_id === event.toolCallId)) {
    throw new Error('Pi duplicated an executed tool observation');
  }
  entry.observations.push({ tool_call_id: event.toolCallId,
    content: content.map(block => block.text).join('\n'),
    authored_by: entry.actor === 'teacher' ? 'teacher' : 'student' });
}

function modelEvidence(entry, actor) {
  requireLiteral(entry, actor === 'teacher');
  const request = entry.request;
  if (!Array.isArray(request?.messages) || !request.messages.length || !Array.isArray(request.tools ?? [])) {
    throw new Error('Pi model request has no visited messages or malformed tools');
  }
  return { request_id: entry.request_id, model: request.model, messages: request.messages,
    tools: request.tools ?? [], chat_template_kwargs: request.chat_template_kwargs ?? {},
    assistant: entry.literal.assistant_message,
    prompt_token_ids: entry.literal.prompt_token_ids,
    response_token_ids: entry.literal.completion_token_ids, logprobs: entry.literal.logprobs };
}

function buildTrajectory() {
  if (!records.length) throw new Error('Pi trajectory must include a sampled turn');
  const trialId = process.env.HARBOR_PI_TRIAL_ID ?? dagger?.trial_id ?? critiquePlan?.trial_id;
  const policyStep = process.env.HARBOR_PI_POLICY_STEP === undefined
    ? dagger?.policy_step ?? 0 : Number(process.env.HARBOR_PI_POLICY_STEP);
  if (typeof trialId !== 'string' || !trialId || !Number.isSafeInteger(policyStep) || policyStep < 0 ||
      (dagger && (dagger.trial_id !== trialId || dagger.policy_step !== policyStep)) ||
      (critiquePlan && critiquePlan.trial_id !== trialId)) {
    throw new Error('Pi trajectory has no valid bound trial identity and policy step');
  }
  const turns = records.map((entry, index) => {
    const actor = entry.actor ?? 'student';
    const label = dagger ? daggerTurns[index]?.teacher_label : null;
    const calls = entry.literal?.assistant_message?.tool_calls ?? [];
    const observed = entry.observations ?? [];
    const observations = calls.map(call => observed.find(item => item.tool_call_id === call.id));
    if (calls.length !== observed.length || observations.some(item => item === undefined)) {
      throw new Error('Pi trajectory is missing an executed tool observation');
    }
    if (dagger && (!label || daggerTurns[index]?.actor !== actor ||
        daggerTurns[index]?.executed_request_id !== entry.request_id)) {
      throw new Error('Pi trajectory has no matching private teacher label');
    }
    const nextMessages = records[index + 1]?.request?.messages;
    const executedAssistant = nextMessages?.[entry.request.messages.length] ?? entry.literal.assistant_message;
    return { index, actor, executed: modelEvidence(entry, actor), executed_assistant: executedAssistant,
      observations, teacher_label: label ? modelEvidence(label, 'teacher') : null };
  });
  if (critiquePlan && (!critique || critiqueFailure)) throw new Error('Pi trajectory has no private critique');
  return { schema_version: 1, trial_id: trialId, policy_step: policyStep,
    turns, planned_executors: dagger ? dagger.executors.slice(0, turns.length) : null,
    critique_feedback: critiquePlan ? critique.feedback : null,
    critique: critiquePlan ? critique : null };
}

function requireLiteral(record, teacher = false) {
  const evidence = record.literal;
  const ids = evidence?.completion_token_ids;
  const probs = evidence?.logprobs;
  if (record.status_code !== 200 || !evidence?.assistant_message ||
      !Array.isArray(evidence.prompt_token_ids) ||
      !Array.isArray(ids) || !ids.length ||
      !((teacher && probs === null) || (Array.isArray(probs) && probs.length === ids.length))) {
    throw new Error('Pi DAgger teacher/student model returned incomplete literal evidence');
  }
}

async function requestCritique(toolResults) {
  if (!critiquePlan || critique || records.length !== 1 || critiqueFailure) {
    throw new Error('Pi critique requires exactly one executed student turn');
  }
  const entry = records[0];
  requireLiteral(entry);
  const assistant = entry.literal.assistant_message;
  const calls = assistant.tool_calls ?? [];
  const results = toolResults ?? [];
  if (calls.length !== results.length || calls.some((call, index) =>
    results[index]?.toolCallId !== call.id || !Array.isArray(results[index]?.content))) {
    throw new Error('Pi critique needs the executed observations for every tool call');
  }
  const messages = [
    ...entry.request.messages,
    assistant,
    ...results.map(result => ({ role: 'tool', tool_call_id: result.toolCallId,
      content: result.content.filter(block => block.type === 'text').map(block => block.text).join('\n') })),
    { role: 'user', content: `Critique this student attempt according to this principle: ${critiquePlan.principle}. ` +
      'Give concrete feedback for a student revision; do not take actions yourself.' },
  ];
  const request = { model: critiquePlan.teacher_model, messages, stream: false };
  const response = await fetch(new URL('chat/completions', critiquePlan.teacher_api_base + '/'), {
    method: 'POST', headers: { 'content-type': 'application/json',
      authorization: `Bearer ${process.env.HARBOR_PI_TEACHER_API_KEY ?? 'not-needed'}` },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(`Pi critique teacher returned HTTP ${response.status}`);
  const result = await response.json();
  const text = result?.choices?.[0]?.message?.content;
  if (typeof text !== 'string' || !text.trim()) throw new Error('Pi critique teacher returned no feedback');
  const feedback = `Teacher feedback on your previous attempt (not an executed action):\n${text.trim()}\n` +
    'Revise your answer. The feedback is advice, not an instruction to execute a teacher action.';
  critique = { trial_id: critiquePlan.trial_id, teacher_model: critiquePlan.teacher_model,
    student_request_id: entry.request_id, teacher_request_id: randomUUID(),
    teacher_request: request, teacher_response: result.choices[0].message, feedback };
  return feedback;
}

async function relay(request, response) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  const body = Buffer.concat(chunks).toString('utf8');
  const isModelCall = request.method === 'POST' && request.url === '/v1/chat/completions';
  const payload = body ? JSON.parse(body) : undefined;
  const served = isModelCall ? enrich(payload, captureActor === 'teacher') : payload;
  const headers = {};
  for (const name of ['authorization', 'x-session-id', 'x-ot-trial-id']) {
    if (request.headers[name]) headers[name] = request.headers[name];
  }
  if (served !== undefined) headers['content-type'] = 'application/json';
  if (critiquePlan && isModelCall) {
    if (records.length >= 2) throw new Error('Pi critique permits one student attempt and one revision only');
    if (records.length === 1) {
      const last = served.messages?.at(-1);
      if (!critique || critiqueFailure || last?.role !== 'user' ||
          JSON.stringify(last.content) !== JSON.stringify([{ type: 'text', text: critique.feedback }])) {
        throw new Error('Pi revision may not sample until labeled teacher feedback is in its context');
      }
    }
  }
  if (dagger && isModelCall) {
    const turn = records.length;
    const executor = dagger.executors[turn];
    if (executor !== 'student' && executor !== 'teacher') throw new Error('Pi DAgger exhausted its pre-action plan');
    const teacherPayload = enrich({ ...served, model: dagger.teacher_model }, true);
    const teacherHeaders = { ...headers, authorization: `Bearer ${process.env.HARBOR_PI_TEACHER_API_KEY ?? 'not-needed'}` };
    const label = await capture(request, response, teacherPayload, dagger.teacher_api_base, teacherHeaders, true);
    requireLiteral(label.record, true);
    let executed = label;
    if (executor === 'teacher') {
      response.writeHead(label.status, { 'content-type': label.contentType });
      for (const chunk of label.buffered) response.write(chunk);
    } else {
      executed = await capture(request, response, served, upstream, headers, false);
      requireLiteral(executed.record);
    }
    records.push({ ...executed.record, actor: executor });
    daggerTurns.push({ turn, actor: executor, teacher_label: label.record,
      executed_request_id: executed.record.request_id });
  } else {
    const result = await capture(request, response, served, upstream, headers, false);
    if (isModelCall) records.push(critiquePlan
      ? { ...result.record, actor: records.length === 0 ? 'student' : 'revision' }
      : captureActor ? { ...result.record, actor: captureActor } : result.record);
  }
  response.end();
}

export function startRecorder() {
  const active = new Set();
  const server = http.createServer((request, response) => {
    const job = relay(request, response).catch(error => {
      console.error('Harbor Pi literal recorder:', error);
      if (!response.headersSent) response.writeHead(502);
      response.destroy(error);
    });
    active.add(job);
    void job.finally(() => active.delete(job));
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      resolve({ url: `http://127.0.0.1:${server.address().port}/v1`, records, daggerTurns,
        get critique() { return critique; }, get critiqueFailure() { return critiqueFailure; },
        failCritique(error) { critiqueFailure = String(error); }, requestCritique,
        recordObservation, buildTrajectory, server,
        async waitForIdle() { await Promise.all([...active]); } });
    });
  });
}
"""
