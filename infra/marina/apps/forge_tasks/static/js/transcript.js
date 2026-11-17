// Chat transcripts: OpenAI-style messages (system, user, assistant with reasoning and tool calls,
// tool observations) rendered as a conversation, with shell calls shown as terminal blocks.

import { h, markdown, code, chip, meter, compact, duration, tryJson, tip, tipBody, STOP_REASONS, plural } from "./util.js";

const COLLAPSE_CHARS = 1400;

function contentText(content) {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((part) => part.text ?? part.content ?? "").join("\n");
  return JSON.stringify(content);
}

function collapsible(node, length, startOpen = false) {
  if (length < COLLAPSE_CHARS || startOpen) return node;
  const wrap = h("div", { class: "collapsed" }, node);
  const button = h("button", { class: "ghost expand small" }, `Show all (${compact(length)} characters)`);
  button.addEventListener("click", () => {
    const open = wrap.classList.toggle("collapsed");
    button.textContent = open ? `Show all (${compact(length)} characters)` : "Show less";
  });
  return h("div", {}, wrap, button);
}

function reasoningBlock(text, tokenCount) {
  if (!text || !text.trim()) return null;
  const label = tokenCount ? `Reasoning, ${compact(tokenCount)} tokens` : `Reasoning, ${compact(text.length)} characters`;
  const el = h("details", { class: "reasoning" }, h("summary", {}, `💭 ${label} — show`));
  el.addEventListener("toggle", () => {
    el.querySelector("summary").textContent = `💭 ${label} — ${el.open ? "hide" : "show"}`;
    if (el.open && !el.querySelector(".think")) el.append(h("div", { class: "think" }, text));
  });
  return el;
}

function shellResult(content) {
  const parsed = tryJson(content);
  if (!parsed || typeof parsed !== "object" || !("stdout" in parsed || "stderr" in parsed)) return { raw: contentText(content) };
  return parsed;
}

function terminalBlock(command, result) {
  const block = h("div", { class: "term" }, h("div", { class: "cmd" }, command));
  if (!result) {
    block.append(h("div", { class: "status" }, "no observation recorded"));
    return block;
  }
  if (result.raw !== undefined) {
    block.append(h("div", { class: "out" }, result.raw || "(no output)"));
    return block;
  }
  const out = h("div", { class: "out" });
  if (result.stdout) out.append(result.stdout);
  if (result.stderr) out.append(h("span", { class: "err" }, result.stderr));
  if (!result.stdout && !result.stderr) out.append(h("span", { style: { color: "var(--term-muted)" } }, "(no output)"));
  block.append(out);
  const ok = result.exit_code === 0;
  block.append(h("div", { class: "status" },
    h("span", { class: ok ? "" : "bad" }, ok ? "✓ exit 0" : `✕ exit ${result.exit_code}`),
    result.reason && result.reason !== "exited" ? h("span", { class: "bad" }, result.reason.replaceAll("_", " ")) : null,
    result.truncated ? h("span", { class: "bad" }, "output truncated") : null));
  return block;
}

function argsView(args) {
  if (!args || typeof args !== "object") return h("div", { class: "mono small" }, String(args ?? ""));
  return h("dl", { class: "kv" }, Object.entries(args).flatMap(([key, value]) => [
    h("dt", {}, key.replaceAll("_", " ")),
    h("dd", {}, typeof value === "string" ? (value.length > 300 ? collapsible(h("div", { class: "mono small", style: { whiteSpace: "pre-wrap" } }, value), value.length) : value) : h("span", { class: "mono small" }, JSON.stringify(value))),
  ]));
}

export function toolCallView(call, observation, renderers = {}) {
  const fn = call.function || call;
  const name = fn.name;
  const args = typeof fn.arguments === "string" ? tryJson(fn.arguments) ?? fn.arguments : fn.arguments;
  if (renderers[name]) return renderers[name](args, observation);
  if (name === "shell" && args && typeof args === "object" && "command" in args) {
    return terminalBlock(args.command, observation === undefined ? null : shellResult(observation));
  }
  return h("div", { class: "toolcall" },
    h("div", { class: "tc-name" }, `⚙ ${name}`),
    argsView(args),
    observation !== undefined ? h("div", { style: { marginTop: "6px" } }, h("div", { class: "muted small" }, "Result"), h("div", { class: "mono small", style: { whiteSpace: "pre-wrap" } }, contentText(observation))) : null);
}

function turnMeta(meta) {
  if (!meta) return null;
  const usage = meta.usage || {};
  const rows = [];
  if (usage.completion_tokens) rows.push(`${compact(usage.completion_tokens)} out`);
  if (usage.reasoning_tokens) rows.push(`${compact(usage.reasoning_tokens)} thinking`);
  if (meta.wall_time) rows.push(duration(meta.wall_time));
  const attempts = meta.attempts || [];
  const retried = attempts.length > 1 || attempts.some((a) => a.http_status && a.http_status !== 200);
  return { rows, retried, attempts, finish: meta.finish_reason, usage };
}

// messages: [{role, content, reasoning_content, tool_calls, tool_call_id}]
// opts.turns: per-assistant-message metadata in order; opts.labels: role -> label; opts.renderers: tool name -> fn
export function transcript(messages, opts = {}) {
  const observations = new Map();
  for (const m of messages) if (m.role === "tool" && m.tool_call_id) observations.set(m.tool_call_id, m.content);
  const root = h("div", { class: "transcript" });
  let assistantIndex = 0;
  const lastAssistant = messages.map((m) => m.role).lastIndexOf("assistant");
  const maxOut = Math.max(1, ...(opts.turns || []).map((t) => t?.usage?.completion_tokens || 0));
  messages.forEach((m, idx) => {
    if (m.role === "tool" && observations.has(m.tool_call_id) && messages.some((x) => (x.tool_calls || []).some((c) => c.id === m.tool_call_id))) return;
    const text = contentText(m.content);
    const label = opts.labels?.[idx] || opts.roleLabels?.[m.role] || m.role;
    const who = h("div", { class: "who" }, h("div", { class: "role" }, label));
    const bubble = h("div", { class: "bubble" });
    let cls = m.role;
    if (m.role === "assistant") {
      const meta = turnMeta(opts.turns?.[assistantIndex]);
      if (meta) {
        who.append(h("div", { class: "meta" }, meta.rows.map((r) => h("div", {}, r))));
        if (meta.usage.completion_tokens) who.append(h("div", { class: "turnbar", style: { justifyContent: "flex-end" } }, meter(meta.usage.completion_tokens / maxOut, "var(--phase-solver)", "70px")));
        if (meta.retried) {
          const c = chip("retried", "warn");
          tip(c, tipBody("Transport attempts", ...meta.attempts.map((a, i) => `attempt ${i + 1}: ${a.outcome}${a.http_status ? `, HTTP ${a.http_status}` : ""}${a.duration ? `, ${duration(a.duration)}` : ""}`)));
          who.append(h("div", { style: { marginTop: "4px" } }, c));
        }
        if (meta.finish === "length") who.append(h("div", { style: { marginTop: "4px" } }, chip("cut at max tokens", "bad")));
      }
      const thinking = reasoningBlock(m.reasoning_content || m.reasoning, opts.turns?.[assistantIndex]?.usage?.reasoning_tokens);
      if (thinking) bubble.append(thinking);
      if (text.trim()) bubble.append(opts.renderContent?.(text) || collapsible(markdown(text), text.length, true));
      for (const call of m.tool_calls || []) bubble.append(toolCallView(call, observations.get(call.id), opts.renderers));
      if (!text.trim() && !(m.tool_calls || []).length && !(m.reasoning_content || "").trim()) bubble.append(h("div", { class: "muted small" }, "(empty reply)"));
      if (idx === lastAssistant && opts.markFinal !== false) cls += " final";
      assistantIndex += 1;
    } else if (m.role === "tool") {
      bubble.append(h("div", { class: "mono small", style: { whiteSpace: "pre-wrap" } }, text));
    } else {
      const startOpen = opts.openRoles?.includes(m.role);
      const body = opts.plainRoles?.includes(m.role) ? h("div", { class: "mono small", style: { whiteSpace: "pre-wrap" } }, text) : markdown(text);
      bubble.append(collapsible(body, text.length, startOpen));
    }
    root.append(h("div", { class: `msg ${cls}` }, who, bubble));
  });
  return root;
}

export function transcriptText(messages) {
  return messages.map((m) => {
    const parts = [`### ${m.role}`];
    if (m.reasoning_content) parts.push(`[reasoning]\n${m.reasoning_content}`);
    const text = contentText(m.content);
    if (text) parts.push(text);
    for (const call of m.tool_calls || []) parts.push(`[tool call ${call.function?.name || call.name}] ${call.function?.arguments || call.arguments}`);
    return parts.join("\n");
  }).join("\n\n");
}

export function stopReasonText(reason) {
  return STOP_REASONS[reason] || (reason ? reason.replaceAll("_", " ") : "unknown");
}

export function turnCount(messages) {
  return plural(messages.filter((m) => m.role === "assistant").length, "model turn");
}

export { code };
