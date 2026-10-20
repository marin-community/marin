import DOMPurify from "dompurify";
import { marked } from "marked";

const $ = (id) => document.getElementById(id);
var state = { runs: [], run: null, requestedRun: null, cases: [], ideas: [], summary: null, overviewSummary: null, selected: null, detail: null, tab: "conversation", page: 0, total: 0, compare: [], busy: false, submittingIdea: false };
var categories = ["reasoning", "instruction_following", "factuality", "coding", "data_analysis", "tool_use", "planning", "state_tracking", "formatting", "repetition", "calibration"];
var breakages = ["incorrect_answer", "missed_instruction", "hallucination", "reasoning_loop", "truncation", "invalid_tool_call", "tool_misuse", "state_loss", "false_success", "format_violation", "unhelpful_refusal", "poor_calibration", "infrastructure", "other"];
var colors = ["#b66321", "#197b86", "#567246", "#a65c7a", "#7365ac"];
var human = (value) => String(value || "").replaceAll("_", " ");
var number = (value) => new Intl.NumberFormat("en", { notation: value >= 1e6 ? "compact" : "standard", maximumFractionDigits: 2 }).format(value || 0);
var node = (tag, text2 = "", className = "") => {
  const element = document.createElement(tag);
  element.textContent = text2;
  if (className) element.className = className;
  return element;
};
var append = (parent, ...children) => {
  parent.append(...children.filter(Boolean));
  return parent;
};
var chip = (text2, cls = "") => node("span", text2, `chip ${cls}`);
var candidates = () => state.run?.config?.candidates || [];
var model = (key) => candidates().find((item) => item.key === key);
var label = (key) => model(key)?.label || (key === "glm" ? "GLM-5.3 \xB7 user / reviewer" : key);
var color = (key) => key === "glm" ? "#7365ac" : colors[Math.max(0, candidates().findIndex((item) => item.key === key)) % colors.length];
var tinted = (element, key) => {
  element.style.setProperty("--model-color", color(key));
  return element;
};
function markdown(source, className = "prose") {
  const element = node("div", "", `${className} markdown`);
  element.innerHTML = DOMPurify.sanitize(marked.parse(source, { async: false, gfm: true }), {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ["img", "style", "input", "textarea", "form", "button"],
    FORBID_ATTR: ["style"]
  });
  for (const link of element.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  return element;
}
function rawText(source, key, title) {
  const details = node("details", "", "raw-text");
  details.dataset.key = key;
  return append(details, node("summary", title), node("pre", source, "source-box"));
}
function userTurn(trace, title) {
  const content = trace.payload.content;
  const item = node("div", "", "user-turn");
  item.id = `trace-${trace.message_id}`;
  return append(item, node("p", title, "message-label"), markdown(content), rawText(content, `user-${trace.message_id}`, "Raw user message"));
}
async function api(path, options = {}) {
  const response = await fetch(`api/${path}`, { ...options, headers: { "Content-Type": "application/json", ...options.headers } });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Request failed (${response.status}): ${body.slice(0, 300)}`);
  }
  return response.json();
}
function error(message2) {
  $("error").textContent = message2;
  $("error").hidden = !message2;
}
function updateSelectionUrl(runId, caseId) {
  const url = new URL(location.href);
  if (runId) url.searchParams.set("run", runId);
  else url.searchParams.delete("run");
  if ($("state-filter").value) url.searchParams.set("state", $("state-filter").value);
  else url.searchParams.delete("state");
  if ($("overview-mode").value) url.searchParams.set("overview_mode", $("overview-mode").value);
  else url.searchParams.delete("overview_mode");
  url.hash = caseId;
  history.replaceState(null, "", url);
}
function clearSelection() {
  state.selected = null;
  state.detail = null;
  updateSelectionUrl(state.run?.run_id || state.requestedRun, "");
  $("detail").replaceChildren(node("div", "Select a matching conversation to inspect its paired trace.", "empty"));
}
function fillSelect(select, options, first) {
  const value = select.value;
  select.replaceChildren();
  if (first) select.append(new Option(first, ""));
  for (const [key, text2] of options) select.append(new Option(text2, key));
  if ([...select.options].some((option) => option.value === value)) select.value = value;
}
function stat(title, value, foot) {
  return append(node("div", "", "stat"), node("p", title, "stat-label"), node("p", value, "stat-value"), node("p", foot, "stat-foot"));
}
function renderStatus() {
  const run = state.run, summary = state.summary || {};
  if (!run) return;
  $("conversation-policy").textContent = run.config.conversation_policy ? "Shared opening \xB7 independent follow-ups" : "Shared user turns";
  if (run.config.tool_policy) $("conversation-policy").textContent += " \xB7 tools stay enabled";
  const lag = (Date.now() - Date.parse(run.updated_at)) / 1e3;
  const displayState = lag > 45 && run.state === "running" ? "Heartbeat stale" : human(run.state);
  const budget = stat("Generated tokens", `${number(run.generated_tokens)} / ${number(run.token_cap)}`, `${number(run.reserved_tokens)} reserved \xB7 all models + thinking`);
  const track = node("div", "", "progress-track"), fill = node("div", "", "progress-fill");
  fill.style.width = `${Math.min(100, 100 * run.generated_tokens / run.token_cap)}%`;
  append(budget, append(track, fill));
  $("run-status").replaceChildren(
    stat("Continuous sessions", `${run.active_chats} / ${run.concurrency}`, `${displayState} \xB7 paired conversations`),
    stat("Saved conversations", number(summary.cases), `${number(summary.completed)} reviewed \xB7 ${number(summary.errors)} errors / interrupted`),
    stat("Concrete breakages", number((summary.breakages || []).reduce((sum, item) => sum + item.count, 0)), "Evidence linked to saved messages"),
    budget
  );
  $("pause").disabled = run.state === "stopped" || run.desired_state === "paused";
  $("resume").disabled = run.state === "stopped" || run.desired_state === "running";
  $("stop").disabled = run.state === "stopped";
  if (run.error) error(run.error);
}
function checkpointText(item) {
  const link = node("a", item.checkpoint, "checkpoint");
  link.href = `https://huggingface.co/${item.checkpoint}/tree/${item.revision}`;
  link.target = "_blank";
  link.rel = "noreferrer";
  return link;
}
function renderModels() {
  const cards = candidates().map((item, index) => {
    const card = tinted(node("div", "", "model-card"), item.key);
    append(card, node("p", `Candidate ${index + 1} \xB7 pinned checkpoint`, "model-kicker"), node("h3", item.label, "model-label"), checkpointText(item), node("p", `revision ${item.revision}`, "revision"));
    append(card, append(node("div", "", "model-footer"), chip(item.key), chip("Thinking enabled"), chip(`${number(item.max_context)} context`), chip(`${item.gpu_count} H100 \xB7 TP ${item.tensor_parallel_size} / DP ${item.data_parallel_size}`)));
    if (state.run.config.shellsim_cpu_limit) append(card, chip(`ShellSim CPU \xB7 ${number(state.run.config.shellsim_cpu_limit)} units`));
    if (item.reasoning_format === "grug_inline") append(card, chip("Native Hermes tools \xB7 inline thinking"));
    if (item.reasoning_parser === "grug") append(card, chip("Grug reasoning parser \xB7 Hermes tools"));
    if (item.vllm_release) append(card, node("p", `vLLM ${item.vllm_release}`, "revision"));
    if (item.reasoning_parser_source_commit) append(card, node("p", `Grug parser source ${item.reasoning_parser_source_commit}`, "revision"));
    if (item.weights_revision) append(card, node("p", `Cached weights ${item.weights_revision} \xB7 ${item.weights_equivalence || "equivalence check pending"}`, "revision"));
    if (item.reasoning_format === "grug_inline" && state.run.config.version < 18) append(card, node("p", "Historical chat handling: thinking markers were removed in chat-only requests. Chat scores include private reasoning in the public answer.", "subtle"));
    if (item.reasoning_parser === "grug_optional") append(card, node("p", "Historical tool handling: calls inside an open thinking block may not have executed.", "subtle"));
    if (item.reasoning_parser_revision) append(card, chip(`Parser \xB7 ${item.reasoning_parser_revision.slice(0, 12)}`));
    else if (item.reasoning_parser === "grug_optional" && state.run.config.version < 14) append(card, node("p", "This run predates the thinking parser fix. Saved answers may contain thinking blocks.", "subtle"));
    return card;
  });
  $("models").replaceChildren(...cards);
  fillSelect($("model-filter"), candidates().map((item) => [item.key, item.label]), "Breakages \xB7 any model");
}
function renderIdeas() {
  const remaining = state.ideas.reduce((sum, item) => sum + (item.cancelled ? 0 : item.chats - item.assigned), 0);
  $("idea-count").textContent = `${remaining} chats waiting`;
  const stopped = state.run.state === "stopped" || state.run.desired_state === "stopped";
  $("idea-submit").disabled = state.submittingIdea || stopped;
  $("idea-run-status").hidden = !stopped;
  fillSelect($("idea-filter"), state.ideas.map((item) => [item.idea_id, item.text.slice(0, 70)]), "All queued + random tasks");
  const rows = state.ideas.map((item) => {
    const row = node("div", "", "idea-row"), description = node("div", "", "idea-description");
    append(description, node("p", item.text, "idea-text"), append(
      node("div", "", "case-tags"),
      chip(item.kind === "topic" ? "GLM riffs on topic" : "Exact opening prompt"),
      chip(item.mode === "auto" ? "Chat + agentic" : human(item.mode)),
      chip(`${item.assigned} / ${item.chats} started`),
      chip(`${item.completed} reviewed`, "good"),
      item.errors ? chip(`${item.errors} errors / interrupted`, "bad") : null,
      item.cancelled ? chip("Remaining chats cancelled") : null
    ));
    const actions = node("div", "", "idea-actions"), view = node("button", "View chats");
    view.onclick = () => {
      for (const key of ["category", "mode", "state", "breakage", "model"]) $(`${key}-filter`).value = "";
      $("idea-filter").value = item.idea_id;
      state.page = 0;
      clearSelection();
      refresh();
      document.querySelector(".lab").scrollIntoView({ behavior: "smooth" });
    };
    append(actions, view);
    if (!item.cancelled && item.assigned < item.chats) {
      const cancel = node("button", "Cancel remaining", "text-button");
      cancel.onclick = async () => {
        try {
          await api(`runs/${encodeURIComponent(state.run.run_id)}/ideas/${item.idea_id}/cancel`, { method: "POST" });
          await refresh();
        } catch (exception) {
          error(exception.message);
        }
      };
      append(actions, cancel);
    }
    return append(row, description, actions);
  });
  $("idea-list").replaceChildren(...rows);
}
function scoreSection(title, scoreItems, selectedModels, emptyText) {
  const scores = append(node("div", "", "score-section"), node("h3", title));
  append(scores, append(node("div", "", "score-row header"), node("span", "Category"), ...selectedModels.map((item) => node("span", item.label))));
  const grouped = [...new Set(scoreItems.map((item) => item.category))];
  if (!grouped.length) append(scores, node("p", emptyText, "subtle"));
  for (const category of grouped) {
    const row = append(node("div", "", "score-row"), node("span", human(category)));
    for (const candidate of selectedModels) {
      const item = scoreItems.find((item2) => item2.category === category && item2.model === candidate.key);
      const cell = tinted(node("div", "", "score-cell"), candidate.key);
      const track = node("div", "", "score-track"), bar = node("div", "", "score-bar");
      bar.style.width = `${item ? item.score * 100 : 0}%`;
      append(row, append(cell, append(track, bar), node("span", item ? `${Math.round(item.score * 100)}% \xB7 n=${item.n}` : "unjudged")));
    }
    append(scores, row);
  }
  return scores;
}
function renderOverview() {
  const summary = state.overviewSummary || {}, selectedModels = candidates().slice(0, 2), scores = node("div", "", "overview-panel");
  if (state.run.config.verification_policy) {
    append(scores, scoreSection("Verified opening task \xB7 by category", summary.verified_scores || [], selectedModels, "Checks set before either model answers. Only pairs with conclusive checks are scored."), node("p", "Same opening task and checks for both models. Generated checks remain available for inspection.", "subtle"));
  }
  append(scores, scoreSection("GLM-rated conversation \xB7 by category", summary.scores || [], selectedModels, "Reviews appear here as the first conversations finish."), node("p", "Exploratory judgments across independent followups; these are not benchmark pass rates.", "subtle"));
  const failures = append(node("div", "", "overview-panel"), node("h3", "Breakages \xB7 grouped by model and category"));
  const groups = /* @__PURE__ */ new Map();
  for (const item of summary.breakages || []) {
    const key = `${item.model}/${item.category}`;
    if (!groups.has(key)) groups.set(key, { ...item, count: 0 });
    groups.get(key).count += item.count;
  }
  for (const item of [...groups.values()].sort((a, b) => b.count - a.count).slice(0, 8)) {
    const row = tinted(node("div", "", "failure-row"), item.model), button = node("button", human(item.category));
    button.onclick = () => {
      $("breakage-filter").value = item.category;
      $("model-filter").value = item.model;
      $("mode-filter").value = $("overview-mode").value;
      state.page = 0;
      clearSelection();
      refresh();
    };
    append(failures, append(row, node("span", "", "failure-dot"), append(node("span", "", "failure-label"), button), node("small", label(item.model)), chip(number(item.count), "bad")));
  }
  if (!groups.size) append(failures, node("p", "No verified breakages yet. Incomplete reviews and transport errors stay visible in the conversation list.", "subtle"));
  $("overview").replaceChildren(scores, failures);
}
function renderCases() {
  $("case-count").textContent = number(state.total);
  const buttons = state.cases.map((item) => {
    const button = node("button", "", `case-button ${state.selected === item.case_id ? "selected" : ""}`);
    const stateChip = chip(human(item.state), item.state === "complete" ? "good" : ["generating", "reviewing"].includes(item.state) ? "live" : "bad");
    append(button, append(node("div", "", "case-meta"), node("span", `${item.mode === "agentic" ? "\u2318 ShellSim" : "\u2194 Chat"} \xB7 ${human(item.category)}`), node("span", `slot ${item.slot + 1}`)), node("p", item.title, "case-title"), append(node("div", "", "case-tags"), stateChip, chip(`${item.round + 1} turns`)));
    if (item.review?.summary) append(button, node("p", item.review.summary.slice(0, 135), "case-result"));
    if (item.seed?.queued_idea) append(button, chip(`Queued \xB7 ${item.seed.queued_idea.text.slice(0, 55)}`, "queued-chip"));
    button.onclick = () => {
      state.selected = item.case_id;
      state.detail = null;
      state.tab = "conversation";
      updateSelectionUrl(item.run_id, item.case_id);
      loadDetail();
      renderCases();
    };
    return button;
  });
  $("case-list").replaceChildren(...buttons.length ? buttons : [node("div", "Waiting for matching conversations\u2026", "empty")]);
  $("page").textContent = `${state.page + 1} / ${Math.max(1, Math.ceil(state.total / 60))}`;
  $("previous").disabled = state.page === 0;
  $("next").disabled = (state.page + 1) * 60 >= state.total;
}
function duration(value) {
  return value === void 0 ? "" : `${Number(value).toFixed(1)}s`;
}
function message(trace) {
  const payload = trace.payload, item = tinted(node("div", "", `message ${trace.role}`), trace.model);
  item.id = `trace-${trace.message_id}`;
  const heading = append(node("div", "", "message-label"), node("span", trace.role === "assistant" ? "Assistant" : human(trace.role)), node("span", trace.state === "streaming" ? "\u25CF streaming" : payload.purpose || "", "right"));
  append(item, heading);
  if (trace.role === "assistant" && payload.purpose === "candidate" && !payload.request?.tools?.length && payload.content?.includes("<tool_call>")) {
    append(item, node("p", "Harness had tools disabled for this request. This tool-call text was not executed.", "subtle"));
  }
  if (payload.reasoning) {
    const details = node("details", "", "thinking");
    details.dataset.key = trace.message_id;
    details.dataset.model = trace.model;
    const toggleLabel = node("span", "Expand thinking", "thinking-toggle-label");
    details.addEventListener("toggle", () => toggleLabel.textContent = details.open ? "Collapse thinking" : "Expand thinking");
    const reasoningTokens = payload.reasoning_tokens_counted ?? payload.usage?.completion_tokens_details?.reasoning_tokens;
    const tokenLabel = payload.reasoning_count_source === "retokenized_text" ? `~${number(reasoningTokens)} tokens \xB7 estimated from text` : reasoningTokens === void 0 ? "token count not reported" : `${number(reasoningTokens)} tokens`;
    append(item, append(details, append(node("summary", `Thinking \xB7 ${tokenLabel}`), toggleLabel), markdown(payload.reasoning, "thinking-prose"), rawText(payload.reasoning, `reasoning-${trace.message_id}`, "Raw thinking")));
  }
  if (payload.native_content !== void 0 && payload.native_content !== payload.content) append(item, rawText(payload.native_content, `native-${trace.message_id}`, "Native API content \xB7 retained in model history"));
  if (payload.reasoning_recovery) {
    append(item, node("p", "Thinking separated for display. Original API fields are retained below.", "subtle"), rawText(JSON.stringify(payload.reasoning_recovery, null, 2), `recovery-${trace.message_id}`, "Original parser output"));
  }
  if (payload.tool_result) {
    const result = payload.tool_result;
    append(item, append(node("div", "", "case-tags"), chip(`exit ${result.exit_code ?? "unknown"}`, result.exit_code ? "bad" : "")));
    if (result.stop_reason) append(item, chip(human(result.stop_reason), "bad"));
    if (result.stdout) append(item, node("pre", result.stdout, "tool-code"));
    if (result.stderr) append(item, node("pre", result.stderr, "tool-code"));
    if (result.unsupported?.length) append(item, node("p", `Unsupported: ${result.unsupported.join(", ")}`, "subtle"));
    if (result.output_truncated) append(item, chip("Tool output truncated", "bad"));
    if (result.error) append(item, node("pre", result.error, "tool-code"));
  } else if (payload.content) {
    if (trace.model === "glm") append(item, node("pre", payload.content, "source-box"));
    else append(item, markdown(payload.content), rawText(payload.content, `content-${trace.message_id}`, "Raw answer"));
  } else if (trace.state === "streaming" && !payload.reasoning) append(item, node("p", "Waiting for the first token\u2026", "subtle"));
  else if (!payload.tool_calls?.length) append(item, node("p", "No public answer in this message.", "subtle"));
  for (const call of payload.tool_calls || []) append(item, node("pre", `${call.function.name}(${call.function.arguments})`, "tool-code"));
  if (payload.error) append(item, node("pre", payload.error, "tool-code"));
  const meta = node("div", "", "trace-meta");
  if (payload.usage?.completion_tokens !== void 0) append(meta, node("span", `${number(payload.usage.completion_tokens)} completion tokens`));
  else if (payload.charged_tokens !== void 0) append(meta, node("span", `${number(payload.charged_tokens)} tokens charged conservatively`));
  if (payload.usage?.prompt_tokens !== void 0) append(meta, node("span", `${number(payload.usage.prompt_tokens)} input`));
  if (payload.duration !== void 0) append(meta, node("span", duration(payload.duration)));
  if (payload.first_token_latency !== void 0) append(meta, node("span", `first token ${duration(payload.first_token_latency)}`));
  if (payload.finish_reason) append(meta, node("span", `finish: ${payload.finish_reason}`));
  append(meta, node("span", `id ${trace.message_id.slice(0, 8)}`));
  append(item, meta);
  return item;
}
function comparisonHeader() {
  const header = node("div", "", "comparison-header");
  for (let index = 0; index < 2; index++) {
    const item = model(state.compare[index]);
    if (!item) continue;
    const box = tinted(node("div", "", "compare-model"), item.key), select = node("select");
    fillSelect(select, candidates().map((candidate) => [candidate.key, candidate.label]));
    select.value = item.key;
    select.setAttribute("aria-label", `Comparison candidate ${index + 1}`);
    select.onchange = () => {
      state.compare[index] = select.value;
      renderDetail();
    };
    append(header, append(box, select, checkpointText(item), node("p", `revision ${item.revision}`, "revision")));
  }
  return header;
}
function conversationView(detail) {
  const content = node("div");
  append(content, comparisonHeader());
  const turns = detail.traces.filter((trace) => trace.role === "user");
  const rounds = [...new Set(turns.map((trace) => trace.round))].sort((a, b) => a - b);
  for (const roundNumber of rounds) {
    const round = node("section", "", "round");
    const shared = turns.find((trace) => trace.round === roundNumber && trace.model === "shared");
    if (shared) append(round, userTurn(shared, `Shared user \xB7 turn ${roundNumber + 1} \xB7 GLM-5.3`));
    const pair = node("div", "", "pair");
    for (const key of state.compare.slice(0, 2)) {
      const column = node("div");
      const traces = detail.traces.filter((trace) => trace.model === key && trace.round === roundNumber);
      const user = traces.find((trace) => trace.role === "user");
      if (user) append(column, tinted(userTurn(user, `User to ${label(key)} \xB7 turn ${roundNumber + 1} \xB7 GLM-5.3`), key));
      for (const trace of traces.filter((trace2) => !["user", "end"].includes(trace2.role))) append(column, message(trace));
      const end = detail.traces.find((trace) => trace.model === key && trace.role === "end");
      if (end && end.round === roundNumber) append(column, node("p", end.payload.content, "subtle"));
      if (!traces.length) append(column, node("p", end && end.round < roundNumber ? end.payload.outcome === "failed" ? "This chat has failed." : "This conversation has finished." : `Waiting for ${label(key)}\u2026`, "subtle"));
      append(pair, column);
    }
    append(round, pair);
    append(content, round);
  }
  if (!turns.length) {
    const status = detail.state === "generating" ? "GLM is creating the user\u2019s scenario." : "This conversation ended before a user scenario was ready.";
    append(content, node("p", `${status} Open \u201CGLM driver\u201D to see its thinking and output.`, "subtle"));
    if (detail.review?.error) append(content, node("pre", detail.review.error, "source-box"));
  }
  return content;
}
function reviewView(detail) {
  const content = node("div"), review = detail.review || {};
  const mapping = node("div", "", "case-tags");
  for (const [alias, key] of Object.entries(review.aliases || {})) append(mapping, chip(`Review ${alias} \u2192 ${label(key)}`));
  append(content, mapping);
  for (const result of review.verification || []) {
    const card = tinted(node("div", "", "assessment"), result.model);
    append(card, node("h3", `${label(result.model)} \xB7 opening checks`), node("p", result.score === null ? "Unscored" : result.score === 1 ? "Passed all checks" : "Failed checks", "score"));
    for (const check of result.checks) {
      append(card, append(node("div", "", "case-tags"), chip(human(check.status), check.status === "passed" ? "good" : check.status === "failed" ? "bad" : ""), node("span", check.name)), node("p", check.detail, "subtle"));
      if (check.output) append(card, rawText(JSON.stringify(check.output, null, 2), `check-${result.model}-${check.name}`, "Verification output"));
    }
    if (!result.checks.length) append(card, node("p", "No deterministic checks were declared for this opening task.", "subtle"));
    append(content, card);
  }
  append(content, markdown(review.summary || "The review arrives after this conversation finishes.", "review-summary"));
  const grid = node("div", "", "assessment-grid");
  for (const assessment of review.assessments || []) {
    const card = tinted(node("div", "", "assessment"), assessment.model);
    append(card, node("h3", label(assessment.model)), node("p", assessment.score === null ? "Unjudged" : `${Math.round(assessment.score * 100)}% task success`, "score"));
    for (const field of ["strengths", "weaknesses"]) {
      append(card, node("h4", human(field)), append(node("ul"), ...assessment[field].map((text2) => append(node("li"), markdown(text2)))));
    }
    append(grid, card);
  }
  append(content, grid);
  for (const failure of review.breakages || []) {
    const card = tinted(node("div", "", "breakage"), failure.model);
    append(card, append(node("div", "", "case-tags"), chip(human(failure.category), "bad"), chip(`${failure.severity} severity`), chip(`${Math.round(failure.confidence * 100)}% confidence`)), node("p", label(failure.model), "revision"), markdown(failure.explanation));
    for (const evidence of failure.evidence) {
      const button = append(node("button", `\u201C${evidence.quote}\u201D`, "evidence"), node("small", `Open cited trace \xB7 ${evidence.message_id.slice(0, 8)} \u2197`));
      button.onclick = () => {
        state.tab = "conversation";
        if (!state.compare.includes(failure.model)) state.compare[0] = failure.model;
        renderDetail();
        const element = $(`trace-${evidence.message_id}`);
        if (element) {
          element.scrollIntoView({ behavior: "smooth", block: "center" });
          element.classList.add("flash");
        }
      };
      append(card, button);
    }
    append(content, card);
  }
  if (review.review_error || review.error) append(content, node("pre", review.review_error || review.error, "source-box"));
  if (review.citation_repairs?.length) append(content, append(node("details", "", "workspace"), node("summary", `${review.citation_repairs.length} evidence links corrected \xB7 exact quotes matched within the same candidate`), node("pre", JSON.stringify(review.citation_repairs, null, 2), "source-box")));
  if (review.previous_review_error) append(content, append(node("details", "", "workspace"), node("summary", "Earlier review validation error \xB7 resolved"), node("pre", review.previous_review_error, "source-box")));
  if (review.original_review) append(content, append(node("details", "", "workspace"), node("summary", "Original GLM judgment \xB7 excluded due to harness or serving failure"), node("pre", JSON.stringify(review.original_review, null, 2), "source-box")));
  return content;
}
function workspaceView(detail) {
  const content = node("div"), scenario = detail.traces.find((trace) => trace.role === "scenario")?.payload;
  if (!scenario) return node("p", detail.state === "generating" ? "Scenario preparation is still running." : "No scenario was saved. Open \u201CGLM driver\u201D for its output and \u201CStrengths & breakages\u201D for the failure details.", "subtle");
  append(content, node("h3", "Task objective"), markdown(scenario.objective, "review-summary"), node("h3", "Expected checks"), append(node("ul"), ...scenario.expected_checks.map((text2) => append(node("li"), markdown(text2)))));
  if (scenario.checks?.length) append(content, node("h3", "Opening checks \xB7 defined before answers", "workspace-heading"), rawText(JSON.stringify(scenario.checks, null, 2), "scenario-checks", "Check definitions"));
  append(content, node("h3", `Initial workspace \xB7 ${Object.keys(scenario.files).length} files`, "workspace-heading"));
  for (const [path, text2] of Object.entries(scenario.files)) append(content, append(node("details", "", "workspace"), node("summary", `/work/${path}`), node("pre", text2, "tool-code")));
  append(content, node("h3", "Inspiration provenance", "workspace-heading"), node("p", "GLM invents the task; the source row is inspiration rather than an evaluation prompt.", "subtle"), node("pre", JSON.stringify(detail.seed, null, 2), "source-box"));
  return content;
}
function driverView(detail) {
  const content = node("div");
  append(content, node("p", "GLM acts as the user. Its followup calls receive public answers and tool results; only the separate reviewer receives saved thinking.", "review-summary"));
  for (const trace of detail.traces.filter((trace2) => trace2.model === "glm")) {
    const details = node("details", "", "driver");
    details.open = trace.state === "streaming";
    details.dataset.key = `driver-${trace.message_id}`;
    const target = trace.payload.target_model ? ` \xB7 for ${label(trace.payload.target_model)}` : "";
    append(content, append(details, node("summary", `${human(trace.payload.purpose)}${target} \xB7 turn ${trace.round + 1} \xB7 ${trace.state}`), message(trace)));
  }
  return content;
}
function renderDetail() {
  if (!state.detail) return;
  const detail = state.detail;
  const previous = $("detail").querySelector(".detail-content"), scroll = previous?.scrollTop || 0;
  const open = new Map([...$("detail").querySelectorAll("details[data-key]")].map((item) => [item.dataset.key, item.open]));
  const header = node("div", "", "detail-header");
  append(header, node("p", `${detail.mode === "agentic" ? "AGENTIC \xB7 SHELLSIM" : "CHAT"} / ${human(detail.category).toUpperCase()}`, "eyebrow"), node("h2", detail.title));
  if (detail.seed?.queued_idea) append(header, node("p", `Queued ${detail.seed.queued_idea.kind}: ${detail.seed.queued_idea.text}`, "queued-focus"));
  const actions = node("div", "", "detail-actions"), exportButton = node("button", "Export trace \u2197"), thinkingButton = node("button", "Expand thinking");
  exportButton.onclick = () => {
    const link = node("a");
    link.href = URL.createObjectURL(new Blob([JSON.stringify({ run: state.run, conversation: detail }, null, 2)], { type: "application/json" }));
    link.download = `${state.run.run_id}-${detail.case_id}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };
  const updateThinkingButton = () => {
    const thinking = [...$("detail").querySelectorAll(".thinking")];
    thinkingButton.disabled = !thinking.length;
    const expanded = thinking.some((item) => item.open);
    thinkingButton.textContent = expanded ? "Collapse thinking" : "Expand thinking";
    thinkingButton.setAttribute("aria-expanded", String(expanded));
  };
  thinkingButton.onclick = () => {
    const thinking = [...$("detail").querySelectorAll(".thinking")], expand = !thinking.some((item) => item.open);
    thinking.forEach((item) => item.open = expand);
    updateThinkingButton();
  };
  append(actions, thinkingButton, exportButton);
  append(header, append(node("div", "", "case-tags"), chip(human(detail.state), detail.state === "complete" ? "good" : ""), chip(`slot ${detail.slot + 1}`), chip(`case ${detail.case_id.slice(0, 8)}`), actions));
  if (detail.review?.kind === "cpu_limit") append(header, node("p", detail.review.summary, "subtle"));
  const tabs = node("nav", "", "tabs");
  for (const [key, text2] of [["conversation", "Paired traces"], ["review", "Strengths & breakages"], ["workspace", "Task & workspace"], ["driver", "GLM driver"]]) {
    const button = node("button", text2, `tab ${state.tab === key ? "active" : ""}`);
    button.onclick = () => {
      state.tab = key;
      renderDetail();
    };
    append(tabs, button);
  }
  const content = node("div", "", "detail-content");
  const view = { conversation: conversationView, review: reviewView, workspace: workspaceView, driver: driverView }[state.tab];
  append(content, view(detail));
  $("detail").replaceChildren(header, tabs, content);
  $("detail").querySelectorAll("details[data-key]").forEach((item) => {
    if (open.has(item.dataset.key)) item.open = open.get(item.dataset.key);
  });
  content.addEventListener("toggle", updateThinkingButton, true);
  updateThinkingButton();
  content.scrollTop = scroll;
}
async function loadDetail() {
  if (!state.selected) return;
  const selected = state.selected;
  try {
    const detail = await api(`cases/${selected}`);
    if (selected !== state.selected) return;
    if (detail.run_id !== state.run.run_id) throw new Error("Conversation and selected run differ. Reload this conversation\u2019s link.");
    state.detail = detail;
    renderDetail();
  } catch (exception) {
    error(exception.message);
  }
}
async function refresh() {
  if (document.hidden || state.busy) return;
  state.busy = true;
  try {
    state.runs = await api("runs");
    const id = $("run-select").value || state.requestedRun || state.runs[0]?.run_id;
    if (id && !state.runs.some((run) => run.run_id === id)) state.runs.unshift(await api(`runs/${encodeURIComponent(id)}`));
    fillSelect($("run-select"), state.runs.map((run) => [run.run_id, run.run_id]));
    $("run-select").value = id || "";
    state.run = state.runs.find((run) => run.run_id === id);
    if (!state.run) {
      $("connection").textContent = "Awaiting first run";
      return;
    }
    if (!state.compare.length) state.compare = candidates().slice(0, 2).map((item) => item.key);
    const params = new URLSearchParams({ run_id: id, limit: 60, offset: state.page * 60 });
    for (const key of ["category", "mode", "state", "breakage", "model"]) if ($(`${key}-filter`).value) params.set(key, $(`${key}-filter`).value);
    if ($("idea-filter").value) params.set("idea_id", $("idea-filter").value);
    const summaryRequest = api(`summary?run_id=${encodeURIComponent(id)}`);
    const overviewRequest = $("overview-mode").value ? api(`summary?run_id=${encodeURIComponent(id)}&mode=${encodeURIComponent($("overview-mode").value)}`) : summaryRequest;
    const [list, summary, ideas, overviewSummary] = await Promise.all([api(`cases?${params}`), summaryRequest, api(`runs/${encodeURIComponent(id)}/ideas`), overviewRequest]);
    state.cases = list.cases;
    state.total = list.total;
    state.summary = summary;
    state.overviewSummary = overviewSummary;
    state.ideas = ideas;
    if (!state.selected && state.cases.length) state.selected = state.cases[0].case_id;
    updateSelectionUrl(id, state.selected || "");
    error("");
    renderStatus();
    renderModels();
    renderIdeas();
    renderOverview();
    renderCases();
    await loadDetail();
    $("connection").textContent = document.body.dataset.connectionLabel;
    $("connection").className = "connection live";
  } catch (exception) {
    error(exception.message);
    $("connection").textContent = "Connection interrupted";
    $("connection").className = "connection";
  } finally {
    state.busy = false;
  }
}
fillSelect($("category-filter"), categories.map((key) => [key, human(key)]), "All task categories");
fillSelect($("breakage-filter"), breakages.map((key) => [key, human(key)]), "All outcomes");
for (const key of ["category", "mode", "state", "breakage", "model", "idea"]) $(`${key}-filter`).onchange = () => {
  state.page = 0;
  clearSelection();
  refresh();
};
$("overview-mode").onchange = () => {
  updateSelectionUrl(state.run?.run_id || state.requestedRun, state.selected || "");
  refresh();
};
$("run-select").onchange = () => {
  state.selected = null;
  state.detail = null;
  state.compare = [];
  state.page = 0;
  state.requestedRun = $("run-select").value;
  $("idea-filter").value = "";
  $("idea-feedback").textContent = "";
  updateSelectionUrl(state.requestedRun, "");
  $("detail").replaceChildren(node("div", "Loading this run\u2026", "empty"));
  refresh();
};
$("idea-form").onsubmit = async (event) => {
  event.preventDefault();
  if (!state.run) return;
  const runId = state.run.run_id;
  state.submittingIdea = true;
  $("idea-submit").disabled = true;
  try {
    await api(`runs/${encodeURIComponent(runId)}/ideas`, { method: "POST", body: JSON.stringify({ text: $("idea-text").value, kind: $("idea-kind").value, mode: $("idea-mode").value, chats: Number($("idea-chats").value) }) });
    $("idea-text").value = "";
    $("idea-feedback").textContent = "Queued. GLM will pick this up as sessions become free.";
    await refresh();
  } catch (exception) {
    error(exception.message);
  } finally {
    state.submittingIdea = false;
    $("idea-submit").disabled = state.run?.state === "stopped" || state.run?.desired_state === "stopped";
  }
};
$("previous").onclick = () => {
  state.page--;
  refresh();
};
$("next").onclick = () => {
  state.page++;
  refresh();
};
$("overview-toggle").onclick = () => {
  const hidden = !$("overview").hidden;
  $("overview").hidden = hidden;
  $("overview-toggle").textContent = hidden ? "Show overview" : "Hide overview";
};
for (const [button, value] of [["pause", "paused"], ["resume", "running"], ["stop", "stopped"]]) $(button).onclick = async () => {
  if (!state.run) return;
  try {
    await api(`runs/${encodeURIComponent(state.run.run_id)}/control`, { method: "POST", body: JSON.stringify({ state: value }) });
    await refresh();
  } catch (exception) {
    error(exception.message);
  }
};
async function initialize() {
  state.selected = location.hash.slice(1) || null;
  state.requestedRun = new URLSearchParams(location.search).get("run");
  $("state-filter").value = new URLSearchParams(location.search).get("state") || "";
  $("overview-mode").value = new URLSearchParams(location.search).get("overview_mode") || "";
  if (state.selected) {
    try {
      state.detail = await api(`cases/${encodeURIComponent(state.selected)}`);
      state.requestedRun = state.detail.run_id;
    } catch (exception) {
      error(exception.message);
      state.selected = null;
    }
  }
  await refresh();
  setInterval(refresh, 4e3);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refresh();
  });
}
initialize();
