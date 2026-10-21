const $ = (id) => document.getElementById(id);
const number = (value) => Number(value).toLocaleString();
const node = (tag, content, className) => {
  const result = document.createElement(tag);
  if (content !== undefined) result.textContent = content;
  if (className) result.className = className;
  return result;
};
async function api(path) {
  const response = await fetch(`api/${path}`);
  if (!response.ok) throw new Error((await response.json()).detail || `Request failed (${response.status})`);
  return response.json();
}
function failure(error) { $("error").hidden = false; $("error").textContent = error.message; }
function metric(value, label, caption) {
  const card = node("div", undefined, "metric");
  card.append(node("div", value, "value"), node("div", label, "label"));
  if (caption) card.append(node("div", caption, "caption"));
  return card;
}
function pills(values) {
  const element = node("div", undefined, "pills");
  for (const value of values) element.append(node("span", value, "pill"));
  return element;
}
function section(title, data) {
  const wrapper = node("section");
  wrapper.append(node("h3", title), node("pre", typeof data === "string" ? data : JSON.stringify(data, null, 2)));
  return wrapper;
}
function curriculumBrowser() {
  const state = { offset: 0, limit: 40, selected: null, request: 0 };
  async function showTask(contractId) {
    state.selected = contractId;
    for (const task of $("tasks").children) task.classList.toggle("selected", task.dataset.contract === contractId);
    const detail = await api(`task/${encodeURIComponent(contractId)}`);
    if (state.selected !== contractId) return;
    const panel = $("detail");
    panel.replaceChildren(node("h2", detail.task_id), pills([detail.family, `${detail.difficulty_prior} · prior`, detail.answer_type || "unknown"]));
    panel.append(node("p", "Model-visible instruction and contract. Private reference answers and grading fixtures are excluded.", "note"));
    if (detail.native_contract) {
      const native = detail.native_contract;
      panel.append(node("p", `Native structural task · ${native.runtime_compatibility}`, "note"));
      panel.append(section("Supplied assistant prefix", native.supplied_prefix));
      panel.append(section("Native closure contract", { session_kind: native.session_kind, format: native.format, family: native.family, control: native.control, tokenizer_model: native.tokenizer_model, tokenizer_revision: native.tokenizer_revision }));
      const tokens = node("details");
      tokens.append(node("summary", `Historical token prefix (${number(native.prompt_token_ids.length)} tokens)`), node("pre", JSON.stringify(native.prompt_token_ids)));
      panel.append(tokens);
    }
    for (const event of detail.context.events || []) {
      const message = node("div", undefined, "message");
      message.append(node("strong", event.role || event.type), node("span", event.content || JSON.stringify(event.calls || event, null, 2)));
      panel.append(message);
    }
    panel.append(section("Answer contract", { answer_type: detail.answer_type, answer_format: detail.answer_format, output_paths: detail.output_paths }),
      section("Tool contract", { final_tools: detail.final_tools, interaction_tools: detail.interaction_tools, capabilities: detail.environment_capabilities, grader_kind: detail.grader_kind }),
      section("Identity and lineage", { source_id: detail.source_id, source: detail.source, contract_id: detail.contract_id, contract_kind: detail.contract_kind, public_content_id: detail.public_content_id, aliases: detail.aliases, population: detail.population, prior_admission_status: detail.prior_admission_status, inclusion_probability: detail.inclusion_probability, tags: detail.tags }));
  }
  async function loadTasks() {
    const request = ++state.request;
    const params = new URLSearchParams({ offset: state.offset, limit: state.limit, source: $("source").value, family: $("family").value, difficulty: $("difficulty").value, q: $("query").value });
    $("page-status").textContent = "Loading…";
    const result = await api(`tasks?${params}`);
    if (request !== state.request) return;
    $("result-count").textContent = `${number(result.total)} environments`;
    $("page-status").textContent = result.total ? `${number(result.offset + 1)}–${number(Math.min(result.total, result.offset + result.limit))} of ${number(result.total)} · all selected environments are searchable` : "No environments match these filters.";
    $("previous").disabled = state.offset === 0;
    $("next").disabled = state.offset + state.limit >= result.total;
    $("tasks").replaceChildren();
    for (const row of result.tasks) {
      const card = node("button", undefined, "task");
      card.dataset.contract = row.contract_id;
      card.classList.toggle("selected", row.contract_id === state.selected);
      card.append(node("div", row.task_id, "task-title"), pills([row.family, `${row.difficulty} · prior`]), node("div", row.source_id, "note"), node("div", row.preview, "task-preview"));
      card.onclick = () => showTask(row.contract_id).catch(failure);
      $("tasks").append(card);
    }
  }
  async function init() {
    const overview = await api("overview");
    const meta = overview.metadata;
    const selected = meta.selection.selected_count;
    $("overview").replaceChildren(metric(number(selected), "Selected environments", "Complete frozen pool"), metric(number(meta.selection.target), "Requested target", `${number(meta.selection.shortfall)} below target`), metric(number(overview.groups.source_id.length), "Selected sources"), metric(number(overview.groups.family.length), "Task families", "Difficulty labels are provisional priors"));
    $("provenance").textContent = JSON.stringify({ tasks_sha256: meta.tasks_sha256, source_manifest_sha256: meta.source_manifest_sha256, load_id: overview.load_id, selection: meta.selection, recipe: meta.recipe }, null, 2);
    for (const [name, value] of Object.entries(meta.readiness)) $("readiness").append(node("div", `${name.replaceAll("_", " ")}: ${typeof value === "string" ? value : JSON.stringify(value)}`, "readiness-item"));
    if (meta.artifact_uri) {
      $("artifact").append(node("p", "Full TaskSpec artifact (includes private grading data; retain internal access controls):"));
      if (/^https:\/\//.test(meta.artifact_uri)) {
        const link = node("a", meta.artifact_uri); link.href = meta.artifact_uri; link.rel = "noopener noreferrer"; $("artifact").append(link);
      } else $("artifact").append(node("code", meta.artifact_uri));
    }
    for (const [id, group] of [["source", "source_id"], ["family", "family"], ["difficulty", "difficulty"]]) {
      for (const row of overview.groups[group]) { const option = node("option", `${row.value} (${number(row.count)})`); option.value = row.value; $(id).append(option); }
      $(id).onchange = () => { state.offset = 0; loadTasks().catch(failure); };
    }
    for (const row of overview.groups.source_id) {
      const button = node("button"); button.append(node("span", row.value), node("b", number(row.count)));
      button.onclick = () => { $("source").value = row.value; $("family").value = ""; $("difficulty").value = ""; $("query").value = ""; state.offset = 0; loadTasks().catch(failure); };
      $("allocation").append(button);
    }
    let timer;
    $("query").oninput = () => { clearTimeout(timer); timer = setTimeout(() => { state.offset = 0; loadTasks().catch(failure); }, 250); };
    $("reset").onclick = () => { for (const id of ["query", "family", "source", "difficulty"]) $(id).value = ""; state.offset = 0; loadTasks().catch(failure); };
    $("previous").onclick = () => { state.offset = Math.max(0, state.offset - state.limit); loadTasks().catch(failure); };
    $("next").onclick = () => { state.offset += state.limit; loadTasks().catch(failure); };
    await loadTasks();
  }
  return init();
}
curriculumBrowser().catch(failure);
