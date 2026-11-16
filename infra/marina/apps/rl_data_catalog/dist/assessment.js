/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const source = new URLSearchParams(location.search).get("source");
const labels = {good: "Good", some_issues: "Some issues", bad: "Bad", unknown: "Unknown"};
const defectLabels = {
  missing_context: "Missing context",
  ambiguity: "Ambiguity",
  wrong_reference: "Wrong reference",
  answer_leakage: "Answer leakage",
  malformed: "Malformed",
  rubric_mismatch: "Rubric mismatch",
};
const confidenceLabels = {high: "High", medium: "Medium", low: "Low"};
const referenceLabels = {consistent: "Reference consistent", conflict: "Reference conflict", unknown: "Reference unknown"};
const state = {
  cohort: "all",
  verdict: "all",
  confidence: "all",
  reference: "all",
  defects: new Set(),
  samples: [],
};

async function send(method, url, body) {
  const response = await fetch(url, method === "GET" ? undefined : {
    method,
    headers: {"content-type": "application/json"},
    body: JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

// One chip per recorded defect: count and rate within the cohort's reviewed tasks.
function defectChips(defectCounts, reviewed) {
  const group = node("span", undefined, "evidence-labels");
  for (const [defect, count] of Object.entries(defectCounts)) {
    if (!count) continue;
    const rate = reviewed ? ` · ${(100 * count / reviewed).toFixed(1)}%` : "";
    group.append(node("span", `${defectLabels[defect]} ${count}${rate}`, `chip quality-some_issues defect-chip`));
  }
  return group;
}

function confidenceLine(counts) {
  const known = counts.high + counts.medium + counts.low;
  if (!known) return "Reviewer confidence: not recorded for this assessment.";
  const parts = Object.entries(confidenceLabels).map(([key, label]) => `${label} ${counts[key]}`);
  return `Reviewer confidence: ${parts.join(" · ")}${counts.unrecorded ? ` · unrecorded ${counts.unrecorded}` : ""}`;
}

function summary(report) {
  const target = $("summary");
  for (const cohort of ["admitted", "excluded"]) {
    const data = report.assessment[cohort], section = node("section", undefined, "assessment-card");
    section.append(node("h2", cohort === "admitted" ? "Retained task opinions" : "Excluded row diagnostics"));
    section.append(node("p", `${labels[data.grade]} · ${data.reviewed}/${data.sample_count} reviewed · population ${data.population.toLocaleString()}`));
    section.append(node("p", ["good", "some_issues", "bad", "unknown"].map(verdict => `${labels[verdict]} ${data.verdict_counts[verdict]}`).join(" · ")));
    section.append(defectChips(data.defect_counts || {}, data.reviewed));
    section.append(node("p", confidenceLine(data.confidence_counts || {high: 0, medium: 0, low: 0, unrecorded: 0})));
    section.append(node("p", data.good_fraction === null ? data.grade_reason : `${(100 * data.good_fraction).toFixed(1)}% Good · ${data.grade_reason}`));
    section.append(node("p", `Missing required evidence: ${data.missing_evidence}. Packets with omissions: ${data.packets_with_omissions}.`));
    target.append(section);
  }
  const controls = report.curation.verification, section = node("section", undefined, "assessment-card");
  section.append(node("h2", "Curation and executed controls"), node("p", `Curation: ${report.curation.status}. Scope: ${report.scope}.`));
  section.append(node("p", `Admission counts: ${JSON.stringify(report.curation.admission_counts)}`));
  section.append(node("p", controls ? `Controls: ${controls.status}. Checked / eligible: ${controls.sample_count ?? "—"} / ${controls.eligible_count ?? "—"}. Counts: ${JSON.stringify(controls.counts)}` : "Controls were not measured."));
  section.append(node("p", `Model: ${report.model.model}; revision ${report.model.revision}.`));
  target.append(section);
}

function matches(sample) {
  if (state.cohort !== "all" && sample.cohort !== state.cohort) return false;
  if (state.verdict !== "all" && sample.verdict !== state.verdict) return false;
  if (state.confidence !== "all" && (sample.confidence || "") !== state.confidence) return false;
  if (state.reference !== "all" && (sample.reference_status || "") !== state.reference) return false;
  for (const defect of state.defects) if (!(sample.defects || []).includes(defect)) return false;
  return true;
}

function taskCard(sample) {
  const details = node("details", undefined, "assessment-card"), content = node("div");
  const badges = (sample.defects || []).map(defect => node("span", defectLabels[defect], "chip quality-some_issues"));
  if (sample.confidence) badges.push(node("span", `Confidence: ${confidenceLabels[sample.confidence] || sample.confidence}`, "chip quality-unassessed"));
  if (sample.reference_status) badges.push(node("span", referenceLabels[sample.reference_status] || sample.reference_status, "chip quality-unassessed"));
  details.append(node("summary", `${labels[sample.verdict] || sample.verdict} · ${sample.task_id}`), content);
  details.addEventListener("toggle", async () => {
    if (!details.open || content.childNodes.length) return;
    content.append(node("p", "Loading full review and packet…"));
    try {
      const data = await json(`api/evidence?${new URLSearchParams({source_id: source, task_id: sample.task_id})}`);
      const header = node("div", undefined, "evidence-labels");
      for (const badge of badges) header.append(badge);
      content.replaceChildren(
        node("p", data.judgment.explanation),
        header,
        repairRecommendations(data),
        node("pre", JSON.stringify(data, null, 2)),
      );
    } catch (error) { content.replaceChildren(node("p", error.message, "error")); }
  });
  return details;
}

function browser(report) {
  state.samples = report.evidence.samples;
  const section = $("browser"), filters = $("filters"), reviews = $("reviews");
  section.hidden = false;
  function select(label, key, options) {
    const wrap = node("label", undefined, "review-filter");
    wrap.append(node("span", label));
    const control = node("select");
    for (const option of options) {
      const item = node("option", option[1]);
      item.value = option[0];
      control.append(item);
    }
    control.value = state[key];
    control.addEventListener("change", () => { state[key] = control.value; render(); });
    wrap.append(control);
    return wrap;
  }
  filters.append(
    select("Cohort", "cohort", [["all", "All"], ["admitted", "Retained"], ["excluded", "Excluded"]]),
    select("Verdict", "verdict", [["all", "All"], ...Object.entries(labels)]),
    select("Confidence", "confidence", [["all", "All"], ["", "Not recorded"], ...Object.entries(confidenceLabels)]),
    select("Reference", "reference", [["all", "All"], ["", "Not recorded"], ...Object.entries(referenceLabels)]),
  );
  const defectGroup = node("span", undefined, "review-filter review-defects");
  defectGroup.append(node("span", "Defects"));
  for (const [defect, label] of Object.entries(defectLabels)) {
    const toggle = node("label", undefined, "review-defect");
    const box = node("input");
    box.type = "checkbox";
    box.addEventListener("change", () => {
      if (box.checked) state.defects.add(defect); else state.defects.delete(defect);
      render();
    });
    toggle.append(box, node("span", label));
    defectGroup.append(toggle);
  }
  filters.append(defectGroup);

  function render() {
    const kept = state.samples.filter(matches);
    $("filter-count").textContent = `${kept.length.toLocaleString()} of ${state.samples.length.toLocaleString()} stored reviews shown`;
    reviews.replaceChildren(...kept.map(taskCard));
  }
  render();
}

// Manual audit: draw a seeded sample, record judgments, read agreement.
const auditState = {current: null};

function judgmentForm(task, onSaved) {
  const form = node("form", undefined, "audit-judgment");
  const existing = task.judgment && task.judgment.mine ? task.judgment : null;
  let agree = existing ? existing.agree : null;
  const verdictSelect = node("select");
  for (const [value, label] of Object.entries(labels)) {
    const option = node("option", label);
    option.value = value;
    verdictSelect.append(option);
  }
  verdictSelect.value = existing && existing.corrected_verdict ? existing.corrected_verdict : task.verdict;
  const defectBoxes = {};
  const defectGroup = node("span", undefined, "review-defects");
  for (const [defect, label] of Object.entries(defectLabels)) {
    const toggle = node("label", undefined, "review-defect");
    const box = node("input");
    box.type = "checkbox";
    box.checked = Boolean(existing && existing.corrected_defects.includes(defect));
    defectBoxes[defect] = box;
    toggle.append(box, node("span", label));
    defectGroup.append(toggle);
  }
  const note = node("textarea");
  note.maxLength = 500;
  note.placeholder = "Optional short note";
  note.value = existing ? existing.note || "" : "";
  const corrected = node("fieldset", undefined, "audit-corrected");
  corrected.append(node("legend", "Corrected judgment"), node("label", "Verdict"), verdictSelect, defectGroup, note);
  const agreeButtons = node("span", undefined, "audit-agree");
  const agreeButton = node("button", "Agree with reviewer");
  agreeButton.type = "button";
  const disagreeButton = node("button", "Disagree");
  disagreeButton.type = "button";
  const sync = () => {
    agreeButton.className = agree === true ? "chip quality-good" : "chip quality-unassessed";
    disagreeButton.className = agree === false ? "chip quality-bad" : "chip quality-unassessed";
    corrected.hidden = agree !== false;
  };
  agreeButton.addEventListener("click", () => { agree = true; sync(); });
  disagreeButton.addEventListener("click", () => { agree = false; sync(); });
  agreeButtons.append(agreeButton, disagreeButton);
  sync();
  const save = node("button", existing ? "Update judgment" : "Record judgment");
  save.type = "submit";
  const statusLine = node("span", undefined, "audit-status");
  form.append(
    node("p", `Reviewer: ${labels[task.verdict]}${task.reviewer_defects.length ? " · " + task.reviewer_defects.map(defect => defectLabels[defect]).join(", ") : ""}${task.reviewer_confidence ? ` · ${confidenceLabels[task.reviewer_confidence] || task.reviewer_confidence} confidence` : ""}`),
    agreeButtons, corrected, save, statusLine,
  );
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (agree === null) { statusLine.textContent = "Choose agree or disagree first."; return; }
    if (agree === false && !corrected.hidden && !verdictSelect.value) {
      statusLine.textContent = "Pick a corrected verdict.";
      return;
    }
    save.disabled = true;
    statusLine.textContent = "Saving…";
    try {
      await send("PUT", `api/audit/samples/${auditState.current.sample_id}/judgments`, {
        task_id: task.task_id,
        agree,
        corrected_verdict: agree === false ? verdictSelect.value : null,
        corrected_defects: agree === false ? Object.entries(defectBoxes).filter(([, box]) => box.checked).map(([defect]) => defect) : [],
        note: note.value.trim(),
      });
      statusLine.textContent = "Saved.";
      onSaved();
    } catch (error) {
      statusLine.textContent = error.message;
      save.disabled = false;
    }
  });
  return form;
}

function agreementTable(agreement) {
  const table = node("table", undefined, "audit-agreement");
  function row(label, entry) {
    const tr = node("tr");
    tr.append(node("td", label), node("td", entry.judged ? `${entry.agree}/${entry.judged}` : "—"),
      node("td", entry.agreement === null ? "—" : `${(100 * entry.agreement).toFixed(1)}%`),
      node("td", `${(100 * entry.interval[0]).toFixed(0)}–${(100 * entry.interval[1]).toFixed(0)}%`));
    return tr;
  }
  const head = node("tr");
  for (const label of ["Stratum", "Agree/judged", "Agreement", "95% interval"]) head.append(node("th", label));
  table.append(head);
  if (agreement.overall.judged) table.append(row("Overall", agreement.overall));
  for (const [verdict, entry] of Object.entries(agreement.per_verdict)) if (entry.judged) table.append(row(`Reviewer verdict: ${labels[verdict]}`, entry));
  for (const [defect, entry] of Object.entries(agreement.per_defect)) if (entry.judged) table.append(row(`Reviewer defect: ${defectLabels[defect]}`, entry));
  return table;
}

function auditPanel() {
  const section = $("audit"), card = node("section", undefined, "assessment-card");
  section.append(card);
  card.append(node("h2", "Manual audit"));
  card.append(node("p", "Draw a reproducible audit sample from this source's stored reviews, record one judgment per task, and compare with the reviewer. Judgments are advisory evidence; they never silently change an admission or a quality label."));
  const message = node("p", undefined, "card-intro");
  const form = node("form", undefined, "audit-draw");
  const strategy = node("select");
  for (const [value, label] of [["stratified", "Stratified by verdict and defect"], ["random", "Random"]]) {
    const option = node("option", label);
    option.value = value;
    strategy.append(option);
  }
  const size = node("input");
  size.type = "number";
  size.min = 1;
  size.max = 500;
  size.value = 40;
  const seed = node("input");
  seed.placeholder = "Seed (e.g. audit-2026-11-16)";
  seed.required = true;
  const draw = node("button", "Draw sample");
  draw.type = "submit";
  const existing = node("div", undefined, "audit-existing");
  form.append(node("label", "Strategy"), strategy, node("label", "Sample size"), size, node("label", "Seed"), seed, draw, message);
  card.append(form, existing);
  const work = node("div", undefined, "audit-work");
  card.append(work);

  async function loadSample(sampleId) {
    const data = await send("GET", `api/audit/samples/${sampleId}/tasks`);
    auditState.current = data.sample;
    work.replaceChildren();
    work.append(node("h3", `Sample ${data.sample.strategy} · seed ${data.sample.seed} · ${data.sample.sample_size} tasks`));
    const agreementLine = node("p", undefined, "card-intro");
    const refreshAgreement = async () => {
      try {
        const agreement = await send("GET", `api/audit/samples/${sampleId}/agreement`);
        agreementLine.replaceChildren(
          node("strong", agreement.overall.judged ? `Judged ${agreement.overall.judged} tasks · agreement ${(100 * agreement.overall.agreement).toFixed(0)}%` : "No judgments recorded yet."),
          agreementTable(agreement),
        );
      } catch (error) { agreementLine.textContent = error.message; }
    };
    work.append(agreementLine);
    const refresh = () => loadSample(sampleId).then(refreshAgreement).catch(error => { message.textContent = error.message; });
    for (const task of data.tasks) {
      const item = node("details", undefined, "assessment-card audit-task");
      item.append(node("summary", `Audit · ${labels[task.verdict] || task.verdict} · ${task.task_id}${task.judgment ? " · judged" : ""}`));
      const body = node("div");
      item.append(body);
      item.addEventListener("toggle", () => {
        if (!item.open || body.childNodes.length) return;
        body.append(node("p", task.explanation), judgmentForm(task, refresh));
        const link = node("a", "Open full evidence ↗");
        link.href = `source.html#${new URLSearchParams({source_id: source})}`;
        body.append(link);
      });
      work.append(item);
    }
    await refreshAgreement();
  }

  async function listExisting() {
    existing.replaceChildren();
    try {
      const data = await send("GET", `api/audit/samples?source_id=${encodeURIComponent(source)}`);
      if (!data.samples.length) return;
      existing.append(node("h3", "Existing audit samples"));
      for (const sample of data.samples) {
        const row = node("p");
        const open = node("a", `${sample.strategy} · seed ${sample.seed} · ${sample.sample_size} tasks · ${sample.judged_by_me} judged by you`);
        open.href = "#";
        open.addEventListener("click", event => { event.preventDefault(); loadSample(sample.sample_id).catch(error => { message.textContent = error.message; }); });
        row.append(open);
        existing.append(row);
      }
    } catch (error) {
      existing.append(node("p", error.message, "error"));
    }
  }

  form.addEventListener("submit", async event => {
    event.preventDefault();
    draw.disabled = true;
    message.textContent = "Drawing sample…";
    try {
      const result = await send("POST", "api/audit/samples", {
        source_id: source,
        strategy: strategy.value,
        seed: seed.value.trim(),
        sample_size: Number(size.value),
      });
      message.textContent = result.existing ? "That sample already exists; loading it." : "Sample drawn.";
      await loadSample(result.sample_id);
      await listExisting();
    } catch (error) {
      message.textContent = error.message;
    } finally {
      draw.disabled = false;
    }
  });

  listExisting();
}

(async () => {
  try {
    const data = await json(`api/assessment?source_id=${encodeURIComponent(source)}`);
    $("title").textContent = data.source;
    $("message").textContent = `Sampled model opinions · ${data.report.scope} curation`;
    summary(data.report);
    browser(data.report);
    auditPanel();
    $("report").textContent = JSON.stringify(data.report, null, 2);
  } catch (error) {
    $("title").textContent = "Assessment unavailable";
    $("message").textContent = error.message;
  }
})();
