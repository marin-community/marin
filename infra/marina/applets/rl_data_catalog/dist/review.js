/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const parameters = new URLSearchParams(location.search);
const reviewId = parameters.get("id"), artifactPath = parameters.get("artifact");
const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};
const label = value => String(value).replace(/[_-]/g, " ").replace(/\b\w/g, letter => letter.toUpperCase());
const artifactUrl = path => `api/reviews/${encodeURIComponent(reviewId)}/artifacts/${path.split("/").map(encodeURIComponent).join("/")}`;
const issueKey = key => /^(issues?|errors?|defects?|failures?|error_message)$/i.test(key);
function fold(title, build, open = false, className = "review-section") {
  const details = node("details", undefined, className);
  details.append(node("summary", title));
  let loaded = false;
  const fill = () => {
    if (!loaded && details.open) { loaded = true; details.append(build()); }
  };
  details.addEventListener("toggle", fill);
  details.open = open; fill();
  return details;
}
function parsedText(text) {
  const trimmed = text.trim();
  if (!/^[\[{]/.test(trimmed)) return text;
  try { return JSON.parse(trimmed); }
  catch (error) { if (!(error instanceof SyntaxError)) throw error; return text; }
}
function pretty(value, key = "") {
  if (typeof value === "string") {
    const parsed = parsedText(value);
    if (typeof parsed !== "string") return pretty(parsed, key);
    return node("p", value || "Empty", issueKey(key) ? "formatted-prose issue-highlight" : "formatted-prose");
  }
  if (value === null || value === undefined) return node("span", "Not recorded", "muted");
  if (typeof value !== "object") return node("span", typeof value === "boolean" ? (value ? "Yes" : "No") : String(value));
  if (Array.isArray(value)) {
    if (!value.length) return node("p", "None recorded", "muted");
    const list = node("div", undefined, "formatted-list");
    value.forEach((item, index) => {
      if (typeof item === "object" && item !== null) {
        const title = item.task_id ? `Task ${index + 1} · ${item.task_id}` : item.event ? `Event ${index + 1} · ${label(item.event)}` : `Entry ${index + 1}`;
        list.append(fold(title, () => pretty(item)));
      } else list.append(pretty(item, key));
    });
    return list;
  }
  if (value.kind === "issue" && typeof value.text === "string") return findingView(value);
  const fields = node("dl", undefined, "readable-fields");
  for (const [name, item] of Object.entries(value)) {
    const term = node("dt", label(name)), description = node("dd");
    if (item !== null && typeof item === "object") description.append(fold(`${Array.isArray(item) ? item.length + " entries" : "Details"}`, () => pretty(item, name)));
    else description.append(pretty(item, name));
    fields.append(term, description);
  }
  return fields;
}
function textSection(title, value, open = false) { return fold(title, () => pretty(value), open); }
function artifactLink(path, title = path) {
  const anchor = node("a", title);
  anchor.href = `review.html?id=${encodeURIComponent(reviewId)}&artifact=${encodeURIComponent(path)}`;
  return anchor;
}
function findingView(finding) {
  const item = node("article", undefined, finding.kind === "issue" ? "finding finding-issue" : "finding");
  item.append(node("p", `${label(finding.dimension)} · ${label(finding.severity || "unrated")}`, "finding-label"));
  const paragraph = node("p", undefined, "formatted-prose");
  paragraph.append(node("span", finding.text, finding.kind === "issue" ? "issue-highlight" : ""));
  item.append(paragraph);
  return item;
}
const methodNames = {runtime_execution: "Task attempt & verifier", model_judgment: "Independent judge", synthesis: "Combined review", static_audit: "Imported audit", human_review: "Human review"};
function reviewCard(review, subjects) {
  const subject = subjects.find(item => item.id === review.subject_id);
  const scope = subject?.level === "source" ? "Source" : "Task";
  const issues = review.findings.filter(item => item.kind === "issue");
  const title = `${methodNames[review.method] || label(review.method)} · ${scope} · ${label(review.verdict)}`;
  const card = fold(title, () => {
    const content = node("div", undefined, "review-card-content");
    content.append(node("p", review.reviewer.label, "review-byline"), pretty(review.summary));
    if (issues.length) content.append(fold(`Technical issues (${issues.length})`, () => {
      const findings = node("div"); issues.forEach(finding => findings.append(findingView(finding))); return findings;
    }, true, "review-section technical-issues"));
    const observations = review.findings.filter(item => item.kind !== "issue");
    if (observations.length) content.append(fold(`Observations (${observations.length})`, () => {
      const findings = node("div"); observations.forEach(finding => findings.append(findingView(finding))); return findings;
    }));
    if (review.metrics.length) content.append(textSection("Metrics", Object.fromEntries(review.metrics.map(metric => [metric.key, metric.value]))));
    if (review.evidence.length) content.append(fold(`Evidence (${review.evidence.length})`, () => {
      const list = node("ul", undefined, "evidence-list");
      for (const evidence of review.evidence) {
        const row = node("li");
        if (evidence.url.startsWith("file:")) row.append(artifactLink(evidence.snapshot_path));
        else { const anchor = node("a", evidence.snapshot_path); anchor.href = evidence.url; anchor.target = "_blank"; anchor.rel = "noopener"; row.append(anchor); }
        list.append(row);
      }
      return list;
    }));
    content.append(textSection("Review details & identifiers", {
      reviewed_at: review.reviewed_at, reviewer: review.reviewer, subject: subject || review.subject_id,
      method: methodNames[review.method] || label(review.method), review_id: review.id,
      contributing_reviews: review.derived_from_review_ids, tags: review.tags,
    }));
    return content;
  }, review.method === "synthesis" && subject?.level === "source", "review-card");
  if (issues.length) card.querySelector("summary").append(node("span", `${issues.length} issue${issues.length === 1 ? "" : "s"}`, "issue-badge"));
  return card;
}
async function jsonResponse(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(`Could not load review evidence (${response.status})`);
  return response.json();
}
function download(value, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"}));
  const anchor = node("a"); anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function artifactPage(record) {
  const metadata = record.artifacts.find(item => item.path === artifactPath);
  if (!metadata) throw Error("This artifact is not part of the saved review.");
  const response = await fetch(artifactUrl(artifactPath));
  if (!response.ok) throw Error(`Could not load artifact (${response.status})`);
  const text = await response.text();
  let value = parsedText(text);
  if (artifactPath.endsWith(".jsonl")) value = text.split("\n").filter(line => line.trim()).map(line => JSON.parse(line));
  document.getElementById("title").textContent = artifactPath.split("/").at(-1);
  document.getElementById("provenance").textContent = `${record.source_id} · ${artifactPath} · SHA-256 ${metadata.sha256}`;
  const back = document.getElementById("back-review"); back.hidden = false; back.href = `review.html?id=${encodeURIComponent(reviewId)}`;
  document.getElementById("reviews").append(fold("Evidence contents", () => pretty(value), true, "review-card"));
  const button = document.getElementById("download"); button.textContent = "Download original artifact";
  button.addEventListener("click", () => { const anchor = node("a"); anchor.href = artifactUrl(artifactPath); anchor.download = artifactPath.split("/").at(-1); anchor.click(); });
}
(async () => {
  try {
    if (!reviewId) throw Error("Choose a source review from the Atlas.");
    const record = await jsonResponse(`api/reviews/${encodeURIComponent(reviewId)}`), collection = record.collection;
    if (artifactPath) { await artifactPage(record); return; }
    document.getElementById("title").textContent = record.source_id;
    document.title = `${record.source_id} · Quality review`;
    const provenance = collection.execution_provenance;
    const date = collection.reviews.map(review => review.reviewed_at).filter(Boolean).sort().at(-1) || "Unknown";
    document.getElementById("provenance").textContent = `Reviewed ${date} · MarinSkyRL commit ${provenance?.marinskyrl_commit || "Not recorded in imported review"}`;
    const container = document.getElementById("reviews");
    const issues = collection.reviews.flatMap(review => review.findings.filter(finding => finding.kind === "issue").map(finding => ({review, finding})));
    if (issues.length) container.append(fold(`Reported technical issues (${issues.length})`, () => {
      const content = node("div", undefined, "review-card-content");
      content.append(node("p", "Reports from separate judges can describe the same defect. Open the individual reviews below for context.", "review-byline"));
      for (const {review, finding} of issues) { const item = findingView(finding); item.prepend(node("p", `${methodNames[review.method] || label(review.method)} · ${review.reviewer.label}`, "review-byline")); content.append(item); }
      return content;
    }, true, "review-card technical-issues"));
    container.append(fold("Execution provenance & review applicability", () => pretty({execution_provenance: provenance, source_mappings: collection.source_mappings, created_at: collection.created_at})));
    const difficultyArtifact = record.artifacts.find(item => item.path === "difficulty.json");
    if (difficultyArtifact) {
      const report = await jsonResponse(artifactUrl("difficulty.json"));
      container.append(fold("Paired difficulty estimate", () => {
        const content = node("div", undefined, "review-card-content");
        content.append(node("p", `${report.sampling.task_count} shared tasks · ${report.sampling.method} · split ${report.split}`, "formatted-prose"));
        for (const model of report.models) {
          const interval = model.wilson_95 ? ` · 95% interval ${(100 * model.wilson_95[0]).toFixed(1)}–${(100 * model.wilson_95[1]).toFixed(1)}%` : "";
          content.append(node("h3", `${label(model.size)} · ${model.model}`), node("p", `${model.solved}/${model.verified} solved · ${model.unverified} unverified${interval}`, "formatted-prose"));
          if (model.provider) content.append(node("p", `Provider: ${model.provider} · checkpoint revision: ${model.model_revision || "Not exposed by provider"}`, "review-byline"));
          if (model.generation_parameters) content.append(textSection("Generation settings", model.generation_parameters));
        }
        for (const followup of report.protocol_followups || []) {
          const alternateCheckpoint = followup.kind === "alternate_checkpoint";
          const followupTitle = alternateCheckpoint ? "Hosted model follow-up" : "Generation setting follow-up";
          content.append(fold(`${followupTitle} · ${label(followup.state)}`, () => {
            const section = node("div", undefined, "review-card-content");
            section.append(node("p", "Additional run on the same sampled tasks. The original paired measurements remain above.", "formatted-prose"));
            if (alternateCheckpoint) section.append(node("h3", followup.model), node("p", `Provider: ${followup.provider} · checkpoint revision: ${followup.model_revision || "Not exposed by provider"}`, "review-byline"));
            if (followup.state === "complete") {
              const interval = followup.wilson_95 ? ` · 95% interval ${(100 * followup.wilson_95[0]).toFixed(1)}–${(100 * followup.wilson_95[1]).toFixed(1)}%` : "";
              section.append(node("p", `Large follow-up: ${followup.solved}/${followup.verified} solved · ${followup.unverified} unverified${interval}`, "formatted-prose"));
            }
            const truncated = followup.original_generation_diagnostics?.reasoning_only_truncated_tasks ?? followup.original_reasoning_only_truncations;
            if (truncated !== undefined) section.append(node("p", `${truncated} original task responses hit the output budget while thinking and submitted no final answer.`, "formatted-prose issue-highlight"));
            if (alternateCheckpoint) {
              section.append(textSection("Model & generation changes", {checkpoint_change: followup.checkpoint_change, generation_parameters: followup.generation_parameters}, true));
              section.append(textSection("Verifier judge configuration", followup.verifier_configuration_change));
              section.append(textSection("Comparison limitations", followup.limitations, true));
            } else section.append(textSection("Generation settings", {original_reasoning_effort: followup.original_reasoning_effort, changed_parameter: followup.changed_parameter}, true));
            if (record.artifacts.some(item => item.path === followup.report_path)) section.append(artifactLink(followup.report_path, "Browse follow-up task outcomes & traces"));
            if (record.artifacts.some(item => item.path === followup.original_report_path)) section.append(node("p"), artifactLink(followup.original_report_path, "Browse preserved original report"));
            return section;
          }, true));
        }
        content.append(textSection("Limitations", report.limitations, true), artifactLink("difficulty.json", "Browse task outcomes & estimate details")); return content;
      }, true, "review-card"));
    }
    const ordered = [...collection.reviews].sort((a, b) => Number(b.method === "synthesis" && collection.subjects.find(s => s.id === b.subject_id)?.level === "source") - Number(a.method === "synthesis" && collection.subjects.find(s => s.id === a.subject_id)?.level === "source"));
    ordered.forEach(review => container.append(reviewCard(review, collection.subjects)));
    container.append(fold(`All saved evidence (${record.artifacts.length})`, () => {
      const list = node("ul", undefined, "evidence-list");
      record.artifacts.forEach(item => { const row = node("li"); row.append(artifactLink(item.path)); list.append(row); }); return list;
    }));
    document.getElementById("download").addEventListener("click", () => download(collection, `${reviewId}.json`));
  } catch (error) { document.getElementById("title").textContent = error.message; }
})();
