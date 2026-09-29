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
const methodNames = {runtime_execution: "Task attempt & verifier", model_judgment: "Independent judge", synthesis: "Combined review", static_inspection: "Code / evidence inspection", static_audit: "Imported audit", human_review: "Human review"};
function reviewCard(review, subjects, supersededIds) {
  const subject = subjects.find(item => item.id === review.subject_id);
  const scope = subject?.level === "source" ? "Source" : "Task";
  const issues = review.findings.filter(item => item.kind === "issue");
  const superseded = supersededIds.has(review.id);
  const title = `${methodNames[review.method] || label(review.method)} · ${scope} · ${label(review.verdict)}${superseded ? " · Superseded opinion" : ""}`;
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
      contributing_reviews: review.derived_from_review_ids, tags: review.tags, attributes: review.attributes,
    }));
    return content;
  }, !superseded && review.method === "synthesis" && subject?.level === "source", "review-card");
  if (issues.length) card.querySelector("summary").append(node("span", `${issues.length} issue${issues.length === 1 ? "" : "s"}`, "issue-badge"));
  return card;
}
async function jsonResponse(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(`Could not load review evidence (${response.status})`);
  return response.json();
}
function modelOutput(response) {
  if (!Array.isArray(response.choices)) return pretty(response);
  const content = node("div");
  content.append(pretty({model: response.model, request_id: response.id}), textSection("Token usage", response.usage));
  for (const choice of response.choices) {
    const message = choice.message;
    content.append(node("p", `Finish reason: ${choice.finish_reason || "Not recorded"}`, choice.finish_reason === "length" ? "formatted-prose issue-highlight" : "review-byline"));
    if (message?.reasoning_content || message?.reasoning) content.append(textSection("Thinking", message.reasoning_content || message.reasoning));
    content.append(node("h3", "Assistant response"), pretty(message?.content || "No final text returned"));
    if (message?.tool_calls?.length) content.append(textSection("Tool calls", message.tool_calls, true));
  }
  return content;
}
function savedEvidence(title, path) {
  return fold(title, () => {
    const content = node("div", "Loading saved evidence…");
    (async () => {
      const response = await fetch(artifactUrl(path));
      if (!response.ok) throw Error(`Could not load saved evidence (${response.status})`);
      const text = await response.text();
      const value = path.endsWith(".jsonl") ? text.split("\n").filter(line => line.trim()).map(line => JSON.parse(line)) : parsedText(text);
      content.replaceChildren(path.endsWith("/response.json") ? modelOutput(value) : pretty(value), artifactLink(path, "Open original artifact ↗"));
    })().catch(error => content.replaceChildren(node("p", error.message, "issue-highlight")));
    return content;
  });
}
function difficultyRun(model, identityPath, record) {
  const section = fold(`${label(model.size)} · ${model.model}`, () => {
    const content = node("div", undefined, "review-card-content difficulty-settings");
    content.append(AtlasDifficulty.comparison([model]));
    const interval = model.wilson_95 ? `${(100 * model.wilson_95[0]).toFixed(1)}–${(100 * model.wilson_95[1]).toFixed(1)}%` : "Not recorded";
    content.append(node("p", `${model.attempted} attempts · ${model.verified} usable verifier results · ${model.unverified} unverified · 95% interval ${interval}`, "review-byline"));
    const paths = new Set(record.artifacts.map(item => item.path));
    if (!identityPath || !paths.has(identityPath)) {
      content.append(node("p", "Saved run settings and traces have not been published for this measurement. The verifier outcomes below are retained from its report.", "difficulty-evidence-status"));
    } else {
      const settings = node("div", "Loading recorded run settings…");
      content.append(settings);
      jsonResponse(artifactUrl(identityPath)).then(identity => {
        settings.replaceChildren(node("h3", "Model and applied settings"), pretty({
          model: identity.config.model.name,
          checkpoint_revision: identity.config.model.revision || "Not exposed by provider",
          provider: identity.config.model.provider || "Local model service",
          generation_parameters: identity.config.model.parameters,
          request_timeout: identity.config.model.timeout,
          agent_timeout: identity.config.runtime.agent_timeout,
          maximum_turns: identity.config.runtime.max_turns,
          verifier_configuration: identity.config.runtime.gym_config,
          native_execution: identity.marinskyrl,
          task_sample_sha256: identity.tasks_sha256,
        }), artifactLink(identityPath, "Recorded run manifest ↗"));
      }).catch(error => settings.replaceChildren(node("p", error.message, "issue-highlight")));
    }
    const prefix = identityPath?.slice(0, -"run.json".length);
    const outcomes = model.task_outcomes || model.outcomes || [];
    content.append(fold(`Actual model attempts and verifier outputs (${outcomes.length})`, () => {
      const tasks = node("div");
      for (const outcome of [...outcomes].sort((a, b) => String(a.task_id).localeCompare(String(b.task_id), undefined, {numeric: true}))) {
        const verification = outcome.verification;
        const result = verification.status === "verified" ? `score ${verification.score}` : `${verification.status} · unverified`;
        tasks.append(fold(`Task ${outcome.task_id} · ${result}`, () => {
          const task = node("div");
          task.append(node("h3", "Native verifier result"), pretty(verification), textSection("Attempt metadata", {task_id: outcome.task_id, row_index: outcome.row_index, verifier_executed: outcome.verifier_executed, done: outcome.done, turns: outcome.turns}));
          const execution = prefix && `${prefix}${outcome.execution_path}/`;
          const evidence = execution ? record.artifacts.filter(item => item.path.startsWith(execution)) : [];
          const modelFiles = evidence.filter(item => /solver\/turn-\d+\/(request|response)\.json$/.test(item.path) || /agent\/trajectory\.json$/.test(item.path));
          const verifierFiles = evidence.filter(item => /\/(verifier-trace\.jsonl|verifier\.(stdout|stderr)|harbor-result\.json)$/.test(item.path));
          for (const item of modelFiles) task.append(savedEvidence(`Model ${item.path.includes("request.json") ? "request" : "response"} · ${item.path.split("/").slice(-2).join(" / ")}`, item.path));
          for (const item of verifierFiles) task.append(savedEvidence(`${outcome.verification_followup ? "Original execution verifier output" : "Actual verifier output"} · ${item.path.split("/").at(-1)}`, item.path));
          if (outcome.verification_followup) {
            task.append(node("p", "The scored verifier result includes a separately recorded re-verification of this saved answer. Original execution logs remain available above.", "review-byline"));
            for (const item of outcome.verification_followup.artifacts.filter(item => /\/(result\.json|verifier-trace\.jsonl|verifier\.(stdout|stderr))$/.test(item.path))) task.append(savedEvidence(`Scored re-verification output · ${item.path.split("/").at(-1)}`, item.path));
          }
          if (!modelFiles.length) task.append(node("p", "The model response trace is not published for this attempt.", "difficulty-evidence-status"));
          if (!verifierFiles.length) task.append(node("p", "The raw verifier log is not published; the recorded verifier result appears above.", "difficulty-evidence-status"));
          const input = evidence.find(item => item.path.endsWith("/input.json"));
          if (input) task.append(savedEvidence("Task prompt and execution input", input.path));
          return task;
        }));
      }
      return tasks;
    }));
    return content;
  }, false, "review-card");
  return section;
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
    document.title = `${record.source_id} · Quality and difficulty`;
    const supplemental = record.supplemental_reviews.flatMap(item => item.collection.reviews);
    const reviews = [...collection.reviews, ...supplemental];
    const subjects = [...collection.subjects, ...record.supplemental_reviews.flatMap(item => item.collection.subjects)];
    const provenance = collection.execution_provenance;
    const date = reviews.map(review => review.reviewed_at).filter(Boolean).sort().at(-1) || "Unknown";
    const referenceCommit = reviews.map(review => review.attributes?.inspected_marinskyrl_commit).find(Boolean);
    const codeProvenance = provenance?.marinskyrl_commit ? `MarinSkyRL execution commit ${provenance.marinskyrl_commit}` : referenceCommit ? `MarinSkyRL reference commit ${referenceCommit} · Reused evidence and current inspection` : "MarinSkyRL commit not recorded in imported review";
    document.getElementById("provenance").textContent = `Reviewed ${date} · ${codeProvenance}`;
    const container = document.getElementById("reviews");
    if (record.verifier_issues.length) container.append(fold("Confirmed verifier defects remain unresolved", () => {
      const content = node("div", undefined, "review-card-content");
      content.append(node("p", "The source is capped at Some issues while these defects remain open. Saved difficulty reports below are historical measurements and do not establish a current difficulty estimate.", "formatted-prose issue-highlight"));
      for (const issue of record.verifier_issues) {
        const anchor = node("a", "Open verifier issue ↗");
        anchor.href = issue.issue_url; anchor.target = "_blank"; anchor.rel = "noopener";
        content.append(node("p"), anchor);
      }
      return content;
    }, true, "review-card technical-issues"));
    const issues = reviews.flatMap(review => review.findings.filter(finding => finding.kind === "issue").map(finding => ({review, finding})));
    if (issues.length) container.append(fold(`Reported technical issues (${issues.length})`, () => {
      const content = node("div", undefined, "review-card-content");
      content.append(node("p", "Reports from separate judges can describe the same defect. Open the individual reviews below for context.", "review-byline"));
      for (const {review, finding} of issues) { const item = findingView(finding); item.prepend(node("p", `${methodNames[review.method] || label(review.method)} · ${review.reviewer.label}`, "review-byline")); content.append(item); }
      return content;
    }, true, "review-card technical-issues"));
    container.append(fold("Execution provenance & review applicability", () => pretty({execution_provenance: provenance, source_mappings: collection.source_mappings, created_at: collection.created_at})));
    const datasetProofs = record.artifacts.filter(item => /^publication\/dataset-revision-attestation-[a-f0-9]{40}\.json$/.test(item.path));
    for (const artifact of datasetProofs) {
      const proof = await jsonResponse(artifactUrl(artifact.path));
      container.append(fold("Dataset content equivalence", () => {
        const content = node("div", undefined, "review-card-content");
        content.append(node("p", `The ${proof.file_path} data file is byte-identical at the reviewed revision ${proof.executed_revision} and repository revision ${proof.current_revision}. The ${proof.component_selector} selection retains ${proof.population_count.toLocaleString()} tasks.`, "formatted-prose"));
        content.append(node("p", "These historical judgments describe the same task data. No new model judgments were run; the original review date and execution revision remain above. Verifier code applicability is checked separately.", "formatted-prose"));
        content.append(node("p", `Data SHA-256: ${proof.file_sha256}`, "review-byline"), artifactLink(artifact.path, "Browse content-equivalence proof"));
        return content;
      }, true));
    }
    if (record.artifacts.some(item => item.path === "difficulty/context.json")) {
      const context = await jsonResponse(artifactUrl("difficulty/context.json"));
      container.append(fold("Reviewer solve baseline & historical evidence", () => {
        const content = node("div", undefined, "review-card-content");
        const baseline = context.reviewer_rollout_baseline;
        content.append(node("p", `Quality-review solver: ${baseline.solved}/${baseline.verified} solved · ${baseline.attempted} attempted · ${baseline.unverified} unverified.`, "formatted-prose"));
        content.append(node("p", `Reviewer model: ${baseline.configured_model_reviewers.map(model => model.label).join(", ")}`, "review-byline"));
        content.append(textSection("Quality sample limitations", baseline.limitations, true));
        const historical = context.historical_reference;
        const reference = node("a", "Historical difficulty study · Marin issue #8942");
        reference.href = "https://github.com/marin-community/marin/issues/8942";
        content.append(reference, node("p", historical.applicability, "formatted-prose"));
        content.append(textSection("Historical comparison limitations", historical.limitations));
        content.append(artifactLink("difficulty/context.json", "Browse reviewer task outcomes & provenance"));
        if (record.artifacts.some(item => item.path === historical.artifact_path)) content.append(node("p"), artifactLink(historical.artifact_path, "Browse preserved historical evidence"));
        return content;
      }, true));
    }
    const difficultyArtifact = record.artifacts.find(item => item.path === "difficulty.json");
    if (difficultyArtifact) {
      const report = await jsonResponse(artifactUrl("difficulty.json"));
      const difficulty = fold("Difficulty · model solve rates and saved attempts", () => {
        const content = node("div", undefined, "review-card-content");
        content.append(node("p", `${report.sampling.task_count} shared tasks · ${report.sampling.method} · split ${report.split}`, "formatted-prose"), AtlasDifficulty.comparison(report.models));
        content.append(node("p", "Longer bars mean more tasks solved. Open a model to inspect its exact settings, attempts, and native verifier outputs. Unverified attempts are excluded from the solve rate.", "review-byline"));
        for (const model of report.models) {
          const identityPath = report.protocol?.run_identities?.find(item => item.size === model.size)?.run_identity_path || `difficulty/${model.size}/run.json`;
          content.append(difficultyRun(model, identityPath, record));
        }
        for (const followup of report.protocol_followups || []) {
          content.append(fold(`${followup.kind === "alternate_checkpoint" ? "Hosted comparison" : "Additional settings comparison"} · ${label(followup.state)}`, () => {
            const section = node("div", undefined, "review-card-content");
            section.append(node("p", "Separate run on the original task sample. Its checkpoint and any LLM verifier settings can differ from the paired models.", "formatted-prose"));
            section.append(textSection("Recorded model and setting changes", {model: followup.model, provider: followup.provider, model_revision: followup.model_revision, checkpoint_change: followup.checkpoint_change, generation_parameters: followup.generation_parameters, verifier_configuration_change: followup.verifier_configuration_change}, true));
            section.append(textSection("Comparison limitations", followup.limitations));
            if (followup.state === "complete" && record.artifacts.some(item => item.path === followup.report_path)) {
              const run = node("div", "Loading saved attempts…");section.append(run);
              jsonResponse(artifactUrl(followup.report_path)).then(data => {
                const identityPath = followup.report_path.replace(/difficulty\.json$/, "run.json");
                run.replaceChildren(difficultyRun({...followup, size: followup.kind === "alternate_checkpoint" ? "hosted" : "followup", task_outcomes: data.outcomes}, identityPath, record));
              }).catch(error => run.replaceChildren(node("p", error.message, "issue-highlight")));
            } else section.append(node("p", "This comparison has no complete published attempts yet.", "difficulty-evidence-status"));
            if (record.artifacts.some(item => item.path === followup.original_report_path)) section.append(artifactLink(followup.original_report_path, "Preserved original report ↗"));
            return section;
          }, false, "review-card"));
        }
        content.append(textSection("Sampling and scope", {sampling: report.sampling, population_count: report.population_count, source_revision: report.source_revision, estimated_at: report.estimated_at}), textSection("Limitations", report.limitations, true), artifactLink("difficulty.json", "Full difficulty report ↗"));
        return content;
      }, true, "review-card");
      difficulty.id = "difficulty";container.append(difficulty);
      if (location.hash === "#difficulty") requestAnimationFrame(() => difficulty.scrollIntoView());
    }
    const supersededIds = new Set(reviews.map(review => review.supersedes_review_id).filter(Boolean));
    const sourceSynthesis = review => review.method === "synthesis" && subjects.find(subject => subject.id === review.subject_id)?.level === "source";
    const ordered = [...reviews].sort((a, b) => Number(supersededIds.has(a.id)) - Number(supersededIds.has(b.id)) || Number(sourceSynthesis(b)) - Number(sourceSynthesis(a)) || (b.reviewed_at || "").localeCompare(a.reviewed_at || ""));
    ordered.forEach(review => container.append(reviewCard(review, subjects, supersededIds)));
    container.append(fold(`All saved evidence (${record.artifacts.length})`, () => {
      const list = node("ul", undefined, "evidence-list");
      record.artifacts.forEach(item => { const row = node("li"); row.append(artifactLink(item.path)); list.append(row); }); return list;
    }));
    if (record.supplemental_reviews.length) container.append(fold("Supplemental review collections", () => {
      const list = node("ul", undefined, "evidence-list");
      for (const item of record.supplemental_reviews) {
        const row = node("li"), anchor = node("a", "Download verifier defect review");
        anchor.href = `api/reviews/${encodeURIComponent(item.id)}`;
        row.append(anchor); list.append(row);
      }
      return list;
    }));
    if (record.review_pool.length) container.append(fold(`Source review pool (${record.review_pool.length} other collections)`, () => {
      const list = node("ul", undefined, "evidence-list");
      for (const item of record.review_pool) {
        const row = node("li"), anchor = node("a", `${item.review_count} reviews · ${item.updated_at}`);
        anchor.href = `review.html?id=${encodeURIComponent(item.id)}`;
        row.append(anchor); list.append(row);
      }
      return list;
    }));
    document.getElementById("download").addEventListener("click", () => download(collection, `${reviewId}.json`));
  } catch (error) { document.getElementById("title").textContent = error.message; }
})();
