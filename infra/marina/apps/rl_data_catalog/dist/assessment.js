/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const source = new URLSearchParams(location.search).get("source");
const labels = {good: "Good", some_issues: "Some issues", bad: "Bad", unknown: "Unknown"};
function summary(report) {
  const target = $("summary");
  for (const cohort of ["admitted", "excluded"]) {
    const data = report.assessment[cohort], section = node("section", undefined, "assessment-card");
    section.append(node("h2", cohort === "admitted" ? "Retained task opinions" : "Excluded row diagnostics"));
    section.append(node("p", `${labels[data.grade]} · ${data.reviewed}/${data.sample_count} reviewed · population ${data.population.toLocaleString()}`));
    section.append(node("p", ["good", "some_issues", "bad", "unknown"].map(verdict => `${labels[verdict]} ${data.verdict_counts[verdict]}`).join(" · ")));
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
function samples(report) {
  $('samples').append(taskReviews(source, report.evidence.samples));
}
(async () => {try {const data = await json(`api/assessment?source_id=${encodeURIComponent(source)}`); $("title").textContent = data.source; $("message").textContent = `Sampled model opinions · ${data.report.scope} curation`; summary(data.report); samples(data.report); $("report").textContent = JSON.stringify(data.report, null, 2);} catch (error) {$("title").textContent = "Assessment unavailable"; $("message").textContent = error.message;}})();
