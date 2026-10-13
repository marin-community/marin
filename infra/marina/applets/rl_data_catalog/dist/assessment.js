/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const source = new URLSearchParams(location.search).get("source");
const labels = {good: "Good", some_issues: "Some issues", bad: "Bad", unknown: "Unknown"};
function summary(report) {
  const target = $("summary");
  for (const cohort of ["admitted", "excluded"]) {
    const data = report.assessment[cohort], section = node("section", undefined, "assessment-card");
    section.append(node("h2", cohort === "admitted" ? "Admitted task opinions" : "Excluded row diagnostics"));
    section.append(node("p", `${labels[data.grade]} · ${data.reviewed}/${data.sample_count} reviewed · population ${data.population.toLocaleString()}`));
    section.append(node("p", Object.entries(data.verdict_counts).map(([verdict, count]) => `${labels[verdict]} ${count}`).join(" · ")));
    section.append(node("p", `Missing required evidence: ${data.missing_evidence}. Packets with omissions: ${data.packets_with_omissions}.`));
    target.append(section);
  }
  const controls = report.curation.verification, section = node("section", undefined, "assessment-card");
  section.append(node("h2", "Curation and executed controls"), node("p", `Curation: ${report.curation.status}. Scope: ${report.scope}.`));
  section.append(node("p", `Admission counts: ${JSON.stringify(report.curation.admission_counts)}`));
  section.append(node("p", controls ? `Controls: ${controls.status}. Checked / eligible: ${controls.sample_count ?? "—"} / ${controls.eligible_count ?? "—"}. Counts: ${JSON.stringify(controls.counts)}` : "Controls were not measured."));
  section.append(node("p", `Model: ${report.model.model}; revision ${report.model.revision}. ${report.grade_rule}`));
  const sourceEvidence = report.source_evidence;
  if (sourceEvidence) section.append(node("p", sourceEvidence.reader_declared ? `Original source evidence: ${sourceEvidence.rows} selected rows recovered. Previews are bounded; read omissions in each packet.` : "This pipeline has no original-row reader."));
  target.append(section);
}
function samples(report) {
  for (const cohort of ["admitted", "excluded"]) {
    const section = node("section"); section.append(node("h2", cohort === "admitted" ? "Admitted sample" : "Excluded sample"));
    for (const sample of report.evidence.samples.filter(item => item.cohort === cohort)) {
      const details = node("details", undefined, "assessment-card"), heading = node("summary", `${labels[sample.verdict]} · ${sample.task_id}`), content = node("div");
      details.append(heading, content); section.append(details);
      details.addEventListener("toggle", async () => {
        if (!details.open || content.childNodes.length) return;
        content.append(node("p", "Loading task evidence…"));
        try {
          const data = await json(`api/evidence?source_id=${encodeURIComponent(source)}&task_id=${encodeURIComponent(sample.task_id)}`);
          content.replaceChildren(node("p", data.judgment.explanation), node("p", `Citations: ${data.judgment.evidence_ids.join(", ")}`));
          if (data.omissions.length) content.append(node("p", `Omissions: ${data.omissions.join("; ")}`));
          const packet = node("details"), title = node("summary", "Task and pipeline evidence packet"); packet.append(title, node("pre", JSON.stringify(data.packet, null, 2))); content.append(packet);
        } catch (error) {content.replaceChildren(node("p", error.message));}
      });
    }
    $("samples").append(section);
  }
}
(async () => {try {const data = await json(`api/assessment?source_id=${encodeURIComponent(source)}`); $("title").textContent = data.source; $("message").textContent = `${data.report.basis} · ${data.report.scope} curation`; summary(data.report); samples(data.report); $("report").textContent = JSON.stringify(data.report, null, 2);} catch (error) {$("title").textContent = "Assessment unavailable"; $("message").textContent = error.message;}})();
