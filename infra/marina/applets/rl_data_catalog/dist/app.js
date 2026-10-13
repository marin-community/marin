/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const number = new Intl.NumberFormat("en-US");
const labels = {good: "Good", some_issues: "Some issues", bad: "Bad", unknown: "Unknown", unassessed: "Unassessed"};
const state = {sources: [], sort: "display_name", direction: 1};
const columns = [["display_name", "Source"], ["origin", "Catalog"], ["task_count", "Input rows"], ["admitted", "Admitted rows"], ["quality", "Sampled grade"], ["sample", "Admitted / excluded sample"], ["curation", "Curation"], ["controls", "Executed controls"], ["type", "Type"], ["status", "Availability"]];
function grade(row) {return row.assessment.status === "complete" ? row.assessment.admitted.grade : "unassessed";}
function value(row, key) {
  if (key === "quality") return grade(row);
  if (key === "curation") return row.curation?.status || "inventory";
  if (key === "admitted") return row.curation?.admission_counts?.admitted;
  if (key === "sample") return row.assessment.admitted?.sample_count || 0;
  if (key === "controls") return row.curation?.verification?.status || "Not measured";
  return row[key];
}
function filtered() {
  const search = $("search").value.trim().toLowerCase();
  return state.sources.filter(row => ["origin", "quality", "curation", "type"].every(key => $(key).value === "all" || $(key).value === value(row, key)) && (!search || [row.display_name, row.name, row.family, ...(row.tags || [])].join(" ").toLowerCase().includes(search))).sort((a, b) => {
    const x = value(a, state.sort), y = value(b, state.sort);
    if (x === null || x === undefined) return y === null || y === undefined ? a.id.localeCompare(b.id) : 1;
    if (y === null || y === undefined) return -1;
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y), undefined, {numeric: true})) * state.direction || a.id.localeCompare(b.id);
  });
}
function render() {
  $("headers").replaceChildren();
  for (const [key, title] of columns) {
    const th = node("th"), button = node("button", title);
    th.scope = "col";
    th.setAttribute("aria-sort", state.sort === key ? state.direction === 1 ? "ascending" : "descending" : "none");
    button.addEventListener("click", () => {state.direction = state.sort === key ? -state.direction : 1; state.sort = key; render();});
    th.append(button); $("headers").append(th);
  }
  const rows = filtered(), fragment = document.createDocumentFragment();
  for (const row of rows) {
    const tr = node("tr");
    tr.dataset.sourceId = row.id;
    for (const [key] of columns) {
      const td = node("td"), item = value(row, key);
      if (key === "display_name") {const button = node("button", row.display_name, "text-button"); button.addEventListener("click", () => details(row)); td.append(button, node("div", row.dataset_id, "source-subtitle"));}
      else if (key === "quality") {const badge = node("span", labels[item], `chip quality-${item}`); if (row.assessment.status === "complete") {const anchor = node("a"); anchor.href = `assessment.html?source=${encodeURIComponent(row.id)}`; anchor.append(badge); td.append(anchor);} else td.append(badge);}
      else if (key === "sample") td.textContent = row.assessment.status === "complete" ? `${row.assessment.admitted.sample_count} / ${row.assessment.excluded.sample_count}` : "—";
      else td.textContent = item === null || item === undefined ? "—" : typeof item === "number" ? number.format(item) : item;
      tr.append(td);
    }
    fragment.append(tr);
  }
  $("rows").replaceChildren(fragment); $("empty").hidden = rows.length > 0;
  $("visible-count").textContent = `${number.format(rows.length)} of ${number.format(state.sources.length)} sources · all grades included`;
}
function details(row) {
  $("detail-name").textContent = row.display_name; $("detail-notes").textContent = row.notes;
  const controls = row.curation?.verification;
  const fields = [["Source ID", row.id], ["Dataset revision", row.dataset_revision], ["Pipeline", row.pipeline?.name], ["Grade", labels[grade(row)]], ["Assessment scope", row.assessment.scope], ["Model", row.assessment.model?.model], ["Admitted opinions", row.assessment.admitted?.verdict_counts], ["Excluded opinions", row.assessment.excluded?.verdict_counts], ["Curation", row.curation?.status], ["Admission counts", row.curation?.admission_counts], ["Control status", controls?.status], ["Control counts", controls?.counts], ["Control sample / eligible", controls ? `${controls.sample_count ?? "—"} / ${controls.eligible_count ?? "—"}` : undefined], ["Assessment note", row.assessment.reason || row.assessment.latest_attempt_error]];
  $("detail-fields").replaceChildren();
  for (const [title, item] of fields) {if (item === null || item === undefined) continue; const field = node("dl", undefined, "field"); field.append(node("dt", title), node("dd", typeof item === "object" ? JSON.stringify(item) : String(item))); $("detail-fields").append(field);}
  $("detail-links").replaceChildren();
  if (row.url?.startsWith("https://")) {const anchor = node("a", "Pinned dataset ↗"); anchor.href = row.url; anchor.rel = "noopener"; anchor.target = "_blank"; $("detail-links").append(anchor);}
  if (row.assessment.status === "complete") {const anchor = node("a", "Read assessment and task evidence ↗"); anchor.href = `assessment.html?source=${encodeURIComponent(row.id)}`; $("detail-links").append(anchor);}
  $("details").showModal();
}
async function load() {
  const response = await fetch("api/sources", {cache: "no-store"});
  if (!response.ok) throw new Error(`Catalog read failed (${response.status})`);
  const data = await response.json(); state.sources = data.sources;
  $("source-count").textContent = number.format(data.sources.length);
  $("assessment-count").textContent = `${data.sources.filter(row => row.assessment.status === "complete").length} complete assessments`;
  $("sample-count").textContent = number.format(data.sources.reduce((total, row) => total + (row.assessment.admitted?.sample_count || 0) + (row.assessment.excluded?.sample_count || 0), 0));
  $("snapshot-date").textContent = new Date(data.created_at).toLocaleDateString(); $("snapshot-id").textContent = data.snapshot_id;
  const selection = $("origin").value;
  $("origin").replaceChildren(node("option", "All catalogs")); $("origin").firstChild.value = "all";
  for (const origin of [...new Set(data.sources.map(row => row.origin))].sort()) {const option = node("option", origin); option.value = origin; $("origin").append(option);}
  $("origin").value = [...$("origin").options].some(option => option.value === selection) ? selection : "all"; render();
}
for (const id of ["origin", "quality", "curation", "type"]) $(id).addEventListener("change", render);
$("search").addEventListener("input", render); $("close-details").addEventListener("click", () => $("details").close());
$("reset").addEventListener("click", () => {for (const id of ["origin", "quality", "curation", "type"]) $(id).value = "all"; $("search").value = ""; render();});
$("export").addEventListener("click", () => {
  const quote = item => `"${String(item ?? "").replaceAll('"', '""')}"`;
  const cell = (row, key) => key === "sample" ? `${row.assessment.admitted?.sample_count || 0} / ${row.assessment.excluded?.sample_count || 0}` : value(row, key);
  const csv = [columns.map(([, title]) => quote(title)).join(","), ...filtered().map(row => columns.map(([key]) => quote(cell(row, key))).join(","))].join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
  const anchor = node("a"); anchor.href = url; anchor.download = "rl-data-atlas.csv"; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
load().then(() => {$("message").textContent = "Showing the loaded curation catalog.";}).catch(error => {$("message").textContent = error.message;});
