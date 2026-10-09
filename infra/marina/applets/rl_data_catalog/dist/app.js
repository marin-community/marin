/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const $ = (id) => document.getElementById(id);
const isPublicView = window.location.hostname === "public.applets.marina.oa.dev";
const state = { sources: [], refreshes: [], origin: "all", sort: "display_name", direction: 1, refreshing: false, error: false };
const columns = [
  ["display_name", "Source name"], ["origin", "Catalog"], ["canonical_source", "Dataset"],
  ["task_count", "Input rows"], ["turns", "Interaction"],
  ["type", "Type"], ["is_benchmark", "Benchmark"], ["family", "Family"],
  ["difficulty", "Difficulty"], ["quality", "Quality"], ["review_date", "Review date"], ["status", "Status"],
];
const number = new Intl.NumberFormat("en-US");
const date = (value) => value ? new Date(value).toLocaleDateString("en-US", {month: "short", day: "2-digit", year: "numeric", timeZone: "UTC"}) : "—";
function node(tag, text, className) { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; if (className) element.className = className; return element; }
function link(text, url) { const element = node("a", text); if (url && /^https:\/\//.test(url)) element.href = url; element.target = "_blank"; element.rel = "noopener"; return element; }
function chip(text, className) { return node("span", text || "Unclassified", `chip ${className || ""}`); }
function matchesDifficulty(row) {
  const selection = $("difficulty").value;
  if (selection === "all") return true;
  if (selection === "measured") return row.difficulty_summary?.status === "current";
  if (selection === "historical") return row.difficulty_summary?.status === "historical";
  if (selection === "unmeasured") return row.difficulty_summary?.status !== "current";
  const model = AtlasDifficulty.currentLarge(row.difficulty_summary);
  if (!model?.verified) return false;
  const percent = 100 * model.solved / model.verified;
  const [lower, upper] = selection.split("-").map(Number);
  return (lower === 0 ? percent >= 0 : percent > lower) && percent <= upper;
}
function filtered() {
  const search = $("search").value.trim().toLowerCase();
  const rows = state.sources.filter(row =>
    (state.origin === "all" || row.origin === state.origin) &&
    ($("excluded").checked || row.status !== "Excluded") &&
    ($("show-bad").checked || row.quality !== "bad") &&
    ($("quality").value === "all" || ($("quality").value === "unknown" ? !row.quality : row.quality === $("quality").value)) &&
    matchesDifficulty(row) &&
    ($("type").value === "all" || ($("type").value === "unknown" ? !row.type : row.type === $("type").value)) &&
    ($("turns").value === "all" || row.turns === $("turns").value) &&
    ($("family").value === "all" || ($("family").value === "unknown" ? !row.family : row.family === $("family").value)) &&
    ($("benchmark").value === "all" || String(row.is_benchmark) === $("benchmark").value) &&
    (!search || [row.display_name,row.canonical_source,row.name,row.family,row.dataset_id,row.notes,row.origin,...(row.tags || [])].join(" ").toLowerCase().includes(search))
  );
  return rows.sort((a,b) => {
    const value = row => state.sort === "difficulty" ? AtlasDifficulty.currentLarge(row.difficulty_summary)?.solve_rate : row[state.sort];
    const x=value(a), y=value(b);
    if (x === null || x === undefined || x === "") return y === null || y === undefined || y === "" ? a.id.localeCompare(b.id) : 1;
    if (y === null || y === undefined || y === "") return -1;
    const order = typeof x === "number" || typeof x === "boolean" ? Number(x)-Number(y) : String(x).localeCompare(String(y),undefined,{numeric:true});
    return order * state.direction || a.id.localeCompare(b.id);
  });
}
function render() {
  const rows = filtered();
  const available = state.sources.filter(row => row.status === "Available");
  $("metric-sources").textContent = number.format(available.length);
  const counted = rows;
  $("metric-tasks").textContent = number.format(counted.reduce((total,row)=>total+(row.task_count||0),0));
  const unknownCounts = counted.filter(row=>row.kind === "Dataset" && row.task_count === null).length;
  $("task-count-context").textContent = `${rows.length} visible entries · ${unknownCounts} unknown dataset counts · may overlap`;
  for (const [id,origin] of [["all-count",null],["sky-count","MarinSkyRL"],["trove-count","Task Trove"]]) $(id).textContent = number.format(state.sources.filter(row => (!origin || row.origin === origin) && ($("excluded").checked || row.status !== "Excluded")).length);
  $("headers").replaceChildren();
  for (const [key,title] of columns) {
    const th=node("th"); th.scope="col"; th.setAttribute("aria-sort",state.sort === key ? state.direction === 1 ? "ascending" : "descending" : "none");
    const button=node("button",title); button.append(node("span",state.sort === key ? state.direction === 1 ? "↑" : "↓" : "↕","sort-icon"));
    button.addEventListener("click",()=>{state.direction=state.sort === key ? -state.direction : 1;state.sort=key;render();}); th.append(button); $("headers").append(th);
  }
  const fragment=document.createDocumentFragment();
  for (const row of rows) {
    const tr=node("tr");tr.dataset.sourceId=row.id;
    for (const [key] of columns) {
      const td=node("td");const value=row[key];
      if (key === "display_name") {td.className="name-cell";const line=node("div",undefined,"name-top");const button=node("button","ⓘ","detail-button");button.setAttribute("aria-label",`Details for ${row.display_name}`);button.addEventListener("click",()=>details(row));line.append(link(row.display_name,row.url),button);td.append(line,node("div",row.dataset_id,"source-subtitle"));}
      else if (key === "canonical_source") td.append(link(value,row.canonical_url));
      else if (key === "difficulty") {
        if (row.difficulty_summary) {
          const anchor=node("a",undefined,"difficulty-link");anchor.href=`review.html?id=${encodeURIComponent(row.review_id)}#difficulty`;
          anchor.setAttribute("aria-label",`Difficulty attempts and verifier results for ${row.display_name}`);
          anchor.append(AtlasDifficulty.comparison(row.difficulty_summary.models, row.difficulty_summary, true));td.append(anchor);
        } else {td.textContent=row.verifier_issues.length ? "Withheld · verifier issues" : row.review_stale ? "Needs re-review" : "Not measured";td.className="muted";}
      }
      else if (key === "quality") {const labels={good:"Good",some_issues:"Some issues",bad:"Bad"};const badge=chip(labels[value] || (row.review_stale ? "Needs re-review" : row.review_id ? "Unrated" : "Unreviewed"), `quality-${value || "unknown"}`);badge.title=row.verifier_issues.length ? "Confirmed verifier defect remains unresolved. Open the source's review pool and GitHub issue." : row.review_stale ? "Source data or verifier changed since this review. Open the historical report." : "Sample-based review conclusion; open the report for coverage and limitations.";if(row.review_id){const anchor=node("a");anchor.href=`review.html?id=${encodeURIComponent(row.review_id)}`;anchor.append(badge);td.append(anchor);}else td.append(badge);}
      else if (key === "review_date") {td.textContent=date(value);td.title=value || "No dated review";}
      else if (key === "origin") td.append(chip(value,value === "MarinSkyRL" ? "sky" : "trove"));
      else if (key === "type") td.append(chip(value,(value||"").toLowerCase()));
      else if (key === "status") td.append(chip(value,value === "Excluded" ? "excluded" : ""));
      else if (key === "task_count") {td.textContent=value === null || value === undefined ? (row.kind === "Generator" ? "Generated" : "—") : number.format(value);td.className=`num ${value === null || value === undefined ? "muted" : ""}`;td.title="Selected input rows before conversion or curation";}
      else if (key === "is_benchmark") {td.textContent=value ? "True" : "False";td.className=value ? "benchmark-yes" : "muted";}
      else {td.textContent=value || "—";if (!value || value === "Unknown") td.className="muted";}
      tr.append(td);
    }
    fragment.append(tr);
  }
  $("rows").replaceChildren(fragment);$("empty").hidden=rows.length > 0 || state.sources.length === 0;
  const excluded = state.sources.filter(row=>row.status === "Excluded").length;
  $("result-count").textContent=`${number.format(rows.length)} of ${number.format(state.sources.length)} entries · ${excluded} excluded sources${$("excluded").checked ? " included" : " hidden"} · bad quality ${$("show-bad").checked ? "included" : "hidden"}`;
  const errors=state.refreshes.filter(item=>item.error);
  $("metric-status").replaceChildren(node("span",isPublicView ? "Saved snapshot" : state.refreshing ? "Checking…" : state.error || errors.length ? "Needs attention" : state.refreshes.length > 0 ? "Up to date" : "Awaiting sync"),node("span",undefined,"live-dot"));
  const checked=state.refreshes.map(item=>item.checked_at).filter(Boolean).sort()[0];
  $("checked-at").textContent=checked ? `Last checked ${new Date(checked).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})} · ${date(checked)}` : "No successful sync yet";
}
function details(row) {
  $("detail-name").textContent=row.display_name;$("detail-chips").replaceChildren(chip(row.origin,row.origin === "MarinSkyRL" ? "sky" : "trove"),chip(row.type,(row.type||"").toLowerCase()));
  $("detail-notes").textContent=row.notes;
  const fields=[["Dataset",row.dataset_id],["Dataset revision",row.dataset_revision],["Source ID",row.id],["Input rows",row.task_count === null ? row.kind === "Generator" ? "No fixed count" : "Unknown" : number.format(row.task_count)],["Input files",row.pipeline?.files],["Pipeline",row.pipeline?.name],["Pipeline version",row.pipeline?.version],["Family",row.family],["Tags",row.tags],["Verifier",row.verifier_name],["Verifier revision",row.verifier_revision],["Quality",row.quality||"Not curated"]];
  $("detail-fields").replaceChildren();for(const [title,value] of fields) {if(value === undefined || value === null || value === "" || Array.isArray(value) && !value.length)continue;const dl=node("dl",undefined,"field");dl.append(node("dt",title),node("dd",Array.isArray(value) ? value.join(", ") : String(value)));$("detail-fields").append(dl);}
  if(row.difficulty_summary){const chart=AtlasDifficulty.comparison(row.difficulty_summary.models, row.difficulty_summary, true);$("detail-fields").append(chart);}
  $("detail-links").replaceChildren(link("Pinned dataset ↗",row.url));if(row.verifier_url) $("detail-links").append(link("Verifier code ↗",row.verifier_url));
  if(row.review_id){const anchor=node("a","Read quality review ↗");anchor.href=`review.html?id=${encodeURIComponent(row.review_id)}`;$("detail-links").append(anchor);}
  if(row.difficulty_summary){const anchor=node("a","Read difficulty attempts & verifier results ↗");anchor.href=`review.html?id=${encodeURIComponent(row.review_id)}#difficulty`;$("detail-links").append(anchor);}
  $("details").showModal();
}
async function load() { const response=await fetch("api/sources",{cache:"no-store"});if(!response.ok)throw new Error(`Catalog read failed (${response.status})`);const data=await response.json();state.sources=data.sources;state.refreshes=data.refreshes;for(const [field,title] of [["family","families"]]) {
    const selected=$(field).value;
    const options=[node("option",`All ${title}`)];options[0].value="all";
    for(const name of [...new Set(state.sources.map(row=>row[field]).filter(Boolean))].sort()) {
      const option=node("option",name);option.value=name;options.push(option);
    }
    if(field === "family" && state.sources.some(row=>!row.family)) {const option=node("option","Unknown / unclassified");option.value="unknown";options.push(option);}
    $(field).replaceChildren(...options);$(field).value=options.some(option=>option.value===selected)?selected:"all";
  }
  render(); }
async function refresh(force=false) {
  if(state.refreshing)return;state.refreshing=true;state.error=false;$("refresh").disabled=true;render();$("sync-banner").className="sync-banner";$("sync-banner").textContent="Loading the packaged task-curation catalog. Saved reviews remain available…";
  try {
    const response=await fetch(`api/refresh${force ? "?force=true" : ""}`,{method:"POST"});if(!response.ok)throw new Error(`Refresh failed (${response.status})`);
    const result=await response.json();await load();
    if(result.busy) {$("sync-banner").textContent=result.message;setTimeout(async()=>{try{await load();}catch(error){showError(error);}},2500);}
    else {const errors=state.refreshes.filter(item=>item.error);if(errors.length){$("sync-banner").classList.add("warning");$("sync-banner").textContent=errors.map(item=>`${item.origin}: ${item.error}. Keeping the last successful snapshot.`).join(" ");}else {$("sync-banner").textContent=state.refreshes.map(item=>`${item.origin} ${item.revision.slice(0,8)}`).join("  ·  ")+"  ·  Packaged task-curation catalog loaded.";}}
  }catch(error){showError(error);}finally{state.refreshing=false;$("refresh").disabled=false;render();}
}
function showError(error){state.error=true;$("sync-banner").className="sync-banner warning";$("sync-banner").textContent=isPublicView ? `${error.message}. Reload this page to retry.` : `${error.message}. Saved sources remain available; retry with Refresh sources.`;$("metric-status").textContent="Needs attention";}
for(const id of ["type","turns","family","benchmark","difficulty","excluded"]) $(id).addEventListener("change",render);$("search").addEventListener("input",render);
$("quality").addEventListener("change",()=>{if($("quality").value === "bad")$("show-bad").checked=true;render();});
$("show-bad").addEventListener("change",()=>{if(!$("show-bad").checked && $("quality").value === "bad")$("quality").value="all";render();});
for(const button of document.querySelectorAll(".tab"))button.addEventListener("click",()=>{state.origin=button.dataset.origin;for(const other of document.querySelectorAll(".tab"))other.classList.toggle("selected",other===button);render();});
$("reset").addEventListener("click",()=>{for(const id of ["type","turns","family","benchmark","quality","difficulty"])$(id).value="all";$("search").value="";$("excluded").checked=false;$("show-bad").checked=false;state.origin="all";for(const button of document.querySelectorAll(".tab"))button.classList.toggle("selected",button.dataset.origin==="all");render();});
$("refresh").hidden=isPublicView;
if(!isPublicView)$("refresh").addEventListener("click",()=>refresh(true));
$("close-details").addEventListener("click",()=>$("details").close());
document.addEventListener("keydown",event=>{if(event.key==="/" && !["INPUT","SELECT","TEXTAREA"].includes(document.activeElement.tagName) && !$("details").open){event.preventDefault();$("search").focus();}});
$("export").addEventListener("click",()=>{const quote=value=>`"${String(value ?? "").replaceAll('"','""')}"`;const fields=[...columns,["url","Pinned dataset URL"],["dataset_revision","Dataset revision"],["name","Pipeline / source name"],["tags","Tags"],["verifier_url","Verifier URL"],["verifier_revision","Verifier revision"]];const csv=[fields.map(([,title])=>quote(title)).join(","),...filtered().map(row=>fields.map(([key])=>quote(row[key])).join(","))].join("\r\n");const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));const a=node("a");a.href=url;a.download="rl-data-atlas.csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
(async()=>{try{await load();if(isPublicView){$("sync-banner").textContent="Showing the latest saved catalog. The last catalog sync is shown above.";}else await refresh();}catch(error){showError(error);}})();
