// Render the scorecard table from the JSON embedded by build.py.

const LABEL_TEXT = { official: "Official", unofficial: "Unofficial" };
const LINK_TEXT = { evaldash: "Open the run in EvalDash", storage: "Open the raw results in Filer" };

function element(tag, attributes = {}, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function scoreCell(cell) {
  const td = element("td");
  if ("missing" in cell) {
    td.className = "missing";
    td.textContent = "–";
    if (cell.missing) td.title = cell.missing;
    return td;
  }
  const tooltip = [LINK_TEXT[cell.link.kind]];
  if (cell.run_id) tooltip.push(cell.run_id);
  if (cell.note) tooltip.push(cell.note);
  const link = element("a", { href: cell.link.url, title: tooltip.join("\n"), target: "_blank", rel: "noopener" });
  link.append(element("span", { class: "score" }, cell.score.toFixed(2)));
  if (cell.note) link.append(element("span", { class: "note-mark" }, "*"));
  td.append(link, element("span", { class: `chip ${cell.label}` }, LABEL_TEXT[cell.label]));
  return td;
}

function render(data) {
  const table = document.getElementById("scorecard");
  const head = element("tr");
  head.append(element("th", { class: "model" }, "Model"));
  for (const name of data.evals) head.append(element("th", {}, name));
  table.append(element("thead")).append(head);

  const body = table.appendChild(element("tbody"));
  for (const model of data.models) {
    const row = body.appendChild(element("tr"));
    const name = element("th", { class: "model", scope: "row" });
    name.append(element("a", { href: model.url, title: model.hf_repo, target: "_blank", rel: "noopener" }, model.name));
    row.append(name);
    for (const evalName of data.evals) row.append(scoreCell(model.cells[evalName]));
  }
}

render(JSON.parse(document.getElementById("scorecard-data").textContent));
