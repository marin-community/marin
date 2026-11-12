// Render the scorecard table from the JSON embedded by build.py, with one delegated hover layer.

const LABEL_TEXT = { official: "Official", unofficial: "Unofficial" };
const LINK_TEXT = { evaldash: "Click to open the run in EvalDash", storage: "Click to open the raw results in Filer" };

function element(tag, attributes = {}, text) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function scoreCell(cell, column) {
  const td = element("td", { "data-col": column });
  if ("missing" in cell) {
    td.className = "missing";
    td.textContent = "–";
    return td;
  }
  td.className = `scored ${cell.label}`;
  td.style.setProperty("--heat", String(cell.score / 100));
  const link = element("a", { href: cell.link.url, target: "_blank", rel: "noopener" });
  link.append(element("span", { class: "score" }, cell.score.toFixed(2)));
  if (cell.note) link.append(element("span", { class: "note-mark" }, "*"));
  td.append(link);
  return td;
}

function tooltipLines(model, evalName, cell) {
  const lines = [[`${model.name} · ${evalName}`, "tip-title"]];
  if ("missing" in cell) {
    lines.push(["No score"]);
    if (cell.missing) lines.push([cell.missing]);
    return lines;
  }
  lines.push([cell.score.toFixed(2), "tip-score"], [LABEL_TEXT[cell.label]]);
  if (cell.note) lines.push([cell.note]);
  lines.push([LINK_TEXT[cell.link.kind]]);
  if (cell.run_id) lines.push([cell.run_id, "tip-id"]);
  return lines;
}

function render(data) {
  const table = document.getElementById("scorecard");
  const head = element("tr");
  head.append(element("th", { class: "model", scope: "col" }, "Model"));
  data.evals.forEach((name, column) => head.append(element("th", { scope: "col", "data-col": column }, name)));
  table.appendChild(element("thead")).append(head);

  const body = table.appendChild(element("tbody"));
  data.models.forEach((model, row) => {
    const tr = body.appendChild(element("tr", { "data-row": row }));
    const name = element("th", { class: "model", scope: "row" });
    name.append(element("a", { href: model.url, target: "_blank", rel: "noopener", title: model.hf_repo }, model.name));
    tr.append(name);
    data.evals.forEach((evalName, column) => tr.append(scoreCell(model.cells[evalName], column)));
  });

  const tooltip = document.getElementById("tooltip");
  const headers = head.querySelectorAll("th[data-col]");
  let active = null;

  function clear() {
    if (!active) return;
    active.row.classList.remove("active");
    headers[active.column].classList.remove("active");
    tooltip.hidden = true;
    active = null;
  }

  function show(td) {
    const row = td.parentElement;
    const column = Number(td.dataset.col);
    if (active && active.td === td) return;
    clear();
    row.classList.add("active");
    headers[column].classList.add("active");
    active = { td, row, column };

    const model = data.models[Number(row.dataset.row)];
    const evalName = data.evals[column];
    tooltip.replaceChildren(
      ...tooltipLines(model, evalName, model.cells[evalName]).map(([text, cls]) =>
        element("div", cls ? { class: cls } : {}, text),
      ),
    );
    tooltip.hidden = false;
    const rect = td.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    const left = Math.min(rect.left, window.innerWidth - tip.width - 12);
    const below = rect.bottom + 8 + tip.height < window.innerHeight;
    tooltip.style.left = `${Math.max(12, left)}px`;
    tooltip.style.top = `${below ? rect.bottom + 8 : rect.top - tip.height - 8}px`;
  }

  body.addEventListener("mouseover", (event) => {
    const td = event.target.closest("td");
    if (td) show(td);
  });
  body.addEventListener("focusin", (event) => {
    const td = event.target.closest("td");
    if (td) show(td);
  });
  body.addEventListener("mouseleave", clear);
  body.addEventListener("focusout", clear);
  window.addEventListener("scroll", clear, { passive: true, capture: true });
}

render(JSON.parse(document.getElementById("scorecard-data").textContent));
