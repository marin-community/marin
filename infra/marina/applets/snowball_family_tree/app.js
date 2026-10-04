const DATA = JSON.parse(document.getElementById("graph-data").textContent),
  nodes = DATA.nodes,
  byId = new Map(nodes.map((n) => [n.id, n]));
const $ = (s) => document.querySelector(s),
  esc = (s) =>
    String(s ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
const SEPT21 = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21";
let detailId = null,
  resultsDismissed = false,
  selected = "pretrain",
  zoom = 1,
  expandedNode = null,
  overviewMode = true,
  offsetX = 0,
  offsetY = 0,
  viewMode = "map",
  focusRoot = null,
  visibleNodes = nodes,
  positions = new Map(),
  canvasWidth = 0,
  canvasHeight = 0;
const children = new Map(
  nodes.map((n) => [n.id, nodes.filter((c) => c.parents.includes(n.id))]),
);
function sourceName(url) {
  const m = url.match(/issues\/(\d+)/);
  return m
    ? "Issue #" + m[1]
    : url.includes("discord.com")
      ? "Discord thread"
      : "Experiment record";
}
function ancestors(id, set = new Set()) {
  if (set.has(id)) return set;
  set.add(id);
  for (const p of byId.get(id)?.parents || []) ancestors(p, set);
  return set;
}
function descendants(id, set = new Set()) {
  if (set.has(id)) return set;
  set.add(id);
  for (const n of nodes) if (n.parents.includes(id)) descendants(n.id, set);
  return set;
}
function shortTitle(n) {
  const labels = {
    pretrain: "67B A2B pretrain",
    "base-2.7": "2.7T cooldown",
    "base-5.7": "5.7T cooldown",
    "base-10": "10T cooldown",
    "lce-qk157": "262K · QK 1.57",
    "lce-qk175": "262K · QK 1.75",
    "lce-skew2": "262K · 2× long docs",
    "lce-skew4": "262K · 4× long docs",
    "lce-skew8": "262K · 8× long docs",
    base157000: "September 14 base",
    "antidoom-ftpo-math-step35": "FTPO · math 35",
    "antidoom-ftpo-full-window-step45": "FTPO · full window 45",
    "antidoom-ftpo-reset-math-step5": "FTPO · reset math 5",
    "antidoom-ftpo-mixed-step5": "FTPO · mixed 5",
    "antidoom-ftpo-mixed-step10": "FTPO · mixed 10",
    "loop-length-sweep-six-arms": "Loop / length · 6 arms",
  };
  if (labels[n.id]) return labels[n.id];
  if (n.id.includes("Antidoom-RLVR1-Reference-KL"))
    return "Antidoom · ref/KL 20";
  if (n.id.includes("Antidoom-RLVR1-Step12")) return "Antidoom · RLVR1 12";
  if (n.model?.endsWith("-dr-doom")) return "Dr. Doom · FTPO";
  const date = n.model?.match(/Datakit-SFT-262K-2026\.09\.(\d+)$/);
  if (date) return "09." + date[1] + " · Datakit SFT";
  if (n.id.startsWith("raw-"))
    return n.title.replace("Raw math RL · ", "Raw · ").replace(/-rno2a.*$/, "");
  return n.title
    .replace(
      "Broad Datakit + Terminus, seven token rows · September 21",
      "09.21 · Datakit SFT",
    )
    .replace("2.7T • ", "")
    .replace("5.7T • ", "")
    .replace("September ", "09.")
    .replace("Relay continuation H9 · 999", "Relay H9 · 999")
    .replace("Relay + all Kimi H8 · 1,203", "Relay H8 · 1203")
    .replace("Mini-swe-agent relay SFT · 2,394", "MSA relay · 2394")
    .replace("GLM5.3 rollout SFT · September 23", "GLM5.3 SFT · 09.23");
}
function card(n, reading = false) {
  const expanded = reading || expandedNode === n.id;
  return `<article class="node ${n.org === "community" ? "community" : ""} ${expanded ? "expanded" : "collapsed"} ${selected === n.id ? "selected" : ""}" data-node="${esc(n.id)}" ${reading ? "" : `id="node-${esc(n.id)}"`} aria-label="${esc(n.title)}">
 <h2>${reading ? esc(n.title) : `<button class="node-toggle" data-expand="${esc(n.id)}" aria-expanded="${expanded}" aria-label="${expanded ? "Collapse" : "Expand"} ${esc(n.title)}">${esc(expanded ? n.title : shortTitle(n))}</button>`}</h2>
 <div class="node-brief"><span>${esc(n.stage)}</span><span>${children.get(n.id).length ? "↳ " + children.get(n.id).length : "leaf"}</span></div>
 <div class="node-full"><div class="node-top"><span class="org-label">${esc(n.org)}</span><span class="kind">${esc(n.stage || "Experiment")}</span></div>${n.model ? `<span class="model">${esc(n.model)}</span>` : ""}${/failed|negative result|collapsed final/.test(n.status) ? '<div class="badge">NEGATIVE RESULT / FAILED TRIAL</div>' : ""}${n.confidence !== "documented" ? `<div class="badge">${esc(n.confidence || "Qualified provenance")}</div>` : ""}<p class="summary">${esc(n.summary)}</p><div class="tags">${n.algorithms.map((a) => `<span class="tag">${esc(a)}</span>`).join("")}</div><div class="compute">${esc(n.flops || "FLOPs unknown")}</div><div class="node-foot"><a href="${esc(n.source)}" target="_blank" rel="noopener noreferrer">${sourceName(n.source)} ↗</a>${n.model ? `<a href="https://huggingface.co/${esc(n.model)}" target="_blank" rel="noopener noreferrer">Weights ↗</a>` : ""}<button type="button" data-select="${esc(n.id)}">Details</button></div>${n.parents.length ? `<div class="from">${n.edge_kind === "retained checkpoint" ? "Earlier checkpoint" : "Initialized from"} ${n.parents.map((p) => `<button data-jump="${esc(p)}">${esc(byId.get(p)?.title || p)}</button>`).join(" + ")}</div>` : ""}</div></article>`;
}
function ordered(list) {
  const result = [],
    seen = new Set();
  function add(n) {
    if (seen.has(n.id)) return;
    for (const p of n.parents) {
      const pn = list.find((x) => x.id === p);
      if (pn) add(pn);
    }
    seen.add(n.id);
    result.push(n);
  }
  list.forEach(add);
  return result;
}
function matches(n) {
  const query = $("#search").value.trim().toLowerCase(),
    family = $("#family").value,
    org = $("#org").value;
  return (
    (family === "all" || n.lane === family) &&
    (org === "all" || n.org === org) &&
    (!query || JSON.stringify(n).toLowerCase().includes(query))
  );
}
function treeMembers() {
  const family = $("#family").value;
  const root =
    focusRoot ||
    (family === "all"
      ? null
      : { "2.7T": "base-2.7", "5.7T": "base-5.7", "10T": "base-10" }[family]);
  if (!root) return nodes;
  const keep = descendants(root);
  for (const id of ancestors(root)) keep.add(id);
  return nodes.filter((n) => keep.has(n.id));
}
function render() {
  visibleNodes = treeMembers();
  $("#tree-nodes").innerHTML = visibleNodes.map((n) => card(n)).join("");
  $("#read-grid").innerHTML = ordered(nodes)
    .map((n) => card(n, true))
    .join("");

  const label = focusRoot
    ? byId.get(focusRoot).title
    : $("#family").value === "all"
      ? "Whole family tree"
      : $("#family").value + " cooldown branch";
  $("#tree-caption").textContent =
    `${label} · ${visibleNodes.length} nodes · all checkpoints shown · click a node for its full summary`;
  $("#collapse-node").disabled = !expandedNode;
  requestAnimationFrame(() => {
    detail(selected);
    const workspace = $(".workspace");
    if (innerWidth > 760)
      workspace.style.height =
        Math.max(360, innerHeight - workspace.getBoundingClientRect().top) +
        "px";
    else workspace.style.height = "";
    if (viewMode === "map") {
      layout();
      if (overviewMode) fitTree();
    }
    applyFilter();
  });
}
function layout() {
  const world = $("#world"),
    pad = 32,
    gap = 22,
    levelGap = 30;
  const visible = new Set(visibleNodes.map((n) => n.id));
  const localChildren = (id) =>
    (children.get(id) || []).filter((n) => visible.has(n.id));
  const blocks = new Map();
  function measure(n, depth) {
    $("#world").classList.toggle("overview-dense", !expandedNode);
    const el = document.getElementById("node-" + n.id),
      width = expandedNode === n.id ? 320 : 148;
    el.style.width = width + "px";
    const height = el.offsetHeight,
      kids = localChildren(n.id);
    kids.forEach((c) => measure(c, depth + 1));
    // Broad sibling sets wrap inside their parent's branch, keeping every node visible.
    const columns =
      n.id === "pretrain" ? kids.length : Math.min(2, Math.max(1, kids.length));
    const rows = [];
    for (let i = 0; i < kids.length; i += columns)
      rows.push(kids.slice(i, i + columns));
    const columnWidths = Array(columns).fill(0);
    rows.forEach((row) =>
      row.forEach(
        (c, i) =>
          (columnWidths[i] = Math.max(columnWidths[i], blocks.get(c.id).width)),
      ),
    );
    const childrenWidth = kids.length
      ? columnWidths.reduce((a, b) => a + b, 0) + gap * (columns - 1)
      : 0;
    const rowHeights = rows.map((row) =>
      Math.max(...row.map((c) => blocks.get(c.id).height)),
    );
    const blockWidth = Math.max(width, childrenWidth) + 22;
    const blockHeight =
      height +
      (kids.length
        ? levelGap +
          rowHeights.reduce((a, b) => a + b, 0) +
          gap * (rows.length - 1)
        : 0);
    blocks.set(n.id, {
      width: blockWidth,
      height: blockHeight,
      nodeWidth: width,
      nodeHeight: height,
      rows,
      rowHeights,
      columnWidths,
      childrenWidth,
      depth,
    });
  }
  const roots = visibleNodes.filter(
    (n) => !n.parents.some((p) => visible.has(p)),
  );
  roots.forEach((n) => measure(n, 0));
  positions = new Map();
  const paths = [];
  function place(n, left, top) {
    const block = blocks.get(n.id),
      x = left + (block.width - 22 - block.nodeWidth) / 2,
      el = document.getElementById("node-" + n.id);
    const pos = {
      x,
      y: top,
      width: block.nodeWidth,
      height: block.nodeHeight,
      depth: block.depth,
      subtreeWidth: block.width,
    };
    positions.set(n.id, pos);
    el.style.left = x + "px";
    el.style.top = top + "px";
    let rowTop = top + block.nodeHeight + levelGap;
    const childLeft = left + (block.width - 22 - block.childrenWidth) / 2;
    block.rows.forEach((row, rowIndex) => {
      let colLeft = childLeft;
      row.forEach((c, col) => {
        const childBlock = blocks.get(c.id);
        place(
          c,
          colLeft + (block.columnWidths[col] - childBlock.width) / 2,
          rowTop,
        );
        const target = positions.get(c.id),
          sx = x + pos.width / 2,
          sy = top + pos.height,
          tx = target.x + target.width / 2,
          ty = target.y;
        const firstBus = top + pos.height + levelGap / 2,
          bus = rowTop - levelGap / 2,
          rail = left + block.width - 10;
        const d =
          rowIndex === 0
            ? `M${sx},${sy} V${bus} H${tx} V${ty}`
            : `M${sx},${sy} V${firstBus} H${rail} V${bus} H${tx} V${ty}`;
        const color =
          n.org === "community" ? "var(--community)" : "var(--athena)";
        paths.push(
          `<path class="edge ${c.confidence === "inferred" ? "inferred" : ""}" style="stroke:${color}" data-from="${esc(n.id)}" data-to="${esc(c.id)}" d="${d}" marker-end="url(#arrow)"><title>${esc(n.title)} → ${esc(c.title)}</title></path>`,
        );
        colLeft += block.columnWidths[col] + gap;
      });
      rowTop += block.rowHeights[rowIndex] + gap;
    });
  }
  let cursor = pad;
  roots.forEach((n) => {
    place(n, cursor, pad);
    cursor += blocks.get(n.id).width + gap;
  });
  canvasWidth = cursor - gap + pad;
  canvasHeight =
    Math.max(...roots.map((n) => blocks.get(n.id).height)) + 2 * pad;
  world.style.width = canvasWidth + "px";
  world.style.height = canvasHeight + "px";
  $("#edges").setAttribute("width", canvasWidth);
  $("#edges").setAttribute("height", canvasHeight);
  $("#paths").innerHTML = paths.join("");
  applyZoom();
  highlight();
}
function applyZoom() {
  const view = $("#viewport");
  offsetX =
    Math.max(0, (view.clientWidth - canvasWidth * zoom) / 2) + 11 * zoom;
  offsetY = 18;
  $("#sizer").style.width =
    Math.max(view.clientWidth, canvasWidth * zoom + 2 * offsetX) + "px";
  $("#sizer").style.height =
    Math.max(view.clientHeight, canvasHeight * zoom + offsetY) + "px";
  $("#world").classList.toggle("overview-dense", !expandedNode);
  $("#world").style.setProperty(
    "--label-factor",
    Math.min(1.8, Math.max(1, 0.65 / zoom)),
  );
  $("#world").style.left = offsetX + "px";
  $("#world").style.top = offsetY + "px";
  $("#world").style.transform = `scale(${zoom})`;
  $("#zoom").textContent =
    zoom < 0.1 ? (zoom * 100).toFixed(1) + "%" : Math.round(zoom * 100) + "%";
}
function fitTree() {
  const view = $("#viewport");
  overviewMode = true;
  zoom = Math.min(
    1,
    (view.clientWidth - 28) / canvasWidth,
    (view.clientHeight - 36) / canvasHeight,
  );
  applyZoom();
  view.scrollTo({ left: 0, top: 0, behavior: "instant" });
}
function zoomAt(next, px, py) {
  const view = $("#viewport"),
    worldX = (view.scrollLeft + px - offsetX) / zoom,
    worldY = (view.scrollTop + py - offsetY) / zoom;
  overviewMode = false;
  zoom = Math.max(0.04, Math.min(2.4, next));
  applyZoom();
  view.scrollTo({
    left: worldX * zoom + offsetX - px,
    top: worldY * zoom + offsetY - py,
    behavior: "instant",
  });
}
function highlight() {
  const chain = ancestors(selected),
    offspring = descendants(selected);
  document.querySelectorAll("[data-node]").forEach((e) => {
    e.classList.toggle("selected", e.dataset.node === selected);
    e.classList.toggle(
      "descendant",
      offspring.has(e.dataset.node) && e.dataset.node !== selected,
    );
    e.classList.toggle(
      "ancestor",
      chain.has(e.dataset.node) && e.dataset.node !== selected,
    );
  });
  document.querySelectorAll(".edge").forEach((e) => {
    e.classList.toggle(
      "highlight",
      chain.has(e.dataset.from) && chain.has(e.dataset.to),
    );
    e.classList.toggle(
      "subtree",
      offspring.has(e.dataset.from) && offspring.has(e.dataset.to),
    );
    e.classList.toggle(
      "dim",
      !chain.has(e.dataset.to) && !offspring.has(e.dataset.from),
    );
  });
}
function applyFilter() {
  const matched = nodes.filter(matches),
    relevant = new Set();
  matched.forEach((n) => ancestors(n.id, relevant));
  document
    .querySelectorAll("#world [data-node]")
    .forEach((e) => e.classList.toggle("dim", !relevant.has(e.dataset.node)));
  document
    .querySelectorAll("#read-grid [data-node]")
    .forEach((e) => (e.hidden = !matches(byId.get(e.dataset.node))));
  const filtered = $("#search").value.trim();
  $("#results").hidden = !filtered || viewMode !== "map" || resultsDismissed;
  $("#results").innerHTML =
    `<span>${matched.length} matches</span>` +
    matched
      .slice(0, 8)
      .map((n) => `<button data-jump="${esc(n.id)}">${esc(n.title)}</button>`)
      .join("");
  $("#read-status").textContent =
    `${matched.length} matching nodes · all summaries and checkpoint evidence`;
  $("#filter-status").textContent =
    viewMode === "timeline"
      ? "Play to reveal checkpoints · scrub to explore · click for details"
      : matched.length === nodes.length
        ? "Wheel to zoom · drag to pan · click a node to expand"
        : `${matched.length} matches · ancestors retained for context`;
}
function detail(id) {
  const n = byId.get(id);
  if (!n) return;
  if (detailId !== id) $("#detail").scrollTop = 0;
  detailId = id;
  selected = id;
  const path = ordered(nodes.filter((x) => ancestors(id).has(x.id)));
  const evidence = n.evidence || [];
  const shortLabels = {
    pretrain: "Shared pretrain",
    "base-2.7": "2.7T cooldown",
    "base-5.7": "5.7T cooldown",
    "base-10": "10T cooldown",
    [SEPT21]: "September 21 Datakit SFT",
  };
  $("#tree-path").innerHTML =
    "<span>Selected ancestry:</span>" +
    path
      .map(
        (x) =>
          `<button data-jump="${esc(x.id)}">${esc(shortLabels[x.id] || x.title)}</button>`,
      )
      .join('<span aria-hidden="true">→</span>');
  $("#detail-content").innerHTML =
    `<span class="kicker">Selected lineage / ${esc(n.lane)}</span><h2>${esc(n.title)}</h2><div class="detail-model">${esc(n.model || n.artifact || "Native training run / checkpoint")}</div><div class="tags">${n.algorithms.map((a) => `<span class="tag">${esc(a)}</span>`).join("")}</div><p>${esc(n.summary)}</p><a class="source-link" target="_blank" rel="noopener noreferrer" href="${esc(n.source)}">${sourceName(n.source)} ↗</a>${n.model ? `<a class="source-link" target="_blank" rel="noopener noreferrer" href="https://huggingface.co/${esc(n.model)}">Hugging Face ↗</a>` : ""}<section><h3>Direct children · ${children.get(id).length}</h3>${
      children.get(id).length
        ? `<ul>${children
            .get(id)
            .map(
              (c) =>
                `<li><button class="child-link" data-jump="${esc(c.id)}">${esc(c.title)}</button></li>`,
            )
            .join("")}</ul>`
        : "<p>No documented children in this record.</p>"
    }<button class="btn branch-button" data-focus="${esc(id)}">Focus this branch</button></section><section><h3>Compute / canonical checkpoint</h3><p><strong>${esc(n.flops || "FLOPs unknown")}</strong></p><p>${esc(n.notes || "No additional compute accounting was established in the available source.")}</p></section><section><h3>Weight lineage / saved-state chain</h3><div class="pathline">${path.map((x) => `<button data-jump="${esc(x.id)}">${esc(x.title)}</button>`).join("<br>↓<br>")}</div></section>${(n.retained || []).length ? `<section><h3>Retained checkpoints / aliases</h3><ul>${n.retained.map((x) => (typeof x === "string" ? `<li>${esc(x)}</li>` : `<li><a target="_blank" rel="noopener noreferrer" href="${esc(x.url)}">${esc(x.label || x.model || x.url)}</a>${x.note ? "<br>" + esc(x.note) : ""}</li>`)).join("")}</ul></section>` : ""}<section><h3>Evidence / ${esc(n.confidence || "documented")}</h3>${evidence.length ? `<ul>${evidence.map((x) => `<li><a href="${esc(x.url)}" target="_blank" rel="noopener noreferrer">${esc(x.label || sourceName(x.url))} ↗</a><br>${esc(x.text || x.note || "")}</li>`).join("")}</ul>` : "<p>The canonical experiment record above establishes this branch.</p>"}</section>`;
  highlight();
}
function centerNode(id, behavior = "instant") {
  const pos = positions.get(id);
  if (!pos) return;
  const view = $("#viewport");
  view.scrollTo({
    left: Math.max(
      0,
      (pos.x + pos.width / 2) * zoom + offsetX - view.clientWidth / 2,
    ),
    top: Math.max(0, pos.y * zoom + offsetY - 24),
    behavior,
  });
}
function expandNode(id) {
  selected = id;
  expandedNode = expandedNode === id ? null : id;
  overviewMode = false;
  zoom = Math.max(zoom, 0.95);
  $("#collapse-node").disabled = !expandedNode;
  render();
  requestAnimationFrame(() => centerNode(id));
}
function jump(id) {
  resultsDismissed = true;
  if (viewMode === "timeline") {
    selectTimelineNode(id);
    return;
  }
  if (viewMode === "map") {
    if (!positions.has(id)) {
      focusRoot = null;
      $("#family").value = "all";
    }
    expandedNode = null;
    expandNode(id);
  } else {
    detail(id);
    const el = [...document.querySelectorAll("#read-grid [data-node]")].find(
      (e) => e.dataset.node === id,
    );
    if (el) {
      el.hidden = false;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }
}
function focusBranch(id) {
  focusRoot = id;
  $("#family").value = "all";
  selected = id;
  expandedNode = null;
  overviewMode = true;
  render();
}
document.addEventListener("click", (e) => {
  const expand = e.target.closest("[data-expand]"),
    select = e.target.closest("[data-select]"),
    j = e.target.closest("[data-jump]"),
    focus = e.target.closest("[data-focus]");
  if (expand) expandNode(expand.dataset.expand);
  else if (
    viewMode === "map" &&
    e.target.closest("#tree-nodes .node.collapsed") &&
    !e.target.closest("a,button")
  )
    expandNode(e.target.closest("[data-node]").dataset.node);
  if (select) detail(select.dataset.select);
  if (j) jump(j.dataset.jump);
  if (focus) {
    setMode("map");
    focusBranch(focus.dataset.focus);
  }
});
$("#search").addEventListener("input", () => {
  resultsDismissed = false;
  applyFilter();
});
$("#org").addEventListener("input", applyFilter);
$("#family").addEventListener("input", () => {
  focusRoot = null;
  expandedNode = null;
  overviewMode = true;
  selected =
    $("#family").value === "all"
      ? "pretrain"
      : { "2.7T": "base-2.7", "5.7T": "base-5.7", "10T": "base-10" }[
          $("#family").value
        ];
  render();
});
$("#collapse-node").onclick = () => {
  expandedNode = null;
  $("#collapse-node").disabled = true;
  render();
  requestAnimationFrame(() => centerNode(selected));
};
function setMode(mode) {
  stopTimeline();
  viewMode = mode;
  $("#pane").classList.toggle("mode-read", mode === "read");
  $("#pane").classList.toggle("mode-timeline", mode === "timeline");
  document.body.classList.toggle("timeline-mode", mode === "timeline");
  for (const [id, value] of [
    ["mapbutton", "map"],
    ["readbutton", "read"],
    ["timelinebutton", "timeline"],
  ]) {
    $("#" + id).classList.toggle("active", mode === value);
    $("#" + id).setAttribute("aria-pressed", mode === value);
  }
  applyFilter();
}
$("#mapbutton").onclick = () => {
  setMode("map");
  if (!visibleNodes.some((n) => n.id === selected)) {
    focusRoot = null;
    $("#family").value = "all";
    expandedNode = null;
    overviewMode = true;
  }
  render();
  requestAnimationFrame(() => {
    if (!overviewMode) centerNode(selected);
  });
};
$("#readbutton").onclick = () => {
  setMode("read");
  render();
};
function stepZoom(factor) {
  if (viewMode === "read") {
    setMode("map");
    layout();
    if (overviewMode) fitTree();
  }
  const view = $("#viewport");
  zoomAt(zoom * factor, view.clientWidth / 2, view.clientHeight / 2);
}
$("#minus").onclick = () => stepZoom(1 / 1.25);
$("#plus").onclick = () => stepZoom(1.25);
function overview() {
  setMode("map");
  expandedNode = null;
  $("#collapse-node").disabled = true;
  overviewMode = true;
  render();
}
$("#fit").onclick = overview;
$("#fit-side").onclick = overview;
$("#head").onclick = () => {
  setMode("map");
  detail("pretrain");
  requestAnimationFrame(() => {
    layout();
    if (overviewMode) fitTree();
    centerNode("pretrain");
  });
};
$("#reset-tree").onclick = () => {
  setMode("map");
  focusRoot = null;
  $("#family").value = "all";
  $("#search").value = "";
  $("#org").value = "all";
  selected = "pretrain";
  expandedNode = null;
  $("#collapse-node").disabled = true;
  overviewMode = true;
  render();
};
$("#viewport").addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const view = $("#viewport"),
      r = view.getBoundingClientRect();
    const delta =
      e.deltaY *
      (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? view.clientHeight : 1);
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY) && !e.ctrlKey) {
      view.scrollLeft += e.deltaX;
      return;
    }
    zoomAt(
      zoom * Math.exp(-delta * (e.ctrlKey ? 0.008 : 0.002)),
      e.clientX - r.left,
      e.clientY - r.top,
    );
  },
  { passive: false },
);
let drag = null;
$("#viewport").addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || e.target.closest("button,a,input,select")) return;
  const view = $("#viewport");
  drag = {
    x: e.clientX,
    y: e.clientY,
    left: view.scrollLeft,
    top: view.scrollTop,
  };
  view.setPointerCapture(e.pointerId);
  view.classList.add("dragging");
});
$("#viewport").addEventListener("pointermove", (e) => {
  if (!drag) return;
  $("#viewport").scrollLeft = drag.left - (e.clientX - drag.x);
  $("#viewport").scrollTop = drag.top - (e.clientY - drag.y);
});
function endDrag() {
  drag = null;
  $("#viewport").classList.remove("dragging");
}
$("#viewport").addEventListener("pointerup", endDrag);
$("#viewport").addEventListener("pointercancel", endDrag);
window.addEventListener("resize", () => {
  const workspace = $(".workspace");
  workspace.style.height =
    innerWidth > 760
      ? Math.max(360, innerHeight - workspace.getBoundingClientRect().top) +
        "px"
      : "";
  if (viewMode === "map") {
    if (overviewMode) fitTree();
    else applyZoom();
  }
});
let timelineFamily = "10T",
  timelineNodes = [],
  timelineRoot = null,
  timelineStart = 0,
  timelineEnd = 0,
  timelineTime = 0,
  timelineFrame = null,
  timelineLast = null,
  timelineLastDay = null;
const DAY = 86400000;
const TEN_T_TIMELINE_START = Date.parse("2026-09-07T00:00:00Z");
const fullDate = new Intl.DateTimeFormat("en-US", {
  month: "long",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});
const shortDate = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});
function nodeTime(n) {
  return n.timeline.date
    ? Date.parse(n.timeline.date + "T00:00:00Z")
    : timelineEnd;
}
function stopTimeline() {
  if (timelineFrame !== null) cancelAnimationFrame(timelineFrame);
  timelineFrame = null;
  timelineLast = null;
  if ($("#timeline-play")) {
    $("#timeline-play").textContent = "Play";
    $("#timeline-play").classList.remove("active");
  }
}
function timelineSource(n) {
  const issue = n.source.match(/\/issues\/\d+/)
    ? n.source
    : (n.evidence || [])
        .map((e) => e.url)
        .find((url) =>
          /github\.com\/marin-community\/marin\/issues\/\d+/.test(url),
        );
  return `<a href="${esc(issue || n.source)}" target="_blank" rel="noopener noreferrer">${sourceName(issue || n.source)} ↗</a>${n.model ? `<a href="https://huggingface.co/${esc(n.model)}" target="_blank" rel="noopener noreferrer">Weights ↗</a>` : ""}`;
}
function timelineModels(list, label) {
  $("#timeline-model-label").textContent = label;
  $("#timeline-model-list").innerHTML = list
    .map(
      (n) =>
        `<div><button class="timeline-model-name" data-time-node="${esc(n.id)}" title="${esc(n.model || n.title)}">${esc(n.model || n.title)}</button>${timelineSource(n)}</div>`,
    )
    .join("");
}
function sizeTimeline() {
  const workspace = $(".workspace");
  workspace.style.height =
    innerWidth > 760
      ? Math.max(360, innerHeight - workspace.getBoundingClientRect().top) +
        "px"
      : "";
}
function setupTimeline(autoplay = true) {
  stopTimeline();
  timelineFamily = $("#timeline-family").value;
  timelineRoot = byId.get(
    { "2.7T": "base-2.7", "5.7T": "base-5.7", "10T": "base-10" }[
      timelineFamily
    ],
  );
  timelineNodes = ordered(
    nodes.filter((n) => n.lane === timelineFamily && n.id !== timelineRoot.id),
  );
  const order = new Map(timelineNodes.map((n, i) => [n.id, i]));
  timelineNodes.sort(
    (a, b) =>
      (a.timeline.date || "9999").localeCompare(b.timeline.date || "9999") ||
      order.get(a.id) - order.get(b.id),
  );
  timelineStart =
    timelineFamily === "10T" ? TEN_T_TIMELINE_START : nodeTime(timelineRoot);
  timelineEnd = Math.max(
    timelineStart + DAY,
    ...timelineNodes.filter((n) => n.timeline.date).map(nodeTime),
  );
  timelineTime = timelineStart;
  timelineLastDay = null;
  const columns = Math.ceil(timelineNodes.length / 6);
  $("#timeline-events").style.gridTemplateColumns =
    `repeat(${columns},minmax(0,1fr))`;
  $("#timeline-events").innerHTML = timelineNodes
    .map(
      (n) =>
        `<button class="timeline-event ${n.org === "community" ? "community" : ""}" data-time-node="${esc(n.id)}" aria-label="${esc(n.title)} · ${esc(n.timeline.date || "Date unresolved")}" title="${esc(n.title)}" disabled><strong>${esc(shortTitle(n))}</strong><span>${esc(n.timeline.date ? shortDate.format(nodeTime(n)) : "Undated")} · ${esc(n.stage)}</span></button>`,
    )
    .join("");
  $("#timeline-origin").innerHTML =
    `<span class="kicker">Fixed antecedent</span><button data-time-node="${esc(timelineRoot.id)}">${esc(timelineRoot.title)}</button><p>Shared 67B A2B pretrain<br>↓<br>${esc(timelineRoot.title)}</p>${timelineSource(timelineRoot)}`;
  $("#timeline-ticks").style.gridTemplateColumns =
    `repeat(${columns},minmax(0,1fr))`;
  $("#timeline-ticks").innerHTML = Array.from({ length: columns }, (_, i) => {
    const group = timelineNodes.slice(i * 6, i * 6 + 6),
      first = group[0],
      last = group[group.length - 1];
    return `<span>${first.timeline.date ? shortDate.format(nodeTime(first)) : "Undated"}${last.timeline.date !== first.timeline.date ? " – " + (last.timeline.date ? shortDate.format(nodeTime(last)) : "undated") : ""}</span>`;
  }).join("");
  $("#tree-caption").textContent =
    `${timelineFamily} timeline · ${timelineNodes.length + 1} nodes · temporal grouping under one cooldown; original lineage remains in Details`;
  detail(timelineRoot.id);
  sizeTimeline();
  paintTimeline();
  if (autoplay && !matchMedia("(prefers-reduced-motion: reduce)").matches)
    playTimeline();
}
function paintTimeline() {
  const revealed = timelineNodes.filter((n) => nodeTime(n) <= timelineTime),
    visible = new Set(revealed.map((n) => n.id));
  document.querySelectorAll(".timeline-event").forEach((e) => {
    const shown = visible.has(e.dataset.timeNode);
    e.classList.toggle("revealed", shown);
    e.disabled = !shown;
    e.classList.toggle("selected", selected === e.dataset.timeNode);
  });
  $("#timeline-clock").textContent = fullDate.format(timelineTime);
  $("#timeline-scrub").value = Math.round(
    ((timelineTime - timelineStart) / (timelineEnd - timelineStart)) * 1000,
  );
  $("#timeline-scrub").setAttribute(
    "aria-valuetext",
    fullDate.format(timelineTime),
  );
  $("#timeline-count").textContent =
    `${revealed.length + 1} / ${timelineNodes.length + 1} nodes`;
  $("#timeline-empty").hidden = revealed.length > 0;
  const day = Math.floor(timelineTime / DAY);
  if (day !== timelineLastDay) {
    timelineLastDay = day;
    timelineModels(
      revealed.length ? revealed : [timelineRoot],
      "Model names / source records · newest at right",
    );
    $("#timeline-model-list").scrollLeft = $(
      "#timeline-model-list",
    ).scrollWidth;
  }
}
function playTimeline() {
  if (timelineFrame !== null) return;
  if (timelineTime >= timelineEnd) {
    timelineTime = timelineStart;
    timelineLastDay = null;
    paintTimeline();
  }
  $("#timeline-play").textContent = "Pause";
  $("#timeline-play").classList.add("active");
  timelineLast = null;
  function frame(now) {
    if (timelineLast !== null)
      timelineTime = Math.min(
        timelineEnd,
        timelineTime +
          ((now - timelineLast) * (timelineEnd - timelineStart)) /
            Number($("#timeline-speed").value),
      );
    timelineLast = now;
    paintTimeline();
    if (timelineTime >= timelineEnd) {
      stopTimeline();
      return;
    }
    timelineFrame = requestAnimationFrame(frame);
  }
  timelineFrame = requestAnimationFrame(frame);
}
function selectTimelineNode(id) {
  const n = byId.get(id);
  if (!n) return;
  stopTimeline();
  if (n.lane !== timelineFamily && n.id !== "pretrain") {
    $("#timeline-family").value = n.lane;
    setupTimeline(false);
  }
  if (n.id === "pretrain") {
    detail(id);
    timelineModels([n], "Shared pretraining root");
    return;
  }
  timelineTime = Math.max(timelineTime, nodeTime(n));
  paintTimeline();
  detail(id);
  sizeTimeline();
  document
    .querySelectorAll(".timeline-event")
    .forEach((e) => e.classList.toggle("selected", e.dataset.timeNode === id));
  timelineModels([n], "Selected model / source record");
  $("#detail-content").insertAdjacentHTML(
    "afterbegin",
    `<p class="timeline-date-basis"><strong>${esc(n.timeline.date ? fullDate.format(nodeTime(n)) : "Date unresolved")}</strong><br>${esc(n.timeline.basis)} · <a href="${esc(n.timeline.source)}" target="_blank" rel="noopener noreferrer">Date evidence ↗</a></p>`,
  );
}
$("#timelinebutton").onclick = () => {
  setMode("timeline");
  setupTimeline();
};
$("#timeline-family").onchange = () => setupTimeline();
function toggleTimeline() {
  timelineFrame !== null ? stopTimeline() : playTimeline();
}
$("#timeline-play").onclick = toggleTimeline;
document.addEventListener("keydown", (e) => {
  if (
    viewMode !== "timeline" ||
    e.code !== "Space" ||
    e.altKey ||
    e.ctrlKey ||
    e.metaKey ||
    e.shiftKey
  )
    return;
  if (
    e.target.closest(
      'input,textarea,select,[contenteditable]:not([contenteditable="false"])',
    )
  )
    return;
  e.preventDefault();
  if (!e.repeat) toggleTimeline();
});
$("#timeline-restart").onclick = () => setupTimeline();
$("#timeline-all").onclick = () => {
  stopTimeline();
  timelineTime = timelineEnd;
  paintTimeline();
};
$("#timeline-scrub").oninput = () => {
  stopTimeline();
  timelineTime =
    timelineStart +
    ((timelineEnd - timelineStart) * Number($("#timeline-scrub").value)) / 1000;
  paintTimeline();
};
document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-time-node]");
  if (target) selectTimelineNode(target.dataset.timeNode);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopTimeline();
});

$("#count").textContent = nodes.length;
$("#edgecount").textContent = nodes.reduce((n, x) => n + x.parents.length, 0);
$("#coverage").textContent =
  "Evidence: GitHub issues + comments · Discord threads · model cards\nGraph data and token-count extracts packaged with this applet; source links open the original records.";
$("#unplaced").innerHTML = (DATA.unplaced || []).length
  ? "<strong>Provenance still unresolved:</strong> " +
    DATA.unplaced
      .map(
        (x) =>
          `<a target="_blank" rel="noopener noreferrer" href="${esc(x.url)}">${esc(x.model || x.title)}</a> — ${esc(x.reason)}`,
      )
      .join("; ")
  : "";
render();
