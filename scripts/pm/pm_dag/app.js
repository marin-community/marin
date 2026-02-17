/* global d3, dagreD3, jsyaml */

const state = {
  rawData: null,
  nodes: new Map(),
  edges: [],
  hasFit: false,
};

const issueCache = new Map();
const issueInflight = new Set();
const ISSUE_CACHE_KEY = "pmDagIssueCacheV1";
let scheduledRender = null;
const MAX_ISSUE_LABELS = 3;
const MAX_ISSUE_TITLE = 80;

function scheduleRender() {
  if (scheduledRender) return;
  scheduledRender = window.setTimeout(() => {
    scheduledRender = null;
    renderGraph();
  }, 150);
}

function loadIssueCache() {
  try {
    const raw = localStorage.getItem(ISSUE_CACHE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    Object.entries(parsed).forEach(([key, value]) => {
      const number = Number(key);
      if (!Number.isNaN(number)) {
        issueCache.set(number, value);
      }
    });
  } catch (err) {
    // ignore cache errors
  }
}

function persistIssueCache() {
  try {
    const payload = {};
    issueCache.forEach((value, key) => {
      payload[key] = value;
    });
    localStorage.setItem(ISSUE_CACHE_KEY, JSON.stringify(payload));
  } catch (err) {
    // ignore cache errors
  }
}

const dagSvg = d3.select("#dag");
const inner = dagSvg.select("g");
const zoom = d3.zoom().on("zoom", (event) => {
  inner.attr("transform", event.transform);
});

dagSvg.call(zoom);

const renderer = new dagreD3.render();

const detailsBody = document.getElementById("detailsBody");
const fileInput = document.getElementById("fileInput");
const loadDefault = document.getElementById("loadDefault");
const searchInput = document.getElementById("searchInput");
const timelineOnly = document.getElementById("timelineOnly");
const hideBackburner = document.getElementById("hideBackburner");
const rankDir = document.getElementById("rankDir");

const DEFAULT_PATHS = [
  "/.agents/the_plan.yaml",
  "../../.agents/the_plan.yaml",
  "../../../.agents/the_plan.yaml",
];

function isObject(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function buildGraphData(parsed) {
  const nodes = new Map();
  const edges = [];

  Object.entries(parsed).forEach(([id, value]) => {
    if (id === "meta") return;
    if (!isObject(value)) return;
    const deps = Array.isArray(value.dependencies) ? value.dependencies : [];
    nodes.set(id, {
      id,
      title: value.title || id,
      type: value.type || null,
      status: value.status || null,
      owners: value.owners || [],
      ownerNames: value.owner_names || [],
      targetDate: value.target_date || null,
      labels: value.labels || [],
      issue: value.issue || null,
      description: value.description || null,
      definitionOfDone: value.definition_of_done || null,
      dependencies: deps,
    });

    deps.forEach((dep) => {
      edges.push({ from: dep, to: id });
    });
  });

  return { nodes, edges };
}

function issueNumber(issue) {
  if (!issue) return null;
  if (typeof issue === "number") return issue;
  if (typeof issue === "string" && /^\\d+$/.test(issue)) return Number(issue);
  return null;
}

async function fetchIssue(number) {
  if (!number || issueCache.has(number) || issueInflight.has(number)) return;
  issueInflight.add(number);
  try {
    const res = await fetch(`https://api.github.com/repos/marin-community/marin/issues/${number}`);
    if (!res.ok) return;
    const data = await res.json();
    const assignee = data.assignee || (Array.isArray(data.assignees) ? data.assignees[0] : null);
    issueCache.set(number, {
      title: data.title,
      state: data.state,
      labels: Array.isArray(data.labels) ? data.labels.map((label) => label.name) : [],
      assigneeLogin: assignee ? assignee.login : null,
      assigneeAvatar: assignee ? assignee.avatar_url : null,
      fetchedAt: Date.now(),
    });
    persistIssueCache();
  } catch (err) {
    // ignore transient fetch failures
  } finally {
    issueInflight.delete(number);
    scheduleRender();
  }
}

function ensureIssues(nodes) {
  nodes.forEach((node) => {
    const number = issueNumber(node.issue);
    if (!number) return;
    fetchIssue(number);
  });
}

function makeIssueUrl(issue) {
  if (!issue) return null;
  if (typeof issue === "number") {
    return `https://github.com/marin-community/marin/issues/${issue}`;
  }
  if (typeof issue === "string") {
    if (issue.startsWith("milestone/")) {
      return `https://github.com/marin-community/marin/${issue}`;
    }
    if (/^\d+$/.test(issue)) {
      return `https://github.com/marin-community/marin/issues/${issue}`;
    }
  }
  return null;
}

function nodeClass(node) {
  const classes = ["node"];
  if (node.type) classes.push(node.type);
  if (node.status) classes.push(`status-${node.status}`);
  if (node.labels.includes("backburner")) classes.push("backburner");
  return classes.join(" ");
}

function renderNodeLabel(node) {
  const issueNum = issueNumber(node.issue);
  const issueInfo = issueNum ? issueCache.get(issueNum) : null;
  const badges = [];
  if (node.type) badges.push(node.type);
  if (node.status) badges.push(node.status);
  if (node.labels.includes("backburner")) badges.push("backburner");
  if (node.labels.includes("timeline")) badges.push("timeline");
  if (issueInfo && issueInfo.state) badges.push(`issue-${issueInfo.state}`);

  const issueBadges = issueInfo
    ? issueInfo.labels
        .slice(0, MAX_ISSUE_LABELS)
        .map((label) => `<span class="badge">${escapeHtml(label)}</span>`)
        .join("")
    : "";

  const badgeHtml = badges
    .slice(0, 4)
    .map((badge) => {
      const className = badge.startsWith("issue-") ? `badge ${badge}` : "badge";
      return `<span class="${className}">${badge.replace("issue-", "")}</span>`;
    })
    .join("");

  const meta = node.targetDate ? `target ${node.targetDate}` : "";
  const issueTitle = issueInfo
    ? issueInfo.title.length > MAX_ISSUE_TITLE
      ? `${issueInfo.title.slice(0, MAX_ISSUE_TITLE - 1)}…`
      : issueInfo.title
    : null;
  const issueLine = issueNum ? `#${issueNum}` : "";
  const avatar = issueInfo && issueInfo.assigneeAvatar
    ? `<img class="avatar" src="${issueInfo.assigneeAvatar}" alt="${escapeHtml(issueInfo.assigneeLogin || "assignee")}" />`
    : "";

  return `
    <div class="node-label" style="color:#e5e7eb;font-family:'IBM Plex Sans','Segoe UI',Tahoma,sans-serif;">
      <div class="node-title">${escapeHtml(node.title)}</div>
      ${issueTitle ? `<div class="issue-title">${escapeHtml(issueTitle)}</div>` : ""}
      <div class="node-meta">${escapeHtml(node.id)}${issueLine ? ` - ${issueLine}` : ""}${meta ? ` - ${meta}` : ""}</div>
      <div class="issue-row">${avatar}<div class="badges">${badgeHtml}${issueBadges}</div></div>
    </div>
  `;
}

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function toList(items) {
  if (!items || items.length === 0) return "-";
  return items.map(escapeHtml).join(", ");
}

function renderDetails(node, dependents) {
  const issueUrl = makeIssueUrl(node.issue);
  const issueLine = issueUrl
    ? `<a href="${issueUrl}" target="_blank" rel="noopener">${escapeHtml(String(node.issue))}</a>`
    : escapeHtml(node.issue || "-");

  const body = `
    <div><strong>Title:</strong> ${escapeHtml(node.title)}</div>
    <div><strong>ID:</strong> ${escapeHtml(node.id)}</div>
    <div><strong>Type:</strong> ${escapeHtml(node.type || "-")}</div>
    <div><strong>Status:</strong> ${escapeHtml(node.status || "-")}</div>
    <div><strong>Target date:</strong> ${escapeHtml(node.targetDate || "-")}</div>
    <div><strong>Owners:</strong> ${toList(node.owners)}</div>
    <div><strong>Owner names:</strong> ${toList(node.ownerNames)}</div>
    <div><strong>Labels:</strong> ${toList(node.labels)}</div>
    <div><strong>Issue:</strong> ${issueLine}</div>
    <div><strong>Dependencies:</strong> ${toList(node.dependencies)}</div>
    <div><strong>Dependents:</strong> ${toList(dependents)}</div>
    ${node.description ? `<div class="section"><strong>Description</strong><div>${escapeHtml(node.description)}</div></div>` : ""}
    ${node.definitionOfDone ? `<div class="section"><strong>Definition of done</strong><div>${escapeHtml(node.definitionOfDone)}</div></div>` : ""}
  `;

  detailsBody.innerHTML = body;
}

function buildDependentsMap(edges) {
  const map = new Map();
  edges.forEach(({ from, to }) => {
    if (!map.has(from)) map.set(from, []);
    map.get(from).push(to);
  });
  return map;
}

function applyFilters(nodes) {
  const query = searchInput.value.trim().toLowerCase();
  const requireTimeline = timelineOnly.checked;
  const hideBack = hideBackburner.checked;

  return new Map(
    Array.from(nodes.entries()).filter(([_, node]) => {
      if (requireTimeline && !node.labels.includes("timeline")) return false;
      if (hideBack && node.labels.includes("backburner")) return false;
      if (query) {
        const haystack = `${node.id} ${node.title}`.toLowerCase();
        if (!haystack.includes(query)) return false;
      }
      return true;
    })
  );
}

function fitGraph(g) {
  const svgNode = dagSvg.node();
  if (!svgNode) return;
  const rect = svgNode.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const graphWidth = g.graph().width || 1;
  const graphHeight = g.graph().height || 1;

  const scale = Math.min(rect.width / (graphWidth + 40), rect.height / (graphHeight + 40), 1);
  const minScale = 0.35;
  const nextScale = Math.max(scale, minScale);
  const translateX = (rect.width - graphWidth * nextScale) / 2;
  const translateY = (rect.height - graphHeight * nextScale) / 2;

  dagSvg.call(
    zoom.transform,
    d3.zoomIdentity.translate(translateX, translateY).scale(nextScale)
  );
}

function renderGraph(options = {}) {
  if (!state.rawData) return;
  const { refit = false } = options;

  const { nodes, edges } = state;
  const visibleNodes = applyFilters(nodes);
  ensureIssues(visibleNodes);

  const currentTransform = d3.zoomTransform(dagSvg.node());

  const g = new dagreD3.graphlib.Graph({ multigraph: true }).setGraph({
    rankdir: rankDir.value,
    nodesep: 20,
    ranksep: 40,
    marginx: 20,
    marginy: 20,
  });

  g.setDefaultEdgeLabel(() => ({}));

  visibleNodes.forEach((node) => {
    g.setNode(node.id, {
      labelType: "html",
      label: renderNodeLabel(node),
      rx: 6,
      ry: 6,
      class: nodeClass(node),
    });
  });

  edges.forEach(({ from, to }) => {
    if (!visibleNodes.has(from) || !visibleNodes.has(to)) return;
    g.setEdge(from, to, { arrowhead: "vee" });
  });

  inner.selectAll("*").remove();
  renderer(inner, g);

  const nodesSelection = inner.selectAll("g.node");
  const dependentsMap = buildDependentsMap(edges);

  nodesSelection.on("click", (event, nodeId) => {
    const node = nodes.get(nodeId);
    if (!node) return;
    const dependents = dependentsMap.get(nodeId) || [];
    renderDetails(node, dependents);
  });

  if (!state.hasFit || refit) {
    fitGraph(g);
    state.hasFit = true;
  } else {
    dagSvg.call(zoom.transform, currentTransform);
  }
}

function setData(parsed) {
  state.rawData = parsed;
  const { nodes, edges } = buildGraphData(parsed);
  state.nodes = nodes;
  state.edges = edges;
  state.hasFit = false;
  renderGraph({ refit: true });
}

async function loadFromPath(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  const text = await res.text();
  return jsyaml.load(text);
}

async function loadDefaultPlan() {
  for (const path of DEFAULT_PATHS) {
    try {
      const parsed = await loadFromPath(path);
      setData(parsed);
      return;
    } catch (err) {
      // continue
    }
  }
  alert("Unable to load .agents/the_plan.yaml. Use the file picker.");
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = jsyaml.load(reader.result);
      setData(parsed);
    } catch (err) {
      alert("Failed to parse YAML file.");
    }
  };
  reader.readAsText(file);
});

loadDefault.addEventListener("click", () => {
  loadDefaultPlan();
});

[searchInput, timelineOnly, hideBackburner, rankDir].forEach((el) => {
  el.addEventListener("input", () => renderGraph({ refit: true }));
  el.addEventListener("change", () => renderGraph({ refit: true }));
});

loadIssueCache();
loadDefaultPlan();
