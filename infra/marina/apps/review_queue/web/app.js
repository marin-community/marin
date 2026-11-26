const API = "https://api.github.com/repos/marin-community/marin/pulls";
const PAGE_SIZE = 100;
const $ = (id) => document.getElementById(id);

function emptyMessage(list, message) {
  const empty = document.createElement("p");
  empty.className = "empty";
  empty.textContent = message;
  list.append(empty);
}

function link(url, label, className = "") {
  const element = document.createElement("a");
  element.href = url;
  element.target = "_blank";
  element.rel = "noopener noreferrer";
  element.className = className;
  element.textContent = label;
  return element;
}

function prRow(pr) {
  const row = document.createElement("div");
  row.className = "pr-row";
  row.append(link(pr.html_url, `#${pr.number} ${pr.title}`, "pr-title"));
  const meta = document.createElement("span");
  meta.className = "pr-meta";
  const age = Math.max(0, Math.floor((Date.now() - Date.parse(pr.created_at)) / 86400000));
  meta.textContent = `${pr.draft ? "DRAFT · " : ""}${age === 0 ? "opened today" : `opened ${age}d ago`}`;
  row.append(meta);
  return row;
}

function groupBy(prs, key, name) {
  const groups = new Map();
  for (const pr of prs) {
    for (const reviewer of pr[key] || []) {
      const id = reviewer[name];
      if (!id) continue;
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(pr);
    }
  }
  return [...groups].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
}

function reviewerCard(name, requests, max) {
  const card = document.createElement("article");
  card.className = "reviewer-card";
  const heading = document.createElement("div");
  heading.className = "reviewer-heading";
  const identity = document.createElement("div");
  identity.className = "identity";
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = name.slice(0, 2).toUpperCase();
  identity.append(avatar, link(`https://github.com/${encodeURIComponent(name)}`, name, "reviewer-name"));
  const count = document.createElement("div");
  count.className = "count";
  count.textContent = String(requests.length);
  const countLabel = document.createElement("span");
  countLabel.textContent = requests.length === 1 ? " pending PR" : " pending PRs";
  count.append(countLabel);
  heading.append(identity, count);
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", `${requests.length} pending requests`);
  const fill = document.createElement("span");
  fill.style.width = `${(requests.length / max) * 100}%`;
  bar.append(fill);
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "View pull requests";
  details.append(summary);
  for (const pr of requests) details.append(prRow(pr));
  card.append(heading, bar, details);
  return card;
}

function renderReviewers(prs) {
  const groups = groupBy(prs, "requested_reviewers", "login");
  const query = $("search").value.trim().toLowerCase();
  const visible = groups.filter(([name]) => name.toLowerCase().includes(query));
  const list = $("reviewers");
  list.replaceChildren();
  if (!visible.length) {
    emptyMessage(list, query ? "No reviewer matches that search." : "No individual review requests are pending.");
    return;
  }
  const max = groups[0][1].length;
  for (const [name, requests] of visible) list.append(reviewerCard(name, requests, max));
}

function renderCompact(target, groups) {
  const list = $(target);
  list.replaceChildren();
  if (!groups.length) {
    emptyMessage(list, "None right now.");
    return;
  }
  for (const [name, prs] of groups) {
    const item = document.createElement("details");
    item.className = "compact-item";
    const summary = document.createElement("summary");
    const label = document.createElement("span");
    label.textContent = name;
    const count = document.createElement("strong");
    count.textContent = String(prs.length);
    summary.append(label, count);
    item.append(summary);
    for (const pr of prs) item.append(prRow(pr));
    list.append(item);
  }
}

function renderUnassigned(prs) {
  const list = $("unassigned");
  list.replaceChildren();
  if (!prs.length) {
    emptyMessage(list, "None right now.");
    return;
  }
  for (const pr of prs) list.append(prRow(pr));
}

function render(prs) {
  const people = groupBy(prs, "requested_reviewers", "login");
  const teams = groupBy(prs, "requested_teams", "slug");
  const noReviewRequest = prs.filter((pr) => !(pr.requested_reviewers || []).length && !(pr.requested_teams || []).length);
  $("open-count").textContent = String(prs.length);
  $("request-count").textContent = String(people.reduce((sum, [, items]) => sum + items.length, 0));
  $("reviewer-count").textContent = String(people.length);
  $("team-count").textContent = String(teams.length);
  $("unassigned-count").textContent = String(noReviewRequest.length);
  renderReviewers(prs);
  renderCompact("teams", teams);
  renderUnassigned(noReviewRequest);
}

async function fetchOpenPrs() {
  const prs = [];
  for (let page = 1; ; page += 1) {
    const response = await fetch(`${API}?state=open&per_page=${PAGE_SIZE}&page=${page}`, {
      headers: { Accept: "application/vnd.github+json" },
    });
    if (!response.ok) {
      if (response.status === 403 || response.status === 429) throw new Error("GitHub rate limit reached. Try refreshing later.");
      throw new Error(`GitHub returned HTTP ${response.status}.`);
    }
    const batch = await response.json();
    if (!Array.isArray(batch)) throw new Error("GitHub returned an unexpected response.");
    prs.push(...batch);
    if (batch.length < PAGE_SIZE) return prs;
  }
}

function startReviewQueue() {
  let current = [];

  async function refresh() {
    $("refresh").disabled = true;
    $("updated").textContent = "Loading open pull requests…";
    let prs;
    try {
      prs = await fetchOpenPrs();
    } catch (error) {
      $("updated").textContent = `${error.message}${current.length ? " Showing the last loaded queue." : ""}`;
      if (!current.length) {
        for (const id of ["reviewers", "teams", "unassigned"]) $(id).textContent = "Queue unavailable.";
      }
      return;
    } finally {
      $("refresh").disabled = false;
    }
    render(prs);
    current = prs;
    $("updated").textContent = `Updated ${new Date().toLocaleString()}`;
  }

  $("refresh").addEventListener("click", refresh);
  $("search").addEventListener("input", () => renderReviewers(current));
  refresh();
}

startReviewQueue();
