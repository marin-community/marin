const FAVORITES_KEY = "marin.review-queue.favorites.v1";
const GITHUB_HANDLE = /^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,38})$/;
const $ = (id) => document.getElementById(id);
const numberFormat = new Intl.NumberFormat();

function link(url, label, className = "") {
  const element = document.createElement("a");
  element.href = url;
  element.target = "_blank";
  element.rel = "noopener noreferrer";
  element.className = className;
  element.textContent = label;
  return element;
}

function ageDays(pr) {
  return Math.max(0, Math.floor((Date.now() - Date.parse(pr.created_at)) / 86400000));
}

function prRow(pr) {
  const row = document.createElement("div");
  row.className = "pr-row";
  row.append(link(pr.html_url, `#${pr.number} ${pr.title}`, "pr-title"));
  const meta = document.createElement("span");
  meta.className = "pr-meta";
  const age = ageDays(pr);
  const size = pr.lines == null ? "" : ` · ${numberFormat.format(pr.lines)} lines`;
  meta.textContent = `${pr.draft ? "Draft · " : ""}${age === 0 ? "opened today" : `opened ${age}d ago`}${size}`;
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
  return [...groups];
}

function emptyMessage(list, message) {
  const empty = document.createElement("p");
  empty.className = "empty";
  empty.textContent = message;
  list.append(empty);
}

function loadFavorites() {
  const saved = localStorage.getItem(FAVORITES_KEY);
  if (!saved) return new Set();
  let names;
  try {
    names = JSON.parse(saved);
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
    return new Set();
  }
  if (!Array.isArray(names)) return new Set();
  return new Set(names.filter((name) => typeof name === "string" && GITHUB_HANDLE.test(name)).map((name) => name.toLowerCase()));
}

function metric(label, value, className = "") {
  const cell = document.createElement("span");
  cell.className = `metric ${className}`;
  const name = document.createElement("span");
  name.className = "metric-label";
  name.textContent = label;
  cell.append(name, value);
  return cell;
}

function reviewerEntry(name, requests, stats, favorites, expanded, toggleFavorite) {
  const key = name.toLowerCase();
  const entry = document.createElement("div");
  entry.className = "reviewer-entry";
  const line = document.createElement("div");
  line.className = "reviewer-line";

  const identity = document.createElement("div");
  identity.className = "reviewer-identity";
  const star = document.createElement("button");
  star.type = "button";
  star.className = `star${favorites.has(key) ? " is-favorite" : ""}`;
  star.textContent = favorites.has(key) ? "★" : "☆";
  star.setAttribute("aria-pressed", String(favorites.has(key)));
  star.setAttribute("aria-label", `${favorites.has(key) ? "Remove" : "Add"} ${name} ${favorites.has(key) ? "from" : "to"} favorites`);
  star.addEventListener("click", () => toggleFavorite(name));
  identity.append(star, link(`https://github.com/${encodeURIComponent(name)}`, name, "reviewer-name"));

  const sizes = requests.flatMap((pr) => pr.lines == null ? [] : [pr.lines]);
  const subtotal = numberFormat.format(sizes.reduce((sum, lines) => sum + lines, 0));
  const lines = requests.length && sizes.length === 0 ? "—" : subtotal + (sizes.length < requests.length ? "+" : "");
  const response = stats?.avg_response_hours;
  const duration = response == null ? "—" : response < 1 ? `${Math.round(response * 60)}m` : `${Math.round(response)}h`;
  const average = metric("Avg response", duration);
  if (stats?.response_count) average.title = `Based on ${stats.response_count} matched reviews`;
  const weekly = stats ? numberFormat.format(Math.round(stats.lines / 2)) : "—";

  const action = document.createElement("button");
  action.type = "button";
  action.className = "expand";
  if (requests.length) {
    action.textContent = expanded.has(key) ? "Hide PRs ↑" : "View PRs ↓";
    action.setAttribute("aria-expanded", String(expanded.has(key)));
    action.addEventListener("click", () => {
      if (expanded.has(key)) expanded.delete(key);
      else expanded.add(key);
      const isExpanded = expanded.has(key);
      action.textContent = isExpanded ? "Hide PRs ↑" : "View PRs ↓";
      action.setAttribute("aria-expanded", String(isExpanded));
      requestsList.hidden = !isExpanded;
    });
  } else {
    action.textContent = "No requests";
    action.disabled = true;
  }
  line.append(identity, metric("Pending PRs", String(requests.length), "pending"), metric("Pending lines", lines), average, metric("Lines / week", weekly), action);
  entry.append(line);

  const requestsList = document.createElement("div");
  requestsList.className = "request-list";
  requestsList.hidden = !expanded.has(key);
  for (const pr of requests) requestsList.append(prRow(pr));
  entry.append(requestsList);
  return entry;
}

function renderReviewers(prs, stats, favorites, expanded, toggleFavorite) {
  const groups = groupBy(prs, "requested_reviewers", "login");
  const known = new Set(groups.map(([name]) => name.toLowerCase()));
  for (const name of favorites) if (!known.has(name)) groups.push([name, []]);
  groups.sort((a, b) => Number(favorites.has(b[0].toLowerCase())) - Number(favorites.has(a[0].toLowerCase())) || b[1].length - a[1].length || a[0].localeCompare(b[0]));

  const query = $("search").value.trim().toLowerCase();
  const list = $("reviewers");
  list.replaceChildren();
  const visible = groups.filter(([name]) => name.toLowerCase().includes(query));
  if (!visible.length) emptyMessage(list, query ? "No matching reviewer in the queue." : "No review requests are pending.");
  for (const [name, requests] of visible) {
    const handle = name.toLowerCase();
    const history = Object.hasOwn(stats, handle) ? stats[handle] : undefined;
    list.append(reviewerEntry(name, requests, history, favorites, expanded, toggleFavorite));
  }

  const suggestion = $("favorite-suggestion");
  suggestion.replaceChildren();
  suggestion.hidden = !query || !GITHUB_HANDLE.test(query) || groups.some(([name]) => name.toLowerCase() === query);
  if (!suggestion.hidden) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `★ Star @${query} to follow their queue`;
    button.addEventListener("click", () => toggleFavorite(query));
    suggestion.append(button);
  }
}

function renderOtherQueues(prs) {
  const teams = groupBy(prs, "requested_teams", "slug");
  $("team-section").hidden = teams.length === 0;
  $("team-count").textContent = String(teams.length);
  const teamList = $("teams");
  teamList.replaceChildren();
  for (const [name, requests] of teams) {
    const group = document.createElement("div");
    group.className = "team-group";
    const title = document.createElement("strong");
    title.textContent = name;
    group.append(title);
    for (const pr of requests) group.append(prRow(pr));
    teamList.append(group);
  }

  const withoutRequest = prs.filter((pr) => !(pr.requested_reviewers || []).length && !(pr.requested_teams || []).length);
  $("unassigned-count").textContent = String(withoutRequest.length);
  const unassigned = $("unassigned");
  unassigned.replaceChildren();
  if (!withoutRequest.length) emptyMessage(unassigned, "None right now.");
  for (const pr of withoutRequest) unassigned.append(prRow(pr));
}

async function api(path, options) {
  const response = await fetch(`./api/${path}`, { cache: "no-store", ...options });
  if (!response.ok) throw new Error(`Marina returned HTTP ${response.status}.`);
  return response.json();
}

function startReviewQueue() {
  let current = { prs: [], stats: {}, refreshed_at: null };
  let timer;
  const favorites = loadFavorites();
  const expanded = new Set();

  function render() {
    renderReviewers(current.prs, current.stats, favorites, expanded, toggleFavorite);
  }

  function toggleFavorite(name) {
    const key = name.toLowerCase();
    if (favorites.has(key)) favorites.delete(key);
    else favorites.add(key);
    localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites]));
    render();
  }

  async function loadQueue() {
    current = await api("queue");
    if (current.refreshed_at) {
      render();
      renderOtherQueues(current.prs);
      $("updated").textContent = `GitHub snapshot · ${new Date(current.refreshed_at).toLocaleString()}`;
    } else {
      $("updated").textContent = "Getting the first GitHub snapshot…";
    }
  }

  async function refresh() {
    clearTimeout(timer);
    $("refresh").disabled = true;
    try {
      await loadQueue();
      const { retry_in } = await api("sync", { method: "POST", headers: { "X-Review-Queue": "sync" } });
      await loadQueue();
      timer = setTimeout(refresh, retry_in * 1000);
    } catch (error) {
      $("updated").textContent = `${error.message}${current.refreshed_at ? " Showing the last loaded queue." : ""}`;
      if (!current.refreshed_at) {
        $("reviewers").replaceChildren();
        emptyMessage($("reviewers"), "Queue unavailable.");
      }
      return;
    } finally {
      $("refresh").disabled = false;
    }
  }

  $("refresh").addEventListener("click", refresh);
  $("search").addEventListener("input", render);
  refresh();
}

startReviewQueue();
