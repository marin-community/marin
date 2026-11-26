const FAVORITES_KEY = "marin.review-queue.favorites.v1";
const PR_URL = "https://github.com/marin-community/marin/pull/";
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
  const subject = document.createElement("span");
  subject.className = "pr-subject";
  subject.append(link(pr.html_url, `#${pr.number} ${pr.title}`, "pr-title"));
  if (pr.reviewStatus) {
    const status = document.createElement("span");
    status.className = `status-pill ${pr.reviewStatus}`;
    status.textContent = pr.reviewStatus === "pending" ? "Pending" : "Reviewed";
    subject.append(status);
  }
  row.append(subject);
  const meta = document.createElement("span");
  meta.className = "pr-meta";
  const age = ageDays(pr);
  const size = pr.lines == null ? "" : ` · ${numberFormat.format(pr.lines)} lines`;
  meta.textContent = `${pr.draft ? "Draft · " : ""}${age === 0 ? "opened today" : `opened ${age}d ago`}${size}`;
  row.append(meta);
  return row;
}

function reviewedRow(pr) {
  const row = document.createElement("div");
  row.className = "pr-row";
  const label = pr.title ? `#${pr.number} ${pr.title}` : `#${pr.number}`;
  const meta = document.createElement("span");
  meta.className = "pr-meta";
  const date = new Date(pr.submitted_at).toLocaleDateString([], { month: "short", day: "numeric" });
  meta.textContent = `Reviewed ${date} · ${numberFormat.format(pr.lines)} lines`;
  row.append(link(PR_URL + pr.number, label, "pr-title"), meta);
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
    console.warn("Saved review queue favorites could not be read.");
    return new Set();
  }
  if (!Array.isArray(names)) return new Set();
  return new Set(names.filter((name) => typeof name === "string" && GITHUB_HANDLE.test(name)).map((name) => name.toLowerCase()));
}

function metric(label, value, className = "", ratio = null) {
  const cell = document.createElement("span");
  cell.className = `metric ${className}`;
  const name = document.createElement("span");
  name.className = "metric-label";
  name.textContent = label;
  cell.append(name, value);
  if (ratio !== null) {
    const bar = document.createElement("span");
    bar.className = "load-bar";
    bar.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.style.width = `${ratio * 100}%`;
    bar.append(fill);
    cell.append(bar);
  }
  return cell;
}

function favoriteButton(name, starred, toggleFavorite) {
  const star = document.createElement("button");
  star.type = "button";
  star.className = `star${starred ? " is-favorite" : ""}`;
  star.dataset.focus = `star-${name.toLowerCase()}`;
  star.textContent = starred ? "★" : "☆";
  star.setAttribute("aria-pressed", String(starred));
  star.setAttribute("aria-label", starred ? `Remove ${name} from favorites` : `Add ${name} to favorites`);
  star.addEventListener("click", () => toggleFavorite(name));
  return star;
}

function pendingVolume(requests) {
  const sizes = requests.filter((pr) => pr.lines != null);
  const lines = requests.length && !sizes.length ? null : sizes.reduce((sum, pr) => sum + pr.lines, 0);
  return { lines, partial: sizes.length < requests.length };
}

function reviewerMetrics(pendingCount, stats, volume, scale) {
  const lines = volume.lines === null ? "—" : `${numberFormat.format(volume.lines)}${volume.partial ? "+" : ""}`;
  const response = stats?.avg_response_hours;
  const duration = response == null ? "—" : response < 1 ? `${Math.round(response * 60)}m` : `${Math.round(response)}h`;
  const average = metric("Avg response", duration);
  if (stats?.response_count) average.title = `Based on ${stats.response_count} matched reviews`;
  const weekly = stats ? numberFormat.format(stats.lines_per_week) : "—";
  return [
    metric("Pending PRs", String(pendingCount), "pending-count"),
    metric("Pending lines", lines, "pending-lines", scale && volume.lines !== null ? volume.lines / scale.pending : null),
    average,
    metric("Lines / week", weekly, "weekly-lines", scale && stats ? stats.lines_per_week / scale.weekly : null),
  ];
}

function prDisclosure(line, handle, openPulls, history, expanded) {
  const action = document.createElement("span");
  action.className = "expand";
  action.setAttribute("aria-hidden", "true");
  line.append(action);
  if (!openPulls.length && !history.length) {
    action.textContent = "No PRs";
    line.disabled = true;
    return null;
  }

  const requestsList = document.createElement("div");
  requestsList.className = "request-list";
  requestsList.id = `prs-${handle}`;
  line.setAttribute("aria-controls", requestsList.id);
  if (openPulls.length && history.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Open PRs";
    requestsList.append(heading);
  }
  for (const pr of openPulls.toSorted((a, b) => (
    Number(b.reviewStatus === "pending") - Number(a.reviewStatus === "pending") || a.created_at.localeCompare(b.created_at)
  ))) requestsList.append(prRow(pr));
  if (history.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Reviewed in the last 14 days";
    requestsList.append(heading, ...history.map(reviewedRow));
  }
  function renderDisclosure() {
    const open = expanded.has(handle);
    action.textContent = open ? "Hide PRs ↑" : "View PRs ↓";
    line.setAttribute("aria-expanded", String(open));
    requestsList.hidden = !open;
  }
  line.addEventListener("click", () => {
    if (expanded.has(handle)) expanded.delete(handle);
    else expanded.add(handle);
    renderDisclosure();
  });
  renderDisclosure();
  return requestsList;
}

function reviewerEntry({ name, openPulls, pendingCount, stats, volume }, scale, favorites, expanded, toggleFavorite) {
  const handle = name.toLowerCase();
  const entry = document.createElement("div");
  entry.className = "reviewer-entry";
  const line = document.createElement("button");
  line.type = "button";
  line.dataset.focus = `row-${handle}`;
  line.className = "reviewer-line";
  const identity = document.createElement("span");
  identity.className = "reviewer-name";
  identity.textContent = name;
  line.append(identity, ...reviewerMetrics(pendingCount, stats, volume, scale));
  const history = scale ? (stats?.pulls ?? []).filter((pr) => !openPulls.some((open) => open.number === pr.number)) : [];
  const requestsList = prDisclosure(line, handle, openPulls, history, expanded);
  entry.append(line, favoriteButton(name, favorites.has(handle), toggleFavorite));
  if (requestsList) entry.append(requestsList);
  return entry;
}

function reviewerGroups(prs, stats, favorites, view) {
  const groups = new Map();
  const extra = view === "load" ? [...favorites, ...Object.keys(stats)] : favorites;
  for (const name of extra) groups.set(name, { name, openPulls: [] });
  for (const pr of prs) {
    const reviewers = new Map();
    for (const { login } of pr.reviewed_reviewers ?? []) reviewers.set(login.toLowerCase(), { login, status: "reviewed" });
    for (const { login } of pr.requested_reviewers) reviewers.set(login.toLowerCase(), { login, status: "pending" });
    for (const [handle, { login, status }] of reviewers) {
      if (!groups.has(handle)) groups.set(handle, { name: login, openPulls: [] });
      groups.get(handle).openPulls.push({ ...pr, reviewStatus: status });
    }
  }
  return [...groups].map(([handle, group]) => {
    const pending = group.openPulls.filter((pr) => pr.reviewStatus === "pending");
    return {
      ...group,
      pendingCount: pending.length,
      volume: pendingVolume(pending),
      stats: Object.hasOwn(stats, handle) ? stats[handle] : undefined,
    };
  });
}

function sortedReviewers(groups, favorites, order) {
  return groups.sort((a, b) => {
    const pinned = Number(favorites.has(b.name.toLowerCase())) - Number(favorites.has(a.name.toLowerCase()));
    if (pinned) return pinned;
    const byName = a.name.localeCompare(b.name, undefined, { sensitivity: "base" });
    if (order.column === "name") return byName * order.direction;
    const [left, right] = [a, b].map((reviewer) => {
      if (order.column === "pending") return reviewer.pendingCount;
      if (order.column === "lines") return reviewer.volume.lines;
      if (order.column === "response") return reviewer.stats?.avg_response_hours;
      return reviewer.stats?.lines_per_week;
    });
    if (left == null) return right == null ? byName : 1;
    if (right == null) return -1;
    return (left - right) * order.direction || byName;
  });
}

function renderReviewers(prs, stats, favorites, expanded, toggleFavorite, view, order) {
  const groups = sortedReviewers(reviewerGroups(prs, stats, favorites, view), favorites, order);
  const scale = view === "load" ? {
    pending: Math.max(1, ...groups.map((item) => item.volume.lines ?? 0)),
    weekly: Math.max(1, ...groups.map((item) => item.stats?.lines_per_week ?? 0)),
  } : null;
  const query = $("search").value.trim().toLowerCase();
  const list = $("reviewers");
  const focused = document.activeElement?.dataset.focus;
  list.replaceChildren();
  const visible = groups.filter(({ name }) => name.toLowerCase().includes(query));
  if (!visible.length) emptyMessage(list, query ? "No matching reviewer in the queue." : "No open reviews.");
  for (const item of visible) list.append(reviewerEntry(item, scale, favorites, expanded, toggleFavorite));
  if (focused) list.querySelector(`[data-focus="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });

  const suggestion = $("favorite-suggestion");
  suggestion.replaceChildren();
  suggestion.hidden = !query || !GITHUB_HANDLE.test(query) || groups.some(({name}) => name.toLowerCase() === query);
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

  const withoutRequest = prs.filter((pr) => !pr.requested_reviewers.length && !pr.requested_teams.length && !pr.reviewed_reviewers?.length);
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
  let view = "queue";
  let order = { column: "name", direction: 1 };
  let timer;
  const favorites = loadFavorites();
  const expanded = new Set();

  function render() {
    renderReviewers(current.prs, current.stats, favorites, expanded, toggleFavorite, view, order);
  }

  function changeSort(column, direction) {
    order = { column, direction };
    $("sort").value = column;
    $("sort-direction").textContent = direction === 1 ? "↑" : "↓";
    $("sort-direction").setAttribute("aria-label", `Reverse sort order, currently ${direction === 1 ? "ascending" : "descending"}`);
    for (const button of document.querySelectorAll("[data-sort]")) {
      const active = button.dataset.sort === column;
      button.setAttribute("aria-pressed", String(active));
      button.dataset.direction = active ? (direction === 1 ? "ascending" : "descending") : "";
      button.setAttribute("aria-label", active ? `${button.textContent}, ${button.dataset.direction}` : button.textContent);
    }
    render();
  }

  function changeView(next) {
    if (view === next) return;
    view = next;
    $("queue-view").setAttribute("aria-pressed", String(view === "queue"));
    $("load-view").setAttribute("aria-pressed", String(view === "load"));
    $("load-note").hidden = view !== "load";
    changeSort(view === "load" ? "load" : "name", view === "load" ? -1 : 1);
  }

  function toggleFavorite(name) {
    const key = name.toLowerCase();
    if (favorites.has(key)) favorites.delete(key);
    else favorites.add(key);
    localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites]));
    render();
    if (document.activeElement === document.body) $("search").focus({ preventScroll: true });
  }

  async function loadQueue() {
    const next = await api("queue");
    const changed = JSON.stringify(next) !== JSON.stringify(current);
    current = next;
    if (current.refreshed_at) {
      if (changed) {
        render();
        renderOtherQueues(current.prs);
      }
      const { completed, total } = current.sync;
      $("sync-count").textContent = total ? `${completed} of ${total} PR details synced` : "Queue up to date";
      const progress = $("sync-progress");
      progress.max = total || 1;
      progress.value = completed;
      progress.hidden = !total;
      const snapshot = new Date(current.refreshed_at);
      const sameDay = snapshot.toDateString() === new Date().toDateString();
      const timestamp = sameDay ? snapshot.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : snapshot.toLocaleDateString([], { month: "short", day: "numeric" });
      $("updated").textContent = `Queue snapshot · ${timestamp}`;
      $("updated").title = snapshot.toLocaleString();
    } else {
      $("updated").textContent = "Getting the first GitHub snapshot…";
    }
  }

  async function refresh() {
    clearTimeout(timer);
    $("refresh").disabled = true;
    try {
      await loadQueue();
      if (current.refreshed_at && current.sync.completed < current.sync.total) {
        $("updated").textContent = "Checking for missing PR details…";
      }
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
  $("queue-view").addEventListener("click", () => changeView("queue"));
  $("load-view").addEventListener("click", () => changeView("load"));
  const defaultDirection = (column) => column === "name" || column === "response" ? 1 : -1;
  for (const button of document.querySelectorAll("[data-sort]")) {
    button.addEventListener("click", () => {
      const column = button.dataset.sort;
      changeSort(column, column === order.column ? -order.direction : defaultDirection(column));
    });
  }
  $("sort").addEventListener("change", (event) => changeSort(event.target.value, defaultDirection(event.target.value)));
  $("sort-direction").addEventListener("click", () => changeSort(order.column, -order.direction));
  refresh();
}

startReviewQueue();
