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
  return `<a href="${esc(issue || n.source)}" target="_blank" rel="noopener noreferrer">${sourceName(issue || n.source)} ↗</a>${n.model ? `<a href="${HF_URL}${esc(n.model)}" target="_blank" rel="noopener noreferrer">Weights ↗</a>` : ""}`;
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
function setupTimeline(autoplay = true) {
  stopTimeline();
  timelineFamily = $("#timeline-family").value;
  timelineRoot = byId.get(COOLDOWN_ROOTS[timelineFamily]);
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
  const columns = Math.ceil(timelineNodes.length / TIMELINE_ROWS);
  $("#timeline-events").style.setProperty("--timeline-rows", TIMELINE_ROWS);
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
    const group = timelineNodes.slice(
        i * TIMELINE_ROWS,
        (i + 1) * TIMELINE_ROWS,
      ),
      first = group[0],
      last = group[group.length - 1];
    return `<span>${first.timeline.date ? shortDate.format(nodeTime(first)) : "Undated"}${last.timeline.date !== first.timeline.date ? " – " + (last.timeline.date ? shortDate.format(nodeTime(last)) : "undated") : ""}</span>`;
  }).join("");
  $("#tree-caption").textContent =
    `${timelineFamily} timeline · ${timelineNodes.length + 1} nodes · temporal grouping under one cooldown; original lineage remains in Details`;
  detail(timelineRoot.id);
  sizeWorkspace();
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
    ((timelineTime - timelineStart) / (timelineEnd - timelineStart)) *
      SCRUBBER_STEPS,
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
  if (n.lane !== timelineFamily && n.id !== PRETRAIN) {
    $("#timeline-family").value = n.lane;
    setupTimeline(false);
  }
  if (n.id === PRETRAIN) {
    detail(id);
    timelineModels([n], "Shared pretraining root");
    return;
  }
  timelineTime = Math.max(timelineTime, nodeTime(n));
  paintTimeline();
  detail(id);
  sizeWorkspace();
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
    ((timelineEnd - timelineStart) * Number($("#timeline-scrub").value)) /
      SCRUBBER_STEPS;
  paintTimeline();
};
document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-time-node]");
  if (target) selectTimelineNode(target.dataset.timeNode);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopTimeline();
});
