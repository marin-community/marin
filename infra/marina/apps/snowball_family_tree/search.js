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

$("#search").addEventListener("input", () => {
  resultsDismissed = false;
  applyFilter();
});
$("#org").addEventListener("input", applyFilter);
