/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const state = {reviews: []};
function render() {
  const search = $("search").value.toLowerCase(), fragment = document.createDocumentFragment();
  for (const review of state.reviews.filter(item => `${item.source_id} ${item.id}`.toLowerCase().includes(search))) {
    const paragraph = node("p"), anchor = node("a", `${review.source_id} · ${new Date(review.updated_at).toLocaleDateString()} · ${review.review_count ?? "—"} reviews`);
    anchor.href = `archive.html?id=${encodeURIComponent(review.id)}`; paragraph.append(anchor); fragment.append(paragraph);
  }
  $("reviews").replaceChildren(fragment);
}
async function selected(id) {
  const review = await json(`api/archive/reviews/${encodeURIComponent(id)}`), target = $("selected");
  $("reviews").hidden = true; $("search").hidden = true;
  target.append(node("h2", review.source_id), node("p", `Review ID: ${review.id}`));
  const collection = node("details"); collection.append(node("summary", "Original review collection"), node("pre", JSON.stringify(review.collection, null, 2))); target.append(collection);
  target.append(node("h2", "Saved artifacts"));
  for (const artifact of review.artifacts) {const paragraph = node("p"), anchor = node("a", artifact.path); anchor.href = `api/archive/reviews/${encodeURIComponent(id)}/artifacts/${artifact.path.split("/").map(encodeURIComponent).join("/")}`; anchor.target = "_blank"; anchor.rel = "noopener"; paragraph.append(anchor); target.append(paragraph);}
}
$("search").addEventListener("input", render);
(async () => {try {const id = new URLSearchParams(location.search).get("id"); if (id) await selected(id); else {const data = await json("api/archive"); state.reviews = data.reviews; render();} $("message").textContent = "Archived material is preserved as recorded.";} catch (error) {$("message").textContent = error.message;}})();
