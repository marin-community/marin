/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
const $ = id => document.getElementById(id);
function node(tag, value, className) {
  const item = document.createElement(tag);
  if (value !== undefined) item.textContent = value;
  if (className) item.className = className;
  return item;
}
async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}
