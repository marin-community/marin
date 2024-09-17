export function viewUrl(params) {
  // Encode arrays (e.g., `paths`) as JSON
  params = {...params, paths: JSON.stringify(params.paths)};
  return "/view?" + new URLSearchParams(params);
}

export function apiViewUrl(params) {
  return "/api/view?" + new URLSearchParams(params);
}

export function viewSingleUrl(path) {
  return viewUrl({paths: [path]});
}
