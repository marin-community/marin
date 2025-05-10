export function apiConfigUrl() {
  return "/api/config";
}

export function viewUrl(params) {
  // Encode arrays (e.g., `paths`) as JSON
  params = {...params, paths: JSON.stringify(params.paths)};
  return "/view?" + new URLSearchParams(params);
}

export function apiViewUrl(params) {
  return "/api/view?" + new URLSearchParams(params);
}

export function experimentUrl(params) {
  return "/experiment?" + new URLSearchParams(params);
}

export function viewSingleUrl(path) {
  return viewUrl({paths: [path]});
}

export function renderLink(item, updateUrlParams) {
  return (<div className="clickable" onClick={() => updateUrlParams({paths: JSON.stringify([item])})}>
    {renderText(item)}
  </div>);
}

export function renderError(error) {
  return (<div className="error">{error}</div>);
}

export function renderText(str) {
  return str.split("\n").map((line, i) => <div key={i}>{line}</div>);
}

export function renderDate(date) {
  return <span className="date">{new Date(date).toLocaleString()}</span>;
}

export function renderDuration(seconds) {
  const hours = Math.floor(seconds / 60 / 60);
  seconds -= hours * 60 * 60;
  const minutes = Math.floor(seconds / 60);
  seconds -= minutes * 60;
  return (hours > 0 ? hours + "h" : "") +
         (minutes > 0 ? minutes + "m" : "") +
         Math.round(seconds) + "s";
}

export function isUrl(str) {
  return str.startsWith("http://") || str.startsWith("https://");
}

/**
 * Navigates to a new URL with updated URL parameters (`urlParams + delta`).
 */
export function navigateToUrl(urlParams, delta, location, navigate) {
  for (const key in delta) {
    if (delta[key] === null || delta[key] === false) {
      urlParams.delete(key);
    } else {
      urlParams.set(key, delta[key]);
    }
  }
  navigate({
    pathname: location.pathname,
    search: urlParams.toString(),
  });
}
