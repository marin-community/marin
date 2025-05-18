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
  const options = {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  };
  return new Date(date).toLocaleString('sv-SE', options);
}

export function round(value, precision) {
  return Math.round(value * Math.pow(10, precision)) / Math.pow(10, precision);
}

export function renderDuration(seconds) {
  const secondsPerDay = 24 * 60 * 60;
  if (seconds > secondsPerDay) {
    return round(seconds / secondsPerDay, 1) + "d";
  }
  const secondsPerHour = 60 * 60;
  if (seconds > secondsPerHour) {
    return round(seconds / secondsPerHour, 1) + "h";
  }
  const secondsPerMinute = 60;
  if (seconds > secondsPerMinute) {
    return round(seconds / secondsPerMinute, 1) + "m";
  }
  return seconds + "s";
}

export function joinSpans(spans, separator) {
  return spans.map((span, i) => <span key={i}>{span}{i < spans.length - 1 ? separator : ""}</span>);
}

export function renderSciNotation(value) {
  if (value === 0) {
    return value
  }
  // Render using scientific notation
  return value.toExponential(1);
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
