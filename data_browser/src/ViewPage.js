import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { apiViewUrl } from './utils';

function ViewPage() {
  /**
   * ViewPage is a swiss-army knife for browsing any data assets generated in Marin.
   * It acts like a file explorer, JSON viewer, allowing the user to render/page
   * through data.
   *
   * The main GET URL parameter is:
   * - `paths`: list of paths (e.g., gs://marin-us-central2/...) to display
   * The paths can be directories or files, and files can be either JSON, text, zipped, etc.
   * 
   * Case 1: each `path` in `paths` represents a list of records (jsonl or parquet).
   * Then we display the records from `offset` to `offset + count` for each of the paths.
   * For example, if we have `paths = [path1, path2]` and `offset = 3`, `count = 5`,
   * then we render the items * below:
   * 
   *                    | path1 |  path2 |
   * offset          3  |   *   |    *   |
   *                 4  |   *   |    *   |
   *                 5  |   *   |    *   |
   *                 6  |   *   |    *   |
   *                 7  |   *   |    *   |
   * offset + count  8  |   *   |    *   |
   * 
   * Case 2: each `path` in `paths` is rendered separately, which is when:
   * - If `path` is a directory, then we provide a directory listing, allowing
   *   the user to navigate to subdirectories or files.
   * - If `path` is a JSON file, then we render the JSON content in a structured way.
   * - If `path` is a text file, then we render the text content.
   * 
   * Case 3: when `paths` contains a combination of the above.  In this case, we
   * simply render the subset of `paths` that corresonding to items as a block,
   * and then render everything else separately.  This flexibility provides the
   * user to start with two paths of case 1 and navigate only one of them (which
   * goes through a hybrid situation) until we get back to case 1 with another
   * file.
   * 
   * In the Marin processing pipeline, we store most data files as jsonl, and
   * various stages of processing preserve the same filenames/order, but just
   * with other generated (meta)data.  An example:

   * paths = [
   *   <path to jsonl file containing raw data>,
   *   <path to jsonl file containing transformed data>,
   *   <path to jsonl file containing attributes data produced by quality classifier>,
   *   ...
   * ]
   * 
   * The data browser effectively does an easy join of all these files.
   */

  // Get URL parameters
  const location = useLocation();
  const navigate = useNavigate();
  const urlParams = new URLSearchParams(location.search);
  const pathsResult = getPaths(urlParams.get("paths"));
  const paths = pathsResult.paths;
  const offset = parseInt(urlParams.get("offset")) || 0;
  const count = parseInt(urlParams.get("count")) || 1;

  // State
  const [error, setError] = useState(null);
  const [payloads, setPayloads] = useState({});  // payloads[path] = what server returns

  // Set the error in paths (don't do this directly to avoid infinite rendering loop)
  useEffect(() => {
    if (pathsResult.error) {
      setError(pathsResult.error);
      console.error(pathsResult.error);
    }
  }, [pathsResult.error]);

  // Fetch data from backend (payload for each path)
  useEffect(() => {
    const fetchData = async () => {
      const promises = paths.map(async (path) => {
        try {
          const response = await axios.get(apiViewUrl({path, offset, count}));
          setPayloads((prevPayloads) => ({...prevPayloads, [path]: response.data}));
        } catch (error) {
          console.error(error);
          setError(error.message);
        }
      });
      await Promise.all(promises);
    };
    fetchData();
    // TODO: For some reason, `paths` causes infinite rendering loop, so we have to use `location`
  }, [location, offset, count]);

  if (error) {
    return renderError(error);
  }

  // The main payload (first one must be ready)
  const mainPayload = paths && payloads[paths[0]];
  if (!mainPayload) {
    return "Loading...";
  }

  function updateUrlParams(delta) {
    navigateToUrl(urlParams, delta, location, navigate);
  }

  return (<div>
    {renderOffsetCount({offset, count, mainPayload, updateUrlParams})}
    {renderPaths({paths, updateUrlParams})}
    <hr />
    {renderPayloads({paths, payloads, offset, updateUrlParams})}
  </div>);
}

export default ViewPage;

////////////////////////////////////////////////////////////

/**
 * Parses the URL parameter `paths`.
 * 
 * @param {str} pathsStr - the paths URL parameter, encoded as a JSON string
 * @returns {paths: string[], error: string}
 */
function getPaths(pathsStr) {
  try {
    const result = JSON.parse(pathsStr);
    if (!Array.isArray(result)) {
      return {paths: [], error: "Invalid paths (should be JSON array): " + pathsStr};
    }
    return {paths: result};
  } catch (e) {
    return {paths: [], error: "Invalid paths (invalid JSON): " + pathsStr};
  }
}

function renderError(error) {
  return (<div className="error">{error}</div>);
}

/**
 * Navigates to a new URL with updated URL parameters (`urlParams + delta`).
 */
function navigateToUrl(urlParams, delta, location, navigate) {
  for (const key in delta) {
    urlParams.set(key, delta[key]);
  } 
  navigate({
    pathname: location.pathname,
    search: urlParams.toString(),
  });
}

/**
 * Renders the `offset` (which item we're starting at to render).
 * Allow it to get changed.
 */
function renderOffset(offset, updateUrlParams) {
  function updateOffset() {
    const newOffset = prompt("Enter new offset", offset);
    if (!newOffset) {
      return;
    }
    updateUrlParams({offset: parseInt(newOffset)});
  }
  return <span className="editable" onClick={updateOffset}>{offset}</span>;
}

/**
 * Renders the `count` (number of items we're displaying).
 * Allow it to get changed.
 */
function renderCount(actualCount, count, updateUrlParams) {
  function updateCount() {
    const newCount = prompt("Enter new count", count);
    if (!newCount) {
      return;
    }
    updateUrlParams({count: parseInt(newCount)});
  }

  const countSpan = (
    <span onClick={updateCount}>
      <span className="editable">{actualCount}</span> items
    </span>
  );
  return countSpan;
}

/**
 * Render the offset and count, and allow the user to navigate to the previous/next items.
 * Example:
 * 
 *   Showing 5 : 10 [5 items] | Prev Next
 */
function renderOffsetCount(args) {
  const {offset, count, mainPayload, updateUrlParams} = args;

  // We're displaying items from offset to offset + count.
  // However, the payload might have fewer than that, so figure out the actual number
  const items = mainPayload.items;
  if (items === undefined) {
    return null;
  }

  const actualCount = items.length;
  const thisRange = (
    <span>
      Showing {renderOffset(offset, updateUrlParams)} : {offset + actualCount} [{renderCount(actualCount, count, updateUrlParams)}]
    </span>
  );

  // Navigate offset with Prev/Next
  const prev = offset > 0 ?
    (<button onClick={() => updateUrlParams({offset: offset - count})}>Prev</button>) :
    <button disabled>Prev</button>;
  const next = actualCount === count ?
    (<button onClick={() => updateUrlParams({offset: offset + count})}>Next</button>) :
    <button disabled>Next</button>;

  return (<div>
    {thisRange} | {prev} {next}
  </div>);
}

function updatePath(paths, index, newPath, updateUrlParams) {
  const newPaths = [...paths.slice(0, index), newPath, ...paths.slice(index + 1)]
  updateUrlParams({paths: JSON.stringify(newPaths)});
}

function removePath(paths, index, updateUrlParams) {
  const newPaths = [...paths.slice(0, index), ...paths.slice(index + 1)]
  updateUrlParams({paths: JSON.stringify(newPaths)});
}

/**
 * Render `paths[index]`
 * Example: "gs://marin-us-west4/tokenized/gpt_neo_tokenizer"
 * Allow user to click on any ancestral path (e.g., tokenized) to navigate to it.
 * 
 * @param {*} args 
 * @returns 
 */
function renderPath(args) {
  const {paths, index, updateUrlParams} = args;
  
  // Find all the indices of '/' (ignoring the gs:// protocol part)
  const path = paths[index];
  const endIndices = [];  // 1:1 correspondence with ancestral paths we can click on
  for (let i = 0; i < path.length; i++) {
    if (path[i] === "/" && (i > 0 && path[i - 1] !== "/") && path[i + 1] !== '/') {
      endIndices.push(i);
    }
  }
  endIndices.push(path.length);

  // For each of these end indices, render the last segment, but link to the
  // entire prefix (`newPath`).
  let prefix = "";
  const parts = endIndices.map((endIndex, i) => {
    const part = path.substring(endIndices[i - 1], endIndices[i]);
    prefix += part;
    const newPath = prefix;  // Need to link to it since prefix is changing
    return <span key={i} className="clickable" onClick={() => updatePath(paths, index, newPath, updateUrlParams)}>{part}</span>;
  });

  return <span className="path">{parts}</span>;
}

/**
 * Show the list of `paths` whose contents we're viewing.
 */
function renderPaths(args) {
  const {paths, updateUrlParams} = args;

  return (<div>
    Paths:
    <ul>
      {paths.map((path, index) => {
        const downloadUrl = path.replace("gs://", "https://storage.cloud.google.com/");
        const downloadLink = <Link to={downloadUrl} title="Download this link.">download</Link>;
        const cloneLink =
          <span className="clickable"
                title="Append this path so you can edit it."
                onClick={() => updatePath(paths, paths.length, path, updateUrlParams)}>clone</span>;
        const hideLink =
          <span className="clickable"
                title="Don't show this path (removing it from the list)."
                onClick={() => removePath(paths, index, updateUrlParams)}>hide</span>;
        const extra = <span> [{downloadLink}] [{cloneLink}] [{hideLink}]</span>;
        return <li key={index}>{renderPath({paths, index, updateUrlParams})}{extra}</li>;
      })}
    </ul>
  </div>);
}

/**
 * Render `paths[index]` as a directory with `files`.
 */
function renderDirectory(args) {
  const {files, paths, index, updateUrlParams} = args;
  return (<div>
    {paths.length > 1 ? renderPath({paths, index, updateUrlParams}) : null}
    <ul>
      {files.map((file, i) => {
        const newPath = "gs://" + file.name;  // TODO: assume gs:// for now, might want to push to backend
        return <li key={i}>
          <span className="clickable" onClick={() => updatePath(paths, index, newPath, updateUrlParams)}>
            {file.name}
          </span>
        </li>;
      })}
    </ul>
  </div>);
}

function renderText(str) {
  return str.split("\n").map((line, i) => <div key={i}>{line}</div>);
}

function isUrl(str) {
  return str.startsWith("http://") || str.startsWith("https://");
}

/**
 * Recursively render `item`, which is an arbitrary JSON object.
 */
function renderItem(item) {
  if (item === null) {
    return "<null>";
  }
  if (typeof item === "string") {
    if (isUrl(item)) {
      return <a href={item} target="_blank">{item}</a>;
    } else {
      return renderText(item);
    }
  }
  if (typeof item === "object") {
    const rows = Object.entries(item).map(([key, value], i) => {
      return (<tr key={i}>
        <td>{key}</td>
        <td>{renderItem(value)}</td>
      </tr>);
    });
    return <table><tbody>{rows}</tbody></table>;
  }

  // Fallback - raw string
  return JSON.stringify(item);
}

/**
 * Main method that renders all the `paths` that have items jointly.  Note that
 * we first go down the items (using the first `path` with items as a guide),
 * and for each item, we render it along with the corresponding items from all
 * the other `paths`.
 */
function renderItems(args) {
  const {paths, payloads, offset, updateUrlParams} = args;

  // Assume each payload is a list of items
  const hasItems = paths.map((path) => payloads[path] && Array.isArray(payloads[path].items));

  const pathIndices = paths.map((path, index) => [path, index]).filter(([path, index]) => hasItems[index]);
  if (pathIndices.length === 0) {
    // No paths with items
    return null;
  }
  const firstItems = payloads[pathIndices[0][0]].items;

  const rows = firstItems.map((_, r) => {
    // Render each payload's r-th item
    const rendered = pathIndices.map(([path, index]) => {
      const items = payloads[path].items;
      return (<div key={index}>
        {paths.length > 1 ? renderPath({paths, index, updateUrlParams}) : null}
        {renderItem(items[r])}
      </div>);
    });
    return (<tr key={r}>
      <td>[{offset + r}]</td>
      <td>{rendered}</td>
    </tr>);
  });
  return <table className="items-table"><tbody>{rows}</tbody></table>;
}

/**
 * Render everything (assume Case 3).
 */
function renderPayloads(args) {
  const {paths, payloads, offset, updateUrlParams} = args;

  const rendered = [];
  
  // Case 2: Render the payloads of each of the paths separately.
  paths.forEach((path, index) => {
    const payload = payloads[path];
    if (!payload) {
      return;
    }
    if (payload.error) {
      rendered.push(renderError(payload.error));
      return;
    }

    if (payload.type === "directory") {
      rendered.push(renderDirectory({files: payload.files, paths, index, updateUrlParams}));
    } else if (payload.type === "json") {
      rendered.push(renderItem(payload.data));
    }
  });

  // Case 1: render all the paths with items in one table.
  rendered.push(renderItems({paths, payloads, offset, updateUrlParams}));

  return rendered.map((item, i) => <div key={i}>{item}</div>);
}