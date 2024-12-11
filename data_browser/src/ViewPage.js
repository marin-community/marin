import React, { useEffect, useState } from 'react';
import axios from 'axios';
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import { useLocation, useNavigate } from 'react-router-dom';
import { apiViewUrl, experimentUrl, renderError, renderLink, renderText, navigateToUrl, isUrl } from './utils';

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

  const pathsResult = parsePaths(urlParams.get("paths"));
  const paths = pathsResult.paths;
  const offset = parseInt(urlParams.get("offset")) || 0;
  const count = parseInt(urlParams.get("count")) || 5;
  const filters = parseFilters(urlParams.get("filters"));
  const sort = parseSort(urlParams.get("sort"));
  const reverse = urlParams.get("reverse");
  const highlights = parseHighlights(urlParams.get("highlights"));
  const showOnlyHighlights = urlParams.get("showOnlyHighlights");

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
    {renderFilters(filters, updateUrlParams)}
    {renderSort(sort, reverse, updateUrlParams)}
    {renderHighlights(highlights, showOnlyHighlights, updateUrlParams)}
    {renderPaths({paths, updateUrlParams})}
    <hr />
    {renderPayloads({paths, payloads, offset, filters, sort, reverse, highlights, showOnlyHighlights, updateUrlParams})}
  </div>);
}

export default ViewPage;

////////////////////////////////////////////////////////////

const upArrow = "↑";
const downArrow = "↓";

/**
 * Parses the URL parameter `paths`.
 *
 * @param {str} str - the paths URL parameter, encoded as a JSON string
 * @returns {paths: string[], error: string}
 */
function parsePaths(str) {
  try {
    const result = JSON.parse(str);
    if (!Array.isArray(result)) {
      return {paths: [], error: "Invalid paths (should be JSON array): " + str};
    }
    return {paths: result};
  } catch (e) {
    return {paths: [], error: "Invalid paths (invalid JSON): " + str};
  }
}

/**
 * Parses the URL parameter `filters`.
 *
 * @param {str} str - the filters URL parameter, encoded as a JSON string
 * @returns {filters: string[], error: string}
 */
function parseFilters(str) {
  if (!str) {
    return {filters: []};
  }
  try {
    const result = JSON.parse(str);
    if (!Array.isArray(result)) {
      return {filters: [], error: "Invalid filters (should be JSON array): " + str};
    }
    return {filters: result};
  } catch (e) {
    return {filters: [], error: "Invalid filters (invalid JSON): " + str};
  }
}

/**
 * Parses the URL parameter `sort`.
 *
 * @param {str} str - the sort URL parameter, encoded as a JSON string
 * @returns {key: string[], error: string}
 */
function parseSort(str) {
  if (!str) {
    return {key: null};
  }
  try {
    const result = JSON.parse(str);
    if (!Array.isArray(result)) {
      return {key: null, error: "Invalid sort (should be JSON array): " + str};
    }
    return {key: result};
  } catch (e) {
    return {key: null, error: "Invalid sort (invalid JSON): " + str};
  }
}

/**
 * Parses the URL parameter `highlights`.
 *
 * @param {str} str - the highlights URL parameter, encoded as a JSON string
 * @returns {highlights: string[], error: string}
 */
function parseHighlights(str) {
  if (!str) {
    return {highlights: []};
  }
  try {
    const result = JSON.parse(str);
    if (!Array.isArray(result)) {
      return {highlights: [], error: "Invalid highlights (should be JSON array): " + str};
    }
    return {highlights: result};
  } catch (e) {
    return {highlights: [], error: "Invalid highlights (invalid JSON): " + str};
  }
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
    (<Button size="small" onClick={() => updateUrlParams({offset: offset - count})}>Prev</Button>) :
    <Button size="small" disabled>Prev</Button>;
  const next = actualCount === count ?
    (<Button size="small" onClick={() => updateUrlParams({offset: offset + count})}>Next</Button>) :
    <Button size="small" disabled>Next</Button>;

  return (<div>
    {thisRange} <ButtonGroup>{prev}{next}</ButtonGroup>
  </div>);
}

function renderKey(key) {
  return "[" + key.join("→") + "]";
}

function removeFromList(name, list, i, updateUrlParams) {
  // Example: name = "filters" or "highlights"
  const newList = [...list.slice(0, i), ...list.slice(i + 1)];
  updateUrlParams({[name]: JSON.stringify(newList)});
}

function renderFilters(filters, updateUrlParams) {
  if (filters.error) {
    return renderError(filters.error);
  }
  if (filters.filters.length === 0) {
    return null;
  }
  return (<Paper className="block">
    Filters
      {filters.filters.map((filter, i) => {
        const label = `${renderKey(filter.key)} ${filter.rel} ${JSON.stringify(filter.value)}`;
        return (<div key={i}>
          <Chip label={label} onDelete={() => removeFromList("filters", filters.filters, i, updateUrlParams)} />
        </div>);
      })}
  </Paper>);
}

function renderSort(sort, reverse, updateUrlParams) {
  if (sort.error) {
    return renderError(sort.error);
  }
  if (!sort.key) {
    return null;
  }
  const label = `${renderKey(sort.key)} ${reverse ? downArrow : upArrow}`;
  return (<Paper className="block">
    Sort
    <div>
      <Chip label={label} onDelete={() => updateUrlParams({sort: null, reverse: null})} />
    </div>
  </Paper>);
}

function renderHighlights(highlights, showOnlyHighlights, updateUrlParams) {
  if (highlights.error) {
    return renderError(highlights.error);
  }
  if (highlights.highlights.length === 0) {
    return null;
  }

  // Toggle showing only highlights or showing all
  const showOnlyButton = showOnlyHighlights ?
    <Button size="small" disabled>Show only highlights</Button> :
    <Button size="small" onClick={() => updateUrlParams({showOnlyHighlights: true})}>Show only highlights</Button>;
  const showAllButton = showOnlyHighlights ?
    <Button size="small" onClick={() => updateUrlParams({showOnlyHighlights: null})}>Show all</Button> :
    <Button size="small" disabled>Show all</Button>;

  return (<Paper className="block">
    Highlights &nbsp;
    <ButtonGroup>{showOnlyButton}{showAllButton}</ButtonGroup>
    <div>
      {highlights.highlights.map((highlight, i) => {
        const label = renderKey(highlight);
        return (<div key={i}>
          <Chip label={label} onDelete={() => removeFromList("highlights", highlights.highlights, i, updateUrlParams)} />
        </div>);
      })}
    </div>
  </Paper>);
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

  function appendNewPath() {
    const newPath = prompt("Enter new path", paths[paths.length - 1]);
    if (!newPath) {
      return;
    }
    updateUrlParams({paths: JSON.stringify([...paths, newPath])});
  }

  const addButton = <Button onClick={appendNewPath}>Add</Button>;

  return (<Paper className="block">
    Paths
    <ul>
      {paths.map((path, index) => {
        const downloadUrl = path.replace("gs://", "https://storage.cloud.google.com/");
        const downloadLink = (
          <Button title="Download this link."
                  size="small"
                  href={downloadUrl}>
            Download
          </Button>
        );
        const cloneButton = (
          <Button title="Add this path again at the end (so you can edit it)."
                  size="small"
                  onClick={() => updatePath(paths, paths.length, path, updateUrlParams)}>
            Clone
          </Button>
        );
        const removeButton = (
          <Button title="Remove this path from the display list (does not delete anything onn disk)."
                  size="small"
                  onClick={() => removePath(paths, index, updateUrlParams)}>
            Remove
          </Button>
        );
        const extra = <ButtonGroup>{downloadLink}{cloneButton}{removeButton}</ButtonGroup>;
        return <li key={index}>{renderPath({paths, index, updateUrlParams})} {extra}</li>;
      })}
    </ul>
    {addButton}
  </Paper>);
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
        const newPath = file.path;
        return <li key={i}>
          <span className="clickable" onClick={() => updatePath(paths, index, newPath, updateUrlParams)}>
            {file.name}
          </span>
        </li>;
      })}
    </ul>
  </div>);
}

/**
 * When click on an sub-item with `itemKey`, we can ask the user to specify
 * sort/filter to that path.  The user types in a space-separated list of tokens,
 * where each token can be:
 * - a filter (e.g., >5, <=0.5, =foo, !=bar, =~baz)
 * - sort (to sort by this key)
 * - asc (to sort in ascending order)
 * - desc (to sort in descending order)
 */
function onItemClick(itemKey, item, oldHighlights, updateUrlParams) {
  const response = prompt("Enter filter (e.g., >5) or 'sort' or 'desc' or 'asc' or 'hi'/'highlight'");
  if (!response) {
    return;
  }

  function matchType(token) {
    if (typeof item === "number") {
      return parseFloat(token);
    }
    return item;
  }

  // Update to URL parameters
  const filters = [];
  let sort = null;
  let reverse = null;
  const highlights = oldHighlights.highlights.slice();

  // The input is a space-separated list of tokens which contribute to filters/sort/reverse.
  response.trim().split(" ").forEach((token) => {
    if (token === "sort") {
      sort = itemKey;
    } else if (token === "asc") {
      reverse = false;
    } else if (token === "desc") {
      reverse = true;
    } else if (token === "highlight" || token === "hi") {
      highlights.push(itemKey);
    } else {
      const m = token.match(/^(<|<=|>|>=|=|!=|=~)(.*)$/);
      if (m) {
        filters.push({key: itemKey, rel: m[1], value: matchType(m[2])});
      } else {
        filters.push({key: itemKey, rel: "=", value: matchType(token)});
      }
    }
  });

  // Package it up into delta
  const delta = {};
  if (filters.length > 0) {
    delta.filters = JSON.stringify(filters);
  }
  if (sort !== null) {
    delta.sort = JSON.stringify(sort);
  }
  if (reverse !== null) {
    delta.reverse = reverse;
  }
  if (highlights.length > 0) {
    delta.highlights = JSON.stringify(highlights);
  }

  // Update the URL parameters
  updateUrlParams(delta);
}

/**
 * Return whether if `highlights` is specified, we want to show at least some
 * things under `itemKey`.
 */
function highlightsMatches(highlights, itemKey) {
  // If no highlights are provided, then it's okay
  if (highlights.highlights.length === 0) {
    return true;
  }

  // Check that some highlight matches in the sense that the itemKey is a prefix
  // of the highlight or vice-versa.
  return highlights.highlights.some((highlight) => {
    const n = Math.min(highlight.length, itemKey.length);
    for (let i = 0; i < n; i++) {
      if (highlight[i] !== itemKey[i]) {
        return false;
      }
    }
    return true;
  });
}

/**
 * Recursively render `item`, which is an arbitrary JSON object.
 * `key` references `item`, so that when people click on part of an item, we can modify the key.
 */
function renderItem(args) {
  const {item, itemKey, highlights, showOnlyHighlights, updateUrlParams} = args;

  if (item === null) {
    return "<null>";
  }

  if (typeof item === "string") {
    if (isUrl(item)) {
      return <a href={item} target="_blank" rel="noreferrer">{item}</a>;
    } else if (item.startsWith("gs://")) {
      return renderLink(item, updateUrlParams);
    } else {
      // Small values we assume can sort by
      const MAX_CLICKABLE_LENGTH = 10;
      const clickable = item.length <= MAX_CLICKABLE_LENGTH ? "clickable" : "";
      return (<div className={clickable} onClick={() => onItemClick(itemKey, item, highlights, updateUrlParams)}>
        {renderText(item)}
      </div>);
    }
  }

  if (typeof item === "number") {
    return (<div className="clickable" onClick={() => onItemClick(itemKey, item, highlights, updateUrlParams)}>
      {item}
    </div>);
  }

  if (typeof item === "object") {
    const rows = Object.entries(item).map(([key, value], i) => {
      const newItemKey = itemKey.concat([key]);

      // Only show if this is consistent with some path in highlight
      if (showOnlyHighlights && !highlightsMatches(highlights, newItemKey)) {
        return null;
      }

      const renderedKey = Array.isArray(item) ? `[${key}]` : key;

      return (<tr key={i}>
        <td>{renderedKey}</td>
        <td>:</td>
        <td>{renderItem({item: value, itemKey: newItemKey, highlights, showOnlyHighlights, updateUrlParams})}</td>
      </tr>);
    });

    return <table className="item-table"><tbody>{rows}</tbody></table>;
  }

  // Fallback - raw string
  return JSON.stringify(item);
}

function isExperiment(item) {
  // Return whether `item` (read from a JSON file) represents an instance of `ExecutorMainConfig`.
  return item.prefix && item.steps;
}

/**
 * Returns whether `item` (an object) matches `filter`.
 * Example: filter = {key: ["a", "b"], rel: "<=", value: 0.5}
 */
function itemMatchesFilter(item, filter) {
  // Base case
  if (filter.key.length === 0) {
    switch (filter.rel) {
      case '=':
        return item === filter.value;
      case '<':
        return typeof item === 'number' && item < filter.value;
      case '>':
        return typeof item === 'number' && item > filter.value;
      case '<=':
        return typeof item === 'number' && item <= filter.value;
      case '>=':
        return typeof item === 'number' && item >= filter.value;
      case '!=':
        return typeof item === 'number' && item !== filter.value;
      case '=~':
        return typeof item === 'string' && item.includes(filter.value);
      default:
        throw new Error(`Unknown rel: ${filter.rel}`);
    }
  }

  // Recurse
  const [first, ...rest] = filter.key;
  if (typeof item === 'object' && item !== null) {
      if (first in item) {
          return itemMatchesFilter(item[first], {...filter, key: rest});
      }
  }
  return false;
}

/**
 * Given a list of `rowIndices`, return the ones that match.
 */
function filterRowIndices(args) {
  const {rowIndices, paths, payloads, filters} = args;
  function rowMatches(r) {
    return filters.filters.every((filter) => {
      // Strip off the first element of `key`, which is the `index`.
      const [index, ...key] = filter.key;
      const payload = payloads[paths[index]];
      return payload && itemMatchesFilter(payload.items[r], {...filter, key});
    });
  }
  return rowIndices.filter(rowMatches);
}

/**
 * Return `rowIndices` sorted by what is specified `sort` (which acesses items in `payloads`).
 */
function sortRowIndices(args) {
  const {rowIndices, paths, payloads, sort} = args;

  // First element is the index of the payload
  if (!sort.key) {
    return rowIndices;
  }
  const [index, ...key] = sort.key;

  // Check that the `index`-th payload has items, which are used to get values
  const payload = payloads[paths[index]];
  const items = payload && payload.items;
  if (!items) {
    return rowIndices;
  }

  // Traverse `item` with `key` to get the subpart (usually a leaf value)
  function getSortValue(item, key) {
    if (key.length === 0) {
      return item;
    }
    const [first, ...rest] = key;
    return item[first] && getSortValue(item[first], rest);
  }

  // Sort the rows by the values
  return rowIndices.slice().sort((r1, r2) => {
    const value1 = getSortValue(items[r1], key);
    const value2 = getSortValue(items[r2], key);
    if (typeof value1 === 'number' && typeof value2 === 'number') {
      return value1 - value2;
    }
    if (typeof value1 === 'string' && typeof value2 === 'string') {
      return value1.localeCompare(value2);
    }
    return 0;  // Values have different type, don't know how to compare so give up
  });
}


/**
 * Main method that renders all the `paths` that have items jointly.  Note that
 * we first go down the items (using the first `path` with items as a guide),
 * and for each item, we render it along with the corresponding items from all
 * the other `paths`.
 */
function renderItems(args) {
  const {paths, payloads, offset, filters, sort, reverse, highlights, showOnlyHighlights, updateUrlParams} = args;

  // Assume each payload is a list of items
  const hasItems = paths.map((path) => payloads[path] && Array.isArray(payloads[path].items));

  const pathIndices = paths.map((path, index) => [path, index]).filter(([path, index]) => hasItems[index]);
  if (pathIndices.length === 0) {
    // No paths with items
    return null;
  }
  const firstItems = payloads[pathIndices[0][0]].items;

  // Filter and sort the row indices
  let rowIndices = firstItems.map((_, r) => r);
  rowIndices = filterRowIndices({rowIndices, paths, payloads, filters});
  rowIndices = sortRowIndices({rowIndices, paths, payloads, sort});
  if (reverse) {
    rowIndices = rowIndices.slice().reverse();
  }

  // Render the rows
  const rows = rowIndices.map((r) => {
    // Render each payload's r-th item
    const rendered = pathIndices.map(([path, index]) => {
      const items = payloads[path].items;
      return (<div key={index}>
        {paths.length > 1 ? renderPath({paths, index, updateUrlParams}) : null}
        {renderItem({item: items[r], itemKey: [index], highlights, showOnlyHighlights, updateUrlParams})}
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
  const {paths, payloads, offset, filters, sort, reverse, highlights, showOnlyHighlights, updateUrlParams} = args;

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
      let item = renderItem({item: payload.data, highlights, itemKey: [], updateUrlParams});
      if (isExperiment(payload.data)) {
        item = (<div>
          <Button variant="contained" size="small" href={experimentUrl({path})}>Go to experiment view</Button>
          {item}
        </div>);
      }
      rendered.push(item);
    }
  });

  // Case 1: render all the paths with items in one table.
  const table = renderItems({paths, payloads, offset, filters, sort, reverse, highlights, showOnlyHighlights, updateUrlParams});
  if (table) {
    rendered.push(table);
  }

  return rendered.map((item, i) => <Paper className="block" key={i}>{item}</Paper>);
}
